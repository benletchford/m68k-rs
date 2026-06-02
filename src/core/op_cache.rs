//! Decoded simple operation cache.
//!
//! This is the first JIT-facing execution substrate: cache a small decoded micro-op for
//! simple one-word instructions that do not touch memory, read extension words, trap, or need
//! rollback. One-word short branches are included because their fetch timing is fully local to
//! the current instruction.
//! Instruction fetch still occurs at the normal instruction boundary, so bus/address-error timing
//! stays aligned with the interpreter.

use super::cpu::CpuCore;
use super::execute::{RUN_MODE_BERR_AERR_RESET, RUN_MODE_NORMAL};
use super::memory::AddressBus;
use super::types::{CpuType, Size};

pub(crate) const DECODED_OP_CACHE_SIZE: usize = 4096;

#[derive(Debug, Clone, Copy)]
pub(crate) struct DecodedOpCacheEntry {
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
    op: DecodedSimpleOp,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DecodedSimpleOp {
    Nop,
    MoveReg {
        src: DirectReg,
        dst: DirectReg,
        size: Size,
    },
    Moveq {
        reg: u8,
        data: u32,
    },
    UnaryDataReg {
        op: UnaryOp,
        reg: u8,
        size: Size,
    },
    Swap {
        reg: u8,
    },
    Ext {
        reg: u8,
        size: Size,
    },
    Extb {
        reg: u8,
    },
    AddqSubqReg {
        reg: u8,
        data: u32,
        size: Size,
        is_sub: bool,
    },
    AddqSubqAddr {
        reg: u8,
        data: u32,
        is_sub: bool,
    },
    BinaryDataReg {
        op: BinaryOp,
        src: DirectReg,
        dst: u8,
        size: Size,
        cycles: i32,
    },
    AddrDataReg {
        op: AddrOp,
        src: DirectReg,
        dst: u8,
        size: Size,
    },
    AddSubxReg {
        src: u8,
        dst: u8,
        size: Size,
        is_sub: bool,
    },
    BitReg {
        op: BitOp,
        bit_reg: u8,
        dst: u8,
    },
    BcdReg {
        src: u8,
        dst: u8,
        is_sub: bool,
    },
    Exg {
        opcode: u16,
    },
    SccDataReg {
        condition: u8,
        reg: u8,
    },
    ShiftReg {
        reg: u8,
        size: Size,
        count_or_reg: u8,
        count_is_register: bool,
        direction: u8,
        op: u8,
    },
    BranchShort {
        condition: u8,
        displacement: i8,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum DirectReg {
    Data(u8),
    Addr(u8),
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum UnaryOp {
    Clr,
    Neg,
    Negx,
    Not,
    Tst,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BinaryOp {
    Add,
    Sub,
    And,
    Or,
    Eor,
    Cmp,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum AddrOp {
    Adda,
    Suba,
    Cmpa,
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum BitOp {
    Test,
    Change,
    Clear,
    Set,
}

pub(crate) enum CachedRunResult {
    Ran,
    Miss(u16),
    Fault,
}

impl DecodedSimpleOp {
    #[inline]
    pub(crate) fn decode(cpu_type: CpuType, opcode: u16) -> Option<Self> {
        let group = (opcode >> 12) & 0xF;

        if (1..=3).contains(&group) {
            return decode_move_reg(
                opcode,
                match group {
                    1 => Size::Byte,
                    2 => Size::Long,
                    3 => Size::Word,
                    _ => unreachable!(),
                },
            );
        }

        if group == 0x7 {
            return Some(Self::Moveq {
                reg: ((opcode >> 9) & 7) as u8,
                data: (opcode & 0xFF) as i8 as i32 as u32,
            });
        }

        if opcode == 0x4E71 {
            return Some(Self::Nop);
        }

        if group == 0x4
            && let Some(op) = decode_group_4_reg(cpu_type, opcode)
        {
            return Some(op);
        }

        if group == 0x0 && (opcode & 0x0100) != 0 && ((opcode >> 3) & 7) == 0 {
            let op = match (opcode >> 6) & 3 {
                0 => BitOp::Test,
                1 => BitOp::Change,
                2 => BitOp::Clear,
                3 => BitOp::Set,
                _ => unreachable!(),
            };
            return Some(Self::BitReg {
                op,
                bit_reg: ((opcode >> 9) & 7) as u8,
                dst: (opcode & 7) as u8,
            });
        }

        if (opcode & 0xFFF8) == 0x4840 {
            return Some(Self::Swap {
                reg: (opcode & 7) as u8,
            });
        }

        if (opcode & 0xFFF8) == 0x4880 {
            return Some(Self::Ext {
                reg: (opcode & 7) as u8,
                size: Size::Word,
            });
        }

        if (opcode & 0xFFF8) == 0x48C0 {
            return Some(Self::Ext {
                reg: (opcode & 7) as u8,
                size: Size::Long,
            });
        }

        if (opcode & 0xFFF8) == 0x49C0 && !is_pre_68020(cpu_type) {
            return Some(Self::Extb {
                reg: (opcode & 7) as u8,
            });
        }

        if group == 0x5 && ((opcode >> 6) & 3) != 3 {
            let ea_mode = (opcode >> 3) & 7;
            if ea_mode <= 1 {
                let data = ((opcode >> 9) & 7) as u32;
                let data = if data == 0 { 8 } else { data };
                let is_sub = (opcode & 0x100) != 0;
                let reg = (opcode & 7) as u8;
                if ea_mode == 1 {
                    return Some(Self::AddqSubqAddr { reg, data, is_sub });
                }
                return Some(Self::AddqSubqReg {
                    reg,
                    data,
                    size: decode_size_00((opcode >> 6) & 3),
                    is_sub,
                });
            }
        }

        if group == 0x5 && ((opcode >> 6) & 3) == 3 && ((opcode >> 3) & 7) == 0 {
            return Some(Self::SccDataReg {
                condition: ((opcode >> 8) & 0xF) as u8,
                reg: (opcode & 7) as u8,
            });
        }

        if matches!(group, 0x8 | 0x9 | 0xB | 0xC | 0xD)
            && let Some(op) = decode_group_alu_reg(opcode)
        {
            return Some(op);
        }

        if group == 0xE && (opcode & 0x00C0) != 0x00C0 {
            return Some(Self::ShiftReg {
                reg: (opcode & 7) as u8,
                size: decode_size_00((opcode >> 6) & 3),
                count_or_reg: ((opcode >> 9) & 7) as u8,
                count_is_register: (opcode & 0x20) != 0,
                direction: ((opcode >> 8) & 1) as u8,
                op: ((opcode >> 3) & 3) as u8,
            });
        }

        if group == 0x6 {
            let condition = ((opcode >> 8) & 0xF) as u8;
            let displacement = (opcode & 0xFF) as u8;
            if condition != 1 && displacement != 0 && displacement != 0xFF {
                return Some(Self::BranchShort {
                    condition,
                    displacement: displacement as i8,
                });
            }
        }

        None
    }

    #[inline]
    pub(crate) fn execute(self, cpu: &mut CpuCore) -> i32 {
        match self {
            Self::Nop => 4,
            Self::MoveReg { src, dst, size } => {
                let value = read_direct_reg(cpu, src, size);
                match dst {
                    DirectReg::Data(reg) => {
                        let reg = reg as usize;
                        write_data_reg(cpu, reg, size, value);
                        cpu.set_logic_flags(value, size);
                    }
                    DirectReg::Addr(reg) => {
                        let reg = reg as usize;
                        let value = if size == Size::Word {
                            value as i16 as i32 as u32
                        } else {
                            value
                        };
                        cpu.dar[8 + reg] = value;
                    }
                }
                4
            }
            Self::Moveq { reg, data } => {
                let reg = reg as usize;
                cpu.dar[reg] = data;
                cpu.n_flag = if (data as i32) < 0 { 0x80 } else { 0 };
                cpu.not_z_flag = data;
                cpu.v_flag = 0;
                cpu.c_flag = 0;
                4
            }
            Self::UnaryDataReg { op, reg, size } => {
                let reg = reg as usize;
                let mask = size.mask();
                let src = cpu.dar[reg] & mask;
                match op {
                    UnaryOp::Clr => {
                        write_data_reg(cpu, reg, size, 0);
                        cpu.n_flag = 0;
                        cpu.not_z_flag = 0;
                        cpu.v_flag = 0;
                        cpu.c_flag = 0;
                    }
                    UnaryOp::Neg => {
                        let result = 0u32.wrapping_sub(src);
                        write_data_reg(cpu, reg, size, result);
                        cpu.set_sub_flags(src, 0, result, size);
                    }
                    UnaryOp::Negx => {
                        let result = cpu.exec_subx(size, src, 0);
                        write_data_reg(cpu, reg, size, result);
                    }
                    UnaryOp::Not => {
                        let result = !src & mask;
                        write_data_reg(cpu, reg, size, result);
                        cpu.set_logic_flags(result, size);
                    }
                    UnaryOp::Tst => {
                        cpu.set_logic_flags(src, size);
                    }
                }
                4
            }
            Self::Swap { reg } => cpu.exec_swap(reg as usize),
            Self::Ext { reg, size } => cpu.exec_ext(size, reg as usize),
            Self::Extb { reg } => cpu.exec_extb(reg as usize),
            Self::AddqSubqAddr { reg, data, is_sub } => {
                let reg = 8 + reg as usize;
                if is_sub {
                    cpu.dar[reg] = cpu.dar[reg].wrapping_sub(data);
                } else {
                    cpu.dar[reg] = cpu.dar[reg].wrapping_add(data);
                }
                4
            }
            Self::AddqSubqReg {
                reg,
                data,
                size,
                is_sub,
            } => {
                let reg = reg as usize;
                let mask = size.mask();
                let dst = cpu.dar[reg] & mask;
                let result = if is_sub {
                    let result = dst.wrapping_sub(data);
                    cpu.set_sub_flags(data, dst, result, size);
                    result & mask
                } else {
                    let result = dst.wrapping_add(data);
                    cpu.set_add_flags(data, dst, result, size);
                    result & mask
                };
                cpu.dar[reg] = (cpu.dar[reg] & !mask) | result;
                4
            }
            Self::BinaryDataReg {
                op,
                src,
                dst,
                size,
                cycles,
            } => {
                let dst = dst as usize;
                let mask = size.mask();
                let src = read_direct_reg(cpu, src, size);
                let dst_value = cpu.dar[dst] & mask;
                match op {
                    BinaryOp::Add => {
                        let result = dst_value.wrapping_add(src);
                        cpu.set_add_flags(src, dst_value, result, size);
                        write_data_reg(cpu, dst, size, result);
                    }
                    BinaryOp::Sub => {
                        let result = dst_value.wrapping_sub(src);
                        cpu.set_sub_flags(src, dst_value, result, size);
                        write_data_reg(cpu, dst, size, result);
                    }
                    BinaryOp::And => {
                        let result = (src & dst_value) & mask;
                        cpu.set_logic_flags(result, size);
                        write_data_reg(cpu, dst, size, result);
                    }
                    BinaryOp::Or => {
                        let result = (src | dst_value) & mask;
                        cpu.set_logic_flags(result, size);
                        write_data_reg(cpu, dst, size, result);
                    }
                    BinaryOp::Eor => {
                        let result = (src ^ dst_value) & mask;
                        cpu.set_logic_flags(result, size);
                        write_data_reg(cpu, dst, size, result);
                    }
                    BinaryOp::Cmp => {
                        let result = dst_value.wrapping_sub(src);
                        cpu.set_cmp_flags(src, dst_value, result, size);
                    }
                }
                cycles
            }
            Self::AddrDataReg { op, src, dst, size } => {
                let dst = dst as usize;
                let mut src = read_direct_reg(cpu, src, size);
                if size == Size::Word {
                    src = src as i16 as i32 as u32;
                }
                let dst_value = cpu.dar[8 + dst];
                match op {
                    AddrOp::Adda => {
                        cpu.dar[8 + dst] = dst_value.wrapping_add(src);
                        8
                    }
                    AddrOp::Suba => {
                        cpu.dar[8 + dst] = dst_value.wrapping_sub(src);
                        8
                    }
                    AddrOp::Cmpa => {
                        let result = dst_value.wrapping_sub(src);
                        cpu.set_cmp_flags(src, dst_value, result, Size::Long);
                        6
                    }
                }
            }
            Self::AddSubxReg {
                src,
                dst,
                size,
                is_sub,
            } => {
                let src = src as usize;
                let dst = dst as usize;
                let mask = size.mask();
                let src = cpu.dar[src] & mask;
                let dst_value = cpu.dar[dst] & mask;
                let result = if is_sub {
                    cpu.exec_subx(size, src, dst_value)
                } else {
                    cpu.exec_addx(size, src, dst_value)
                };
                write_data_reg(cpu, dst, size, result);
                4
            }
            Self::BitReg { op, bit_reg, dst } => {
                let bit = cpu.dar[bit_reg as usize] & 31;
                let mask = 1u32 << bit;
                let dst = dst as usize;
                let value = cpu.dar[dst];
                cpu.not_z_flag = if value & mask != 0 { 1 } else { 0 };
                match op {
                    BitOp::Test => 6,
                    BitOp::Change => {
                        cpu.dar[dst] = value ^ mask;
                        8
                    }
                    BitOp::Clear => {
                        cpu.dar[dst] = value & !mask;
                        10
                    }
                    BitOp::Set => {
                        cpu.dar[dst] = value | mask;
                        8
                    }
                }
            }
            Self::BcdReg { src, dst, is_sub } => {
                if is_sub {
                    cpu.exec_sbcd_rr(src as usize, dst as usize)
                } else {
                    cpu.exec_abcd_rr(src as usize, dst as usize)
                }
            }
            Self::Exg { opcode } => cpu.exec_exg(opcode),
            Self::SccDataReg { condition, reg } => {
                let reg = reg as usize;
                let value = if cpu.test_condition(condition) {
                    0xFF
                } else {
                    0
                };
                write_data_reg(cpu, reg, Size::Byte, value);
                4
            }
            Self::ShiftReg {
                reg,
                size,
                count_or_reg,
                count_is_register,
                direction,
                op,
            } => {
                let reg = reg as usize;
                let shift = if count_is_register {
                    cpu.dar[count_or_reg as usize] & 63
                } else {
                    let c = count_or_reg as u32;
                    if c == 0 { 8 } else { c }
                };
                let value = cpu.dar[reg] & size.mask();
                let (result, cycles) = match (op, direction) {
                    (0, 0) => cpu.exec_asr(size, shift, value),
                    (0, 1) => cpu.exec_asl(size, shift, value),
                    (1, 0) => cpu.exec_lsr(size, shift, value),
                    (1, 1) => cpu.exec_lsl(size, shift, value),
                    (2, 0) => cpu.exec_roxr(size, shift, value),
                    (2, 1) => cpu.exec_roxl(size, shift, value),
                    (3, 0) => cpu.exec_ror(size, shift, value),
                    (3, 1) => cpu.exec_rol(size, shift, value),
                    _ => unreachable!(),
                };
                let mask = size.mask();
                cpu.dar[reg] = (cpu.dar[reg] & !mask) | result;
                cycles
            }
            Self::BranchShort {
                condition,
                displacement,
            } => {
                if condition == 0 || cpu.test_condition(condition) {
                    cpu.change_of_flow = true;
                    cpu.pc = (cpu.pc as i32).wrapping_add(displacement as i32) as u32;
                    10
                } else {
                    8
                }
            }
        }
    }
}

#[inline]
fn decode_move_reg(opcode: u16, size: Size) -> Option<DecodedSimpleOp> {
    let src = direct_reg((opcode >> 3) & 7, opcode & 7)?;
    let dst_reg = ((opcode >> 9) & 7) as u8;
    let dst_mode = (opcode >> 6) & 7;

    let dst = match dst_mode {
        0 => DirectReg::Data(dst_reg),
        1 if size != Size::Byte => DirectReg::Addr(dst_reg),
        _ => return None,
    };

    Some(DecodedSimpleOp::MoveReg { src, dst, size })
}

#[inline]
fn decode_group_4_reg(_cpu_type: CpuType, opcode: u16) -> Option<DecodedSimpleOp> {
    let ea_mode = (opcode >> 3) & 7;
    if ea_mode != 0 {
        return None;
    }

    let size_bits = (opcode >> 6) & 3;
    if size_bits == 3 {
        return None;
    }

    let op = match (opcode >> 8) & 0xF {
        0x0 => UnaryOp::Negx,
        0x2 => UnaryOp::Clr,
        0x4 => UnaryOp::Neg,
        0x6 => UnaryOp::Not,
        0xA => UnaryOp::Tst,
        _ => return None,
    };

    Some(DecodedSimpleOp::UnaryDataReg {
        op,
        reg: (opcode & 7) as u8,
        size: decode_size_00(size_bits),
    })
}

#[inline]
fn decode_group_alu_reg(opcode: u16) -> Option<DecodedSimpleOp> {
    let group = (opcode >> 12) & 0xF;
    let reg = ((opcode >> 9) & 7) as u8;
    let ea_mode = (opcode >> 3) & 7;
    let ea_reg = (opcode & 7) as u8;
    let op_mode = (opcode >> 6) & 7;
    let src = direct_reg(ea_mode, opcode & 7);

    match group {
        0x8 => {
            if op_mode <= 2 {
                Some(DecodedSimpleOp::BinaryDataReg {
                    op: BinaryOp::Or,
                    src: src?,
                    dst: reg,
                    size: decode_size_012(op_mode),
                    cycles: 4,
                })
            } else if op_mode == 4 && ea_mode == 0 {
                Some(DecodedSimpleOp::BcdReg {
                    src: ea_reg,
                    dst: reg,
                    is_sub: true,
                })
            } else {
                None
            }
        }
        0x9 => match op_mode {
            0..=2 => Some(DecodedSimpleOp::BinaryDataReg {
                op: BinaryOp::Sub,
                src: src?,
                dst: reg,
                size: decode_size_012(op_mode),
                cycles: 4,
            }),
            3 | 7 => Some(DecodedSimpleOp::AddrDataReg {
                op: AddrOp::Suba,
                src: src?,
                dst: reg,
                size: if op_mode == 3 { Size::Word } else { Size::Long },
            }),
            4..=6 if ea_mode == 0 => Some(DecodedSimpleOp::AddSubxReg {
                src: ea_reg,
                dst: reg,
                size: decode_size_012(op_mode - 4),
                is_sub: true,
            }),
            _ => None,
        },
        0xB => match op_mode {
            0..=2 => Some(DecodedSimpleOp::BinaryDataReg {
                op: BinaryOp::Cmp,
                src: src?,
                dst: reg,
                size: decode_size_012(op_mode),
                cycles: 4,
            }),
            3 | 7 => Some(DecodedSimpleOp::AddrDataReg {
                op: AddrOp::Cmpa,
                src: src?,
                dst: reg,
                size: if op_mode == 3 { Size::Word } else { Size::Long },
            }),
            4..=6 if ea_mode == 0 => Some(DecodedSimpleOp::BinaryDataReg {
                op: BinaryOp::Eor,
                src: DirectReg::Data(reg),
                dst: ea_reg,
                size: decode_size_012(op_mode - 4),
                cycles: 8,
            }),
            _ => None,
        },
        0xC => {
            if op_mode <= 2 {
                return Some(DecodedSimpleOp::BinaryDataReg {
                    op: BinaryOp::And,
                    src: src?,
                    dst: reg,
                    size: decode_size_012(op_mode),
                    cycles: 4,
                });
            }

            if op_mode == 4 && ea_mode == 0 {
                return Some(DecodedSimpleOp::BcdReg {
                    src: ea_reg,
                    dst: reg,
                    is_sub: false,
                });
            }

            let mode_field = (opcode >> 3) & 0x1F;
            if matches!(mode_field, 0x08 | 0x09 | 0x11) {
                Some(DecodedSimpleOp::Exg { opcode })
            } else {
                None
            }
        }
        0xD => match op_mode {
            0..=2 => Some(DecodedSimpleOp::BinaryDataReg {
                op: BinaryOp::Add,
                src: src?,
                dst: reg,
                size: decode_size_012(op_mode),
                cycles: 4,
            }),
            3 | 7 => Some(DecodedSimpleOp::AddrDataReg {
                op: AddrOp::Adda,
                src: src?,
                dst: reg,
                size: if op_mode == 3 { Size::Word } else { Size::Long },
            }),
            4..=6 if ea_mode == 0 => Some(DecodedSimpleOp::AddSubxReg {
                src: ea_reg,
                dst: reg,
                size: decode_size_012(op_mode - 4),
                is_sub: false,
            }),
            _ => None,
        },
        _ => None,
    }
}

impl CpuCore {
    #[inline]
    pub(crate) fn clear_decoded_op_cache(&mut self) {
        for entry in self.decoded_op_cache.iter_mut() {
            *entry = None;
        }
    }

    #[inline]
    pub(crate) fn can_run_decoded_simple_ops(&self) -> bool {
        self.run_mode == RUN_MODE_NORMAL
            && self.stopped == 0
            && self.int_level == 0
            && self.t1_flag == 0
            && self.t0_flag == 0
    }

    pub(crate) fn execute_decoded_simple_run<B: AddressBus>(
        &mut self,
        bus: &mut B,
    ) -> CachedRunResult {
        let cpu_type = self.cpu_type;

        while self.cycles_remaining > 0 {
            self.ppc = self.pc;
            let opcode = self.read_opcode_16(bus);
            if self.run_mode == RUN_MODE_BERR_AERR_RESET {
                return CachedRunResult::Fault;
            }
            self.ir = opcode as u32;

            let Some(op) = self.decoded_simple_op(self.ppc, opcode, cpu_type) else {
                return CachedRunResult::Miss(opcode);
            };
            let cycles = op.execute(self);
            self.cycles_remaining -= cycles;
        }

        CachedRunResult::Ran
    }

    #[inline]
    fn decoded_simple_op(
        &mut self,
        pc: u32,
        opcode: u16,
        cpu_type: CpuType,
    ) -> Option<DecodedSimpleOp> {
        let idx = decoded_op_cache_index(pc);
        if let Some(entry) = self.decoded_op_cache[idx]
            && entry.pc == pc
            && entry.opcode == opcode
            && entry.cpu_type == cpu_type
        {
            return Some(entry.op);
        }

        let op = DecodedSimpleOp::decode(cpu_type, opcode)?;
        self.decoded_op_cache[idx] = Some(DecodedOpCacheEntry {
            pc,
            opcode,
            cpu_type,
            op,
        });
        Some(op)
    }
}

#[inline]
pub(crate) fn decoded_op_cache_index(pc: u32) -> usize {
    ((pc >> 1) as usize) & (DECODED_OP_CACHE_SIZE - 1)
}

#[inline]
fn decode_size_00(bits: u16) -> Size {
    match bits {
        0 => Size::Byte,
        1 => Size::Word,
        2 => Size::Long,
        _ => Size::Byte,
    }
}

#[inline]
fn decode_size_012(bits: u16) -> Size {
    match bits {
        0 => Size::Byte,
        1 => Size::Word,
        2 => Size::Long,
        _ => Size::Byte,
    }
}

#[inline]
fn direct_reg(mode: u16, reg: u16) -> Option<DirectReg> {
    match mode {
        0 => Some(DirectReg::Data(reg as u8)),
        1 => Some(DirectReg::Addr(reg as u8)),
        _ => None,
    }
}

#[inline]
fn read_direct_reg(cpu: &CpuCore, reg: DirectReg, size: Size) -> u32 {
    match reg {
        DirectReg::Data(reg) => cpu.dar[reg as usize] & size.mask(),
        DirectReg::Addr(reg) => cpu.dar[8 + reg as usize] & size.mask(),
    }
}

#[inline]
fn write_data_reg(cpu: &mut CpuCore, reg: usize, size: Size, value: u32) {
    let mask = size.mask();
    cpu.dar[reg] = (cpu.dar[reg] & !mask) | (value & mask);
}

#[inline]
fn is_pre_68020(cpu_type: CpuType) -> bool {
    matches!(
        cpu_type,
        CpuType::M68000 | CpuType::M68010 | CpuType::SCC68070
    )
}

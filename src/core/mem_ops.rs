//! Fastmem-decoded memory-operand instructions for the batch loop.
//!
//! These ops execute against a contiguous RAM window exposed by the bus
//! ([`AddressBus::fast_mem`](super::memory::AddressBus::fast_mem)). They are
//! used only by the instruction-budgeted [`CpuCore::run_batch`] loop: the
//! cycle-budgeted paths never populate the window, so cache entries tagged
//! `Mem` fall back to full dispatch there and cycle accounting is unaffected.
//!
//! Correctness rules:
//! - An op either executes completely or reports "not handled" with **zero**
//!   state changes (registers, flags, memory, PC) — the caller then runs the
//!   same opcode through full dispatch, which handles faults, address errors
//!   and cycle-exact semantics.
//! - All guards (window range, pre-68020 alignment) run before any commit.
//!   Window reads are side-effect-free per the bus contract, so a read
//!   followed by fallback is harmless.
//! - Effective addresses are masked with `CpuCore::address` before the
//!   window check, exactly like bus accesses in the interpreter.

use super::cpu::CpuCore;
use super::op_cache::{AddrOp, BinaryOp, BitOp, is_pre_68020};
use super::types::{CpuType, Size};

/// Effective-address forms the fastmem path understands. Everything else
/// (68020 full-format indexing, register pairs, …) falls back to dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FastEa {
    DataReg(u8),
    AddrReg(u8),
    /// (An)
    AnInd(u8),
    /// (An)+
    AnPostInc(u8),
    /// -(An)
    AnPreDec(u8),
    /// (d16,An)
    AnDisp(u8),
    /// (d8,An,Xn) — brief extension word only
    AnIndex(u8),
    /// (xxx).W
    AbsW,
    /// (xxx).L
    AbsL,
    /// (d16,PC)
    PcDisp,
    /// (d8,PC,Xn) — brief extension word only
    PcIndex,
    /// #imm
    Imm,
}

impl FastEa {
    #[inline]
    fn decode(mode: u16, reg: u16) -> Option<FastEa> {
        Some(match mode & 7 {
            0 => FastEa::DataReg(reg as u8),
            1 => FastEa::AddrReg(reg as u8),
            2 => FastEa::AnInd(reg as u8),
            3 => FastEa::AnPostInc(reg as u8),
            4 => FastEa::AnPreDec(reg as u8),
            5 => FastEa::AnDisp(reg as u8),
            6 => FastEa::AnIndex(reg as u8),
            7 => match reg & 7 {
                0 => FastEa::AbsW,
                1 => FastEa::AbsL,
                2 => FastEa::PcDisp,
                3 => FastEa::PcIndex,
                4 => FastEa::Imm,
                _ => return None,
            },
            _ => unreachable!(),
        })
    }

    /// Data-alterable EA: legal destination for MOVE/CLR/ALU-to-mem/….
    #[inline]
    fn is_data_alterable(self) -> bool {
        !matches!(
            self,
            FastEa::AddrReg(_) | FastEa::PcDisp | FastEa::PcIndex | FastEa::Imm
        )
    }

    /// Memory EA usable as a control address (LEA/PEA/JMP/JSR).
    #[inline]
    fn is_control(self) -> bool {
        matches!(
            self,
            FastEa::AnInd(_)
                | FastEa::AnDisp(_)
                | FastEa::AnIndex(_)
                | FastEa::AbsW
                | FastEa::AbsL
                | FastEa::PcDisp
                | FastEa::PcIndex
        )
    }

    #[inline]
    fn is_memory(self) -> bool {
        !matches!(self, FastEa::DataReg(_) | FastEa::AddrReg(_) | FastEa::Imm)
    }
}

/// Bit-number source for the BTST/BCHG/BCLR/BSET memory forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BitSource {
    Reg(u8),
    /// Bit number in an immediate extension word (read before the EA words).
    Imm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecodedMemOp {
    /// MOVE/MOVEA. `dst == AddrReg` is MOVEA (sign-extends word, no flags).
    Move {
        size: Size,
        src: FastEa,
        dst: FastEa,
    },
    /// ADD/SUB/AND/OR/CMP `<ea>,Dn`
    AluToReg {
        op: BinaryOp,
        size: Size,
        src: FastEa,
        dst: u8,
    },
    /// ADD/SUB/AND/OR/EOR `Dn,<ea>` (memory-alterable EA)
    AluToMem {
        op: BinaryOp,
        size: Size,
        src: u8,
        dst: FastEa,
    },
    /// ADDA/SUBA/CMPA `<ea>,An`
    AluAddr {
        op: AddrOp,
        size: Size,
        src: FastEa,
        dst: u8,
    },
    /// ADDI/SUBI/ANDI/ORI/EORI/CMPI `#imm,<ea>`
    AluImm {
        op: BinaryOp,
        size: Size,
        dst: FastEa,
    },
    /// ADDQ/SUBQ `#q,<ea>` (memory EA; register forms are simple ops)
    AddqSubq {
        data: u32,
        size: Size,
        ea: FastEa,
        is_sub: bool,
    },
    Tst {
        size: Size,
        ea: FastEa,
    },
    Clr {
        size: Size,
        ea: FastEa,
    },
    Neg {
        size: Size,
        ea: FastEa,
    },
    Not {
        size: Size,
        ea: FastEa,
    },
    /// BTST/BCHG/BCLR/BSET on a memory byte.
    BitMem {
        op: BitOp,
        bit: BitSource,
        ea: FastEa,
    },
    /// CMPM (Ay)+,(Ax)+
    CmpM {
        size: Size,
        src: u8,
        dst: u8,
    },
    Lea {
        reg: u8,
        ea: FastEa,
    },
    Pea {
        ea: FastEa,
    },
    Jmp {
        ea: FastEa,
    },
    Jsr {
        ea: FastEa,
    },
    Rts,
    /// BSR with byte or word displacement (`length` covers opcode + ext).
    Bsr {
        displacement: i32,
        length: u32,
    },
    /// Bcc/BRA with word displacement.
    BranchWord {
        condition: u8,
    },
    Dbcc {
        condition: u8,
        reg: u8,
    },
}

// ============================================================================
// Decode
// ============================================================================

impl DecodedMemOp {
    pub(crate) fn decode(cpu_type: CpuType, opcode: u16) -> Option<Self> {
        let group = (opcode >> 12) & 0xF;
        match group {
            0x0 => decode_group_0(opcode),
            0x1 => decode_move(opcode, Size::Byte),
            0x2 => decode_move(opcode, Size::Long),
            0x3 => decode_move(opcode, Size::Word),
            0x4 => decode_group_4(cpu_type, opcode),
            0x5 => decode_group_5(opcode),
            0x6 => decode_group_6(opcode),
            0x8 | 0x9 | 0xB | 0xC | 0xD => decode_alu(opcode),
            _ => None,
        }
    }
}

#[inline]
fn decode_size_00(bits: u16) -> Option<Size> {
    match bits {
        0 => Some(Size::Byte),
        1 => Some(Size::Word),
        2 => Some(Size::Long),
        _ => None,
    }
}

fn decode_move(opcode: u16, size: Size) -> Option<DecodedMemOp> {
    let src = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
    let dst = FastEa::decode((opcode >> 6) & 7, (opcode >> 9) & 7)?;

    // An as byte source is illegal; MOVEA.B does not exist.
    if size == Size::Byte
        && (matches!(src, FastEa::AddrReg(_)) || matches!(dst, FastEa::AddrReg(_)))
    {
        return None;
    }
    if !dst.is_data_alterable() && !matches!(dst, FastEa::AddrReg(_)) {
        return None;
    }
    // Register-to-register MOVEs are covered by the simple-op path; only
    // accept forms that touch memory or read extension/immediate words.
    Some(DecodedMemOp::Move { size, src, dst })
}

fn decode_group_0(opcode: u16) -> Option<DecodedMemOp> {
    // Dynamic bit ops: 0000 rrr1 xxmm mrrr with memory EA.
    if (opcode & 0x0100) != 0 {
        let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
        // MOVEP shares this encoding with mode 1 (An).
        if !ea.is_memory() || !ea.is_data_alterable() {
            return None;
        }
        let op = match (opcode >> 6) & 3 {
            0 => BitOp::Test,
            1 => BitOp::Change,
            2 => BitOp::Clear,
            3 => BitOp::Set,
            _ => unreachable!(),
        };
        return Some(DecodedMemOp::BitMem {
            op,
            bit: BitSource::Reg(((opcode >> 9) & 7) as u8),
            ea,
        });
    }

    // Static bit ops: 0000 1000 xxmm mrrr (#imm form).
    if (opcode & 0xFF00) == 0x0800 {
        let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
        if !ea.is_memory() || !ea.is_data_alterable() {
            return None;
        }
        let op = match (opcode >> 6) & 3 {
            0 => BitOp::Test,
            1 => BitOp::Change,
            2 => BitOp::Clear,
            3 => BitOp::Set,
            _ => unreachable!(),
        };
        return Some(DecodedMemOp::BitMem {
            op,
            bit: BitSource::Imm,
            ea,
        });
    }

    // Immediate ALU: 0000 ooo0 ssmm mrrr.
    let size = decode_size_00((opcode >> 6) & 3)?;
    let op = match (opcode >> 9) & 7 {
        0 => BinaryOp::Or,
        1 => BinaryOp::And,
        2 => BinaryOp::Sub,
        3 => BinaryOp::Add,
        5 => BinaryOp::Eor,
        6 => BinaryOp::Cmp,
        _ => return None,
    };
    let dst = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
    // Excludes An (illegal), #imm (this is ORI-to-CCR/SR space) and PC-rel
    // destinations; CMPI on PC-relative EAs (68020+) also falls back.
    if !dst.is_data_alterable() {
        return None;
    }
    Some(DecodedMemOp::AluImm { op, size, dst })
}

fn decode_group_4(cpu_type: CpuType, opcode: u16) -> Option<DecodedMemOp> {
    if opcode == 0x4E75 {
        return Some(DecodedMemOp::Rts);
    }

    // LEA: 0100 rrr1 11mm mrrr
    if (opcode & 0xF1C0) == 0x41C0 {
        let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
        if !ea.is_control() {
            return None;
        }
        return Some(DecodedMemOp::Lea {
            reg: ((opcode >> 9) & 7) as u8,
            ea,
        });
    }

    // PEA: 0100 1000 01mm mrrr
    if (opcode & 0xFFC0) == 0x4840 {
        let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
        if !ea.is_control() {
            return None;
        }
        return Some(DecodedMemOp::Pea { ea });
    }

    // JSR: 0100 1110 10mm mrrr / JMP: 0100 1110 11mm mrrr
    if (opcode & 0xFFC0) == 0x4E80 || (opcode & 0xFFC0) == 0x4EC0 {
        let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
        if !ea.is_control() {
            return None;
        }
        return Some(if (opcode & 0x0040) != 0 {
            DecodedMemOp::Jmp { ea }
        } else {
            DecodedMemOp::Jsr { ea }
        });
    }

    // TST / CLR / NEG / NOT: 0100 xxxx ssmm mrrr
    let size = decode_size_00((opcode >> 6) & 3)?;
    let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
    // Memory forms only: register forms are already simple ops, and the
    // odd corners (TST.W An on 68020+, TST #imm, …) keep exact dispatch.
    if !ea.is_memory() || !ea.is_data_alterable() {
        // TST on 68020+ additionally allows PC-relative and immediate
        // sources; keep those on the dispatch path.
        return None;
    }
    let _ = cpu_type;
    match opcode & 0xFF00 {
        0x4A00 => Some(DecodedMemOp::Tst { size, ea }),
        0x4200 => Some(DecodedMemOp::Clr { size, ea }),
        0x4400 => Some(DecodedMemOp::Neg { size, ea }),
        0x4600 => Some(DecodedMemOp::Not { size, ea }),
        _ => None,
    }
}

fn decode_group_5(opcode: u16) -> Option<DecodedMemOp> {
    let size_bits = (opcode >> 6) & 3;
    if size_bits == 3 {
        // DBcc: 0101 cccc 1100 1rrr (Scc with mode 1)
        if ((opcode >> 3) & 7) == 1 {
            return Some(DecodedMemOp::Dbcc {
                condition: ((opcode >> 8) & 0xF) as u8,
                reg: (opcode & 7) as u8,
            });
        }
        return None;
    }

    // ADDQ/SUBQ #q,<ea> with a memory EA (register forms are simple ops).
    let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;
    if !ea.is_memory() || !ea.is_data_alterable() {
        return None;
    }
    let size = decode_size_00(size_bits)?;
    let data = ((opcode >> 9) & 7) as u32;
    Some(DecodedMemOp::AddqSubq {
        data: if data == 0 { 8 } else { data },
        size,
        ea,
        is_sub: (opcode & 0x100) != 0,
    })
}

fn decode_group_6(opcode: u16) -> Option<DecodedMemOp> {
    let condition = ((opcode >> 8) & 0xF) as u8;
    let displacement = (opcode & 0xFF) as u8;
    if condition == 1 {
        // BSR — byte or word displacement (long form falls back).
        return match displacement {
            0 => Some(DecodedMemOp::Bsr {
                displacement: 0, // read from the extension word at exec time
                length: 4,
            }),
            0xFF => None,
            d => Some(DecodedMemOp::Bsr {
                displacement: d as i8 as i32,
                length: 2,
            }),
        };
    }
    // Word-displacement Bcc/BRA; short forms are simple ops, long falls back.
    if displacement == 0 {
        return Some(DecodedMemOp::BranchWord { condition });
    }
    None
}

fn decode_alu(opcode: u16) -> Option<DecodedMemOp> {
    let group = (opcode >> 12) & 0xF;
    let reg = ((opcode >> 9) & 7) as u8;
    let op_mode = (opcode >> 6) & 7;
    let ea = FastEa::decode((opcode >> 3) & 7, opcode & 7)?;

    // ADDA/SUBA/CMPA: op_mode 3 (word) / 7 (long).
    if op_mode == 3 || op_mode == 7 {
        let op = match group {
            0x9 => AddrOp::Suba,
            0xB => AddrOp::Cmpa,
            0xD => AddrOp::Adda,
            _ => return None,
        };
        return Some(DecodedMemOp::AluAddr {
            op,
            size: if op_mode == 3 { Size::Word } else { Size::Long },
            src: ea,
            dst: reg,
        });
    }

    let size = decode_size_00(op_mode & 3)?;
    let to_ea = (op_mode & 4) != 0;

    if !to_ea {
        // <ea> op Dn → Dn. Register-direct sources are simple ops already,
        // but immediate/memory sources come through here.
        let op = match group {
            0x8 => BinaryOp::Or,
            0x9 => BinaryOp::Sub,
            0xB => BinaryOp::Cmp,
            0xC => BinaryOp::And,
            0xD => BinaryOp::Add,
            _ => return None,
        };
        // An source is only legal for word/long ADD/SUB/CMP.
        if matches!(ea, FastEa::AddrReg(_))
            && (size == Size::Byte || matches!(op, BinaryOp::Or | BinaryOp::And))
        {
            return None;
        }
        return Some(DecodedMemOp::AluToReg {
            op,
            size,
            src: ea,
            dst: reg,
        });
    }

    // Dn op <ea> → <ea> forms, plus CMPM and EOR.
    match group {
        0xB => {
            // EOR Dn,<ea> — mode 1 here is CMPM (Ay)+,(Ax)+.
            if let FastEa::AddrReg(src) = ea {
                return Some(DecodedMemOp::CmpM {
                    size,
                    src,
                    dst: reg,
                });
            }
            if matches!(ea, FastEa::DataReg(_)) {
                // EOR Dn,Dm is a simple op.
                return None;
            }
            if !ea.is_data_alterable() || !ea.is_memory() {
                return None;
            }
            Some(DecodedMemOp::AluToMem {
                op: BinaryOp::Eor,
                size,
                src: reg,
                dst: ea,
            })
        }
        0x8 | 0x9 | 0xC | 0xD => {
            // OR/SUB/AND/ADD Dn,<ea>: memory-alterable EAs only (mode 0/1
            // here encode SBCD/ABCD/EXG and friends).
            if !ea.is_memory() || !ea.is_data_alterable() {
                return None;
            }
            let op = match group {
                0x8 => BinaryOp::Or,
                0x9 => BinaryOp::Sub,
                0xC => BinaryOp::And,
                0xD => BinaryOp::Add,
                _ => unreachable!(),
            };
            Some(DecodedMemOp::AluToMem {
                op,
                size,
                src: reg,
                dst: ea,
            })
        }
        _ => None,
    }
}

// ============================================================================
// Execute
// ============================================================================

/// The fastmem window, copied out of the `CpuCore` scratch fields.
#[derive(Clone, Copy)]
struct Win {
    ptr: *mut u8,
    base: u32,
    len: u32,
}

impl Win {
    #[inline]
    fn from_cpu(cpu: &CpuCore) -> Option<Win> {
        if cpu.fm_len == 0 {
            return None;
        }
        Some(Win {
            ptr: cpu.fm_ptr as *mut u8,
            base: cpu.fm_base,
            len: cpu.fm_len,
        })
    }

    /// Window offset for an access of `bytes` at (masked) `addr`, or None
    /// if any part of the access falls outside the window.
    #[inline]
    fn off(self, addr: u32, bytes: u32) -> Option<usize> {
        let o = addr.wrapping_sub(self.base);
        // `fm_len >= 4` is guaranteed at capture, so `len - bytes` can't wrap.
        if o <= self.len - bytes {
            Some(o as usize)
        } else {
            None
        }
    }

    #[inline]
    fn read(self, off: usize, size: Size) -> u32 {
        unsafe {
            let p = self.ptr.add(off);
            match size {
                Size::Byte => *p as u32,
                Size::Word => u16::from_be_bytes([*p, *p.add(1)]) as u32,
                Size::Long => u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]),
            }
        }
    }

    #[inline]
    fn write(self, off: usize, size: Size, value: u32) {
        unsafe {
            let p = self.ptr.add(off);
            match size {
                Size::Byte => *p = value as u8,
                Size::Word => {
                    let b = (value as u16).to_be_bytes();
                    *p = b[0];
                    *p.add(1) = b[1];
                }
                Size::Long => {
                    let b = value.to_be_bytes();
                    *p = b[0];
                    *p.add(1) = b[1];
                    *p.add(2) = b[2];
                    *p.add(3) = b[3];
                }
            }
        }
    }
}

/// Deferred post-increment/pre-decrement address-register updates. Two
/// slots cover the worst case (CMPM, mem-to-mem MOVE).
#[derive(Clone, Copy, Default)]
struct Pending {
    slots: [(u8, u32); 2],
    count: u8,
}

impl Pending {
    #[inline]
    fn a(&self, cpu: &CpuCore, reg: u8) -> u32 {
        // Later pendings shadow earlier ones (e.g. CMPM (A0)+,(A0)+).
        for i in (0..self.count as usize).rev() {
            if self.slots[i].0 == reg {
                return self.slots[i].1;
            }
        }
        cpu.dar[8 + reg as usize]
    }

    #[inline]
    fn push(&mut self, reg: u8, value: u32) {
        self.slots[self.count as usize] = (reg, value);
        self.count += 1;
    }

    #[inline]
    fn commit(&self, cpu: &mut CpuCore) {
        for i in 0..self.count as usize {
            cpu.dar[8 + self.slots[i].0 as usize] = self.slots[i].1;
        }
    }
}

/// Per-instruction EA/extension-word cursor. `next_ext` is the guest
/// address of the next unread extension word (starts at PC, which the
/// fetch already advanced past the opcode).
struct Ctx {
    next_ext: u32,
    pending: Pending,
    aligned_only: bool,
}

/// A resolved operand location.
enum Loc {
    /// dar index (0-7 data, 8-15 address)
    Reg(usize),
    /// window offset
    Mem(usize),
    Imm(u32),
}

/// Specialized MOVE for the register / register-indirect EA forms
/// (`Dn`/`An`/`(An)`/`(An)+`/`-(An)`) — no extension words, so no
/// `Ctx`/`Pending` bookkeeping is needed and register adjustments can be
/// staged in locals. Returns `None` when either EA has another form (the
/// generic path handles it), `Some(false)` on window/alignment misses
/// (full dispatch takes over with nothing committed), `Some(true)` when
/// fully executed.
#[inline]
fn fast_move(cpu: &mut CpuCore, win: Win, size: Size, src: FastEa, dst: FastEa) -> Option<bool> {
    // (addr-register index, staged new value) for post-inc/pre-dec.
    let mut src_adj: Option<(usize, u32)> = None;
    let aligned_only = cpu.is_pre_68020;

    // Window offset for an access at `raw`, or an alignment/range miss.
    let locate = |cpu: &CpuCore, raw: u32| -> Result<usize, ()> {
        if aligned_only && size != Size::Byte && (raw & 1) != 0 {
            return Err(());
        }
        win.off(cpu.address(raw), size.bytes()).ok_or(())
    };

    let value = match src {
        FastEa::DataReg(r) => cpu.dar[r as usize] & size.mask(),
        FastEa::AddrReg(r) => cpu.dar[8 + r as usize] & size.mask(),
        FastEa::AnInd(r) => {
            let Ok(off) = locate(cpu, cpu.dar[8 + r as usize]) else {
                return Some(false);
            };
            win.read(off, size)
        }
        FastEa::AnPostInc(r) => {
            let a = cpu.dar[8 + r as usize];
            let Ok(off) = locate(cpu, a) else {
                return Some(false);
            };
            src_adj = Some((8 + r as usize, a.wrapping_add(ea_step(size, r))));
            win.read(off, size)
        }
        FastEa::AnPreDec(r) => {
            let a = cpu.dar[8 + r as usize].wrapping_sub(ea_step(size, r));
            let Ok(off) = locate(cpu, a) else {
                return Some(false);
            };
            src_adj = Some((8 + r as usize, a));
            win.read(off, size)
        }
        _ => return None,
    };

    // An address-register base must see a same-register source adjustment
    // (e.g. `MOVE.L (A0)+,(A0)+`).
    let addr_base = |cpu: &CpuCore, r: u8| -> u32 {
        match src_adj {
            Some((idx, val)) if idx == 8 + r as usize => val,
            _ => cpu.dar[8 + r as usize],
        }
    };

    match dst {
        FastEa::DataReg(r) => {
            if let Some((idx, val)) = src_adj {
                cpu.dar[idx] = val;
            }
            let mask = size.mask();
            let r = r as usize;
            cpu.dar[r] = (cpu.dar[r] & !mask) | value;
            cpu.set_logic_flags(value, size);
        }
        FastEa::AddrReg(r) => {
            // MOVEA: sign-extend word, no flags.
            if let Some((idx, val)) = src_adj {
                cpu.dar[idx] = val;
            }
            cpu.dar[8 + r as usize] = if size == Size::Word {
                value as u16 as i16 as i32 as u32
            } else {
                value
            };
        }
        FastEa::AnInd(r) => {
            let Ok(off) = locate(cpu, addr_base(cpu, r)) else {
                return Some(false);
            };
            if let Some((idx, val)) = src_adj {
                cpu.dar[idx] = val;
            }
            win.write(off, size, value);
            cpu.set_logic_flags(value, size);
        }
        FastEa::AnPostInc(r) => {
            let a = addr_base(cpu, r);
            let Ok(off) = locate(cpu, a) else {
                return Some(false);
            };
            if let Some((idx, val)) = src_adj {
                cpu.dar[idx] = val;
            }
            cpu.dar[8 + r as usize] = a.wrapping_add(ea_step(size, r));
            win.write(off, size, value);
            cpu.set_logic_flags(value, size);
        }
        FastEa::AnPreDec(r) => {
            let a = addr_base(cpu, r).wrapping_sub(ea_step(size, r));
            let Ok(off) = locate(cpu, a) else {
                return Some(false);
            };
            if let Some((idx, val)) = src_adj {
                cpu.dar[idx] = val;
            }
            cpu.dar[8 + r as usize] = a;
            win.write(off, size, value);
            cpu.set_logic_flags(value, size);
        }
        _ => return None,
    }
    Some(true)
}

/// Specialized DBcc/BSR/RTS/Bcc.W execution: at most one extension word,
/// no EA resolution, so the `Ctx` plumbing is skipped. Mirrors the generic
/// arms exactly (which remain the single source of truth for the other
/// ops). Returns `false` on any window/alignment miss so full dispatch
/// takes over with nothing committed.
#[inline]
fn fast_flow(cpu: &mut CpuCore, win: Win, op: DecodedMemOp) -> bool {
    // One extension word at `pc` (the word after the opcode).
    let read_ext = |cpu: &CpuCore| -> Option<u32> {
        let off = win.off(cpu.address(cpu.pc), 2)?;
        Some(win.read(off, Size::Word))
    };

    match op {
        DecodedMemOp::Dbcc { condition, reg } => {
            let Some(disp) = read_ext(cpu) else {
                return false;
            };
            if !cpu.test_condition(condition) {
                let reg = reg as usize;
                let counter = (cpu.dar[reg] as u16).wrapping_sub(1);
                cpu.dar[reg] = (cpu.dar[reg] & 0xFFFF_0000) | counter as u32;
                if counter != 0xFFFF {
                    // Displacement is relative to the displacement word.
                    cpu.pc = (cpu.pc as i32).wrapping_add(disp as u16 as i16 as i32) as u32;
                    return true;
                }
            }
            cpu.pc = cpu.pc.wrapping_add(2);
            true
        }
        DecodedMemOp::Rts => {
            let sp = cpu.dar[15];
            if cpu.is_pre_68020 && (sp & 1) != 0 {
                return false;
            }
            let Some(off) = win.off(cpu.address(sp), 4) else {
                return false;
            };
            let ret = win.read(off, Size::Long);
            cpu.dar[15] = sp.wrapping_add(4);
            cpu.change_of_flow = true;
            cpu.pc = ret;
            true
        }
        DecodedMemOp::Bsr {
            displacement,
            length,
        } => {
            let base = cpu.pc;
            let disp = if length == 4 {
                let Some(v) = read_ext(cpu) else {
                    return false;
                };
                v as u16 as i16 as i32
            } else {
                displacement
            };
            let ret = base.wrapping_add(length - 2);
            let sp = cpu.dar[15].wrapping_sub(4);
            if cpu.is_pre_68020 && (sp & 1) != 0 {
                return false;
            }
            let Some(off) = win.off(cpu.address(sp), 4) else {
                return false;
            };
            win.write(off, Size::Long, ret);
            cpu.dar[15] = sp;
            cpu.change_of_flow = true;
            cpu.pc = (base as i32).wrapping_add(disp) as u32;
            true
        }
        DecodedMemOp::BranchWord { condition } => {
            let base = cpu.pc;
            let Some(disp) = read_ext(cpu) else {
                return false;
            };
            if condition == 0 || cpu.test_condition(condition) {
                cpu.change_of_flow = true;
                cpu.pc = (base as i32).wrapping_add(disp as u16 as i16 as i32) as u32;
            } else {
                cpu.pc = base.wrapping_add(2);
            }
            true
        }
        _ => unreachable!("fast_flow only handles flow-control ops"),
    }
}

fn ea_step(size: Size, reg: u8) -> u32 {
    // Byte accesses through (A7)+ / -(A7) keep the stack pointer even.
    if size == Size::Byte && reg == 7 {
        2
    } else {
        size.bytes()
    }
}

#[inline]
fn read_ext_word(cpu: &CpuCore, win: Win, ctx: &mut Ctx) -> Option<u32> {
    let addr = cpu.address(ctx.next_ext);
    let off = win.off(addr, 2)?;
    ctx.next_ext = ctx.next_ext.wrapping_add(2);
    Some(win.read(off, Size::Word))
}

/// Brief-format index computation, mirroring `CpuCore::compute_index`.
/// Returns None for the 68020+ full extension format.
///
/// The index register is read through `pending` so an earlier EA's
/// post-increment/pre-decrement of the same address register is visible
/// (e.g. `MOVE.L -(A0),(d8,A4,A0.W)` must index with the decremented A0).
#[inline]
fn brief_index(cpu: &CpuCore, pending: &Pending, base: u32, ext: u32) -> Option<u32> {
    let is_020_plus = !is_pre_68020(cpu.cpu_type);
    if (ext & 0x0100) != 0 && is_020_plus {
        return None;
    }
    let d8 = (ext & 0xFF) as u8 as i8 as i32;
    let idx_reg = ((ext >> 12) & 0xF) as usize;
    let idx_is_addr = (ext & 0x8000) != 0;
    let idx_is_long = (ext & 0x0800) != 0;
    let scale = if is_020_plus {
        1i32 << ((ext >> 9) & 0x3)
    } else {
        1i32
    };
    let idx_val = if idx_is_addr {
        pending.a(cpu, (idx_reg & 7) as u8)
    } else {
        cpu.dar[idx_reg & 7]
    };
    let idx_val = if idx_is_long {
        idx_val as i32
    } else {
        (idx_val as i16) as i32
    };
    let idx_val = idx_val.wrapping_mul(scale);
    Some((base as i32).wrapping_add(d8).wrapping_add(idx_val) as u32)
}

/// Resolve one EA to a location, reading extension words and recording
/// pending register updates. No CPU state is modified.
fn resolve(cpu: &CpuCore, win: Win, ea: FastEa, size: Size, ctx: &mut Ctx) -> Option<Loc> {
    let mem = |cpu: &CpuCore, ctx: &Ctx, raw: u32| -> Option<Loc> {
        if ctx.aligned_only && size != Size::Byte && (raw & 1) != 0 {
            return None;
        }
        let addr = cpu.address(raw);
        Some(Loc::Mem(win.off(addr, size.bytes())?))
    };

    match ea {
        FastEa::DataReg(r) => Some(Loc::Reg(r as usize)),
        FastEa::AddrReg(r) => Some(Loc::Reg(8 + r as usize)),
        FastEa::AnInd(r) => mem(cpu, ctx, ctx.pending.a(cpu, r)),
        FastEa::AnPostInc(r) => {
            let a = ctx.pending.a(cpu, r);
            let loc = mem(cpu, ctx, a)?;
            ctx.pending.push(r, a.wrapping_add(ea_step(size, r)));
            Some(loc)
        }
        FastEa::AnPreDec(r) => {
            let a = ctx.pending.a(cpu, r).wrapping_sub(ea_step(size, r));
            let loc = mem(cpu, ctx, a)?;
            ctx.pending.push(r, a);
            Some(loc)
        }
        FastEa::AnDisp(r) => {
            let d = read_ext_word(cpu, win, ctx)? as u16 as i16 as i32;
            let a = (ctx.pending.a(cpu, r) as i32).wrapping_add(d) as u32;
            mem(cpu, ctx, a)
        }
        FastEa::AnIndex(r) => {
            let ext = read_ext_word(cpu, win, ctx)?;
            let a = brief_index(cpu, &ctx.pending, ctx.pending.a(cpu, r), ext)?;
            mem(cpu, ctx, a)
        }
        FastEa::AbsW => {
            let a = read_ext_word(cpu, win, ctx)? as u16 as i16 as i32 as u32;
            mem(cpu, ctx, a)
        }
        FastEa::AbsL => {
            let hi = read_ext_word(cpu, win, ctx)?;
            let lo = read_ext_word(cpu, win, ctx)?;
            mem(cpu, ctx, (hi << 16) | lo)
        }
        FastEa::PcDisp => {
            // Base is the address of the extension word itself.
            let base = ctx.next_ext;
            let d = read_ext_word(cpu, win, ctx)? as u16 as i16 as i32;
            mem(cpu, ctx, (base as i32).wrapping_add(d) as u32)
        }
        FastEa::PcIndex => {
            let base = ctx.next_ext;
            let ext = read_ext_word(cpu, win, ctx)?;
            let a = brief_index(cpu, &ctx.pending, base, ext)?;
            mem(cpu, ctx, a)
        }
        FastEa::Imm => {
            let v = match size {
                Size::Byte => read_ext_word(cpu, win, ctx)? & 0xFF,
                Size::Word => read_ext_word(cpu, win, ctx)?,
                Size::Long => {
                    let hi = read_ext_word(cpu, win, ctx)?;
                    let lo = read_ext_word(cpu, win, ctx)?;
                    (hi << 16) | lo
                }
            };
            Some(Loc::Imm(v))
        }
    }
}

/// Resolve a control EA (LEA/PEA/JMP/JSR) to a guest address (unmasked).
fn resolve_control_addr(cpu: &CpuCore, win: Win, ea: FastEa, ctx: &mut Ctx) -> Option<u32> {
    match ea {
        FastEa::AnInd(r) => Some(cpu.dar[8 + r as usize]),
        FastEa::AnDisp(r) => {
            let d = read_ext_word(cpu, win, ctx)? as u16 as i16 as i32;
            Some((cpu.dar[8 + r as usize] as i32).wrapping_add(d) as u32)
        }
        FastEa::AnIndex(r) => {
            let ext = read_ext_word(cpu, win, ctx)?;
            brief_index(cpu, &ctx.pending, cpu.dar[8 + r as usize], ext)
        }
        FastEa::AbsW => Some(read_ext_word(cpu, win, ctx)? as u16 as i16 as i32 as u32),
        FastEa::AbsL => {
            let hi = read_ext_word(cpu, win, ctx)?;
            let lo = read_ext_word(cpu, win, ctx)?;
            Some((hi << 16) | lo)
        }
        FastEa::PcDisp => {
            let base = ctx.next_ext;
            let d = read_ext_word(cpu, win, ctx)? as u16 as i16 as i32;
            Some((base as i32).wrapping_add(d) as u32)
        }
        FastEa::PcIndex => {
            let base = ctx.next_ext;
            let ext = read_ext_word(cpu, win, ctx)?;
            brief_index(cpu, &ctx.pending, base, ext)
        }
        _ => None,
    }
}

#[inline]
fn load(cpu: &CpuCore, win: Win, loc: &Loc, size: Size) -> u32 {
    match *loc {
        Loc::Reg(i) => cpu.dar[i] & size.mask(),
        Loc::Mem(off) => win.read(off, size),
        Loc::Imm(v) => v & size.mask(),
    }
}

#[inline]
fn store(cpu: &mut CpuCore, win: Win, loc: &Loc, size: Size, value: u32) {
    match *loc {
        Loc::Reg(i) => {
            let mask = size.mask();
            cpu.dar[i] = (cpu.dar[i] & !mask) | (value & mask);
        }
        Loc::Mem(off) => win.write(off, size, value),
        Loc::Imm(_) => unreachable!("immediate is never a destination"),
    }
}

/// Execute a decoded memory op against the fastmem window.
///
/// Returns `false` (with zero state changes) when the op must fall back to
/// full dispatch: window misses, pre-68020 misaligned accesses, 68020+
/// full-format extensions, or no window at all.
pub(crate) fn execute_mem_op(cpu: &mut CpuCore, op: DecodedMemOp) -> bool {
    let Some(win) = Win::from_cpu(cpu) else {
        return false;
    };

    // MOVE between registers and register-indirect memory is by far the
    // hottest mem op (block copies, loads, stores); handle it without the
    // generic Ctx/Pending resolution machinery. `None` means "not this
    // shape", falling through to the generic path below.
    if let DecodedMemOp::Move { size, src, dst } = op
        && let Some(handled) = fast_move(cpu, win, size, src, dst)
    {
        return handled;
    }

    // Flow-control ops (loop closers, calls, returns) are the other hot
    // class; they need at most one extension word and no EA resolution.
    match op {
        DecodedMemOp::Dbcc { .. }
        | DecodedMemOp::Bsr { .. }
        | DecodedMemOp::Rts
        | DecodedMemOp::BranchWord { .. } => return fast_flow(cpu, win, op),
        _ => {}
    }

    let mut ctx = Ctx {
        next_ext: cpu.pc,
        pending: Pending::default(),
        aligned_only: is_pre_68020(cpu.cpu_type),
    };

    macro_rules! finish_pc {
        () => {
            cpu.pc = ctx.next_ext;
        };
    }

    match op {
        DecodedMemOp::Move { size, src, dst } => {
            let Some(src_loc) = resolve(cpu, win, src, size, &mut ctx) else {
                return false;
            };
            let value = load(cpu, win, &src_loc, size);
            if let FastEa::AddrReg(r) = dst {
                // MOVEA: sign-extend word, no flags.
                let value = if size == Size::Word {
                    value as u16 as i16 as i32 as u32
                } else {
                    value
                };
                ctx.pending.commit(cpu);
                cpu.dar[8 + r as usize] = value;
                finish_pc!();
                return true;
            }
            let Some(dst_loc) = resolve(cpu, win, dst, size, &mut ctx) else {
                return false;
            };
            ctx.pending.commit(cpu);
            store(cpu, win, &dst_loc, size, value);
            cpu.set_logic_flags(value, size);
            finish_pc!();
            true
        }
        DecodedMemOp::AluToReg { op, size, src, dst } => {
            let Some(src_loc) = resolve(cpu, win, src, size, &mut ctx) else {
                return false;
            };
            let src_val = load(cpu, win, &src_loc, size);
            ctx.pending.commit(cpu);
            let dst_val = cpu.dar[dst as usize] & size.mask();
            apply_binary_to_reg(cpu, op, size, src_val, dst_val, dst as usize);
            finish_pc!();
            true
        }
        DecodedMemOp::AluToMem { op, size, src, dst } => {
            let Some(dst_loc) = resolve(cpu, win, dst, size, &mut ctx) else {
                return false;
            };
            let dst_val = load(cpu, win, &dst_loc, size);
            let src_val = cpu.dar[src as usize] & size.mask();
            let result = match op {
                BinaryOp::Add => src_val.wrapping_add(dst_val),
                BinaryOp::Sub => dst_val.wrapping_sub(src_val),
                BinaryOp::And => src_val & dst_val,
                BinaryOp::Or => src_val | dst_val,
                BinaryOp::Eor => src_val ^ dst_val,
                BinaryOp::Cmp => unreachable!("CMP has no to-memory form"),
            };
            ctx.pending.commit(cpu);
            store(cpu, win, &dst_loc, size, result & size.mask());
            match op {
                BinaryOp::Add => cpu.set_add_flags(src_val, dst_val, result, size),
                BinaryOp::Sub => cpu.set_sub_flags(src_val, dst_val, result, size),
                _ => cpu.set_logic_flags(result, size),
            }
            finish_pc!();
            true
        }
        DecodedMemOp::AluAddr { op, size, src, dst } => {
            let Some(src_loc) = resolve(cpu, win, src, size, &mut ctx) else {
                return false;
            };
            let raw = load(cpu, win, &src_loc, size);
            // Address ALU always operates on the full 32-bit register with
            // a sign-extended word source.
            let src_val = if size == Size::Word {
                raw as u16 as i16 as i32 as u32
            } else {
                raw
            };
            ctx.pending.commit(cpu);
            let dst_i = 8 + dst as usize;
            let dst_val = cpu.dar[dst_i];
            match op {
                AddrOp::Adda => cpu.dar[dst_i] = dst_val.wrapping_add(src_val),
                AddrOp::Suba => cpu.dar[dst_i] = dst_val.wrapping_sub(src_val),
                AddrOp::Cmpa => {
                    let result = dst_val.wrapping_sub(src_val);
                    cpu.set_cmp_flags(src_val, dst_val, result, Size::Long);
                }
            }
            finish_pc!();
            true
        }
        DecodedMemOp::AluImm { op, size, dst } => {
            let imm = {
                let Some(Loc::Imm(v)) = resolve(cpu, win, FastEa::Imm, size, &mut ctx) else {
                    return false;
                };
                v
            };
            let Some(dst_loc) = resolve(cpu, win, dst, size, &mut ctx) else {
                return false;
            };
            let dst_val = load(cpu, win, &dst_loc, size);
            let result = match op {
                BinaryOp::Add => imm.wrapping_add(dst_val),
                BinaryOp::Sub | BinaryOp::Cmp => dst_val.wrapping_sub(imm),
                BinaryOp::And => imm & dst_val,
                BinaryOp::Or => imm | dst_val,
                BinaryOp::Eor => imm ^ dst_val,
            };
            ctx.pending.commit(cpu);
            if op != BinaryOp::Cmp {
                store(cpu, win, &dst_loc, size, result & size.mask());
            }
            match op {
                BinaryOp::Add => cpu.set_add_flags(imm, dst_val, result, size),
                BinaryOp::Sub => cpu.set_sub_flags(imm, dst_val, result, size),
                BinaryOp::Cmp => cpu.set_cmp_flags(imm, dst_val, result, size),
                _ => cpu.set_logic_flags(result, size),
            }
            finish_pc!();
            true
        }
        DecodedMemOp::AddqSubq {
            data,
            size,
            ea,
            is_sub,
        } => {
            let Some(loc) = resolve(cpu, win, ea, size, &mut ctx) else {
                return false;
            };
            let dst_val = load(cpu, win, &loc, size);
            let result = if is_sub {
                dst_val.wrapping_sub(data)
            } else {
                dst_val.wrapping_add(data)
            };
            ctx.pending.commit(cpu);
            store(cpu, win, &loc, size, result & size.mask());
            if is_sub {
                cpu.set_sub_flags(data, dst_val, result, size);
            } else {
                cpu.set_add_flags(data, dst_val, result, size);
            }
            finish_pc!();
            true
        }
        DecodedMemOp::Tst { size, ea } => {
            let Some(loc) = resolve(cpu, win, ea, size, &mut ctx) else {
                return false;
            };
            let value = load(cpu, win, &loc, size);
            ctx.pending.commit(cpu);
            cpu.set_logic_flags(value, size);
            finish_pc!();
            true
        }
        DecodedMemOp::Clr { size, ea } => {
            let Some(loc) = resolve(cpu, win, ea, size, &mut ctx) else {
                return false;
            };
            ctx.pending.commit(cpu);
            store(cpu, win, &loc, size, 0);
            cpu.n_flag = 0;
            cpu.not_z_flag = 0;
            cpu.v_flag = 0;
            cpu.c_flag = 0;
            finish_pc!();
            true
        }
        DecodedMemOp::Neg { size, ea } => {
            let Some(loc) = resolve(cpu, win, ea, size, &mut ctx) else {
                return false;
            };
            let src = load(cpu, win, &loc, size);
            let result = 0u32.wrapping_sub(src);
            ctx.pending.commit(cpu);
            store(cpu, win, &loc, size, result & size.mask());
            cpu.set_sub_flags(src, 0, result, size);
            finish_pc!();
            true
        }
        DecodedMemOp::Not { size, ea } => {
            let Some(loc) = resolve(cpu, win, ea, size, &mut ctx) else {
                return false;
            };
            let src = load(cpu, win, &loc, size);
            let result = !src & size.mask();
            ctx.pending.commit(cpu);
            store(cpu, win, &loc, size, result);
            cpu.set_logic_flags(result, size);
            finish_pc!();
            true
        }
        DecodedMemOp::BitMem { op, bit, ea } => {
            let bit_num = match bit {
                BitSource::Reg(r) => cpu.dar[r as usize],
                BitSource::Imm => {
                    let Some(v) = read_ext_word(cpu, win, &mut ctx) else {
                        return false;
                    };
                    v
                }
            } & 7;
            let Some(loc) = resolve(cpu, win, ea, Size::Byte, &mut ctx) else {
                return false;
            };
            let value = load(cpu, win, &loc, Size::Byte);
            ctx.pending.commit(cpu);
            cpu.not_z_flag = if value & (1 << bit_num) != 0 { 1 } else { 0 };
            match op {
                BitOp::Test => {}
                BitOp::Set => store(cpu, win, &loc, Size::Byte, value | (1 << bit_num)),
                BitOp::Clear => store(cpu, win, &loc, Size::Byte, value & !(1 << bit_num)),
                BitOp::Change => store(cpu, win, &loc, Size::Byte, value ^ (1 << bit_num)),
            }
            finish_pc!();
            true
        }
        DecodedMemOp::CmpM { size, src, dst } => {
            let Some(src_loc) = resolve(cpu, win, FastEa::AnPostInc(src), size, &mut ctx) else {
                return false;
            };
            let src_val = load(cpu, win, &src_loc, size);
            let Some(dst_loc) = resolve(cpu, win, FastEa::AnPostInc(dst), size, &mut ctx) else {
                return false;
            };
            let dst_val = load(cpu, win, &dst_loc, size);
            ctx.pending.commit(cpu);
            let result = dst_val.wrapping_sub(src_val);
            cpu.set_cmp_flags(src_val, dst_val, result, size);
            finish_pc!();
            true
        }
        DecodedMemOp::Lea { reg, ea } => {
            let Some(addr) = resolve_control_addr(cpu, win, ea, &mut ctx) else {
                return false;
            };
            cpu.dar[8 + reg as usize] = addr;
            finish_pc!();
            true
        }
        DecodedMemOp::Pea { ea } => {
            let Some(addr) = resolve_control_addr(cpu, win, ea, &mut ctx) else {
                return false;
            };
            let sp = cpu.dar[15].wrapping_sub(4);
            if ctx.aligned_only && (sp & 1) != 0 {
                return false;
            }
            let Some(off) = win.off(cpu.address(sp), 4) else {
                return false;
            };
            win.write(off, Size::Long, addr);
            cpu.dar[15] = sp;
            finish_pc!();
            true
        }
        DecodedMemOp::Jmp { ea } => {
            let Some(addr) = resolve_control_addr(cpu, win, ea, &mut ctx) else {
                return false;
            };
            // The interpreter assigns the EA unmasked; the mask is applied
            // at the next fetch. Mirror that so pc values stay identical.
            cpu.change_of_flow = true;
            cpu.pc = addr;
            true
        }
        DecodedMemOp::Jsr { ea } => {
            let Some(addr) = resolve_control_addr(cpu, win, ea, &mut ctx) else {
                return false;
            };
            let ret = ctx.next_ext;
            let sp = cpu.dar[15].wrapping_sub(4);
            if ctx.aligned_only && (sp & 1) != 0 {
                return false;
            }
            let Some(off) = win.off(cpu.address(sp), 4) else {
                return false;
            };
            win.write(off, Size::Long, ret);
            cpu.dar[15] = sp;
            cpu.change_of_flow = true;
            cpu.pc = addr;
            true
        }
        DecodedMemOp::Rts => {
            let sp = cpu.dar[15];
            if ctx.aligned_only && (sp & 1) != 0 {
                return false;
            }
            let Some(off) = win.off(cpu.address(sp), 4) else {
                return false;
            };
            let ret = win.read(off, Size::Long);
            cpu.dar[15] = sp.wrapping_add(4);
            cpu.change_of_flow = true;
            cpu.pc = ret;
            true
        }
        DecodedMemOp::Bsr {
            displacement,
            length,
        } => {
            let base = cpu.pc;
            let disp = if length == 4 {
                let Some(v) = read_ext_word(cpu, win, &mut ctx) else {
                    return false;
                };
                v as u16 as i16 as i32
            } else {
                displacement
            };
            let ret = ctx.next_ext;
            let sp = cpu.dar[15].wrapping_sub(4);
            if ctx.aligned_only && (sp & 1) != 0 {
                return false;
            }
            let Some(off) = win.off(cpu.address(sp), 4) else {
                return false;
            };
            win.write(off, Size::Long, ret);
            cpu.dar[15] = sp;
            cpu.change_of_flow = true;
            cpu.pc = (base as i32).wrapping_add(disp) as u32;
            true
        }
        DecodedMemOp::BranchWord { condition } => {
            let base = cpu.pc;
            let Some(disp) = read_ext_word(cpu, win, &mut ctx) else {
                return false;
            };
            if condition == 0 || cpu.test_condition(condition) {
                cpu.change_of_flow = true;
                cpu.pc = (base as i32).wrapping_add(disp as u16 as i16 as i32) as u32;
            } else {
                cpu.pc = ctx.next_ext;
            }
            true
        }
        DecodedMemOp::Dbcc { condition, reg } => {
            let Some(disp) = read_ext_word(cpu, win, &mut ctx) else {
                return false;
            };
            if !cpu.test_condition(condition) {
                let reg = reg as usize;
                let counter = (cpu.dar[reg] as u16).wrapping_sub(1);
                cpu.dar[reg] = (cpu.dar[reg] & 0xFFFF_0000) | counter as u32;
                if counter != 0xFFFF {
                    // Displacement is relative to the displacement word.
                    cpu.pc = (cpu.pc as i32).wrapping_add(disp as u16 as i16 as i32) as u32;
                    return true;
                }
            }
            cpu.pc = ctx.next_ext;
            true
        }
    }
}

#[inline]
fn apply_binary_to_reg(
    cpu: &mut CpuCore,
    op: BinaryOp,
    size: Size,
    src: u32,
    dst: u32,
    reg: usize,
) {
    let mask = size.mask();
    match op {
        BinaryOp::Add => {
            let result = src.wrapping_add(dst);
            cpu.dar[reg] = (cpu.dar[reg] & !mask) | (result & mask);
            cpu.set_add_flags(src, dst, result, size);
        }
        BinaryOp::Sub => {
            let result = dst.wrapping_sub(src);
            cpu.dar[reg] = (cpu.dar[reg] & !mask) | (result & mask);
            cpu.set_sub_flags(src, dst, result, size);
        }
        BinaryOp::Cmp => {
            let result = dst.wrapping_sub(src);
            cpu.set_cmp_flags(src, dst, result, size);
        }
        BinaryOp::And => {
            let result = src & dst;
            cpu.dar[reg] = (cpu.dar[reg] & !mask) | (result & mask);
            cpu.set_logic_flags(result, size);
        }
        BinaryOp::Or => {
            let result = src | dst;
            cpu.dar[reg] = (cpu.dar[reg] & !mask) | (result & mask);
            cpu.set_logic_flags(result, size);
        }
        BinaryOp::Eor => {
            let result = src ^ dst;
            cpu.dar[reg] = (cpu.dar[reg] & !mask) | (result & mask);
            cpu.set_logic_flags(result, size);
        }
    }
}

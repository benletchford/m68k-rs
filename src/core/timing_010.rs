//! MC68010 instruction timing.
//!
//! M68000UM Section 9 ("MC68010 Instruction Execution Times") publishes the
//! time of every instruction as a total number of clocks. The tables differ
//! from the MC68000's in a handful of places: long operations on a register
//! are two clocks shorter, CLR, Scc, TAS and MOVE from SR no longer read
//! their destination and use the "no fetch" address times of Table 9-1, the
//! immediate-to-CCR/SR forms take 16 clocks, and so on.
//!
//! The instruction implementations are shared with the MC68000 and bill
//! their bus cycles and most of their internal clocks through
//! [`AddressBus`](super::memory::AddressBus) as they run. This module
//! supplies the instruction's total, so that a host which advances time
//! from the bus tops the instruction up to its table time and no further.
//!
//! Four entries follow the hardware where it departs from the manual. They
//! were measured from photographs of an A500 with an MC68010 running
//! programs that mark the end of an instruction on the raster (vAmigaTS
//! `CPU/Timing`, `*_A500_ECS_68010.JPG`):
//!
//! - `ADDQ`/`SUBQ.W #,An` take 8 clocks, as the long form does, where Table
//!   9-8 prints 4 (`ADDQ3w`).
//! - `CMP.L` with a memory operand takes 4 clocks plus the fetch, where
//!   Table 9-6 prints 6 (`CMP1l`, `CMP2l`, `CMP3l`, `CMP4l`). A register or
//!   immediate operand takes the printed 6.
//! - `ADDA.L`/`SUBA.L` with a register or immediate operand take 8 clocks
//!   plus the fetch, as on the MC68000, where Table 9-6 prints 6 (`ADDA1l`,
//!   `ADDA2l`). A memory operand takes the printed 6.
//! - `MOVE.L Rn,-(An)` takes 12 clocks, the same as to `(An)`, where Table
//!   9-4 prints 14 (`MOVE4L`).
//!
//! Times that depend on a run-time value the retired instruction no longer
//! exposes stay with the implementation: a shift by a register count, a bit
//! operation on a register, MOVEM's register count, the multiplies and
//! divides, the branches, loop mode and every exception.

use super::cpu::CpuCore;

/// Table 9-1, "Fetch" columns: calculate the address and read the operand.
fn fetch(mode: u16, reg: u16, long: bool) -> Option<i32> {
    let word = match (mode, reg) {
        (0 | 1, _) => return Some(0),
        (2 | 3, _) | (7, 4) => 4,
        (4, _) => 6,
        (5, _) | (7, 0 | 2) => 8,
        (6, _) | (7, 3) => 10,
        (7, 1) => 12,
        _ => return None,
    };
    Some(if long { word + 4 } else { word })
}

/// Table 9-1, "No Fetch" columns: calculate the address only.
fn no_fetch(mode: u16, reg: u16) -> Option<i32> {
    match (mode, reg) {
        (2, _) => Some(2),
        (3..=5, _) | (7, 0) => Some(4),
        (6, _) | (7, 1) => Some(8),
        _ => None,
    }
}

/// A source that is a register or immediate data.
fn register_or_immediate(mode: u16, reg: u16) -> bool {
    mode <= 1 || (mode == 7 && reg == 4)
}

/// A memory-alterable destination: (An) through (xxx).L.
fn memory_alterable(mode: u16, reg: u16) -> bool {
    matches!(mode, 2..=6) || (mode == 7 && reg <= 1)
}

/// Tables 9-2 and 9-4: MOVE and MOVEA.
fn move_time(op: u16, long: bool) -> Option<i32> {
    let (src_mode, src_reg) = ((op >> 3) & 7, op & 7);
    let (dst_mode, dst_reg) = ((op >> 6) & 7, (op >> 9) & 7);
    let destination = match (dst_mode, dst_reg) {
        (0 | 1, _) => return Some(4 + fetch(src_mode, src_reg, long)?),
        (2..=4, _) => 4,
        (5, _) | (7, 0) => 8,
        (6, _) => 10,
        (7, 1) => 12,
        _ => return None,
    };
    let destination = if long { destination + 4 } else { destination };
    Some(4 + fetch(src_mode, src_reg, long)? + destination)
}

/// Table 9-16: the control addressing modes of JMP, JSR, LEA and PEA.
fn control_time(mode: u16, reg: u16, times: [i32; 5]) -> Option<i32> {
    let [indirect, displacement, index, absolute_short, absolute_long] = times;
    match (mode, reg) {
        (2, _) => Some(indirect),
        (5, _) | (7, 2) => Some(displacement),
        (6, _) | (7, 3) => Some(index),
        (7, 0) => Some(absolute_short),
        (7, 1) => Some(absolute_long),
        _ => None,
    }
}

/// Tables 9-8 and 9-14 and the immediate rows of Table 9-18.
fn group_0(op: u16) -> Option<i32> {
    let (mode, reg) = ((op >> 3) & 7, op & 7);
    let kind = usize::from((op >> 6) & 3);
    if op & 0x0100 != 0 {
        if mode == 1 {
            // MOVEP: 16 clocks for a word, 24 for a long.
            return Some(if op & 0x0040 != 0 { 24 } else { 16 });
        }
        if mode == 0 {
            return None;
        }
        // Dynamic bit operation on memory.
        return Some([4, 8, 10, 8][kind] + fetch(mode, reg, false)?);
    }
    let operation = (op >> 9) & 7;
    match operation {
        4 => {
            if mode == 0 {
                return None;
            }
            // Static bit operation on memory.
            Some([8, 12, 14, 12][kind] + fetch(mode, reg, false)?)
        }
        7 => None,
        _ => {
            if kind == 3 {
                return None;
            }
            let long = kind == 2;
            let compare = operation == 6;
            if mode == 7 && reg == 4 {
                // ORI, ANDI and EORI to CCR or SR.
                return (!compare && !long).then_some(16);
            }
            if mode == 0 {
                return Some(match (long, compare) {
                    (false, _) => 8,
                    (true, true) => 12,
                    (true, false) => 14,
                });
            }
            if !memory_alterable(mode, reg) {
                return None;
            }
            let base = match (long, compare) {
                (false, true) => 8,
                (true, true) | (false, false) => 12,
                (true, false) => 20,
            };
            Some(base + fetch(mode, reg, long)?)
        }
    }
}

/// Tables 9-9, 9-10, 9-16 and 9-18.
fn group_4(op: u16) -> Option<i32> {
    let (mode, reg) = ((op >> 3) & 7, op & 7);
    let size = (op >> 6) & 3;
    let long = size == 2;
    if op & 0x01C0 == 0x0180 {
        // CHK without a trap.
        return Some(8 + fetch(mode, reg, false)?);
    }
    if op & 0x01C0 == 0x01C0 {
        return control_time(mode, reg, [4, 8, 12, 8, 12]);
    }
    if op & 0x0100 != 0 {
        return None;
    }
    // NEGX, NEG and NOT; TST is shorter and CLR has its own table.
    let unary = |register: i32, memory: i32| {
        if mode == 0 {
            Some(register)
        } else if memory_alterable(mode, reg) {
            Some(memory + fetch(mode, reg, long)?)
        } else {
            None
        }
    };
    // MOVE from SR and MOVE from CCR do not read their destination.
    let store_status = || {
        if mode == 0 {
            Some(4)
        } else {
            Some(8 + no_fetch(mode, reg)?)
        }
    };
    match (op >> 9) & 7 {
        0 if size == 3 => store_status(),
        1 if size == 3 => store_status(),
        1 => {
            // CLR, Table 9-10.
            let word = match (mode, reg) {
                (0, _) => return Some(if long { 6 } else { 4 }),
                (2 | 3, _) => 8,
                (4, _) => 10,
                (5, _) | (7, 0) => 12,
                (6, _) | (7, 1) => 16,
                _ => return None,
            };
            Some(if long { word + 4 } else { word })
        }
        2 | 3 if size == 3 => {
            // MOVE to CCR and MOVE to SR.
            if mode == 1 {
                return None;
            }
            Some(12 + fetch(mode, reg, false)?)
        }
        0 | 2 | 3 => unary(if long { 6 } else { 4 }, if long { 12 } else { 8 }),
        4 => match size {
            0 => {
                // NBCD
                if mode == 0 {
                    Some(6)
                } else if memory_alterable(mode, reg) {
                    Some(8 + fetch(mode, reg, false)?)
                } else {
                    None
                }
            }
            1 if mode == 0 => Some(4), // SWAP
            1 => control_time(mode, reg, [12, 16, 20, 16, 20]),
            _ if mode == 0 => Some(4), // EXT
            _ => None,                 // MOVEM register to memory
        },
        5 if size == 3 => {
            // TAS
            if mode == 0 {
                Some(4)
            } else {
                Some(14 + no_fetch(mode, reg)?)
            }
        }
        5 => {
            // TST
            if mode == 1 || (mode == 7 && reg > 1) {
                return None;
            }
            Some(4 + fetch(mode, reg, long)?)
        }
        7 => match op & 0x00FF {
            0x50..=0x57 => Some(16), // LINK
            0x58..=0x5F => Some(12), // UNLK
            0x60..=0x6F => Some(6),  // MOVE to and from USP
            0x70 => Some(130),       // RESET
            0x71 | 0x72 => Some(4),  // NOP, STOP
            0x74 | 0x75 => Some(16), // RTD, RTS
            0x76 => Some(4),         // TRAPV without a trap
            0x77 => Some(20),        // RTR
            0x7A => Some(12),        // MOVEC from a control register
            0x7B => Some(10),        // MOVEC to a control register
            0x80..=0xBF => control_time(mode, reg, [16, 18, 22, 18, 20]),
            0xC0..=0xFF => control_time(mode, reg, [8, 10, 14, 10, 12]),
            _ => None,
        },
        _ => None,
    }
}

/// ADDQ, SUBQ and Scc (Tables 9-8 and 9-9).
fn group_5(op: u16) -> Option<i32> {
    let (mode, reg) = ((op >> 3) & 7, op & 7);
    let size = (op >> 6) & 3;
    if size == 3 {
        return match mode {
            0 => Some(4),
            1 => None, // DBcc
            _ => Some(8 + no_fetch(mode, reg)?),
        };
    }
    let long = size == 2;
    match mode {
        0 => Some(if long { 8 } else { 4 }),
        1 if size == 0 => None,
        // An address register is always adjusted as a long.
        1 => Some(8),
        _ if memory_alterable(mode, reg) => {
            Some(if long { 12 } else { 8 } + fetch(mode, reg, long)?)
        }
        _ => None,
    }
}

/// OR, SUB, CMP, EOR, AND and ADD with their address-register, extended,
/// decimal and exchange forms (Tables 9-6, 9-17 and 9-18).
fn dyadic(op: u16) -> Option<i32> {
    let group = op >> 12;
    let (mode, reg) = ((op >> 3) & 7, op & 7);
    let opmode = (op >> 6) & 7;
    let logical = matches!(group, 0x8 | 0xC);
    if opmode == 3 || opmode == 7 {
        if logical {
            // DIVU, DIVS, MULU and MULS.
            return None;
        }
        // ADDA and SUBA: 8 word, 6 long. CMPA: 6 for both.
        let long = opmode == 7;
        let base = if group == 0xB {
            6
        } else if !long || register_or_immediate(mode, reg) {
            8
        } else {
            6
        };
        return Some(base + fetch(mode, reg, long)?);
    }
    let long = opmode & 3 == 2;
    if opmode < 3 {
        if mode == 1 && (logical || opmode == 0) {
            return None;
        }
        // A long compare with a memory operand takes 4 clocks, not 6.
        let compare_from_memory = group == 0xB && !register_or_immediate(mode, reg);
        let base = if long && !compare_from_memory { 6 } else { 4 };
        return Some(base + fetch(mode, reg, long)?);
    }
    match (group, mode) {
        // SBCD and ABCD.
        (0x8 | 0xC, 0) if opmode == 4 => Some(6),
        (0x8 | 0xC, 1) if opmode == 4 => Some(18),
        // EXG.
        (0xC, 0 | 1) => Some(6),
        (0x8, 0 | 1) => None,
        // SUBX and ADDX.
        (0x9 | 0xD, 0) => Some(if long { 6 } else { 4 }),
        (0x9 | 0xD, 1) => Some(if long { 30 } else { 18 }),
        // EOR Dn,Dn and CMPM.
        (0xB, 0) => Some(if long { 6 } else { 4 }),
        (0xB, 1) => Some(if long { 20 } else { 12 }),
        _ if memory_alterable(mode, reg) => {
            Some(if long { 12 } else { 8 } + fetch(mode, reg, long)?)
        }
        _ => None,
    }
}

/// Table 9-12: shifts and rotates with an immediate count or on memory.
fn group_e(op: u16) -> Option<i32> {
    let size = (op >> 6) & 3;
    if size == 3 {
        if op & 0x0800 != 0 {
            return None;
        }
        let (mode, reg) = ((op >> 3) & 7, op & 7);
        if !memory_alterable(mode, reg) {
            return None;
        }
        return Some(8 + fetch(mode, reg, false)?);
    }
    if op & 0x0020 != 0 {
        // The count is in a register.
        return None;
    }
    let count = match (op >> 9) & 7 {
        0 => 8,
        count => i32::from(count),
    };
    Some(if size == 2 { 8 } else { 6 } + 2 * count)
}

fn table(op: u16) -> Option<i32> {
    match op >> 12 {
        0x0 => group_0(op),
        0x1 | 0x3 => move_time(op, false),
        0x2 => move_time(op, true),
        0x4 => group_4(op),
        0x5 => group_5(op),
        0x7 => (op & 0x0100 == 0).then_some(4),
        0x8 | 0x9 | 0xB | 0xC | 0xD => dyadic(op),
        0xE => group_e(op),
        _ => None,
    }
}

impl CpuCore {
    /// Return the Section 9 time of the MC68010 instruction that just
    /// retired, or `raw` when the tables do not fix it from the opcode
    /// alone.
    pub(crate) fn cycles_010(&self, raw: i32) -> i32 {
        if self.instruction_exception_vector.is_some() || self.loop_mode {
            return raw;
        }
        table(self.ir as u16).unwrap_or(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_register_operations_are_two_clocks_shorter_than_the_68000() {
        assert_eq!(table(0xD081), Some(6)); // add.l d1,d0
        assert_eq!(table(0xB380), Some(6)); // eor.l d1,d0
        assert_eq!(table(0x0680), Some(14)); // addi.l #,d0
        assert_eq!(table(0x0C80), Some(12)); // cmpi.l #,d0
        assert_eq!(table(0xD181), Some(6)); // addx.l d1,d0
    }

    #[test]
    fn destinations_that_are_not_read_use_the_no_fetch_times() {
        assert_eq!(table(0x4210), Some(8)); // clr.b (a0)
        assert_eq!(table(0x42A8), Some(16)); // clr.l (d16,a0)
        assert_eq!(table(0x50D0), Some(10)); // st (a0)
        assert_eq!(table(0x40D0), Some(10)); // move sr,(a0)
    }

    #[test]
    fn measured_times_replace_the_printed_ones() {
        assert_eq!(table(0x5049), Some(8)); // addq.w #8,a1
        assert_eq!(table(0x5149), Some(8)); // subq.w #8,a1
        assert_eq!(table(0xB090), Some(12)); // cmp.l (a0),d0
        assert_eq!(table(0xB081), Some(6)); // cmp.l d1,d0
        assert_eq!(table(0xB0BC), Some(14)); // cmp.l #,d0
        assert_eq!(table(0xD090), Some(14)); // add.l (a0),d0
        assert_eq!(table(0xD1C1), Some(8)); // adda.l d1,a0
        assert_eq!(table(0xD1FC), Some(16)); // adda.l #,a0
        assert_eq!(table(0xD1D0), Some(14)); // adda.l (a0),a0
        assert_eq!(table(0x2101), Some(12)); // move.l d1,-(a0)
        assert_eq!(table(0x4AD0), Some(16)); // tas (a0)
        assert_eq!(table(0x4AD8), Some(18)); // tas (a0)+
    }

    #[test]
    fn run_time_dependent_times_are_left_to_the_implementation() {
        assert_eq!(table(0xE0A8), None); // lsr.l d0,d0
        assert_eq!(table(0x0100), None); // btst d0,d0
        assert_eq!(table(0x48D0), None); // movem.l regs,(a0)
        assert_eq!(table(0xC0C1), None); // mulu d1,d0
        assert_eq!(table(0x6000), None); // bra
        assert_eq!(table(0x51C8), None); // dbra
        assert_eq!(table(0x4E73), None); // rte
    }
}

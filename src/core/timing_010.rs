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
//! divides, the branches and every exception.
//!
//! The table fixes totals; `tail_clocks` is the part of a total that an
//! instruction ending on a register spends after its final prefetch, which
//! the core bills in place.
//!
//! In loop mode an instruction and the DBcc behind it are timed as a pair
//! (Tables 9-3, 9-5, 9-7, 9-11, 9-13 and 9-17): `loop_times` gives the
//! pair's three times and the DBcc takes its share from them. The body's
//! internal clocks fall between its last read and its write, or after its
//! last read when it writes nothing.

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

/// Loop-mode times of a loopable instruction and the DBcc that closes its
/// loop, taken together: the loop continuing, the loop ending with the
/// condition true, and the loop ending with the count expired (Tables 9-3,
/// 9-5, 9-7, 9-11, 9-13 and 9-17).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct LoopTimes {
    pub(crate) continued: i32,
    pub(crate) condition_true: i32,
    pub(crate) expired: i32,
}

impl LoopTimes {
    const fn new(continued: i32, condition_true: i32, expired: i32) -> Self {
        Self {
            continued,
            condition_true,
            expired,
        }
    }

    /// The tables add two clocks for a -(An) operand.
    const fn predecrement(self, apply: bool) -> Self {
        if apply {
            Self::new(
                self.continued + 2,
                self.condition_true + 2,
                self.expired + 2,
            )
        } else {
            self
        }
    }
}

/// The looping DBcc's own share of a loop iteration: it re-enters the body
/// in 6 clocks without touching the bus.
pub(crate) const LOOP_DBCC_CONTINUED: i32 = 6;

/// Loop-mode times for the body instruction `op`, or `None` when the tables
/// have no entry for it.
pub(crate) fn loop_times(op: u16) -> Option<LoopTimes> {
    let mode = (op >> 3) & 7;
    let t = LoopTimes::new;
    match op >> 12 {
        // Tables 9-3 and 9-5: MOVE to (An), (An)+ or -(An).
        group @ 0x1..=0x3 => {
            let long = group == 0x2;
            let dst_mode = (op >> 6) & 7;
            if !(2..=4).contains(&dst_mode) {
                return None;
            }
            let to_predecrement = dst_mode == 4;
            let times = match (mode, long) {
                (0 | 1, _) if to_predecrement => return None,
                (0 | 1, false) => t(10, 18, 16),
                (0 | 1, true) => t(14, 20, 18),
                (2 | 3, false) => t(14, 20, 18),
                (2 | 3, true) => t(22, 28, 24),
                (4, false) => t(16, 22, 20),
                (4, true) => t(24, 30, 26),
                _ => return None,
            };
            Some(times.predecrement(to_predecrement && mode >= 2))
        }
        // Table 9-11 (single operand) and, for TST, the same table.
        0x4 => {
            if !(2..=4).contains(&mode) {
                return None;
            }
            let size = (op >> 6) & 3;
            let long = size == 2;
            let times = match (op >> 8) & 0xF {
                // NEGX, NEG and NOT
                0x0 | 0x4 | 0x6 if size != 3 => {
                    if long {
                        t(24, 30, 28)
                    } else {
                        t(16, 22, 20)
                    }
                }
                // CLR
                0x2 if size != 3 => {
                    if long {
                        t(14, 22, 20)
                    } else {
                        t(10, 18, 16)
                    }
                }
                // NBCD
                0x8 if size == 0 => t(18, 24, 22),
                // TST
                0xA if size != 3 => {
                    if long {
                        t(18, 24, 20)
                    } else {
                        t(12, 18, 16)
                    }
                }
                _ => return None,
            };
            Some(times.predecrement(mode == 4))
        }
        // Tables 9-7 and 9-17.
        group @ (0x8 | 0x9 | 0xB | 0xC | 0xD) => {
            let opmode = (op >> 6) & 7;
            let logical = matches!(group, 0x8 | 0xC);
            // ABCD and SBCD -(Ay),-(Ax).
            if logical && op & 0x01F8 == 0x0108 {
                return Some(t(24, 30, 28));
            }
            let long = opmode & 3 == 2;
            if op & 0x0138 == 0x0108 && opmode & 3 != 3 {
                return match group {
                    // ADDX and SUBX -(Ay),-(Ax).
                    0x9 | 0xD => Some(if long { t(32, 38, 36) } else { t(22, 28, 26) }),
                    // CMPM (Ay)+,(Ax)+.
                    0xB => Some(if long { t(24, 30, 26) } else { t(14, 20, 18) }),
                    _ => None,
                };
            }
            if !(2..=4).contains(&mode) {
                return None;
            }
            let compare = group == 0xB;
            let times = match opmode {
                // op <ea>,An: ADDA, SUBA and CMPA.
                3 | 7 if logical => return None,
                3 if compare => t(12, 18, 16),
                7 if compare => t(18, 24, 20),
                3 => t(18, 24, 22),
                7 => t(22, 28, 26),
                // op <ea>,Dn. Table 9-7 prints 20/26/24 for SUB.L; ADD.L's
                // row is used for both.
                0..=2 if compare => {
                    if long {
                        t(18, 24, 20)
                    } else {
                        t(12, 18, 16)
                    }
                }
                0..=2 => {
                    if long {
                        t(22, 28, 26)
                    } else {
                        t(16, 22, 20)
                    }
                }
                // op Dn,<ea> (EOR for the compare group).
                _ => {
                    if long {
                        t(24, 30, 28)
                    } else {
                        t(16, 22, 20)
                    }
                }
            };
            Some(times.predecrement(mode == 4))
        }
        // Table 9-13: word shifts and rotates on memory.
        0xE => {
            if (op >> 6) & 3 != 3 || op & 0x0800 != 0 || !(2..=4).contains(&mode) {
                return None;
            }
            Some(t(18, 24, 22).predecrement(mode == 4))
        }
        _ => None,
    }
}

/// Bus cycles of the loop body `op` in loop mode, where it fetches nothing:
/// its operand reads and writes.
fn loop_bus_cycles(op: u16) -> i32 {
    let mode = (op >> 3) & 7;
    let per_operand = |long: bool| if long { 2 } else { 1 };
    match op >> 12 {
        group @ 0x1..=0x3 => {
            let n = per_operand(group == 0x2);
            if mode <= 1 { n } else { 2 * n }
        }
        0x4 => {
            let n = per_operand((op >> 6) & 3 == 2);
            // CLR only writes and TST only reads.
            if matches!((op >> 8) & 0xF, 0x2 | 0xA) {
                n
            } else {
                2 * n
            }
        }
        0xE => 2,
        group => {
            let opmode = (op >> 6) & 7;
            let n = per_operand(opmode == 7 || opmode & 3 == 2);
            match opmode {
                // <ea>,Dn and <ea>,An read their source.
                0..=3 | 7 => n,
                // CMPM reads twice; ADDX, SUBX, ABCD and SBCD also write.
                _ if mode == 1 => {
                    if group == 0xB {
                        2 * n
                    } else {
                        3 * n
                    }
                }
                // Dn,<ea> reads and writes.
                _ => 2 * n,
            }
        }
    }
}

/// Clocks the loop body `op` already spends forming a -(An) address before
/// its operand cycles: two, once, for a predecremented source or single
/// operand and for the -(Ay),-(Ax) pairs. A predecremented MOVE destination
/// has none of its own.
fn loop_predecrement_clocks(op: u16) -> i32 {
    let mode = (op >> 3) & 7;
    let pair = matches!(op >> 12, 0x8 | 0x9 | 0xC | 0xD) && op & 0x0138 == 0x0108;
    if mode == 4 || pair { 2 } else { 0 }
}

/// Whether the loop body `op` writes to memory.
fn loop_body_writes(op: u16) -> bool {
    match op >> 12 {
        0x1..=0x3 | 0xE => true,
        // Every single-operand body but TST.
        0x4 => (op >> 8) & 0xF != 0xA,
        // op Dn,<ea> and the -(Ay),-(Ax) pairs; not CMPM or op <ea>,An.
        group => {
            let opmode = (op >> 6) & 7;
            (4..=6).contains(&opmode) && !(group == 0xB && (op >> 3) & 7 == 1)
        }
    }
}

/// Internal clocks a loop body that writes spends between its last read and
/// its write (measured: the ADD Dn,(An) loops of vAmigaTS CPU/68010/Loopmode
/// keep their time under bus contention only with the clocks there).
pub(crate) fn loop_write_lead(op: u16) -> u32 {
    if loop_body_writes(op) {
        loop_body_clocks(op) as u32
    } else {
        0
    }
}

/// Internal clocks a loop body that only reads spends after its last read.
pub(crate) fn loop_tail_clocks(op: u16) -> i32 {
    if loop_body_writes(op) {
        0
    } else {
        loop_body_clocks(op)
    }
}

/// Internal clocks the loop body `op` spends in loop mode besides its
/// operand cycles: its share of a continuing iteration less those cycles and
/// the address clocks that precede them.
fn loop_body_clocks(op: u16) -> i32 {
    loop_times(op).map_or(0, |times| {
        let body = times.continued - LOOP_DBCC_CONTINUED;
        (body - 4 * loop_bus_cycles(op) - loop_predecrement_clocks(op)).max(0)
    })
}

/// Internal clocks an instruction that ends on a register spends after its
/// final prefetch: its table time less its bus cycles. Shifts and rotates of
/// a register depend on the count and are billed where it is known.
pub(crate) fn tail_clocks(op: u16) -> i32 {
    let (mode, reg) = ((op >> 3) & 7, op & 7);
    let size = (op >> 6) & 3;
    match op >> 12 {
        // ORI, ANDI, SUBI, ADDI and EORI.L #,Dn: 14(3/0).
        0x0 => {
            let operation = (op >> 9) & 7;
            let arithmetic = operation <= 3 || operation == 5;
            i32::from(op & 0x0100 == 0 && arithmetic && size == 2 && mode == 0) * 2
        }
        0x4 if op & 0x0100 == 0 => match (op >> 9) & 7 {
            // NEGX, CLR, NEG and NOT.L Dn: 6(1/0).
            0..=3 if size == 2 && mode == 0 => 2,
            // NBCD Dn: 6(1/0).
            4 if size == 0 && mode == 0 => 2,
            // MOVEC: 12(2/0) from a control register, 10(2/0) to one.
            7 => match op & 0x00FF {
                0x7A => 4,
                0x7B => 2,
                _ => 0,
            },
            _ => 0,
        },
        // ADDQ and SUBQ: 8(1/0) for a long data register and for an
        // address register.
        0x5 if size != 3 => match mode {
            0 if size == 2 => 4,
            1 if size != 0 => 4,
            _ => 0,
        },
        group @ (0x8 | 0x9 | 0xB | 0xC | 0xD) => {
            let opmode = (op >> 6) & 7;
            let logical = matches!(group, 0x8 | 0xC);
            match opmode {
                // MULU, MULS, DIVU and DIVS bill themselves.
                3 | 7 if logical => 0,
                // CMPA: 6 plus the fetch.
                3 | 7 if group == 0xB => 2,
                // ADDA and SUBA: 8 plus the fetch, or 6 for a long from
                // memory.
                7 if !register_or_immediate(mode, reg) => 2,
                3 | 7 => 4,
                // Long op <ea>,Dn: 6 plus the fetch; a compare with memory
                // has no tail.
                2 if group == 0xB && !register_or_immediate(mode, reg) => 0,
                2 if !(logical && mode == 1) => 2,
                // ADDX.L, SUBX.L and EOR.L Dn,Dn: 6(1/0).
                6 if mode == 0 && !logical => 2,
                _ => 0,
            }
        }
        _ => 0,
    }
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
        if self.instruction_exception_vector.is_some() {
            return raw;
        }
        let op = self.ir as u16;
        if self.loop_mode {
            // The body of a loop in loop mode: its share of an iteration is
            // the table's time less the looping DBcc's. The DBcc itself and
            // the loop's last pass bill their own time.
            if op & 0xF0F8 == 0x50C8 {
                return raw;
            }
            return loop_times(op).map_or(raw, |times| times.continued - LOOP_DBCC_CONTINUED);
        }
        table(op).unwrap_or(raw)
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

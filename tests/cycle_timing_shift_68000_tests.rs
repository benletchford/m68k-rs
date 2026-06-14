//! MC68000 register shift/rotate cycle timing.
//!
//! On the 68000 a register shift/rotate costs `base + 2 * count` clocks, where
//! the base is 6 for byte/word and 8 for long. The variable part counts the
//! full shift count, even for ROXL/ROXR where the rotation itself wraps every
//! (operand_bits + 1) positions.

use m68k::{CpuCore, CpuType, Size};

fn cpu(kind: CpuType) -> CpuCore {
    let mut c = CpuCore::new();
    c.set_cpu_type(kind);
    c
}

#[test]
fn long_shift_base_is_eight_on_68000() {
    let mut c = cpu(CpuType::M68000);
    // ASL/LSL/ROL .l #1 = 8 (long base) + 2 = 10.
    assert_eq!(c.exec_asl(Size::Long, 1, 0x1).1, 10);
    assert_eq!(c.exec_lsl(Size::Long, 1, 0x1).1, 10);
    assert_eq!(c.exec_rol(Size::Long, 1, 0x1).1, 10);
    // Byte/word keep the base of 6.
    assert_eq!(c.exec_asl(Size::Word, 1, 0x1).1, 8);
    assert_eq!(c.exec_asl(Size::Byte, 1, 0x1).1, 8);
    // A larger count scales by 2 per step on top of the long base.
    assert_eq!(c.exec_asl(Size::Long, 8, 0x1).1, 8 + 16);
}

#[test]
fn long_shift_base_unchanged_on_68020() {
    // 68020+ is not affected by the 68000-only long-base correction.
    let mut c = cpu(CpuType::M68020);
    assert_eq!(c.exec_asl(Size::Long, 1, 0x1).1, 8);
    assert_eq!(c.exec_asl(Size::Word, 1, 0x1).1, 8);
}

#[test]
fn roxl_roxr_timing_counts_full_shift() {
    let mut c = cpu(CpuType::M68000);
    // A word ROXL by 17 rotates through 17 positions; 17 mod (16 + 1) == 0, so
    // there is no net rotation, but timing is still 6 + 2 * 17 = 40.
    assert_eq!(c.exec_roxl(Size::Word, 17, 0x1234).1, 40);
    assert_eq!(c.exec_roxr(Size::Word, 17, 0x1234).1, 40);
    // A plain count is base + 2 * count.
    assert_eq!(c.exec_roxl(Size::Word, 3, 0x1).1, 6 + 6);
    assert_eq!(c.exec_roxl(Size::Long, 3, 0x1).1, 8 + 6);
}

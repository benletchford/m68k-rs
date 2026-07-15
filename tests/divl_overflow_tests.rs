//! DIVS.L overflow edge cases.
//!
//! The 64-bit dividend form can present the one division that does not fit
//! i64 (`i64::MIN / -1`). Hardware flags overflow and leaves the registers
//! alone; a native Rust division would panic with an arithmetic overflow,
//! taking the host process down with it.

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, StepResult};

const SR_V: u16 = 0x0002;

/// Reset into supervisor mode with PC at 0x0200 and one DIVS.L there.
///
/// Encoding: opcode 0x4C40 | EA (register-direct D1 divisor), extension
/// 0x0800 (signed) | optional 0x0400 (64-bit dividend) | Dq in bits 14..12,
/// Dr in bits 2..0.
fn setup(ext: u16) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, 0x0800); // SSP
    bus.write_long(0x04, 0x0200); // initial PC
    bus.write_word(0x0200, 0x4C41); // DIVS.L D1, ...
    bus.write_word(0x0202, ext);

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68020);
    cpu.reset(&mut bus);
    (cpu, bus)
}

#[test]
fn divs_l_64bit_min_dividend_by_minus_one_sets_v_instead_of_panicking() {
    // DIVS.L D1, D2:D0 -- dividend D2:D0 = i64::MIN, divisor D1 = -1.
    let (mut cpu, mut bus) = setup(0x0C02);
    cpu.set_d(2, 0x8000_0000); // dividend high (Dr)
    cpu.set_d(0, 0x0000_0000); // dividend low (Dq)
    cpu.set_d(1, 0xFFFF_FFFF); // divisor -1

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.pc, 0x0204, "overflow must not trap");
    assert_ne!(cpu.get_sr() & SR_V, 0, "V must be set on overflow");
    assert_eq!(cpu.d(0), 0x0000_0000, "quotient register left alone");
    assert_eq!(cpu.d(2), 0x8000_0000, "remainder register left alone");
}

#[test]
fn divs_l_32bit_min_dividend_by_minus_one_sets_v() {
    // DIVS.L D1, D0 -- dividend D0 = i32::MIN, divisor D1 = -1: the +2^31
    // quotient does not fit i32, so V is set and D0 is left alone.
    let (mut cpu, mut bus) = setup(0x0800);
    cpu.set_d(0, 0x8000_0000);
    cpu.set_d(1, 0xFFFF_FFFF);

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.pc, 0x0204, "overflow must not trap");
    assert_ne!(cpu.get_sr() & SR_V, 0, "V must be set on overflow");
    assert_eq!(cpu.d(0), 0x8000_0000, "quotient register left alone");
}

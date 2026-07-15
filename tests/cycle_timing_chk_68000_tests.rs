//! MC68000 CHK trap-path cycle timing.
//!
//! The microcode runs the upper-bound comparison first as a signed word
//! subtract. Two trap shapes fall out of it, register-direct costs below:
//!
//! - upper-bound path: 38 clocks. Taken by `Dn > bound`, and also by a
//!   negative `Dn` whose `Dn - bound` subtraction overflows (the subtract
//!   result looks "above the bound" to the microcode).
//! - negative path: 40 clocks. Taken by the remaining `Dn < 0` cases.
//!
//! These values match all SingleStepTests `m68000` CHK cases.

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, StepResult};

/// Reset a 68000 into supervisor mode with `CHK D1, D0` at 0x0200.
fn setup(dn: u32, bound: u32) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, 0x0800); // SSP
    bus.write_long(0x04, 0x0200); // initial PC
    bus.write_long(0x18, 0x0400); // CHK vector (6)
    bus.write_word(0x0200, 0x4181); // CHK.W D1, D0

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.reset(&mut bus);
    cpu.set_d(0, dn);
    cpu.set_d(1, bound);
    (cpu, bus)
}

fn chk_cycles(dn: u32, bound: u32) -> i32 {
    let (mut cpu, mut bus) = setup(dn, bound);
    match cpu.step(&mut bus) {
        StepResult::Ok { cycles } => cycles,
        other => panic!("unexpected step result: {other:?}"),
    }
}

#[test]
fn chk_pass_is_10() {
    assert_eq!(chk_cycles(3, 100), 10);
}

#[test]
fn chk_above_bound_takes_the_38_clock_upper_bound_path() {
    assert_eq!(chk_cycles(200, 100), 38);
}

#[test]
fn chk_negative_value_above_negative_bound_takes_the_upper_bound_path() {
    // -1 > -100: still the cheaper upper-bound trap despite being negative.
    assert_eq!(chk_cycles(0xFFFF_FFFF, 0xFFFF_FF9C), 38);
}

#[test]
fn chk_negative_value_takes_the_40_clock_negative_path() {
    assert_eq!(chk_cycles(0xFFFF_FFFF, 100), 40);
}

#[test]
fn chk_negative_value_with_overflowing_subtract_takes_the_upper_bound_path() {
    // -32768 - 32767 overflows a signed word, so the microcode's bound
    // comparison sends this out on the 38-clock upper-bound path even
    // though the architectural trap condition is Dn < 0.
    assert_eq!(chk_cycles(0xFFFF_8000, 0x0000_7FFF), 38);
}

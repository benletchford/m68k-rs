//! MC68000 multiply/divide cycle timing.
//!
//! On the 68000 MULU/MULS/DIVU/DIVS take a data-dependent number of clocks
//! rather than a flat worst-case value. The values below match the
//! SingleStepTests `m68000` fixtures for register-direct operands (the residual
//! for memory operands is the separate effective-address cost, which this core
//! does not model for any instruction).

use m68k::core::ea::AddressingMode;
use m68k::{CpuCore, CpuType, LinearMemoryBus};

fn cpu(kind: CpuType) -> CpuCore {
    let mut c = CpuCore::new();
    c.set_cpu_type(kind);
    c
}

// Source operand in a data register (no bus access), destination in D0.
fn src_dn(reg: u8) -> AddressingMode {
    AddressingMode::DataDirect(reg)
}

#[test]
fn mulu_is_38_plus_two_per_source_one_bit() {
    let mut bus = LinearMemoryBus::new(0x100);
    let mut c = cpu(CpuType::M68000);
    c.set_d(1, 0x0000); // 0 one-bits
    assert_eq!(c.exec_mulu(&mut bus, src_dn(1), 0), 38);
    c.set_d(1, 0xFFFF); // 16 one-bits
    assert_eq!(c.exec_mulu(&mut bus, src_dn(1), 0), 38 + 2 * 16);
}

#[test]
fn muls_is_38_plus_two_per_source_bit_transition() {
    let mut bus = LinearMemoryBus::new(0x100);
    let mut c = cpu(CpuType::M68000);
    c.set_d(1, 0x0000); // no transitions in [0..0, 0]
    assert_eq!(c.exec_muls(&mut bus, src_dn(1), 0), 38);
    c.set_d(1, 0xFFFF); // one transition in [1..1, 0]
    assert_eq!(c.exec_muls(&mut bus, src_dn(1), 0), 38 + 2);
}

#[test]
fn divu_is_data_dependent() {
    let mut bus = LinearMemoryBus::new(0x100);
    let mut c = cpu(CpuType::M68000);
    // 0 / 1: 38 + 15*2 internal cycles, doubled = 136.
    c.set_d(0, 0);
    c.set_d(1, 1);
    assert_eq!(c.exec_divu(&mut bus, src_dn(1), 0), 136);
    // Quotient overflows 16 bits: the early-overflow path is short (10).
    c.set_d(0, 0xFFFF_0000);
    c.set_d(1, 1);
    assert_eq!(c.exec_divu(&mut bus, src_dn(1), 0), 10);
}

#[test]
fn divs_overflow_paths_are_short() {
    let mut bus = LinearMemoryBus::new(0x100);
    let mut c = cpu(CpuType::M68000);
    // |quotient| cannot fit in 15 bits: early-overflow termination.
    c.set_d(0, 0x1000_0000);
    c.set_d(1, 1);
    assert_eq!(c.exec_divs(&mut bus, src_dn(1), 0), 16);
    // 0x80000000 / -1 also overflows and terminates early (negative dividend).
    c.set_d(0, 0x8000_0000);
    c.set_d(1, 0xFFFF); // -1 as a word
    assert_eq!(c.exec_divs(&mut bus, src_dn(1), 0), 18);
}

#[test]
fn divs_sign_adjustment_matches_register_direct_fixtures() {
    let mut bus = LinearMemoryBus::new(0x100);
    let mut c = cpu(CpuType::M68000);
    // SingleStepTests DIVS fixture 036: positive dividend / positive divisor.
    c.set_d(0, 0x462A_588A);
    c.set_d(1, 0x7F67_5925);
    assert_eq!(c.exec_divs(&mut bus, src_dn(1), 0), 130);
    // SingleStepTests DIVS fixture 070: negative dividend / positive divisor.
    c.set_d(0, 0xA0D3_273D);
    c.set_d(1, 0x8E8C_7902);
    assert_eq!(c.exec_divs(&mut bus, src_dn(1), 0), 142);
}

#[test]
fn flat_timing_retained_off_the_68000() {
    let mut bus = LinearMemoryBus::new(0x100);
    let mut c = cpu(CpuType::M68020);
    c.set_d(1, 0xFFFF);
    assert_eq!(c.exec_mulu(&mut bus, src_dn(1), 0), 38);
    c.set_d(0, 0);
    c.set_d(1, 1);
    assert_eq!(c.exec_divu(&mut bus, src_dn(1), 0), 140);
}

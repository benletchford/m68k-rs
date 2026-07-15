//! Bcc/BSR/BRA displacement-byte decoding across CPU generations.
//!
//! A displacement byte of $FF selects the 32-bit branch displacement only on
//! the 68020+ (Bcc.L). The 68000 and 68010 have no long form: $FF is the
//! ordinary signed 8-bit displacement -1, and no extension word is consumed.

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, StepResult};

/// Reset into supervisor mode with PC at 0x0200 and the given opcode there.
fn setup(cpu_type: CpuType, opcode: u16) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, 0x0800); // SSP
    bus.write_long(0x04, 0x0200); // initial PC
    bus.write_word(0x0200, opcode);
    // A plausible 32-bit displacement after the opcode: a pre-68020 CPU must
    // NOT consume it, a 68020+ must.
    bus.write_long(0x0202, 0x0000_0100);

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    (cpu, bus)
}

#[test]
fn m68000_not_taken_bcc_ff_is_short_and_consumes_no_extension() {
    // BNE $FF with Z set: not taken. The 68000 falls through to the next
    // word; decoding $FF as a long displacement would skip 4 extra bytes.
    let (mut cpu, mut bus) = setup(CpuType::M68000, 0x66FF);
    cpu.set_sr(0x2704); // Z set

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { cycles: 8 }));
    assert_eq!(cpu.pc, 0x0202, "fall-through must be opcode + 2");
}

#[test]
fn m68000_bsr_ff_pushes_return_after_opcode_and_branches_minus_one() {
    let (mut cpu, mut bus) = setup(CpuType::M68000, 0x61FF);

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(
        bus.read_long(0x07FC),
        0x0202,
        "return address is the word after the opcode, not opcode + 6"
    );
    assert_eq!(
        cpu.pc, 0x0201,
        "displacement is -1 (the odd target faults on the next fetch, as on hardware)"
    );
}

#[test]
fn m68020_taken_bcc_ff_consumes_long_displacement() {
    // BNE.L with Z clear: taken, PC = base + 32-bit displacement.
    let (mut cpu, mut bus) = setup(CpuType::M68020, 0x66FF);
    cpu.set_sr(0x2700); // Z clear

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.pc, 0x0302, "0x0202 + 0x100");
}

//! CAS and CAS2 set the condition codes as CMP Dc,<ea> does: the memory
//! operand minus the compare register (M68000PRM CAS/CAS2: "destination -
//! compare operand -> cc"). X is not affected.

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, StepResult};

const SR_X: u16 = 0x0010;
const SR_N: u16 = 0x0008;
const SR_Z: u16 = 0x0004;
const SR_V: u16 = 0x0002;
const SR_C: u16 = 0x0001;
const CCR: u16 = 0x001f;

const PC: u32 = 0x0200;
const MEM: u32 = 0x0600;

fn setup(cpu_type: CpuType, words: &[u16]) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, 0x0800);
    bus.write_long(0x04, PC);
    for (i, w) in words.iter().enumerate() {
        bus.write_word(PC + 2 * i as u32, *w);
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    cpu.set_a(0, MEM);
    cpu.set_a(1, MEM + 8);
    (cpu, bus)
}

fn step(cpu: &mut CpuCore, bus: &mut LinearMemoryBus) {
    assert!(matches!(cpu.step(bus), StepResult::Ok { .. }));
}

#[test]
fn cas_mismatch_flags_are_memory_minus_compare_register() {
    for cpu_type in [CpuType::M68020, CpuType::M68030, CpuType::M68040] {
        // CAS.B D0,D1,(A0): memory $00, compare $01 -> $00 - $01 = $FF
        // borrows, so N and C are set; the reverse subtraction sets neither.
        let (mut cpu, mut bus) = setup(cpu_type, &[0x0ad0, 0x0040, 0x4e71, 0x4e71]);
        bus.write_byte(MEM, 0x00);
        cpu.set_d(0, 0xffff_ff01);
        cpu.set_d(1, 0x5a3c_965a);
        cpu.set_sr(0x2700 | CCR);

        step(&mut cpu, &mut bus);

        assert_eq!(cpu.get_sr() & CCR, SR_X | SR_N | SR_C, "{cpu_type:?}");
        assert_eq!(cpu.d(0), 0xffff_ff00, "{cpu_type:?}: memory loaded into Dc");
        assert_eq!(bus.read_byte(MEM), 0x00, "{cpu_type:?}: memory unchanged");
    }
}

#[test]
fn cas_mismatch_overflow_follows_memory_minus_compare_register() {
    // CAS.W D0,D1,(A0): memory $8000, compare $0001 -> $8000 - $0001 =
    // $7FFF overflows without a borrow (V set, N and C clear).
    let (mut cpu, mut bus) = setup(CpuType::M68020, &[0x0cd0, 0x0040, 0x4e71, 0x4e71]);
    bus.write_word(MEM, 0x8000);
    cpu.set_d(0, 0x0000_0001);
    cpu.set_sr(0x2700);

    step(&mut cpu, &mut bus);

    assert_eq!(cpu.get_sr() & CCR, SR_V);
    assert_eq!(cpu.d(0), 0x0000_8000);
}

#[test]
fn cas_match_sets_z_and_stores_the_update_register() {
    // CAS.L D0,D1,(A0) with equal operands.
    let (mut cpu, mut bus) = setup(CpuType::M68020, &[0x0ed0, 0x0040, 0x4e71, 0x4e71]);
    bus.write_long(MEM, 0x1234_5678);
    cpu.set_d(0, 0x1234_5678);
    cpu.set_d(1, 0xcafe_f00d);
    cpu.set_sr(0x2700 | SR_X | SR_N | SR_C);

    step(&mut cpu, &mut bus);

    assert_eq!(cpu.get_sr() & CCR, SR_X | SR_Z);
    assert_eq!(bus.read_long(MEM), 0xcafe_f00d);
    assert_eq!(cpu.d(0), 0x1234_5678);
}

#[test]
fn cas2_first_pair_mismatch_flags_are_memory_minus_compare_register() {
    // CAS2.W D0:D1,D2:D3,(A0):(A1): memory $1234 against compare $1235
    // borrows (N and C set).
    let (mut cpu, mut bus) = setup(CpuType::M68020, &[0x0cfc, 0x8080, 0x90c1, 0x4e71, 0x4e71]);
    bus.write_word(MEM, 0x1234);
    bus.write_word(MEM + 8, 0x8000);
    cpu.set_d(0, 0xffff_1235);
    cpu.set_d(1, 0x5a3c_8000);
    cpu.set_d(2, 0xffff_aaaa);
    cpu.set_d(3, 0x0000_5555);
    cpu.set_sr(0x2700 | CCR);

    step(&mut cpu, &mut bus);

    assert_eq!(cpu.get_sr() & CCR, SR_X | SR_N | SR_C);
    assert_eq!(cpu.d(0), 0xffff_1234);
    assert_eq!(cpu.d(1), 0x5a3c_8000);
    assert_eq!(bus.read_word(MEM), 0x1234);
    assert_eq!(bus.read_word(MEM + 8), 0x8000);
}

#[test]
fn cas2_second_pair_mismatch_flags_come_from_the_second_compare() {
    // First pair equal, second pair memory $0000 against compare $0001.
    let (mut cpu, mut bus) = setup(CpuType::M68020, &[0x0cfc, 0x8080, 0x90c1, 0x4e71, 0x4e71]);
    bus.write_word(MEM, 0x1234);
    bus.write_word(MEM + 8, 0x0000);
    cpu.set_d(0, 0x0000_1234);
    cpu.set_d(1, 0x0000_0001);
    cpu.set_sr(0x2700);

    step(&mut cpu, &mut bus);

    assert_eq!(cpu.get_sr() & CCR, SR_N | SR_C);
    assert_eq!(cpu.d(1), 0x0000_0000);
    assert_eq!(bus.read_word(MEM), 0x1234, "no update on a mismatch");
}

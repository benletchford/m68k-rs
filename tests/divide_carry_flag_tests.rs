//! The divide instructions always clear C, including when the quotient
//! overflows and when the divisor is zero (M68000PRM DIVS/DIVU condition
//! codes: "C - always cleared"). X is never affected.

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, StepResult};

const SR_X: u16 = 0x0010;
const SR_V: u16 = 0x0002;
const SR_C: u16 = 0x0001;

const SSP: u32 = 0x0800;
const PC: u32 = 0x0200;
const HANDLER: u32 = 0x0400;

/// Reset into supervisor mode with `words` at the reset PC and the
/// zero-divide vector pointing at `HANDLER`.
fn setup(cpu_type: CpuType, words: &[u16]) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, SSP);
    bus.write_long(0x04, PC);
    bus.write_long(5 * 4, HANDLER);
    for (i, w) in words.iter().enumerate() {
        bus.write_word(PC + 2 * i as u32, *w);
    }
    bus.write_word(HANDLER, 0x4e71);
    bus.write_word(HANDLER + 2, 0x4e71);

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    (cpu, bus)
}

fn step(cpu: &mut CpuCore, bus: &mut LinearMemoryBus) {
    assert!(matches!(cpu.step(bus), StepResult::Ok { .. }));
}

const WORD_MODELS: [CpuType; 5] = [
    CpuType::M68000,
    CpuType::M68010,
    CpuType::M68020,
    CpuType::M68040,
    CpuType::M68060,
];

const LONG_MODELS: [CpuType; 4] = [
    CpuType::M68020,
    CpuType::M68030,
    CpuType::M68040,
    CpuType::M68060,
];

#[test]
fn word_divide_overflow_clears_carry_and_keeps_x() {
    // DIVU.W D1,D0 and DIVS.W D1,D0 with a quotient of $10000.
    for opcode in [0x80c1u16, 0x81c1] {
        for cpu_type in WORD_MODELS {
            let (mut cpu, mut bus) = setup(cpu_type, &[opcode, 0x4e71, 0x4e71]);
            cpu.set_d(0, 0x0001_0000);
            cpu.set_d(1, 0x0000_0001);
            cpu.set_sr(0x2700 | SR_X | SR_C);

            step(&mut cpu, &mut bus);

            let sr = cpu.get_sr();
            assert_eq!(cpu.pc, PC + 2, "{cpu_type:?} {opcode:04x}: no trap");
            assert_eq!(cpu.d(0), 0x0001_0000, "{cpu_type:?} {opcode:04x}");
            assert_ne!(sr & SR_V, 0, "{cpu_type:?} {opcode:04x}: V set");
            assert_eq!(sr & SR_C, 0, "{cpu_type:?} {opcode:04x}: C cleared");
            assert_ne!(sr & SR_X, 0, "{cpu_type:?} {opcode:04x}: X kept");
        }
    }
}

#[test]
fn word_divide_by_zero_stacks_a_cleared_carry() {
    for opcode in [0x80c1u16, 0x81c1] {
        for cpu_type in WORD_MODELS {
            let (mut cpu, mut bus) = setup(cpu_type, &[opcode, 0x4e71, 0x4e71]);
            cpu.set_d(0, 0x1234_5678);
            cpu.set_d(1, 0);
            cpu.set_sr(0x2700 | SR_X | SR_C);

            step(&mut cpu, &mut bus);

            assert_eq!(cpu.pc, HANDLER, "{cpu_type:?} {opcode:04x}: vector 5");
            let stacked_sr = bus.read_word(cpu.sp());
            assert_eq!(stacked_sr & SR_C, 0, "{cpu_type:?} {opcode:04x}: C");
            assert_ne!(stacked_sr & SR_X, 0, "{cpu_type:?} {opcode:04x}: X");
            assert_eq!(cpu.d(0), 0x1234_5678, "{cpu_type:?} {opcode:04x}");
        }
    }
}

#[test]
fn long_divide_by_zero_stacks_a_cleared_carry() {
    // DIVU.L D1,D0 / DIVS.L D1,D0 / DIVUL.L D1,D2:D0 / DIVSL.L D1,D2:D0.
    for ext in [0x0000u16, 0x0800, 0x0002, 0x0802] {
        for cpu_type in LONG_MODELS {
            let (mut cpu, mut bus) = setup(cpu_type, &[0x4c41, ext, 0x4e71, 0x4e71]);
            cpu.set_d(0, 0x1234_5678);
            cpu.set_d(1, 0);
            cpu.set_d(2, 0x0bad_f00d);
            cpu.set_sr(0x2700 | SR_X | SR_C);

            step(&mut cpu, &mut bus);

            assert_eq!(cpu.pc, HANDLER, "{cpu_type:?} {ext:04x}: vector 5");
            let stacked_sr = bus.read_word(cpu.sp());
            assert_eq!(stacked_sr & SR_C, 0, "{cpu_type:?} {ext:04x}: C");
            assert_ne!(stacked_sr & SR_X, 0, "{cpu_type:?} {ext:04x}: X");
            assert_eq!(cpu.d(0), 0x1234_5678, "{cpu_type:?} {ext:04x}");
            assert_eq!(cpu.d(2), 0x0bad_f00d, "{cpu_type:?} {ext:04x}");
        }
    }
}

#[test]
fn long_divide_overflow_clears_carry() {
    // DIVU.L D1,D2:D0 with D2:D0 / D1 = $1_0000_0000.
    for cpu_type in [CpuType::M68020, CpuType::M68030, CpuType::M68040] {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x4c41, 0x0402, 0x4e71, 0x4e71]);
        cpu.set_d(0, 0);
        cpu.set_d(1, 1);
        cpu.set_d(2, 1);
        cpu.set_sr(0x2700 | SR_X | SR_C);

        step(&mut cpu, &mut bus);

        let sr = cpu.get_sr();
        assert_eq!(cpu.pc, PC + 4, "{cpu_type:?}: no trap");
        assert_ne!(sr & SR_V, 0, "{cpu_type:?}: V set");
        assert_eq!(sr & SR_C, 0, "{cpu_type:?}: C cleared");
        assert_ne!(sr & SR_X, 0, "{cpu_type:?}: X kept");
    }
}

#[test]
fn most_negative_dividend_by_minus_one_overflows_on_every_model() {
    // DIVS.W D1,D0 with D0 = $80000000 and D1 = -1: the quotient is +2^31,
    // far outside 16 bits. V is set, C cleared, D0 left unchanged.
    for cpu_type in WORD_MODELS {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x81c1, 0x4e71, 0x4e71]);
        cpu.set_d(0, 0x8000_0000);
        cpu.set_d(1, 0x0000_ffff);
        cpu.set_sr(0x2700 | SR_X | SR_C);

        step(&mut cpu, &mut bus);

        let sr = cpu.get_sr();
        assert_eq!(cpu.pc, PC + 2, "{cpu_type:?}: no trap");
        assert_eq!(cpu.d(0), 0x8000_0000, "{cpu_type:?}: D0 unchanged");
        assert_ne!(sr & SR_V, 0, "{cpu_type:?}: V set");
        assert_eq!(sr & SR_C, 0, "{cpu_type:?}: C cleared");
        assert_ne!(sr & SR_X, 0, "{cpu_type:?}: X kept");
    }
}

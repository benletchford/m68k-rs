//! Which exception stack frame formats RTE accepts on each model, and how
//! many words it pops for them.
//!
//! - 68010: $0, $8
//! - 68020/68030: $0, $1 (throwaway), $2, $9, $A, $B
//! - 68040: $0, $1, $2, $3, $7 (68LC040/68EC040 also $4)
//! - 68060: $0, $2, $3, $4 (M68060UM 8.2.7); no throwaway frame
//!
//! Any other format code takes the format-error exception (vector 14) with
//! a format $0 frame whose PC is the RTE and whose SR is the SR the RTE
//! began with.

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, NoOpHleHandler, StepResult};

const SSP: u32 = 0x0e00;
const PC: u32 = 0x0200;
const RETURN_PC: u32 = 0x0600;
const FORMAT_ERROR_HANDLER: u32 = 0x0400;

/// Build an RTE at `PC` with a frame of `format` at `SSP` returning to
/// `RETURN_PC` with SR $2015, then execute the RTE.
fn run_rte(cpu_type: CpuType, format: u16) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, SSP);
    bus.write_long(0x04, PC);
    bus.write_long(14 * 4, FORMAT_ERROR_HANDLER);
    bus.write_word(PC, 0x4e73);
    bus.write_word(FORMAT_ERROR_HANDLER, 0x4e71);
    bus.write_word(RETURN_PC, 0x4e71);
    bus.write_word(SSP, 0x2015);
    bus.write_long(SSP + 2, RETURN_PC);
    bus.write_word(SSP + 6, format << 12);

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    cpu.set_sr(0x2700);
    cpu.set_a(7, SSP);

    let mut hle = NoOpHleHandler;
    let r = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(r, StepResult::Ok { .. }), "step result {r:?}");
    (cpu, bus)
}

fn assert_accepted(cpu_type: CpuType, format: u16, frame_words: u32) {
    let (cpu, _) = run_rte(cpu_type, format);
    let tag = format!("{cpu_type:?} format ${format:X}");
    assert_eq!(cpu.pc, RETURN_PC, "{tag}: returns to the stacked PC");
    assert_eq!(cpu.get_sr(), 0x2015, "{tag}: SR restored");
    assert_eq!(cpu.sp(), SSP + 2 * frame_words, "{tag}: frame size");
}

fn assert_format_error(cpu_type: CpuType, format: u16) {
    let (cpu, mut bus) = run_rte(cpu_type, format);
    let tag = format!("{cpu_type:?} format ${format:X}");
    assert_eq!(cpu.pc, FORMAT_ERROR_HANDLER, "{tag}: vector 14");
    let sp = cpu.sp();
    assert_eq!(sp, SSP - 8, "{tag}: format $0 frame below the bad one");
    assert_eq!(bus.read_word(sp), 0x2700, "{tag}: SR the RTE began with");
    assert_eq!(bus.read_long(sp + 2), PC, "{tag}: stacked PC is the RTE");
    assert_eq!(bus.read_word(sp + 6), 14 << 2, "{tag}: format/vector");
}

const FAMILY_020_030: [CpuType; 4] = [
    CpuType::M68EC020,
    CpuType::M68020,
    CpuType::M68EC030,
    CpuType::M68030,
];
const FAMILY_040: [CpuType; 3] = [CpuType::M68EC040, CpuType::M68LC040, CpuType::M68040];

#[test]
fn format_0_and_2_pop_four_and_six_words_on_every_68020_plus() {
    for cpu_type in FAMILY_020_030
        .into_iter()
        .chain(FAMILY_040)
        .chain([CpuType::M68060])
    {
        assert_accepted(cpu_type, 0, 4);
        assert_accepted(cpu_type, 2, 6);
    }
}

#[test]
fn format_3_is_a_six_word_frame_on_the_68040_and_68060_only() {
    for cpu_type in FAMILY_040.into_iter().chain([CpuType::M68060]) {
        assert_accepted(cpu_type, 3, 6);
    }
    for cpu_type in FAMILY_020_030 {
        assert_format_error(cpu_type, 3);
    }
}

#[test]
fn format_4_is_an_eight_word_frame_on_the_68060_and_fpu_less_68040s() {
    for cpu_type in [CpuType::M68060, CpuType::M68LC040, CpuType::M68EC040] {
        assert_accepted(cpu_type, 4, 8);
    }
    for cpu_type in FAMILY_020_030 {
        assert_format_error(cpu_type, 4);
    }
}

#[test]
fn format_9_is_a_ten_word_frame_on_the_68020_and_68030_only() {
    for cpu_type in FAMILY_020_030 {
        assert_accepted(cpu_type, 9, 10);
    }
    for cpu_type in FAMILY_040.into_iter().chain([CpuType::M68060]) {
        assert_format_error(cpu_type, 9);
    }
}

#[test]
fn throwaway_frame_is_a_format_error_on_the_68060() {
    assert_format_error(CpuType::M68060, 1);
}

#[test]
fn throwaway_frame_continues_with_the_frame_below_before_the_68060() {
    for cpu_type in FAMILY_020_030.into_iter().chain(FAMILY_040) {
        let mut bus = LinearMemoryBus::new(0x1000);
        bus.write_long(0x00, SSP);
        bus.write_long(0x04, PC);
        bus.write_word(PC, 0x4e73);
        bus.write_word(RETURN_PC, 0x4e71);
        // Throwaway frame, then a format $0 frame below it.
        bus.write_word(SSP, 0x2715);
        bus.write_long(SSP + 2, 0x0bad_f00c);
        bus.write_word(SSP + 6, 0x1000);
        bus.write_word(SSP + 8, 0x2700);
        bus.write_long(SSP + 10, RETURN_PC);
        bus.write_word(SSP + 14, 0x0000);

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(cpu_type);
        cpu.reset(&mut bus);
        cpu.set_sr(0x2700);
        cpu.set_a(7, SSP);
        let mut hle = NoOpHleHandler;
        let _ = cpu.step_with_hle_handler(&mut bus, &mut hle);

        assert_eq!(cpu.pc, RETURN_PC, "{cpu_type:?}");
        assert_eq!(cpu.sp(), SSP + 16, "{cpu_type:?}: both frames popped");
    }
}

#[test]
fn unassigned_formats_take_the_format_error() {
    for cpu_type in FAMILY_020_030 {
        for format in [5, 6, 7, 8, 0xc, 0xd, 0xe, 0xf] {
            assert_format_error(cpu_type, format);
        }
    }
    for format in [5, 6, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf] {
        assert_format_error(CpuType::M68040, format);
    }
    for format in [1, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe, 0xf] {
        assert_format_error(CpuType::M68060, format);
    }
}

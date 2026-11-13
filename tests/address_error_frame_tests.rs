//! Address-error exception stack frames per model.
//!
//! - 68000: the seven-word group 0 frame. From the new stack pointer up:
//!   status word (FC in bits 2-0, I/N in bit 3, R/W in bit 4, 1 = read),
//!   32-bit access address, instruction register, SR, PC (M68000UM 6.3.9.1,
//!   figure 6-7). The stacked PC is the prefetch address at the fault.
//! - 68010: format $8 (29 words) with the special status word and the
//!   fault address.
//! - 68020/68030: only an instruction fetch can be misaligned; it stacks
//!   the bus fault frame (format $B here), as a bus error does.
//! - 68040/68060: the six-word format $2 frame whose address field is the
//!   fetch address with bit 0 cleared.

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, NoOpHleHandler, StepResult};

const SSP: u32 = 0x0f00;
const USP: u32 = 0x0c00;
const PC: u32 = 0x0200;
const HANDLER: u32 = 0x0400;
const ODD_DATA: u32 = 0x0601;
const ODD_CODE: u32 = 0x0701;

fn setup(cpu_type: CpuType, words: &[u16]) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, SSP);
    bus.write_long(0x04, PC);
    bus.write_long(3 * 4, HANDLER);
    bus.write_word(HANDLER, 0x4e71);
    bus.write_word(HANDLER + 2, 0x4e71);
    for (i, w) in words.iter().enumerate() {
        bus.write_word(PC + 2 * i as u32, *w);
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    cpu.set_a(7, SSP);
    cpu.set_sr(0x2704);
    cpu.set_a(0, ODD_DATA);
    (cpu, bus)
}

fn step(cpu: &mut CpuCore, bus: &mut LinearMemoryBus) {
    let mut hle = NoOpHleHandler;
    let r = cpu.step_with_hle_handler(bus, &mut hle);
    assert!(matches!(r, StepResult::Ok { .. }), "step result {r:?}");
}

struct Group0 {
    status: u16,
    address: u32,
    ir: u16,
    sr: u16,
    pc: u32,
}

fn group0(cpu: &CpuCore, bus: &mut LinearMemoryBus) -> Group0 {
    assert_eq!(cpu.pc, HANDLER, "address-error handler entered");
    assert_eq!(cpu.sp(), SSP - 14, "seven-word frame");
    let sp = cpu.sp();
    Group0 {
        status: bus.read_word(sp),
        address: bus.read_long(sp + 2),
        ir: bus.read_word(sp + 6),
        sr: bus.read_word(sp + 8),
        pc: bus.read_long(sp + 10),
    }
}

#[test]
fn m68000_word_read_stacks_the_group_0_frame_in_hardware_order() {
    // MOVE.W (A0),D0
    let (mut cpu, mut bus) = setup(CpuType::M68000, &[0x3010, 0x4e71, 0x4e71]);
    step(&mut cpu, &mut bus);
    let f = group0(&cpu, &mut bus);
    assert_eq!(f.status & 0x1f, 0x15, "read, instruction, supervisor data");
    assert_eq!(f.status & !0x1f, 0x3010 & !0x1f, "upper bits carry the IR");
    assert_eq!(f.address, ODD_DATA);
    assert_eq!(f.ir, 0x3010);
    assert_eq!(f.sr, 0x2704);
    assert_eq!(f.pc, PC + 2);
    assert_eq!(cpu.get_sr() & 0xa000, 0x2000, "supervisor, trace off");
}

#[test]
fn m68000_access_address_keeps_all_32_bits() {
    let (mut cpu, mut bus) = setup(CpuType::M68000, &[0x3010, 0x4e71, 0x4e71]);
    cpu.set_a(0, 0x2500_0000 | ODD_DATA);
    step(&mut cpu, &mut bus);
    assert_eq!(group0(&cpu, &mut bus).address, 0x2500_0000 | ODD_DATA);
}

#[test]
fn m68000_long_read_and_writes_report_direction() {
    // MOVE.L (A0),D0 / MOVE.W D1,(A0) / MOVE.L D1,(A0)
    for (opcode, status, pc_offset) in [
        (0x2010u16, 0x15u16, 2u32),
        (0x3081, 0x05, 4),
        (0x2081, 0x05, 4),
    ] {
        let (mut cpu, mut bus) = setup(CpuType::M68000, &[opcode, 0x4e71, 0x4e71]);
        step(&mut cpu, &mut bus);
        let f = group0(&cpu, &mut bus);
        assert_eq!(f.status & 0x1f, status, "{opcode:04x}");
        assert_eq!(f.address, ODD_DATA, "{opcode:04x}");
        assert_eq!(f.ir, opcode, "{opcode:04x}");
        assert_eq!(f.pc, PC + pc_offset, "{opcode:04x}");
    }
}

#[test]
fn m68000_stacked_pc_follows_the_addressing_mode() {
    // (opcode words, A0, expected PC offset from the opcode)
    let cases: [(&[u16], u32, u32); 6] = [
        // MOVE.W -(A0),D0: word predecrement prefetches first.
        (&[0x3020, 0x4e71, 0x4e71], ODD_DATA + 2, 4),
        // MOVE.L -(A0),D0
        (&[0x2020, 0x4e71, 0x4e71], ODD_DATA + 4, 2),
        // MOVE.W 2(A0),D0: the displacement word is the prefetch address.
        (&[0x3028, 0x0002, 0x4e71], ODD_DATA - 2, 2),
        // MOVE.W $0601.W,D0
        (&[0x3038, 0x0601, 0x4e71], 0, 4),
        // MOVE.W $00000601.L,D0
        (&[0x3039, 0x0000, 0x0601, 0x4e71], 0, 6),
        // MOVEM.W (A0),D0-D1
        (&[0x4c90, 0x0003, 0x4e71], ODD_DATA, 6),
    ];
    for (words, a0, pc_offset) in cases {
        let (mut cpu, mut bus) = setup(CpuType::M68000, words);
        cpu.set_a(0, a0);
        step(&mut cpu, &mut bus);
        let f = group0(&cpu, &mut bus);
        assert_eq!(f.address, ODD_DATA, "{:04x}", words[0]);
        assert_eq!(f.pc, PC + pc_offset, "{:04x}", words[0]);
    }
}

#[test]
fn m68000_user_mode_read_reports_user_data_space() {
    let (mut cpu, mut bus) = setup(CpuType::M68000, &[0x3010, 0x4e71, 0x4e71]);
    cpu.set_sr(0x0704);
    cpu.set_a(7, USP);
    step(&mut cpu, &mut bus);
    let f = group0(&cpu, &mut bus);
    assert_eq!(f.status & 0x1f, 0x11, "read, instruction, user data");
    assert_eq!(f.sr, 0x0704);
}

#[test]
fn m68000_pc_relative_read_reports_program_space() {
    // MOVE.W $401(PC),D0 -> PC + 2 + $401 is odd.
    let (mut cpu, mut bus) = setup(CpuType::M68000, &[0x303a, 0x0401, 0x4e71]);
    step(&mut cpu, &mut bus);
    let f = group0(&cpu, &mut bus);
    assert_eq!(
        f.status & 0x1f,
        0x16,
        "read, instruction, supervisor program"
    );
    assert_eq!(f.address, PC + 2 + 0x401);
}

#[test]
fn m68000_jump_to_an_odd_address_faults_on_the_fetch() {
    // JMP (A0)
    let (mut cpu, mut bus) = setup(CpuType::M68000, &[0x4ed0, 0x4e71, 0x4e71]);
    cpu.set_a(0, ODD_CODE);
    step(&mut cpu, &mut bus);
    if cpu.pc != HANDLER {
        step(&mut cpu, &mut bus);
    }
    let f = group0(&cpu, &mut bus);
    assert_eq!(
        f.status & 0x1f,
        0x16,
        "read, instruction, supervisor program"
    );
    assert_eq!(f.address, ODD_CODE);
    assert_eq!(f.ir, 0x4ed0);
    assert_eq!(f.sr & 0xff00, 0x2700);
}

#[test]
fn m68010_stacks_format_8_with_status_word_and_fault_address() {
    // (opcode, SSW IF/DF/RW/FC bits)
    for (opcode, ssw) in [
        (0x3010u16, 0x1105u16), // MOVE.W (A0),D0: data read
        (0x3081, 0x0005),       // MOVE.W D1,(A0): write
    ] {
        let (mut cpu, mut bus) = setup(CpuType::M68010, &[opcode, 0x4e71, 0x4e71]);
        step(&mut cpu, &mut bus);
        let sp = cpu.sp();
        assert_eq!(cpu.pc, HANDLER, "{opcode:04x}");
        assert_eq!(sp, SSP - 58, "{opcode:04x}: 29-word frame");
        assert_eq!(bus.read_word(sp), 0x2704, "{opcode:04x}: SR");
        assert_eq!(bus.read_word(sp + 6), 0x800c, "{opcode:04x}: format/vector");
        assert_eq!(bus.read_word(sp + 8) & 0x3107, ssw, "{opcode:04x}: SSW");
        assert_eq!(
            bus.read_long(sp + 10),
            ODD_DATA,
            "{opcode:04x}: fault address"
        );
        let pc = bus.read_long(sp + 2);
        assert!(
            (PC..=PC + 10).contains(&pc),
            "{opcode:04x}: stacked PC {pc:#x}"
        );
    }

    // JMP (A0) to an odd address: instruction fetch.
    let (mut cpu, mut bus) = setup(CpuType::M68010, &[0x4ed0, 0x4e71, 0x4e71]);
    cpu.set_a(0, ODD_CODE);
    step(&mut cpu, &mut bus);
    if cpu.pc != HANDLER {
        step(&mut cpu, &mut bus);
    }
    let sp = cpu.sp();
    assert_eq!(sp, SSP - 58);
    assert_eq!(
        bus.read_word(sp + 8) & 0x3107,
        0x2106,
        "IF, read, supervisor program"
    );
    assert_eq!(bus.read_long(sp + 10), ODD_CODE);
}

fn jump_to_odd(cpu_type: CpuType) -> (CpuCore, LinearMemoryBus) {
    let (mut cpu, mut bus) = setup(cpu_type, &[0x4ed0, 0x4e71, 0x4e71]);
    cpu.set_a(0, ODD_CODE);
    step(&mut cpu, &mut bus);
    if cpu.pc != HANDLER {
        step(&mut cpu, &mut bus);
    }
    assert_eq!(cpu.pc, HANDLER, "{cpu_type:?}: vector 3");
    (cpu, bus)
}

#[test]
fn m68020_and_m68030_stack_the_bus_fault_frame_for_an_odd_fetch() {
    for cpu_type in [
        CpuType::M68EC020,
        CpuType::M68020,
        CpuType::M68EC030,
        CpuType::M68030,
    ] {
        let (cpu, mut bus) = jump_to_odd(cpu_type);
        let sp = cpu.sp();
        assert_eq!(sp, SSP - 92, "{cpu_type:?}: 46-word format $B frame");
        assert_eq!(bus.read_word(sp) & 0xff00, 0x2700, "{cpu_type:?}: SR");
        assert_eq!(bus.read_word(sp + 6), 0xb00c, "{cpu_type:?}: format/vector");
        let ssw = bus.read_word(sp + 0x0a);
        assert_eq!(
            ssw & 0x5000,
            0x5000,
            "{cpu_type:?}: stage B fault and rerun"
        );
        assert_eq!(ssw & 0x0100, 0, "{cpu_type:?}: not a data fault");
        assert_eq!(ssw & 7, 6, "{cpu_type:?}: supervisor program space");
        assert_eq!(
            bus.read_long(sp + 0x24),
            ODD_CODE,
            "{cpu_type:?}: stage B address"
        );
    }
}

#[test]
fn m68020_data_accesses_may_be_misaligned() {
    for cpu_type in [CpuType::M68020, CpuType::M68030, CpuType::M68040] {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x3010, 0x4e71, 0x4e71]);
        bus.write_byte(ODD_DATA, 0x12);
        bus.write_byte(ODD_DATA + 1, 0x34);
        step(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, PC + 2, "{cpu_type:?}: no exception");
        assert_eq!(cpu.d(0) & 0xffff, 0x1234, "{cpu_type:?}");
    }
}

#[test]
fn m68040_and_m68060_stack_format_2_for_an_odd_fetch() {
    for cpu_type in [
        CpuType::M68EC040,
        CpuType::M68LC040,
        CpuType::M68040,
        CpuType::M68060,
    ] {
        let (cpu, mut bus) = jump_to_odd(cpu_type);
        let sp = cpu.sp();
        assert_eq!(sp, SSP - 12, "{cpu_type:?}: six-word frame");
        assert_eq!(bus.read_word(sp) & 0xff00, 0x2700, "{cpu_type:?}: SR");
        assert_eq!(bus.read_word(sp + 6), 0x200c, "{cpu_type:?}: format/vector");
        assert_eq!(
            bus.read_long(sp + 8),
            ODD_CODE & !1,
            "{cpu_type:?}: address field"
        );
    }
}

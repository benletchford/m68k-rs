//! How a pending T1 trace interacts with an exception raised by the traced
//! instruction.
//!
//! - An instruction that does not execute (illegal, line A, privilege
//!   violation) is not traced on any model: only its own exception frame is
//!   stacked, holding the traced SR (M68000UM 6.3.8).
//! - An instruction that completes by trapping (TRAP #n, zero divide, CHK,
//!   TRAPV) is traced on the 68000 through 68030: the trap is processed
//!   first and the trace exception right after it, so the trace frame sits
//!   on top with the trap handler's address and the trap-entry SR
//!   (M68020UM 6.1.8).
//! - The 68040 and 68060 do not report a trace that is pending together
//!   with such a trap (M68040UM 8.3, M68060UM 8.2.6).

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, LinearMemoryBus, NoOpHleHandler, StepResult};

const SSP: u32 = 0x0f00;
const USP: u32 = 0x0c00;
const PC: u32 = 0x0200;
const TRACE_HANDLER: u32 = 0x0400;

const ALL_CPUS: [CpuType; 6] = [
    CpuType::M68000,
    CpuType::M68010,
    CpuType::M68020,
    CpuType::M68030,
    CpuType::M68040,
    CpuType::M68060,
];

fn handler(vector: u32) -> u32 {
    if vector == 9 {
        TRACE_HANDLER
    } else {
        0x0500 + vector * 4
    }
}

fn setup(cpu_type: CpuType, words: &[u16], sr: u16) -> (CpuCore, LinearMemoryBus) {
    let mut bus = LinearMemoryBus::new(0x1000);
    bus.write_long(0x00, SSP);
    bus.write_long(0x04, PC);
    for vector in 2..64 {
        bus.write_long(vector * 4, handler(vector));
        bus.write_word(handler(vector), 0x4e71);
        bus.write_word(handler(vector) + 2, 0x4e71);
    }
    for (i, w) in words.iter().enumerate() {
        bus.write_word(PC + 2 * i as u32, *w);
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    cpu.set_a(7, SSP);
    if sr & 0x2000 == 0 {
        cpu.set_sr(sr);
        cpu.set_a(7, USP);
    } else {
        cpu.set_sr(sr);
    }
    (cpu, bus)
}

fn step(cpu: &mut CpuCore, bus: &mut LinearMemoryBus) {
    let mut hle = NoOpHleHandler;
    let r = cpu.step_with_hle_handler(bus, &mut hle);
    assert!(matches!(r, StepResult::Ok { .. }), "step result {r:?}");
}

/// Bytes of the frame the model stacks for `vector`.
fn frame_size(cpu_type: CpuType, vector: u32) -> u32 {
    let format_2 = matches!(vector, 5 | 6 | 7 | 9);
    match cpu_type {
        CpuType::M68000 => 6,
        CpuType::M68010 => 8,
        _ if format_2 => 12,
        _ => 8,
    }
}

fn is_040_or_060(cpu_type: CpuType) -> bool {
    matches!(cpu_type, CpuType::M68040 | CpuType::M68060)
}

/// The instruction took `vector` and nothing else: one frame holding the
/// traced SR, and the CPU sits at that vector's handler with tracing off.
fn assert_only_exception(
    cpu: &CpuCore,
    bus: &mut LinearMemoryBus,
    cpu_type: CpuType,
    vector: u32,
    traced_sr: u16,
) {
    let tag = format!("{cpu_type:?} vector {vector}");
    assert_eq!(cpu.pc, handler(vector), "{tag}: handler entered");
    assert_eq!(
        cpu.sp(),
        SSP - frame_size(cpu_type, vector),
        "{tag}: exactly one frame"
    );
    assert_eq!(bus.read_word(cpu.sp()), traced_sr, "{tag}: stacked SR");
    assert_eq!(cpu.get_sr() & 0xc000, 0, "{tag}: trace cleared in handler");
}

/// The instruction took `vector` and was then traced: the trace frame is on
/// top, stacking the trap handler's address and the trap-entry SR.
fn assert_exception_then_trace(
    cpu: &CpuCore,
    bus: &mut LinearMemoryBus,
    cpu_type: CpuType,
    vector: u32,
    traced_sr: u16,
) {
    let tag = format!("{cpu_type:?} vector {vector}");
    let trap_frame = SSP - frame_size(cpu_type, vector);
    let trace_frame = trap_frame - frame_size(cpu_type, 9);
    assert_eq!(cpu.pc, TRACE_HANDLER, "{tag}: trace handler runs first");
    assert_eq!(cpu.sp(), trace_frame, "{tag}: two frames");
    assert_eq!(
        bus.read_word(trace_frame),
        0x2000 | (traced_sr & 0x071f),
        "{tag}: trace frame holds the trap-entry SR (S set, T clear)"
    );
    assert_eq!(
        bus.read_long(trace_frame + 2),
        handler(vector),
        "{tag}: trace frame PC is the trap handler"
    );
    assert_eq!(bus.read_word(trap_frame), traced_sr, "{tag}: trap frame SR");
}

#[test]
fn illegal_instruction_is_not_traced() {
    for cpu_type in ALL_CPUS {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x4afc, 0x4e71], 0xa700);
        step(&mut cpu, &mut bus);
        assert_only_exception(&cpu, &mut bus, cpu_type, 4, 0xa700);
        assert_eq!(bus.read_long(cpu.sp() + 2), PC, "{cpu_type:?}: stacked PC");
    }
}

#[test]
fn line_a_instruction_is_not_traced() {
    for cpu_type in ALL_CPUS {
        let (mut cpu, mut bus) = setup(cpu_type, &[0xa123, 0x4e71], 0xa700);
        step(&mut cpu, &mut bus);
        assert_only_exception(&cpu, &mut bus, cpu_type, 10, 0xa700);
    }
}

#[test]
fn privilege_violation_is_not_traced() {
    for cpu_type in ALL_CPUS {
        // RESET in user mode.
        let (mut cpu, mut bus) = setup(cpu_type, &[0x4e70, 0x4e71], 0x8000);
        step(&mut cpu, &mut bus);
        assert_only_exception(&cpu, &mut bus, cpu_type, 8, 0x8000);
    }
}

#[test]
fn trap_is_traced_after_its_exception_before_the_68040() {
    for cpu_type in ALL_CPUS {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x4e45, 0x4e71], 0xa700);
        step(&mut cpu, &mut bus);
        if is_040_or_060(cpu_type) {
            assert_only_exception(&cpu, &mut bus, cpu_type, 37, 0xa700);
        } else {
            assert_exception_then_trace(&cpu, &mut bus, cpu_type, 37, 0xa700);
        }
        assert_eq!(
            bus.read_long(SSP - frame_size(cpu_type, 37) + 2),
            PC + 2,
            "{cpu_type:?}: the TRAP frame stacks the next instruction"
        );
    }
}

#[test]
fn zero_divide_is_traced_after_its_exception_before_the_68040() {
    for cpu_type in ALL_CPUS {
        // DIVU.W D1,D0 with D1 = 0.
        let (mut cpu, mut bus) = setup(cpu_type, &[0x80c1, 0x4e71], 0xa700);
        cpu.set_d(0, 1);
        cpu.set_d(1, 0);
        step(&mut cpu, &mut bus);
        if is_040_or_060(cpu_type) {
            assert_only_exception(&cpu, &mut bus, cpu_type, 5, 0xa700);
        } else {
            assert_exception_then_trace(&cpu, &mut bus, cpu_type, 5, 0xa700);
        }
    }
}

#[test]
fn trapv_is_traced_after_its_exception_before_the_68040() {
    for cpu_type in ALL_CPUS {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x4e76, 0x4e71], 0xa702);
        step(&mut cpu, &mut bus);
        if is_040_or_060(cpu_type) {
            assert_only_exception(&cpu, &mut bus, cpu_type, 7, 0xa702);
        } else {
            assert_exception_then_trace(&cpu, &mut bus, cpu_type, 7, 0xa702);
        }
    }
}

#[test]
fn execute_loop_does_not_trace_an_illegal_instruction() {
    for cpu_type in ALL_CPUS {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x4afc, 0x4e71], 0xa700);
        // Enough for the exception entry and a few handler NOPs.
        cpu.execute(&mut bus, 60);
        let frame = SSP - frame_size(cpu_type, 4);
        assert_eq!(cpu.sp(), frame, "{cpu_type:?}: exactly one frame");
        assert_eq!(bus.read_word(frame), 0xa700, "{cpu_type:?}: stacked SR");
        assert!(
            (handler(4)..handler(4) + 0x40).contains(&cpu.pc),
            "{cpu_type:?}: running the illegal-instruction handler"
        );
    }
}

#[test]
fn completed_instruction_is_still_traced() {
    for cpu_type in ALL_CPUS {
        let (mut cpu, mut bus) = setup(cpu_type, &[0x4e71, 0x4e71], 0xa700);
        step(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, TRACE_HANDLER, "{cpu_type:?}");
        assert_eq!(cpu.sp(), SSP - frame_size(cpu_type, 9), "{cpu_type:?}");
        assert_eq!(bus.read_long(cpu.sp() + 2), PC + 2, "{cpu_type:?}");
    }
}

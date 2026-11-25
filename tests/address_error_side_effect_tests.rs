//! What an MC68000 and an MC68010 leave behind after an address error.
//!
//! M68000UM 6.3.9.1 describes the group 0 frame (figure 6-7) but only bounds
//! the stacked PC ("advanced 2 to 10 bytes" past the opcode) and says
//! nothing about address registers the instruction had already updated. The
//! rules checked here were measured on an Amiga 500 with an MC68000 and
//! with an MC68010: the vAmigaTS suite (Dirk Hoffmann,
//! `CPU/Exceptions/StackFrame/addrerr11` to `addrerr43`) records the frame
//! and the registers of one faulting instruction per test and publishes a
//! photograph of each result on the real machine. The comment on each case
//! names the measurement it rests on.
//!
//! The programs, addresses and register values below are this crate's own.
//! Each case runs one instruction from `CODE` with the registers preset,
//! and every expectation is written as the rule: where the register ends
//! up, which address the frame reports, and how far past the opcode the
//! stacked PC is.
//!
//! The 68000 group 0 frame is, from SP up: the status word (R/W in bit 4,
//! I/N in bit 3, FC2-FC0 in bits 2-0, the instruction register in the bits
//! above), the 32-bit access address, the instruction register, SR and PC.

use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus, NoOpHleHandler, StepResult};

/// Supervisor stack pointer the reset vector loads.
const SSP: u32 = 0x8000;
/// Address of the instruction under test.
const CODE: u32 = 0x2000;
/// Address-error handler (vector 3).
const HANDLER: u32 = 0x0400;
/// An odd data address.
const ODD: u32 = 0x3001;
/// An even data address inside the RAM.
const EVEN: u32 = 0x4000;
/// Supervisor mode, interrupts masked, X set and the other flags clear.
const SUPERVISOR: u16 = 0x2710;
const NOP: u16 = 0x4e71;

/// Function codes the frames report.
const USER_DATA: u16 = 1;
const SUPERVISOR_DATA: u16 = 5;
const SUPERVISOR_PROGRAM: u16 = 6;

struct Machine {
    cpu: CpuCore,
    bus: LinearMemoryBus,
}

/// A machine about to execute `code` at `CODE`, followed by NOPs.
fn machine(cpu_type: CpuType, code: &[u16]) -> Machine {
    let mut bus = LinearMemoryBus::new(0x1_0000);
    bus.write_long(0, SSP);
    bus.write_long(4, CODE);
    bus.write_long(3 * 4, HANDLER);
    bus.write_word(HANDLER, NOP);
    // A positive, non-zero operand at the even data address, so an
    // instruction that sets the condition codes from it leaves N and Z
    // clear.
    bus.write_long(EVEN - 4, 0x0123_4567);
    bus.write_long(EVEN, 0x0123_4567);
    for (i, word) in code.iter().chain([NOP, NOP].iter()).enumerate() {
        bus.write_word(CODE + 2 * i as u32, *word);
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    cpu.set_sr(SUPERVISOR);
    Machine { cpu, bus }
}

/// Step until the address-error handler is entered.
fn run_to_handler(m: &mut Machine, what: &str) {
    let mut hle = NoOpHleHandler;
    for _ in 0..4 {
        if m.cpu.pc == HANDLER {
            return;
        }
        let r = m.cpu.step_with_hle_handler(&mut m.bus, &mut hle);
        assert!(matches!(r, StepResult::Ok { .. }), "{what}: {r:?}");
    }
    assert_eq!(m.cpu.pc, HANDLER, "{what}: no address error");
}

/// The fields of a 68000 group 0 frame.
#[derive(Clone, Copy)]
struct Group0 {
    /// The faulting bus cycle was a read.
    read: bool,
    /// Function code of the faulting cycle.
    fc: u16,
    /// Address of the faulting cycle.
    access: u32,
    /// Instruction register.
    ir: u16,
    sr: u16,
    pc: u32,
}

/// Compare the 68000 frame on the supervisor stack with `want`. Nothing
/// but the 14-byte frame may be on the stack, and I/N must be clear.
fn assert_group0(m: &mut Machine, what: &str, want: Group0) {
    let sp = m.cpu.sp();
    assert_eq!(sp, SSP - 14, "{what}: stack pointer");
    let status = (want.ir & !0x001f) | if want.read { 0x0010 } else { 0 } | want.fc;
    assert_eq!(m.bus.read_word(sp), status, "{what}: status word");
    assert_eq!(
        m.bus.read_long(sp + 2),
        want.access,
        "{what}: access address"
    );
    assert_eq!(
        m.bus.read_word(sp + 6),
        want.ir,
        "{what}: instruction register"
    );
    assert_eq!(m.bus.read_word(sp + 8), want.sr, "{what}: stacked SR");
    assert_eq!(m.bus.read_long(sp + 10), want.pc, "{what}: stacked PC");
}

/// One instruction faulting on a data access in supervisor mode, with A2
/// and A3 preset.
struct DataFault {
    what: &'static str,
    code: &'static [u16],
    a2: u32,
    a3: u32,
    read: bool,
    access: u32,
    /// Stacked PC, in bytes past the opcode.
    pc_past_opcode: u32,
    /// A2 and A3 when the handler is entered.
    a2_after: u32,
    a3_after: u32,
}

fn check_data_faults_68000(cases: &[DataFault]) {
    for case in cases {
        let mut m = machine(CpuType::M68000, case.code);
        m.cpu.set_a(2, case.a2);
        m.cpu.set_a(3, case.a3);
        m.cpu.set_d(1, 0x0000_0102);
        run_to_handler(&mut m, case.what);
        assert_group0(
            &mut m,
            case.what,
            Group0 {
                read: case.read,
                fc: SUPERVISOR_DATA,
                access: case.access,
                ir: case.code[0],
                sr: SUPERVISOR,
                pc: CODE + case.pc_past_opcode,
            },
        );
        assert_eq!(m.cpu.a(2), case.a2_after, "{}: A2", case.what);
        assert_eq!(m.cpu.a(3), case.a3_after, "{}: A3", case.what);
    }
}

#[test]
fn m68000_predecrement_read_fault_leaves_the_register_decremented() {
    // The decremented address is written back to An before the operand
    // cycle starts, so the fault leaves An at the faulting address.
    // Measured: addrerr13, 14, 18, 19 and 39.
    check_data_faults_68000(&[
        DataFault {
            what: "move.w -(a2),d1",
            code: &[0x3222],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD - 2,
            pc_past_opcode: 4,
            a2_after: ODD - 2,
            a3_after: EVEN,
        },
        DataFault {
            what: "move.l -(a2),d1",
            code: &[0x2222],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD - 4,
            pc_past_opcode: 2,
            a2_after: ODD - 4,
            a3_after: EVEN,
        },
        DataFault {
            what: "add.l d1,-(a2)",
            code: &[0xd3a2],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD - 4,
            pc_past_opcode: 2,
            a2_after: ODD - 4,
            a3_after: EVEN,
        },
        DataFault {
            what: "tst.l -(a2)",
            code: &[0x4aa2],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD - 4,
            pc_past_opcode: 2,
            a2_after: ODD - 4,
            a3_after: EVEN,
        },
        DataFault {
            what: "movea.w -(a2),a3",
            code: &[0x3662],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD - 2,
            pc_past_opcode: 4,
            a2_after: ODD - 2,
            a3_after: EVEN,
        },
    ]);
}

#[test]
fn m68000_postincrement_and_long_predecrement_write_faults_leave_the_register_unchanged() {
    // An (An)+ access that faults has not incremented An, and MOVE.L to
    // -(An) does not write the decremented address back before its first
    // (low word) write. Measured: addrerr11, 12, 15, 16, 17, 22, 27, 29, 31.
    check_data_faults_68000(&[
        DataFault {
            what: "move.w (a2)+,d1",
            code: &[0x321a],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 2,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "move.l (a2)+,d1",
            code: &[0x221a],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 2,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "move.w d1,(a2)+",
            code: &[0x34c1],
            a2: ODD,
            a3: EVEN,
            read: false,
            access: ODD,
            pc_past_opcode: 4,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "move.l d1,-(a2)",
            code: &[0x2501],
            a2: ODD,
            a3: EVEN,
            read: false,
            access: ODD - 2,
            pc_past_opcode: 4,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "add.w d1,(a2)+",
            code: &[0xd35a],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 2,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "movem.l (a2)+,d1-d2",
            code: &[0x4cda, 0x0006],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 6,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "divu.w (a2)+,d1",
            code: &[0x82da],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 2,
            a2_after: ODD,
            a3_after: EVEN,
        },
        // The 68000 reads the destination of MOVE from SR and of CLR
        // before writing it, so the faulting cycle is a read.
        DataFault {
            what: "move.w sr,(a2)+",
            code: &[0x40da],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 2,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "clr.l (xxx).L",
            code: &[0x42b9, (ODD >> 16) as u16, ODD as u16],
            a2: EVEN,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 6,
            a2_after: EVEN,
            a3_after: EVEN,
        },
    ]);
}

#[test]
fn m68000_source_register_update_survives_a_destination_fault() {
    // Once the source operand has been read, its (An)+ or -(An) update is
    // not undone by a fault on the destination. A fault on the source
    // leaves both registers alone; ADDX.L -(Ay),-(Ax) reads the low word
    // first. Measured: addrerr20, 21, 23, 24, 25 and 26.
    check_data_faults_68000(&[
        DataFault {
            what: "move.w (a2)+,(a3)+, source odd",
            code: &[0x36da],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 2,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "move.w (a2)+,(a3)+, destination odd",
            code: &[0x36da],
            a2: EVEN,
            a3: ODD,
            read: false,
            access: ODD,
            pc_past_opcode: 4,
            a2_after: EVEN + 2,
            a3_after: ODD,
        },
        DataFault {
            what: "addx.l -(a2),-(a3), source odd",
            code: &[0xd78a],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD - 2,
            pc_past_opcode: 4,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "addx.l -(a2),-(a3), destination odd",
            code: &[0xd78a],
            a2: EVEN,
            a3: ODD,
            read: true,
            access: ODD - 2,
            pc_past_opcode: 4,
            a2_after: EVEN - 4,
            a3_after: ODD,
        },
        DataFault {
            what: "cmpm.l (a2)+,(a3)+, source odd",
            code: &[0xb78a],
            a2: ODD,
            a3: EVEN,
            read: true,
            access: ODD,
            pc_past_opcode: 4,
            a2_after: ODD,
            a3_after: EVEN,
        },
        DataFault {
            what: "cmpm.w (a2)+,(a3)+, destination odd",
            code: &[0xb74a],
            a2: EVEN,
            a3: ODD,
            read: true,
            access: ODD,
            pc_past_opcode: 4,
            a2_after: EVEN + 2,
            a3_after: ODD,
        },
    ]);
}

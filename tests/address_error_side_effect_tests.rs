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

/// JMP, JSR and DBcc to an odd target: the instruction, its preset A2 and
/// D1, and the odd target.
struct OddTarget {
    what: &'static str,
    code: &'static [u16],
    a2: u32,
    d1: u32,
    target: u32,
}

const ODD_TARGETS: [OddTarget; 4] = [
    OddTarget {
        what: "jmp (a2)",
        code: &[0x4ed2],
        a2: ODD,
        d1: 0,
        target: ODD,
    },
    OddTarget {
        what: "jsr 6(a2)",
        code: &[0x4eaa, 0x0006],
        a2: ODD,
        d1: 0,
        target: ODD + 6,
    },
    // The displacement is relative to the extension word.
    OddTarget {
        what: "dbra d1,*+5, branch taken",
        code: &[0x51c9, 0x0003],
        a2: EVEN,
        d1: 5,
        target: CODE + 5,
    },
    OddTarget {
        what: "dbra d1,*+5, counter expired",
        code: &[0x51c9, 0x0003],
        a2: EVEN,
        d1: 0,
        target: CODE + 5,
    },
];

fn odd_target_machine(cpu_type: CpuType, case: &OddTarget) -> Machine {
    let mut m = machine(cpu_type, case.code);
    m.cpu.set_a(2, case.a2);
    m.cpu.set_d(1, case.d1);
    m
}

fn assert_odd_target_frame_68000(m: &mut Machine, case: &OddTarget) {
    assert_group0(
        m,
        case.what,
        Group0 {
            read: true,
            fc: SUPERVISOR_PROGRAM,
            access: case.target,
            ir: case.code[0],
            sr: SUPERVISOR,
            pc: CODE + 2,
        },
    );
}

#[test]
fn m68000_odd_flow_target_faults_inside_the_instruction() {
    // The fault is a program-space read of the target, taken before the
    // instruction changes anything: the PC stacked is the address of the
    // word after the opcode, not the target; JSR has pushed no return
    // address (the stack holds the frame alone); and DBcc faults whether
    // the branch is taken or the counter has expired.
    // Measured: addrerr28, 40, 41 and 42.
    for case in &ODD_TARGETS {
        let mut m = odd_target_machine(CpuType::M68000, case);
        run_to_handler(&mut m, case.what);
        assert_odd_target_frame_68000(&mut m, case);
    }
}

#[test]
fn m68000_decoded_fast_path_raises_the_same_odd_flow_target_fault() {
    // run_batch executes JMP, JSR and DBcc through the decoded fast path;
    // an odd target must be left to full dispatch.
    for case in &ODD_TARGETS {
        let mut m = odd_target_machine(CpuType::M68000, case);
        let result = m.cpu.run_batch(&mut m.bus, 16, &[HANDLER]);
        assert_eq!(m.cpu.pc, HANDLER, "{}: {result:?}", case.what);
        assert_odd_target_frame_68000(&mut m, case);
    }
}

/// Size of the 68010 format $8 bus and address error frame.
const FORMAT_8_BYTES: u32 = 58;

/// Compare the architecturally defined words of a 68010 address-error
/// frame: SR, the format and vector word, the special status word and the
/// fault address. The stacked PC is not compared: this core stacks the
/// instruction's own address and restarts the instruction from it on RTE,
/// where the hardware continues it from the frame's internal words.
fn assert_format_8(m: &mut Machine, what: &str, sr: u16, ssw: u16, fault_address: u32) {
    let sp = m.cpu.sp();
    assert_eq!(sp, SSP - FORMAT_8_BYTES, "{what}: stack pointer");
    assert_eq!(m.bus.read_word(sp), sr, "{what}: stacked SR");
    assert_eq!(m.bus.read_word(sp + 6), 0x800c, "{what}: format and vector");
    assert_eq!(m.bus.read_word(sp + 8), ssw, "{what}: special status word");
    assert_eq!(
        m.bus.read_long(sp + 10),
        fault_address,
        "{what}: fault address"
    );
}

#[test]
fn m68010_odd_flow_target_faults_before_the_instruction_changes_anything() {
    // As on the 68000, with the 68010's frame: SSW $2106 (an instruction
    // fetch, read, supervisor program), the target as the fault address,
    // and nothing pushed by JSR. Measured: addrerr28, 40, 41 and 42 on the
    // MC68010.
    for case in &ODD_TARGETS {
        let mut m = odd_target_machine(CpuType::M68010, case);
        run_to_handler(&mut m, case.what);
        assert_format_8(&mut m, case.what, SUPERVISOR, 0x2106, case.target);
    }
}

/// An odd user stack pointer.
const ODD_USP: u32 = 0x5001;

/// Drop to user mode with the given condition codes and user stack.
fn enter_user_mode(m: &mut Machine, ccr: u16, usp: u32) {
    m.cpu.set_usp(usp);
    m.cpu.set_sr(ccr);
}

#[test]
fn m68000_pea_push_fault_stacks_the_pc_past_the_final_prefetch() {
    // PEA (An) and PEA (d16,An) make their final prefetch before the push,
    // so a fault on the push stacks the address two bytes past that
    // prefetch; PEA (xxx).W pushes first. Measured: addrerr32, 33 and 34.
    let cases: [(&str, &[u16], u32); 3] = [
        ("pea (a1)", &[0x4851], 4),
        ("pea 4(a1)", &[0x4869, 0x0004], 6),
        ("pea (xxx).W", &[0x4878, 0x1234], 4),
    ];
    for (what, code, pc_past_opcode) in cases {
        let mut m = machine(CpuType::M68000, code);
        m.cpu.set_a(1, EVEN);
        enter_user_mode(&mut m, 0, ODD_USP);
        run_to_handler(&mut m, what);
        assert_group0(
            &mut m,
            what,
            Group0 {
                read: false,
                fc: USER_DATA,
                access: ODD_USP - 4,
                ir: code[0],
                sr: 0,
                pc: CODE + pc_past_opcode,
            },
        );
    }
}

#[test]
fn m68000_movem_indexed_fault_stacks_the_extension_word_address() {
    // movem.l (4,a2,d1.w),d3-d5: the stacked PC is the address of the
    // brief extension word, four bytes past the opcode, where MOVEM (An)+
    // stacks the address after its register mask plus two. Measured:
    // addrerr43 (and addrerr22 for the (An)+ form).
    let code = [0x4cf2, 0x0038, 0x1004];
    let what = "movem.l (4,a2,d1.w),d3-d5";
    let mut m = machine(CpuType::M68000, &code);
    m.cpu.set_a(2, ODD);
    m.cpu.set_d(1, 0);
    run_to_handler(&mut m, what);
    assert_group0(
        &mut m,
        what,
        Group0 {
            read: true,
            fc: SUPERVISOR_DATA,
            access: ODD + 4,
            ir: code[0],
            sr: SUPERVISOR,
            pc: CODE + 4,
        },
    );
    assert_eq!(m.cpu.a(2), ODD, "{what}: A2");
}

/// move.w #$0123,-(a1): a positive, non-zero source.
const MOVE_WORD_TO_PREDECREMENT: [u16; 2] = [0x333c, 0x0123];
const CARRY: u16 = 0x0001;

#[test]
fn m68000_move_word_to_predecrement_fault_stacks_the_next_opcode_and_the_new_ccr() {
    // MOVE.W to -(An) makes its final prefetch before the write, so the
    // instruction register (and with it the upper bits of the status word)
    // already holds the next opcode. The condition codes are set from the
    // source before the write: carry set beforehand is clear in the
    // stacked SR. Measured: addrerr35 (carry clear) and addrerr36 (carry
    // set), both in user mode.
    for ccr in [0, CARRY] {
        let what = "move.w #$0123,-(a1)";
        let mut m = machine(CpuType::M68000, &MOVE_WORD_TO_PREDECREMENT);
        m.cpu.set_a(1, ODD);
        enter_user_mode(&mut m, ccr, EVEN);
        run_to_handler(&mut m, what);
        assert_group0(
            &mut m,
            what,
            Group0 {
                read: false,
                fc: USER_DATA,
                access: ODD - 2,
                ir: NOP,
                sr: 0,
                pc: CODE + 6,
            },
        );
    }
}

#[test]
fn m68010_sets_the_ccr_before_a_faulting_write() {
    // CLR on the 68010 does not read its destination first, and sets Z
    // before the write faults: the frame stacks Z set, with X kept, and a
    // write in the special status word. The first cycle of the long write
    // is the low word, so the fault address is two past the operand
    // address. Measured: addrerr31 on the MC68010.
    let what = "clr.l (xxx).L";
    let mut m = machine(CpuType::M68010, &[0x42b9, (ODD >> 16) as u16, ODD as u16]);
    run_to_handler(&mut m, what);
    assert_format_8(&mut m, what, SUPERVISOR | 0x0004, SUPERVISOR_DATA, ODD + 2);

    // MOVE.W to -(An) with carry set beforehand stacks it clear.
    // Measured: addrerr36 on the MC68010.
    let what = "move.w #$0123,-(a1)";
    let mut m = machine(CpuType::M68010, &MOVE_WORD_TO_PREDECREMENT);
    m.cpu.set_a(1, ODD);
    enter_user_mode(&mut m, CARRY, EVEN);
    run_to_handler(&mut m, what);
    assert_format_8(&mut m, what, 0, USER_DATA, ODD - 2);
}

#[test]
fn m68000_and_m68010_divu_overflow_sets_n_and_clears_z() {
    // divu.w d2,d1 with a quotient that does not fit: V set, C clear, X
    // kept, and of the two flags M68000PRM leaves undefined, N set and Z
    // clear. Measured: addrerr30 on the MC68000 and on the MC68010.
    for cpu_type in [CpuType::M68000, CpuType::M68010] {
        let mut m = machine(cpu_type, &[0x82c2]);
        m.cpu.set_d(1, 0xffff_0000);
        m.cpu.set_d(2, 0x0000_8000);
        let mut hle = NoOpHleHandler;
        let r = m.cpu.step_with_hle_handler(&mut m.bus, &mut hle);
        assert!(matches!(r, StepResult::Ok { .. }), "{cpu_type:?}: {r:?}");
        assert_eq!(m.cpu.d(1), 0xffff_0000, "{cpu_type:?}: D1 is unchanged");
        // X | N | V.
        assert_eq!(m.cpu.get_sr() & 0x001f, 0x001a, "{cpu_type:?}: CCR");
    }
}

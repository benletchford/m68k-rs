//! 68040 floating-point exceptions the M68040FPSP services, and the FSAVE
//! state frames it reads: the unimplemented-instruction frame ($4130), the
//! busy frame ($4160), the operands and tags they carry, FRESTORE's
//! completion of an instruction the handler repaired (CU_SAVEPC = $FE), and
//! the pending exception a restored frame carries to the next floating-point
//! instruction.
//!
//! Frame field offsets are those of the FPSP's `fpsp.h`, counted here from
//! the frame's header long.

use m68k::core::memory::AddressBus;
use m68k::fpu::FloatX80;
use m68k::{CpuCore, CpuType, NoOpHleHandler, StepResult};

struct TestBus {
    memory: Vec<u8>,
}

impl TestBus {
    fn new() -> Self {
        Self {
            memory: vec![0; 0x10000],
        }
    }

    fn write_word_at(&mut self, addr: u32, value: u16) {
        self.memory[addr as usize..addr as usize + 2].copy_from_slice(&value.to_be_bytes());
    }

    fn write_long_at(&mut self, addr: u32, value: u32) {
        self.memory[addr as usize..addr as usize + 4].copy_from_slice(&value.to_be_bytes());
    }

    fn long_at(&self, addr: u32) -> u32 {
        let i = addr as usize;
        u32::from_be_bytes(self.memory[i..i + 4].try_into().unwrap())
    }

    fn word_at(&self, addr: u32) -> u16 {
        let i = addr as usize;
        u16::from_be_bytes(self.memory[i..i + 2].try_into().unwrap())
    }

    fn byte_at(&self, addr: u32) -> u8 {
        self.memory[addr as usize]
    }

    fn write_extended(&mut self, addr: u32, v: FloatX80) {
        let (exp, mantissa) = v.to_extended();
        self.write_long_at(addr, u32::from(exp) << 16);
        self.write_long_at(addr + 4, (mantissa >> 32) as u32);
        self.write_long_at(addr + 8, mantissa as u32);
    }

    fn extended_at(&self, addr: u32) -> FloatX80 {
        FloatX80::from_extended(
            self.word_at(addr),
            (u64::from(self.long_at(addr + 4)) << 32) | u64::from(self.long_at(addr + 8)),
        )
    }
}

impl AddressBus for TestBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[(address as usize) & 0xFFFF]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        self.word_at(address & 0xFFFF)
    }

    fn read_long(&mut self, address: u32) -> u32 {
        self.long_at(address & 0xFFFF)
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[(address as usize) & 0xFFFF] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        self.write_word_at(address & 0xFFFF, value);
    }

    fn write_long(&mut self, address: u32, value: u32) {
        self.write_long_at(address & 0xFFFF, value);
    }
}

const LINE_F: u32 = 0x0340;
const UNSUPP: u32 = 0x0380;
/// Handler address of arithmetic vector `n` (48-54).
const fn arith(n: u32) -> u32 {
    0x0400 + (n - 48) * 0x10
}

const OPERAND: u32 = 0x4000;
const FRAME: u32 = 0x5000;

// Frame field offsets from the header long of a busy frame; an
// unimplemented-instruction frame holds the fields from CMDREG3B on, $30
// lower.
const CU_SAVEPC: u32 = 0x08;
const WBTEMP: u32 = 0x18;
const FPSR_SHADOW: u32 = 0x24;
const FPIARCU: u32 = 0x28;
const CMDREG3B: u32 = 0x34;
const NMNEXC: u32 = 0x38;
const STAG: u32 = 0x3C;
const CMDREG1B: u32 = 0x40;
const DTAG: u32 = 0x44;
const E_BYTE: u32 = 0x48;
const T_BYTE: u32 = 0x49;
const FPTEMP: u32 = 0x4C;
const ETEMP: u32 = 0x58;
const UNIMP: u32 = 0x30;

const E1: u8 = 0x04;
const E3: u8 = 0x02;

fn one() -> FloatX80 {
    FloatX80::from_f64(1.0)
}

/// An extended denormal: zero exponent, integer bit clear.
fn denormal() -> FloatX80 {
    FloatX80::from_extended(0x0000, 0x4000_0000_0000_0000)
}

/// A 68040 in supervisor mode with every vector this file uses pointed at
/// its own handler address and A0 at an operand buffer.
fn setup(traps: bool) -> (CpuCore, TestBus) {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68040);
    let mut bus = TestBus::new();
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0200);
    bus.write_long_at(11 * 4, LINE_F);
    bus.write_long_at(55 * 4, UNSUPP);
    for n in 48..=54 {
        bus.write_long_at(n * 4, arith(n));
    }
    cpu.reset(&mut bus);
    cpu.set_sr(0x2700);
    cpu.emulate_unimplemented_040 = !traps;
    cpu.dar[8] = OPERAND;
    for r in 0..8 {
        cpu.fpr[r] = one();
    }
    (cpu, bus)
}

/// Run one instruction placed at `pc` on a fresh supervisor stack.
fn run_at(cpu: &mut CpuCore, bus: &mut TestBus, pc: u32, words: &[u16]) -> StepResult {
    for (i, w) in words.iter().enumerate() {
        bus.write_word_at(pc + 2 * i as u32, *w);
    }
    cpu.pc = pc;
    cpu.dar[15] = 0x1000;
    let mut hle = NoOpHleHandler;
    cpu.step_with_hle_handler(bus, &mut hle)
}

fn run(cpu: &mut CpuCore, bus: &mut TestBus, words: &[u16]) -> StepResult {
    run_at(cpu, bus, 0x0200, words)
}

/// FSAVE (A1) with A1 = FRAME, returning the header long.
fn fsave(cpu: &mut CpuCore, bus: &mut TestBus) -> u32 {
    for a in FRAME..FRAME + 0x80 {
        bus.memory[a as usize] = 0xEE;
    }
    cpu.dar[9] = FRAME;
    run_at(cpu, bus, 0x0280, &[0xF311]);
    assert_eq!(cpu.pc, 0x0282, "FSAVE completes");
    bus.long_at(FRAME)
}

/// FRESTORE (A1) with A1 = FRAME.
fn frestore(cpu: &mut CpuCore, bus: &mut TestBus) {
    cpu.dar[9] = FRAME;
    run_at(cpu, bus, 0x0290, &[0xF351]);
    assert_eq!(cpu.pc, 0x0292, "FRESTORE completes");
}

/// FNOP: reports a pending pre-instruction exception.
fn fnop(cpu: &mut CpuCore, bus: &mut TestBus) {
    run_at(cpu, bus, 0x02A0, &[0xF280, 0x0000]);
}

/// The exception stack frame at the supervisor stack pointer: (format/vector
/// word, stacked PC).
fn stacked(cpu: &CpuCore, bus: &TestBus) -> (u16, u32) {
    let sp = cpu.dar[15];
    (bus.word_at(sp + 6), bus.long_at(sp + 2))
}

/// Normalize the operand at `addr` in a saved frame the way the FPSP's
/// mk_norm does, returning the tag byte it writes: the mantissa is shifted
/// up, the exponent goes negative and is stored in fifteen bits, and bit 4
/// (exponent bit 15) is set because the exponent is not above $3FFF.
fn fpsp_normalize(bus: &mut TestBus, addr: u32) -> u8 {
    let v = bus.extended_at(addr);
    let lz = v.mantissa.leading_zeros();
    let exp = i32::from(v.biased_exp()) - lz as i32;
    let word = (v.sign_exp & 0x8000) | (exp as u16 & 0x7FFF);
    bus.write_extended(addr, FloatX80::from_extended(word, v.mantissa << lz));
    u8::from(exp <= 0x3FFF) << 4
}

// ============================================================================
// Unimplemented instructions: Line-F format $2 and the $4130 frame
// ============================================================================

#[test]
fn unimplemented_instruction_saves_the_unimp_frame() {
    let (mut cpu, mut bus) = setup(true);
    cpu.fpr[1] = FloatX80::from_f64(0.5);
    // FSIN.X FP1,FP0
    run(&mut cpu, &mut bus, &[0xF200, 0x040E]);
    assert_eq!(cpu.pc, LINE_F);
    assert_eq!(stacked(&cpu, &bus), (0x202C, 0x0204));

    // FSAVE -(A1): 4 + $30 bytes.
    cpu.dar[9] = FRAME + 0x34;
    run_at(&mut cpu, &mut bus, 0x0280, &[0xF321]);
    assert_eq!(cpu.dar[9], FRAME, "the unimp frame is $34 bytes");
    assert_eq!(bus.long_at(FRAME), 0x4130_0000);
    let f = FRAME - UNIMP;
    assert_eq!(bus.word_at(f + CMDREG1B), 0x040E, "CMDREG1B");
    assert_eq!(bus.extended_at(f + ETEMP), FloatX80::from_f64(0.5));
    assert_eq!(bus.extended_at(f + FPTEMP), one());
    assert_eq!(bus.byte_at(f + STAG), 0x10, "normalized, exponent <= $3FFF");
    assert_eq!(bus.byte_at(f + E_BYTE), 0x00, "no E1/E3");
    assert_eq!(bus.byte_at(f + T_BYTE), 0x20, "U flag");

    // The FSAVE collected the state: the FPU is idle again.
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000);
    assert_eq!(bus.long_at(FRAME + 4), 0xEEEE_EEEE, "idle frame is 4 bytes");
}

#[test]
fn unimp_frame_carries_the_converted_source_and_its_tag() {
    // FSIN.L (A0),FP0: the long is converted to extended; 2.0 has an
    // exponent above $3FFF, so exponent bit 15 is clear.
    let (mut cpu, mut bus) = setup(true);
    bus.write_long_at(OPERAND, 2);
    run(&mut cpu, &mut bus, &[0xF210, 0x400E]);
    assert_eq!(stacked(&cpu, &bus), (0x202C, 0x0204));
    assert_eq!(bus.long_at(cpu.dar[15] + 8), OPERAND, "EA field");
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    let f = FRAME - UNIMP;
    assert_eq!(bus.extended_at(f + ETEMP), FloatX80::from_f64(2.0));
    assert_eq!(bus.byte_at(f + STAG), 0x00);

    // Zero, infinity and NaN sources: tags 1, 2, 3.
    for (value, stag) in [
        (FloatX80::zero(false), 0x30),
        (FloatX80::infinity(true), 0x40),
        (FloatX80::default_nan(), 0x60),
    ] {
        let (mut cpu, mut bus) = setup(true);
        bus.write_extended(OPERAND, value);
        run(&mut cpu, &mut bus, &[0xF210, 0x480E]); // FSIN.X (A0),FP0
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
        assert_eq!(bus.byte_at(f + STAG), stag, "{value:?}");
        assert_eq!(bus.extended_at(f + ETEMP), value);
    }

    // An extended denormal source: tag 4, operand as fetched.
    let (mut cpu, mut bus) = setup(true);
    bus.write_extended(OPERAND, denormal());
    run(&mut cpu, &mut bus, &[0xF210, 0x480E]);
    assert_eq!(cpu.pc, LINE_F, "the unimplemented instruction wins");
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    assert_eq!(bus.byte_at(f + STAG), 0x90);
    assert_eq!(bus.extended_at(f + ETEMP), denormal());

    // A single denormal source: tag 5, the fraction left-justified below a
    // clear integer bit under the unadjusted exponent $3F80 (the FPSP
    // rewrites it to $3F81 and normalizes).
    let (mut cpu, mut bus) = setup(true);
    bus.write_long_at(OPERAND, 0x8000_0003);
    run(&mut cpu, &mut bus, &[0xF210, 0x440E]); // FSIN.S (A0),FP0
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    assert_eq!(bus.byte_at(f + STAG), 0xB0);
    assert_eq!(bus.word_at(f + ETEMP), 0xBF80);
    assert_eq!(bus.long_at(f + ETEMP + 4), 0x0000_0300);
    assert_eq!(bus.long_at(f + ETEMP + 8), 0);

    // A single signaling NaN source: tag 3, still signaling in ETEMP.
    let (mut cpu, mut bus) = setup(true);
    bus.write_long_at(OPERAND, 0x7F80_0001);
    run(&mut cpu, &mut bus, &[0xF210, 0x440E]);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    assert_eq!(bus.byte_at(f + STAG), 0x60);
    assert_eq!(bus.word_at(f + ETEMP), 0x7FFF);
    assert_eq!(bus.long_at(f + ETEMP + 4), 0x8000_0100);

    // A double denormal source: tag 5 under $3C00.
    let (mut cpu, mut bus) = setup(true);
    bus.write_long_at(OPERAND, 0x0000_0000);
    bus.write_long_at(OPERAND + 4, 0x0000_0001);
    run(&mut cpu, &mut bus, &[0xF210, 0x540E]); // FSIN.D (A0),FP0
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    assert_eq!(bus.byte_at(f + STAG), 0xB0);
    assert_eq!(bus.word_at(f + ETEMP), 0x3C00);
    assert_eq!(bus.long_at(f + ETEMP + 4), 0);
    assert_eq!(bus.long_at(f + ETEMP + 8), 0x0000_0800);
}

#[test]
fn unimp_frame_of_a_dyadic_instruction_tags_the_destination() {
    let (mut cpu, mut bus) = setup(true);
    cpu.fpr[2] = denormal();
    cpu.fpr[3] = FloatX80::from_f64(3.0);
    // FMOD.X FP3,FP2
    run(&mut cpu, &mut bus, &[0xF200, 0x0D21]);
    assert_eq!(cpu.pc, LINE_F);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    let f = FRAME - UNIMP;
    assert_eq!(bus.word_at(f + CMDREG1B), 0x0D21);
    assert_eq!(bus.extended_at(f + ETEMP), FloatX80::from_f64(3.0));
    assert_eq!(bus.byte_at(f + STAG), 0x00);
    assert_eq!(bus.extended_at(f + FPTEMP), denormal());
    assert_eq!(bus.byte_at(f + DTAG), 0x90);
}

#[test]
fn unimp_frame_of_a_packed_source_sets_e1_and_holds_the_raw_operand() {
    let (mut cpu, mut bus) = setup(true);
    bus.write_long_at(OPERAND, 0x0001_0001);
    bus.write_long_at(OPERAND + 4, 0x2345_6789);
    bus.write_long_at(OPERAND + 8, 0x1234_5678);
    // FSIN.P (A0),FP0
    run(&mut cpu, &mut bus, &[0xF210, 0x4C0E]);
    assert_eq!(stacked(&cpu, &bus), (0x202C, 0x0204));
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    let f = FRAME - UNIMP;
    assert_eq!(bus.byte_at(f + E_BYTE) & E1, E1, "E1 marks a packed source");
    // The FPSP's get_op copies FPTEMP's last long over ETEMP's first before
    // converting the twelve bytes at ETEMP.
    assert_eq!(bus.long_at(f + FPTEMP + 8), 0x0001_0001);
    assert_eq!(bus.long_at(f + ETEMP + 4), 0x2345_6789);
    assert_eq!(bus.long_at(f + ETEMP + 8), 0x1234_5678);
}

#[test]
fn fmovecr_saves_its_command_word() {
    let (mut cpu, mut bus) = setup(true);
    run(&mut cpu, &mut bus, &[0xF200, 0x5C32]);
    assert_eq!(stacked(&cpu, &bus), (0x202C, 0x0204));
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    assert_eq!(bus.word_at(FRAME - UNIMP + CMDREG1B), 0x5C32);
}

// ============================================================================
// Unsupported data types: vector 55 and the $4160 frame
// ============================================================================

#[test]
fn denormal_memory_source_takes_vector_55_with_a_busy_frame() {
    let (mut cpu, mut bus) = setup(true);
    bus.write_extended(OPERAND, denormal());
    // FADD.X (A0),FP0
    run(&mut cpu, &mut bus, &[0xF210, 0x4822]);
    assert_eq!(cpu.pc, UNSUPP);
    // Format $0; the handler completes the instruction through FRESTORE, so
    // the stacked PC is past it.
    assert_eq!(stacked(&cpu, &bus), (0x00DC, 0x0204));
    assert_eq!(cpu.fpiar, 0x0200);
    assert_eq!(cpu.fpr[0], one(), "the instruction has not executed");

    cpu.dar[9] = FRAME + 0x64;
    run_at(&mut cpu, &mut bus, 0x0280, &[0xF321]); // FSAVE -(A1)
    assert_eq!(cpu.dar[9], FRAME, "the busy frame is $64 bytes");
    assert_eq!(bus.long_at(FRAME), 0x4160_0000);
    assert_eq!(bus.word_at(FRAME + CMDREG1B), 0x4822);
    assert_eq!(bus.byte_at(FRAME + E_BYTE) & (E1 | E3), E1);
    assert_eq!(bus.byte_at(FRAME + T_BYTE), 0x00);
    assert_eq!(bus.extended_at(FRAME + ETEMP), denormal());
    assert_eq!(bus.byte_at(FRAME + STAG), 0x90);
    assert_eq!(bus.extended_at(FRAME + FPTEMP), one());
    assert_eq!(bus.byte_at(FRAME + DTAG), 0x10);
    assert_eq!(bus.long_at(FRAME + FPIARCU), 0x0200);

    assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000, "idle after FSAVE");
}

/// (name, opcode, command word, memory operand longs, FP1, FP0).
type OperandCase = (&'static str, u16, u16, [u32; 3], FloatX80, FloatX80);

#[test]
fn every_unsupported_operand_form_traps() {
    // (opcode, command, memory operand longs, FP1, FP0): source formats and
    // register operands the 68040 hands to software.
    let d = denormal();
    let unnormal = FloatX80::from_extended(0x3FFF, 0x4000_0000_0000_0000);
    let cases: [OperandCase; 9] = [
        ("FMOVE.S denormal", 0xF210, 0x4400, [1, 0, 0], one(), one()),
        ("FMOVE.D denormal", 0xF210, 0x5400, [0, 1, 0], one(), one()),
        (
            "FMOVE.X unnormal",
            0xF210,
            0x4800,
            [0x3FFF_0000, 0x4000_0000, 0],
            one(),
            one(),
        ),
        ("FMOVE.X FP1 denormal", 0xF200, 0x0400, [0; 3], d, one()),
        (
            "FSQRT.X FP1 unnormal",
            0xF200,
            0x0404,
            [0; 3],
            unnormal,
            one(),
        ),
        ("FTST.X FP1 denormal", 0xF200, 0x043A, [0; 3], d, one()),
        (
            "FADD.X FP1,FP0 denormal dst",
            0xF200,
            0x0422,
            [0; 3],
            one(),
            d,
        ),
        (
            "FCMP.X FP1,FP0 denormal dst",
            0xF200,
            0x0438,
            [0; 3],
            one(),
            d,
        ),
        (
            "FDMUL.X FP1,FP0 denormal dst",
            0xF200,
            0x0467,
            [0; 3],
            one(),
            d,
        ),
    ];
    for (name, opcode, command, mem, fp1, fp0) in cases {
        let (mut cpu, mut bus) = setup(true);
        for (i, long) in mem.iter().enumerate() {
            bus.write_long_at(OPERAND + 4 * i as u32, *long);
        }
        cpu.fpr[1] = fp1;
        cpu.fpr[0] = fp0;
        run(&mut cpu, &mut bus, &[opcode, command]);
        assert_eq!(cpu.pc, UNSUPP, "{name}");
        assert_eq!(stacked(&cpu, &bus), (0x00DC, 0x0204), "{name}");
        assert_eq!(cpu.fpr[0], fp0, "{name}: destination untouched");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000, "{name}");
    }
}

#[test]
fn fsqrt_is_saved_with_its_internal_command_encoding() {
    // The 68040 holds FSQRT (opmode $04) as $05 in CMDREG1B; the FPSP's
    // res_func tells the square roots apart by that low bit.
    let (mut cpu, mut bus) = setup(true);
    cpu.fpr[1] = denormal();
    run(&mut cpu, &mut bus, &[0xF200, 0x0404]);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    assert_eq!(bus.word_at(FRAME + CMDREG1B), 0x0405);
}

#[test]
fn supported_operands_do_not_trap() {
    // A zero exponent with the integer bit set is a normalized number, a
    // zero mantissa is a zero whatever its exponent, and a monadic
    // instruction does not look at its destination.
    let min_normal = FloatX80::from_extended(0x0000, 0x8000_0000_0000_0000);
    let odd_zero = FloatX80::from_extended(0x1234, 0);
    for (command, fp1, fp0) in [
        (0x0400u16, min_normal, one()),
        (0x0400, odd_zero, one()),
        (0x0422, min_normal, one()),
        (0x0400, one(), denormal()),
        (0x0404, FloatX80::from_f64(4.0), denormal()),
    ] {
        let (mut cpu, mut bus) = setup(true);
        cpu.fpr[1] = fp1;
        cpu.fpr[0] = fp0;
        run(&mut cpu, &mut bus, &[0xF200, command]);
        assert_eq!(cpu.pc, 0x0204, "{command:#06x} executes in hardware");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000);
    }
}

#[test]
fn packed_source_takes_vector_55_past_the_instruction() {
    let (mut cpu, mut bus) = setup(true);
    bus.write_long_at(OPERAND, 0x0000_0001);
    bus.write_long_at(OPERAND + 4, 0x2000_0000);
    bus.write_long_at(OPERAND + 8, 0x0000_0003);
    cpu.fpr[1] = FloatX80::from_f64(1.5);
    // FADD.P (A0),FP1
    run(&mut cpu, &mut bus, &[0xF210, 0x4CA2]);
    assert_eq!(cpu.pc, UNSUPP);
    assert_eq!(stacked(&cpu, &bus), (0x00DC, 0x0204));
    assert_eq!(cpu.fpiar, 0x0200);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    assert_eq!(bus.word_at(FRAME + CMDREG1B), 0x4CA2);
    assert_eq!(bus.byte_at(FRAME + E_BYTE) & E1, E1);
    assert_eq!(bus.long_at(FRAME + FPTEMP + 8), 0x0000_0001);
    assert_eq!(bus.long_at(FRAME + ETEMP + 4), 0x2000_0000);
    assert_eq!(bus.long_at(FRAME + ETEMP + 8), 0x0000_0003);
    assert_eq!(bus.byte_at(FRAME + DTAG), 0x10, "FP1 = 1.5 is normalized");
}

#[test]
fn move_out_of_an_unsupported_value_is_a_post_instruction_exception() {
    // FMOVE.X, .S, .D, .L, .W, .B FP0,(A0) with FP0 denormal.
    for (fmt, _bytes) in [(2u16, 12u32), (1, 4), (5, 8), (0, 4), (4, 2), (6, 1)] {
        let (mut cpu, mut bus) = setup(true);
        cpu.fpr[0] = denormal();
        bus.write_long_at(OPERAND, 0xAAAA_AAAA);
        run(&mut cpu, &mut bus, &[0xF210, 0x6000 | (fmt << 10)]);
        assert_eq!(cpu.pc, UNSUPP, "format {fmt}");
        assert_eq!(stacked(&cpu, &bus), (0x30DC, 0x0204), "format {fmt}");
        assert_eq!(bus.long_at(cpu.dar[15] + 8), OPERAND, "EA field");
        assert_eq!(bus.long_at(OPERAND), 0xAAAA_AAAA, "nothing stored");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
        assert_eq!(bus.word_at(FRAME + CMDREG1B), 0x6000 | (fmt << 10));
        assert_eq!(bus.byte_at(FRAME + E_BYTE) & E1, E1);
        assert_eq!(bus.byte_at(FRAME + T_BYTE) & 0x10, 0x10, "T: move out");
        assert_eq!(bus.extended_at(FRAME + ETEMP), denormal());
        assert_eq!(bus.byte_at(FRAME + STAG), 0x90);
    }

    // To a data register: no effective address, the register is untouched.
    let (mut cpu, mut bus) = setup(true);
    cpu.fpr[0] = denormal();
    cpu.dar[1] = 0x5555_5555;
    run(&mut cpu, &mut bus, &[0xF201, 0x6000]); // FMOVE.L FP0,D1
    assert_eq!(stacked(&cpu, &bus), (0x30DC, 0x0204));
    assert_eq!(bus.long_at(cpu.dar[15] + 8), 0);
    assert_eq!(cpu.dar[1], 0x5555_5555);

    // A normalized value moves out in hardware.
    let (mut cpu, mut bus) = setup(true);
    run(&mut cpu, &mut bus, &[0xF210, 0x6800]);
    assert_eq!(cpu.pc, 0x0204);
    assert_eq!(bus.extended_at(OPERAND), one());
}

#[test]
fn packed_move_out_saves_a_busy_frame() {
    for command in [0x6C03u16, 0x7C10] {
        let (mut cpu, mut bus) = setup(true);
        cpu.fpr[0] = FloatX80::from_f64(2.5);
        run(&mut cpu, &mut bus, &[0xF210, command]);
        assert_eq!(stacked(&cpu, &bus), (0x30DC, 0x0204));
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
        assert_eq!(bus.word_at(FRAME + CMDREG1B), command);
        assert_eq!(bus.byte_at(FRAME + T_BYTE) & 0x10, 0x10);
        assert_eq!(bus.extended_at(FRAME + ETEMP), FloatX80::from_f64(2.5));
        assert_eq!(bus.byte_at(FRAME + STAG), 0x00);
    }
}

// ============================================================================
// FRESTORE: completing a repaired instruction
// ============================================================================

/// (name, instruction words, memory operand longs, FP1, FP0).
type RepairCase = (&'static str, [u16; 2], [u32; 3], FloatX80, FloatX80);

/// What the FPSP's res_func does to the saved busy frame before restoring
/// it: operands normalized, CU_SAVEPC = $FE, E1 and the non-maskable
/// exception bits cleared.
fn fpsp_fix_frame(bus: &mut TestBus) {
    if bus.byte_at(FRAME + DTAG) & 0x80 != 0 {
        let tag = fpsp_normalize(bus, FRAME + FPTEMP);
        bus.memory[(FRAME + DTAG) as usize] = tag;
    }
    if bus.byte_at(FRAME + STAG) & 0x80 != 0 {
        let tag = fpsp_normalize(bus, FRAME + ETEMP);
        bus.memory[(FRAME + STAG) as usize] = tag;
    }
    bus.memory[(FRAME + CU_SAVEPC) as usize] = 0xFE;
    bus.memory[(FRAME + E_BYTE) as usize] &= !E1;
    bus.write_word_at(FRAME + NMNEXC, 0);
}

/// Run `words` natively (no traps) from the same register state.
fn native(words: &[u16], mem: &[u32], fpr: [FloatX80; 8], fpcr: u32) -> (CpuCore, TestBus) {
    let (mut cpu, mut bus) = setup(false);
    for (i, long) in mem.iter().enumerate() {
        bus.write_long_at(OPERAND + 4 * i as u32, *long);
    }
    cpu.fpr = fpr;
    cpu.fpcr = fpcr;
    run(&mut cpu, &mut bus, words);
    assert_eq!(cpu.pc, 0x0204);
    (cpu, bus)
}

#[test]
fn frestore_completes_an_instruction_the_handler_repaired() {
    let d = denormal();
    let tiny = FloatX80::from_extended(0x8000, 0x0000_0000_0012_3456);
    let unnormal = FloatX80::from_extended(0x4002, 0x0123_4567_89AB_CDEF);
    let cases: [RepairCase; 8] = [
        (
            "FADD.X (A0),FP0",
            [0xF210, 0x4822],
            [0, 0x4000_0000, 0],
            one(),
            one(),
        ),
        ("FADD.X FP1,FP0 both", [0xF200, 0x0422], [0; 3], d, d),
        (
            "FMUL.X FP1,FP0",
            [0xF200, 0x0423],
            [0; 3],
            tiny,
            FloatX80::from_f64(1e30),
        ),
        ("FDIV.X FP1,FP0", [0xF200, 0x0420], [0; 3], one(), tiny),
        ("FSUB.X FP1,FP0", [0xF200, 0x0428], [0; 3], unnormal, one()),
        ("FSQRT.X FP1,FP0", [0xF200, 0x0404], [0; 3], d, one()),
        ("FCMP.X FP1,FP0", [0xF200, 0x0438], [0; 3], d, tiny),
        (
            "FSQRT.S (A0),FP0",
            [0xF210, 0x4404],
            [0x0000_0400, 0, 0],
            one(),
            one(),
        ),
    ];
    for (name, words, mem, fp1, fp0) in cases {
        let (mut cpu, mut bus) = setup(true);
        for (i, long) in mem.iter().enumerate() {
            bus.write_long_at(OPERAND + 4 * i as u32, *long);
        }
        cpu.fpr[1] = fp1;
        cpu.fpr[0] = fp0;
        let fpr = cpu.fpr;
        run(&mut cpu, &mut bus, &words);
        assert_eq!(cpu.pc, UNSUPP, "{name}");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000, "{name}");

        if bus.byte_at(FRAME + STAG) >> 5 == 5 {
            // get_op gives a single denormal its true exponent first.
            let sign = bus.word_at(FRAME + ETEMP) & 0x8000;
            bus.write_word_at(FRAME + ETEMP, sign | 0x3F81);
        }
        fpsp_fix_frame(&mut bus);
        frestore(&mut cpu, &mut bus);

        let (expect, _) = native(&words, &mem, fpr, 0);
        assert_eq!(cpu.fpr[0], expect.fpr[0], "{name}: result");
        assert_eq!(cpu.fpr[1], fp1, "{name}: source register");
        assert_eq!(cpu.fpsr, expect.fpsr, "{name}: FPSR");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000, "{name}: idle");
    }
}

#[test]
fn a_completed_instruction_keeps_the_handlers_shadow_exceptions() {
    // A packed source: the FPSP converts it in software, which is inexact
    // (INEX1), writes the extended value to ETEMP and hands the status to
    // the 68040 in the shadow FPSR with the S and X flags, so that the
    // restarted FADD reports the conversion's exception with its own.
    let (mut cpu, mut bus) = setup(true);
    bus.write_long_at(OPERAND, 0x4001_0001); // 1E-1
    run(&mut cpu, &mut bus, &[0xF210, 0x4C22]); // FADD.P (A0),FP0
    assert_eq!(cpu.pc, UNSUPP);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    let tenth = FloatX80::from_extended(0x3FFB, 0xCCCC_CCCC_CCCC_CCCD);
    bus.write_extended(FRAME + ETEMP, tenth);
    bus.memory[(FRAME + STAG) as usize] = 0x10;
    bus.write_long_at(FRAME + FPTEMP + 4, 0x8000_0000);
    bus.write_long_at(FRAME + FPTEMP + 8, 0);
    bus.memory[(FRAME + CU_SAVEPC) as usize] = 0xFE;
    bus.write_word_at(FRAME + NMNEXC, 0);
    bus.write_long_at(FRAME + FPSR_SHADOW, 0x0000_0108);
    bus.memory[(FRAME + E_BYTE) as usize] = 0x01; // E1 clear, S set
    bus.memory[(FRAME + T_BYTE) as usize] |= 0x80; // X
    cpu.fpsr = 0x0000_0108;
    frestore(&mut cpu, &mut bus);

    let mut f = FloatX80::from_f64(1.0);
    f.mantissa = 0x8CCC_CCCC_CCCC_CCCD; // 1 + 1E-1, rounded to nearest
    assert_eq!(cpu.fpr[0], f);
    assert_eq!(cpu.fpsr & 0xFF00, 0x0300, "INEX1 of the conversion, INEX2");
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000);
}

#[test]
fn a_completed_instruction_reports_its_enabled_exception_at_the_next_one() {
    let (mut cpu, mut bus) = setup(true);
    cpu.fpcr = 0x0200; // INEX2 enabled
    bus.write_extended(OPERAND, denormal());
    run(&mut cpu, &mut bus, &[0xF210, 0x4822]); // FADD.X (A0),FP0
    assert_eq!(cpu.pc, UNSUPP);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    fpsp_fix_frame(&mut bus);
    frestore(&mut cpu, &mut bus);
    assert_eq!(cpu.fpsr & 0x0200, 0x0200, "1 + denormal is inexact");

    // Reported by the next floating-point instruction, which does not run.
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, arith(49));
    assert_eq!(stacked(&cpu, &bus), (49 << 2, 0x02A0));
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    assert_eq!(bus.byte_at(FRAME + E_BYTE) & (E1 | E3), E3);
}

// ============================================================================
// FRESTORE: pending exception state
// ============================================================================

/// A busy frame at FRAME with the given E_BYTE and nothing else set.
fn write_busy_frame(bus: &mut TestBus, e_byte: u8) {
    for a in (FRAME..FRAME + 0x64).step_by(4) {
        bus.write_long_at(a, 0);
    }
    bus.write_long_at(FRAME, 0x4160_0000);
    bus.memory[(FRAME + E_BYTE) as usize] = e_byte;
    bus.write_long_at(FRAME + WBTEMP + 4, 0xCAFE_F00D);
    bus.write_word_at(FRAME + CMDREG3B, 0x0022);
}

#[test]
fn a_restored_exception_is_taken_by_the_next_floating_point_instruction() {
    let (mut cpu, mut bus) = setup(true);
    cpu.fpcr = 0x0200; // INEX2 enabled
    cpu.fpsr = 0x0200;
    write_busy_frame(&mut bus, E3);
    frestore(&mut cpu, &mut bus);

    // FSAVE and FRESTORE do not report it; every other floating-point
    // instruction does, a control-register move and FMOVEM included,
    // because the result of an E3 exception is not in its register (nor
    // its status in FPSR) until the handler has run.
    for words in [
        [0xF200, 0xA800], // FMOVE.L FPSR,D0
        [0xF212, 0xF080], // FMOVEM.X FP0,(A2)
    ] {
        let (mut cpu, mut bus) = setup(true);
        cpu.fpcr = 0x0200;
        cpu.fpsr = 0x0200;
        write_busy_frame(&mut bus, E3);
        frestore(&mut cpu, &mut bus);
        cpu.dar[10] = 0x6000;
        run_at(&mut cpu, &mut bus, 0x02B0, &words);
        assert_eq!(cpu.pc, arith(49), "{words:04X?}");
        assert_eq!(stacked(&cpu, &bus), (49 << 2, 0x02B0), "{words:04X?}");
    }

    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, arith(49), "INEX through vector 49");
    assert_eq!(stacked(&cpu, &bus), (49 << 2, 0x02A0), "pre-instruction");

    // The handler's FSAVE gets the restored frame back, byte for byte.
    let expect: Vec<u8> = bus.memory[FRAME as usize..(FRAME + 0x64) as usize].to_vec();
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    assert_eq!(
        &bus.memory[FRAME as usize..(FRAME + 0x64) as usize],
        &expect[..]
    );

    // Collected: nothing is pending any more.
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, 0x02A4);
}

#[test]
fn restored_overflow_and_underflow_are_taken_even_when_disabled() {
    // The FPSP reports an emulated instruction's overflow by restoring an
    // E3 frame and expects vector 53 whatever the FPCR enables say.
    for (fpsr, vector) in [(0x1000u32, 53u32), (0x0800, 51), (0x1200, 53)] {
        let (mut cpu, mut bus) = setup(true);
        cpu.fpsr = fpsr;
        write_busy_frame(&mut bus, E3);
        frestore(&mut cpu, &mut bus);
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(vector), "FPSR {fpsr:#x}");
        assert_eq!(stacked(&cpu, &bus), ((vector << 2) as u16, 0x02A0));
    }

    // A disabled inexact alone is not an exception.
    let (mut cpu, mut bus) = setup(true);
    cpu.fpsr = 0x0200;
    write_busy_frame(&mut bus, E3);
    frestore(&mut cpu, &mut bus);
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, 0x02A4);
}

#[test]
fn an_unchanged_unsupported_frame_is_taken_again_after_frestore() {
    let (mut cpu, mut bus) = setup(true);
    bus.write_extended(OPERAND, denormal());
    run(&mut cpu, &mut bus, &[0xF210, 0x4822]);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    let saved: Vec<u8> = bus.memory[FRAME as usize..(FRAME + 0x64) as usize].to_vec();
    frestore(&mut cpu, &mut bus);

    // FSAVE/FRESTORE round trip with the state still pending.
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    assert_eq!(
        &bus.memory[FRAME as usize..(FRAME + 0x64) as usize],
        &saved[..]
    );
    frestore(&mut cpu, &mut bus);

    run_at(&mut cpu, &mut bus, 0x02A0, &[0xF200, 0x0400]); // FMOVE.X FP1,FP0
    assert_eq!(cpu.pc, UNSUPP);
    assert_eq!(stacked(&cpu, &bus), (0x00DC, 0x02A0));
}

#[test]
fn idle_and_null_frames_clear_a_pending_exception() {
    for header in [0x4100_0000u32, 0x0000_0000] {
        let (mut cpu, mut bus) = setup(true);
        cpu.fpcr = 0x0200;
        cpu.fpsr = 0x0200;
        write_busy_frame(&mut bus, E3);
        frestore(&mut cpu, &mut bus);
        bus.write_long_at(FRAME, header);
        frestore(&mut cpu, &mut bus);
        cpu.fpcr = 0x0200;
        cpu.fpsr = 0x0200;
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, 0x02A4, "header {header:#010x}");
        let idle = if header == 0 { 0 } else { 0x4100_0000 };
        assert_eq!(fsave(&mut cpu, &mut bus), idle);
    }
}

#[test]
fn a_restored_unimp_frame_round_trips() {
    let (mut cpu, mut bus) = setup(true);
    run(&mut cpu, &mut bus, &[0xF200, 0x040E]); // FSIN.X FP1,FP0
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    let saved: Vec<u8> = bus.memory[FRAME as usize..(FRAME + 0x34) as usize].to_vec();
    frestore(&mut cpu, &mut bus);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000);
    assert_eq!(
        &bus.memory[FRAME as usize..(FRAME + 0x34) as usize],
        &saved[..]
    );
    assert_eq!(bus.long_at(FRAME + 0x34), 0xEEEE_EEEE);
}

#[test]
fn a_busy_frame_with_nothing_pending_leaves_the_fpu_idle() {
    // What the FPSP's overflow handler restores once it has stored the
    // result: E3 cleared.
    let (mut cpu, mut bus) = setup(true);
    cpu.fpsr = 0x1000;
    write_busy_frame(&mut bus, 0x01);
    frestore(&mut cpu, &mut bus);
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, 0x02A4);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000);
}

// ============================================================================
// Arithmetic exceptions leave a state frame for their handlers (M68040UM
// table 9-16), whichever way the FPSP's work is done
// ============================================================================

const WB_BYTE: u32 = 0x45;
const SBIT: u32 = 0x3D;

fn max() -> FloatX80 {
    FloatX80::from_extended(0x7FFE, 0xFFFF_FFFF_FFFF_FFFF)
}

fn tiny() -> FloatX80 {
    FloatX80::from_extended(0x0001, 0x8000_0000_0000_0000)
}

#[test]
fn operand_exceptions_save_the_26_word_frame_with_both_operands() {
    // Divide by zero is found on the operands: E1, with both of them, and
    // the destination register is not written.
    for traps in [true, false] {
        let (mut cpu, mut bus) = setup(traps);
        cpu.fpcr = 0x0400;
        cpu.fpr[1] = FloatX80::zero(false);
        cpu.fpr[0] = FloatX80::from_f64(3.0);
        run(&mut cpu, &mut bus, &[0xF200, 0x0420]); // FDIV.X FP1,FP0
        assert_eq!(cpu.pc, 0x0204, "held for the next FP instruction");
        assert_eq!(cpu.fpr[0], FloatX80::from_f64(3.0), "destination kept");
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(50), "traps {traps}");
        assert_eq!(stacked(&cpu, &bus), (50 << 2, 0x02A0), "pre-instruction");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000, "traps {traps}");
        let f = FRAME - UNIMP;
        assert_eq!(bus.byte_at(f + E_BYTE) & (E1 | E3), E1);
        assert_eq!(bus.byte_at(f + T_BYTE), 0, "not an unimplemented one");
        assert_eq!(bus.word_at(f + CMDREG1B), 0x0420);
        assert_eq!(bus.extended_at(f + ETEMP), FloatX80::zero(false));
        assert_eq!(bus.byte_at(f + STAG), 0x30);
        assert_eq!(bus.extended_at(f + FPTEMP), FloatX80::from_f64(3.0));
    }
}

#[test]
fn an_inexact_result_saves_the_unrounded_result_in_wbtemp() {
    // 1 / 3: E3, the re-encoded command, and the 64 mantissa bits before
    // rounding with guard, round and sticky beside STAG. The register
    // holds the rounded result.
    for traps in [true, false] {
        let (mut cpu, mut bus) = setup(traps);
        cpu.fpcr = 0x0200;
        cpu.fpr[1] = FloatX80::from_f64(3.0);
        run(&mut cpu, &mut bus, &[0xF200, 0x0420]); // FDIV.X FP1,FP0
        assert_eq!(cpu.pc, 0x0204, "held for the next FP instruction");
        assert_eq!(
            cpu.fpr[0],
            FloatX80::from_extended(0x3FFD, 0xAAAA_AAAA_AAAA_AAAB)
        );
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(49), "traps {traps}");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
        assert_eq!(bus.byte_at(FRAME + E_BYTE) & (E1 | E3), E3);
        assert_eq!(bus.word_at(FRAME + CMDREG3B), 0x0010, "FDIV as CMDREG3B");
        assert_eq!(
            bus.extended_at(FRAME + WBTEMP),
            FloatX80::from_extended(0x3FFD, 0xAAAA_AAAA_AAAA_AAAA)
        );
        assert_eq!(bus.byte_at(FRAME + STAG) & 0x03, 0x02, "guard, no round");
        assert_eq!(bus.byte_at(FRAME + SBIT), 0x80, "sticky");
        assert_eq!(bus.long_at(FRAME + FPSR_SHADOW) & 0xFF00, 0x0200);
        assert_eq!(bus.extended_at(FRAME + ETEMP), FloatX80::from_f64(3.0));
        assert_eq!(bus.extended_at(FRAME + FPTEMP), one());
    }
}

#[test]
fn an_overflow_saves_the_rounded_result_with_its_wrapped_exponent() {
    // max * 2 has the exponent $7FFF, one past the format: WBTEMP keeps
    // its low fifteen bits and WBTE15 is clear (M68040UM table 9-16).
    for traps in [true, false] {
        let (mut cpu, mut bus) = setup(traps);
        cpu.fpcr = 0x1000;
        cpu.fpr[1] = FloatX80::from_f64(2.0);
        cpu.fpr[0] = max();
        run(&mut cpu, &mut bus, &[0xF200, 0x0423]); // FMUL.X FP1,FP0
        assert_eq!(cpu.pc, 0x0204);
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(53), "traps {traps}");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000, "traps {traps}");
        assert_eq!(bus.byte_at(FRAME + E_BYTE) & (E1 | E3), E3);
        assert_eq!(bus.word_at(FRAME + CMDREG3B), 0x0013, "FMUL as CMDREG3B");
        assert_eq!(bus.long_at(FRAME + WBTEMP), 0x7FFF_0000);
        assert_eq!(bus.long_at(FRAME + WBTEMP + 4), 0xFFFF_FFFF);
        assert_eq!(bus.long_at(FRAME + WBTEMP + 8), 0xFFFF_FFFF);
        assert_eq!(bus.byte_at(FRAME + WB_BYTE) & 0x10, 0, "WBTE15 clear");
        assert_eq!(bus.extended_at(FRAME + ETEMP), FloatX80::from_f64(2.0));
        assert_eq!(bus.extended_at(FRAME + FPTEMP), max());
    }

    // Rounded to the precision first: at single precision this product,
    // just under 2 at the exponent $8000, rounds up into the next one.
    let (mut cpu, mut bus) = setup(false);
    cpu.fpcr = 0x1040; // OVFL enabled, single precision
    cpu.fpr[1] = FloatX80::from_extended(0x4000, 0xFFFF_FFFF_FFFF_FFFF);
    cpu.fpr[0] = max();
    run(&mut cpu, &mut bus, &[0xF200, 0x0423]);
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, arith(53));
    fsave(&mut cpu, &mut bus);
    assert_eq!(bus.long_at(FRAME + WBTEMP), 0x0001_0000, "exponent $8001");
    assert_eq!(bus.long_at(FRAME + WBTEMP + 4), 0x8000_0000);
    assert_eq!(bus.long_at(FRAME + WBTEMP + 8), 0);
}

#[test]
fn an_underflow_saves_the_normalized_result_with_its_wrapped_exponent() {
    // tiny * tiny: the exponent is 2 - $3FFF, far below the format. The
    // frame keeps the normalized mantissa, the exponent's low fifteen
    // bits and WBTE15 set, which is what the FPSP denormalizes.
    for traps in [true, false] {
        let (mut cpu, mut bus) = setup(traps);
        cpu.fpcr = 0x0800;
        cpu.fpr[1] = tiny();
        cpu.fpr[0] = tiny();
        run(&mut cpu, &mut bus, &[0xF200, 0x0423]); // FMUL.X FP1,FP0
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(51), "traps {traps}");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
        assert_eq!(bus.byte_at(FRAME + E_BYTE) & (E1 | E3), E3);
        assert_eq!(bus.long_at(FRAME + WBTEMP), 0x4003_0000);
        assert_eq!(bus.long_at(FRAME + WBTEMP + 4), 0x8000_0000);
        assert_eq!(bus.long_at(FRAME + WBTEMP + 8), 0);
        assert_eq!(bus.byte_at(FRAME + WB_BYTE) & 0x10, 0x10, "WBTE15 set");
    }
}

#[test]
fn overflow_and_underflow_are_nonmaskable_with_the_fpsp_in_the_guest() {
    // The 68040 takes them whether or not they are enabled and leaves the
    // destination register alone: the result a 6888x would store is the
    // FPSP's to supply (M68040UM 9.7.4.2 and 9.7.5.2).
    let two = FloatX80::from_f64(2.0);
    for (src, dst, vector) in [(two, max(), 53), (tiny(), tiny(), 51)] {
        let (mut cpu, mut bus) = setup(true);
        cpu.fpcr = 0;
        cpu.fpr[1] = src;
        cpu.fpr[0] = dst;
        run(&mut cpu, &mut bus, &[0xF200, 0x0423]); // FMUL.X FP1,FP0
        assert_eq!(cpu.pc, 0x0204, "held for the next FP instruction");
        assert_eq!(cpu.fpr[0], dst, "vector {vector}: register not written");
        assert_ne!(cpu.fpsr & 0x1800, 0, "vector {vector}: FPSR records it");
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(vector));
        assert_eq!(stacked(&cpu, &bus), ((vector << 2) as u16, 0x02A0));
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
        assert_eq!(bus.byte_at(FRAME + E_BYTE) & (E1 | E3), E3);
        assert_eq!(bus.extended_at(FRAME + FPTEMP), dst);
        assert_eq!(bus.extended_at(FRAME + ETEMP), src);

        // Enabled, the register is still the FPSP's to write.
        let (mut cpu, mut bus) = setup(true);
        cpu.fpcr = 0x1800;
        cpu.fpr[1] = src;
        cpu.fpr[0] = dst;
        run(&mut cpu, &mut bus, &[0xF200, 0x0423]);
        assert_eq!(cpu.fpr[0], dst, "vector {vector}: enabled");

        // With the FPSP's work done here the instruction just completes.
        let (mut cpu, mut bus) = setup(false);
        cpu.fpcr = 0;
        cpu.fpr[1] = src;
        cpu.fpr[0] = dst;
        run(&mut cpu, &mut bus, &[0xF200, 0x0423]);
        assert_ne!(cpu.fpr[0], dst, "vector {vector}: result stored");
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, 0x02A4, "vector {vector}: no exception");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000);
    }

    // An operand exception that is enabled comes first; a disabled one
    // changes nothing (infinity * 0 is an operand error, not an overflow).
    let (mut cpu, mut bus) = setup(true);
    cpu.fpcr = 0;
    cpu.fpr[1] = FloatX80::zero(false);
    cpu.fpr[0] = FloatX80::infinity(false);
    run(&mut cpu, &mut bus, &[0xF200, 0x0423]);
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, 0x02A4);
}

#[test]
fn a_move_into_a_register_reports_its_rounding_from_the_conversion_unit() {
    // FSMOVE of a value with more than 24 mantissa bits is inexact. FMOVE,
    // FABS and FNEG do not go through the normalization unit: E1 in the
    // 26-word frame, the unrounded result in FPTEMP.
    let third = FloatX80::from_extended(0x3FFD, 0xAAAA_AAAA_AAAA_AAAB);
    for traps in [true, false] {
        let (mut cpu, mut bus) = setup(traps);
        cpu.fpcr = 0x0200;
        cpu.fpr[1] = third;
        run(&mut cpu, &mut bus, &[0xF200, 0x0440]); // FSMOVE FP1,FP0
        assert_eq!(
            cpu.fpr[0],
            FloatX80::from_extended(0x3FFD, 0xAAAA_AB00_0000_0000),
            "the rounded result is stored"
        );
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(49), "traps {traps}");
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4130_0000, "traps {traps}");
        let f = FRAME - UNIMP;
        assert_eq!(bus.byte_at(f + E_BYTE) & (E1 | E3), E1);
        assert_eq!(bus.word_at(f + CMDREG1B), 0x0440);
        assert_eq!(bus.extended_at(f + FPTEMP), third);
        assert_eq!(bus.byte_at(f + DTAG), 0x10, "normalized, exponent low");
        assert_eq!(bus.extended_at(f + ETEMP), third);
    }
}

#[test]
fn a_held_exception_goes_through_fsave_and_frestore() {
    // FSAVE before the next floating-point instruction collects the
    // frame, and FRESTORE of it holds the exception again.
    for traps in [true, false] {
        let (mut cpu, mut bus) = setup(traps);
        cpu.fpcr = 0x0200;
        cpu.fpr[1] = FloatX80::from_f64(3.0);
        run(&mut cpu, &mut bus, &[0xF200, 0x0420]);
        assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, 0x02A4, "collected by FSAVE");
        frestore(&mut cpu, &mut bus);
        fnop(&mut cpu, &mut bus);
        assert_eq!(cpu.pc, arith(49), "held again, traps {traps}");
    }
}

// ============================================================================
// With the FPSP's work done here (the default), its cases complete, and an
// exception is the one the FPSP would pass on to the user's handler
// ============================================================================

#[test]
fn native_mode_completes_the_fpsp_cases_and_saves_idle_frames() {
    let (mut cpu, mut bus) = setup(false);
    bus.write_extended(OPERAND, denormal());
    run(&mut cpu, &mut bus, &[0xF210, 0x4822]); // FADD.X (A0),FP0
    assert_eq!(cpu.pc, 0x0204, "denormals are handled natively");
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000);

    // An unimplemented instruction runs, and its enabled exception is
    // held with a frame like any other: FSIN of 1 is inexact.
    let (mut cpu, mut bus) = setup(false);
    cpu.fpcr = 0x0200;
    run(&mut cpu, &mut bus, &[0xF200, 0x040E]); // FSIN.X FP1,FP0
    assert_eq!(cpu.pc, 0x0204);
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, arith(49));
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);
    assert_eq!(bus.byte_at(FRAME + E_BYTE) & (E1 | E3), E3);
    assert_eq!(bus.word_at(FRAME + CMDREG1B), 0x040E);
    assert_eq!(bus.extended_at(FRAME + WBTEMP), cpu.fpr[0], "the result");
}

#[test]
fn a_disabled_overflow_is_taken_only_where_the_fpsp_must_supply_the_result() {
    // max * 2 with OVFL disabled and INEX2 enabled. The inexact exception
    // is the one to take (M68040UM 9.7.7).
    let overflow = |traps: bool, fpcr: u32| {
        let (mut cpu, mut bus) = setup(traps);
        cpu.fpcr = fpcr;
        cpu.fpr[1] = FloatX80::from_f64(2.0);
        cpu.fpr[0] = max();
        run(&mut cpu, &mut bus, &[0xF200, 0x0423]); // FMUL.X FP1,FP0
        fnop(&mut cpu, &mut bus);
        cpu.pc
    };
    // Done here: the user's inexact handler, as the FPSP would call it.
    assert_eq!(overflow(false, 0x0200), arith(49));
    // Left to the FPSP: the 68040's own overflow vector, behind which the
    // FPSP stores the result and goes on to the inexact handler.
    assert_eq!(overflow(true, 0x0200), arith(53));
    // Enabled, both take the overflow exception.
    assert_eq!(overflow(false, 0x1000), arith(53));
    assert_eq!(overflow(true, 0x1000), arith(53));
}

#[test]
fn a_restored_frame_is_honoured_in_native_mode_too() {
    let (mut cpu, mut bus) = setup(false);
    cpu.fpcr = 0x0200;
    cpu.fpsr = 0x0200;
    write_busy_frame(&mut bus, E3);
    frestore(&mut cpu, &mut bus);
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, arith(49), "the restored exception is taken");
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4160_0000);

    // A disabled overflow in a restored frame is not an exception here:
    // nothing is left for an FPSP to do.
    let (mut cpu, mut bus) = setup(false);
    cpu.fpsr = 0x1000;
    write_busy_frame(&mut bus, E3);
    frestore(&mut cpu, &mut bus);
    fnop(&mut cpu, &mut bus);
    assert_eq!(cpu.pc, 0x02A4);
}

#[cfg(feature = "serde")]
#[test]
fn exceptional_state_survives_a_snapshot() {
    let (mut cpu, mut bus) = setup(true);
    bus.write_extended(OPERAND, denormal());
    run(&mut cpu, &mut bus, &[0xF210, 0x4822]);
    assert_eq!(cpu.pc, UNSUPP);
    let json = serde_json::to_string(&cpu).unwrap();
    let mut restored: CpuCore = serde_json::from_str(&json).unwrap();
    assert_eq!(fsave(&mut restored, &mut bus), 0x4160_0000);
    assert_eq!(bus.extended_at(FRAME + ETEMP), denormal());
}

#[test]
fn reset_drops_the_exceptional_state() {
    let (mut cpu, mut bus) = setup(true);
    bus.write_extended(OPERAND, denormal());
    run(&mut cpu, &mut bus, &[0xF210, 0x4822]);
    assert_eq!(cpu.pc, UNSUPP);
    cpu.reset(&mut bus);
    cpu.set_sr(0x2700);
    assert_eq!(fsave(&mut cpu, &mut bus), 0x4100_0000);
}

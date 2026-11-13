//! FPU architectural-behaviour tests for the 6888x coprocessor (on a
//! 68030), the 68040 and the 68060: control-register bits, IEEE special
//! operands, conversions, predicates, state frames and exceptions, each
//! checked against the MC68881/MC68882 User's Manual, the M68040 User's
//! Manual (section 9) and the M68060 User's Manual (section 6).

use m68k::NoOpHleHandler;
use m68k::core::cpu::CpuCore;
use m68k::core::memory::AddressBus;
use m68k::core::types::CpuType;
use m68k::fpu::FloatX80;

struct FlatBus {
    mem: Vec<u8>,
}

impl AddressBus for FlatBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.mem.get(address as usize).copied().unwrap_or(0)
    }
    fn read_word(&mut self, address: u32) -> u16 {
        (u16::from(self.read_byte(address)) << 8) | u16::from(self.read_byte(address + 1))
    }
    fn read_long(&mut self, address: u32) -> u32 {
        (u32::from(self.read_word(address)) << 16) | u32::from(self.read_word(address + 2))
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        if let Some(b) = self.mem.get_mut(address as usize) {
            *b = value;
        }
    }
    fn write_word(&mut self, address: u32, value: u16) {
        self.write_byte(address, (value >> 8) as u8);
        self.write_byte(address + 1, value as u8);
    }
    fn write_long(&mut self, address: u32, value: u32) {
        self.write_word(address, (value >> 16) as u16);
        self.write_word(address + 2, value as u16);
    }
}

const CODE: u32 = 0x2000;
const DATA: u32 = 0x3000;
const HANDLERS: u32 = 0x4000;
const STACK: u32 = 0x8000;
/// D7 before the program runs; every exception handler overwrites it with
/// its vector number.
const NO_EXCEPTION: u32 = 0xFFFF_FFFF;

/// FPSR condition codes.
const CC_N: u32 = 0x0800_0000;
const CC_Z: u32 = 0x0400_0000;
const CC_NAN: u32 = 0x0100_0000;
/// FPSR exception-status bits.
const EXC_BSUN: u32 = 0x8000;
const EXC_SNAN: u32 = 0x4000;
/// FPSR accrued-exception bits.
const AEXC_IOP: u32 = 0x0080;
const AEXC_INEX: u32 = 0x0008;
const EXC_INEX2: u32 = 0x0200;
const EXC_INEX1: u32 = 0x0100;

struct Machine {
    cpu: CpuCore,
    bus: FlatBus,
}

impl Machine {
    /// A reset CPU of `cpu_type` (with its FPU present) in supervisor mode.
    /// Vectors 2-63 each point at a stub `MOVEQ #vector,D7 ; STOP #$2700`.
    fn new(cpu_type: CpuType) -> Self {
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(cpu_type);
        cpu.fpu_present = true;
        let mut bus = FlatBus {
            mem: vec![0; 0x10000],
        };
        bus.write_long(0, STACK);
        bus.write_long(4, CODE);
        for v in 2..64u32 {
            let stub = HANDLERS + v * 8;
            bus.write_long(v * 4, stub);
            bus.write_word(stub, 0x7E00 | v as u16); // MOVEQ #v,D7
            bus.write_word(stub + 2, 0x4E72); // STOP #$2700
            bus.write_word(stub + 4, 0x2700);
        }
        cpu.reset(&mut bus);
        // The 68060 FPU comes out of reset disabled (PCR.DFP).
        cpu.pcr &= !0x2;
        cpu.dar[7] = NO_EXCEPTION;
        Machine { cpu, bus }
    }

    /// Run `code` at CODE, followed by STOP #$2700, until the CPU stops.
    fn run(&mut self, code: &[u16]) {
        let mut a = CODE;
        for w in code.iter().chain([0x4E72u16, 0x2700].iter()) {
            self.bus.write_word(a, *w);
            a += 2;
        }
        self.cpu.pc = CODE;
        self.cpu.stopped = 0;
        let mut hle = NoOpHleHandler;
        for _ in 0..256 {
            if self.cpu.is_stopped() {
                return;
            }
            self.cpu.step_with_hle_handler(&mut self.bus, &mut hle);
        }
        panic!("program did not reach STOP");
    }
}

fn x80(sign_exp: u16, mantissa: u64) -> FloatX80 {
    FloatX80 { sign_exp, mantissa }
}

fn bits(v: FloatX80) -> (u16, u64) {
    (v.sign_exp, v.mantissa)
}

/// Extended signaling NaN (quiet bit 62 clear) and its quieted form.
const SNAN_X: (u16, u64) = (0x7FFF, 0xA000_0000_0000_0000);
const SNAN_X_QUIET: (u16, u64) = (0x7FFF, 0xE000_0000_0000_0000);

const ALL_FPUS: [CpuType; 3] = [CpuType::M68030, CpuType::M68040, CpuType::M68060];

#[test]
fn unused_fpcr_and_fpsr_bits_read_as_zero() {
    for cpu in ALL_FPUS {
        let mut m = Machine::new(cpu);
        m.cpu.dar[0] = 0xFFFF_FFFF;
        m.run(&[
            0xF200, 0x9000, // FMOVE.L D0,FPCR
            0xF201, 0xB000, // FMOVE.L FPCR,D1
        ]);
        assert_eq!(m.cpu.dar[1], 0x0000_FFF0, "{cpu:?} FPCR");

        let mut m = Machine::new(cpu);
        m.cpu.dar[0] = 0xFFFF_FFFF;
        m.run(&[
            0xF200, 0x8800, // FMOVE.L D0,FPSR
            0xF201, 0xA800, // FMOVE.L FPSR,D1
            0xF200, 0x8400, // FMOVE.L D0,FPIAR
            0xF202, 0xA400, // FMOVE.L FPIAR,D2
        ]);
        assert_eq!(m.cpu.dar[1], 0x0FFF_FFF8, "{cpu:?} FPSR");
        assert_eq!(m.cpu.dar[2], 0xFFFF_FFFF, "{cpu:?} FPIAR is 32 bits");
    }
}

#[test]
fn fmovem_control_list_masks_each_register() {
    // FMOVEM.L (A0),FPCR/FPSR/FPIAR from three all-ones longs.
    let mut m = Machine::new(CpuType::M68030);
    for i in 0..3 {
        m.bus.write_long(DATA + 4 * i, 0xFFFF_FFFF);
    }
    m.cpu.dar[8] = DATA;
    m.run(&[0xF210, 0x9C00]);
    assert_eq!(m.cpu.fpcr, 0x0000_FFF0);
    assert_eq!(m.cpu.fpsr, 0x0FFF_FFF8);
    assert_eq!(m.cpu.fpiar, 0xFFFF_FFFF);
}

/// Run `FOP FP1,FP0` (register form, `opmode`) with FP1 = `src`, FP0 =
/// `dst` and FPSR cleared.
fn reg_op(cpu: CpuType, opmode: u16, src: (u16, u64), dst: (u16, u64)) -> Machine {
    reg_op_fpcr(cpu, opmode, src, dst, 0)
}

/// `reg_op` with FPCR = `fpcr`.
fn reg_op_fpcr(cpu: CpuType, opmode: u16, src: (u16, u64), dst: (u16, u64), fpcr: u32) -> Machine {
    let mut m = Machine::new(cpu);
    m.cpu.fpr[1] = x80(src.0, src.1);
    m.cpu.fpr[0] = x80(dst.0, dst.1);
    m.cpu.fpsr = 0;
    m.cpu.fpcr = fpcr;
    m.run(&[0xF200, 0x0400 | opmode]);
    m
}

#[test]
fn signaling_nan_operand_raises_snan_but_not_operr() {
    // An SNaN operand is an SNAN exception, not an operand error; the
    // result is the NaN made quiet (MC68881/MC68882UM 4.6.4).
    let one = (0x3FFF, 0x8000_0000_0000_0000);
    for cpu in ALL_FPUS {
        // FMOVE, FADD, FMUL, FDIV, FSQRT, FCMP.
        for opmode in [0x00, 0x22, 0x23, 0x20, 0x04, 0x38] {
            let m = reg_op(cpu, opmode, SNAN_X, one);
            assert_eq!(
                m.cpu.fpsr,
                CC_NAN | EXC_SNAN | AEXC_IOP,
                "{cpu:?} opmode {opmode:#x}"
            );
            if opmode != 0x38 {
                assert_eq!(
                    bits(m.cpu.fpr[0]),
                    SNAN_X_QUIET,
                    "{cpu:?} opmode {opmode:#x}"
                );
            }
        }
        // A signaling NaN destination is quieted and wins.
        let m = reg_op(cpu, 0x22, one, SNAN_X);
        assert_eq!(m.cpu.fpsr, CC_NAN | EXC_SNAN | AEXC_IOP, "{cpu:?}");
        assert_eq!(bits(m.cpu.fpr[0]), SNAN_X_QUIET, "{cpu:?}");
    }
}

#[test]
fn single_and_double_signaling_nan_sources_raise_snan() {
    for cpu in ALL_FPUS {
        // FMOVE.S (A0),FP0 with $7F800001.
        let mut m = Machine::new(cpu);
        m.bus.write_long(DATA, 0x7F80_0001);
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x4400]);
        assert_eq!(
            bits(m.cpu.fpr[0]),
            (0x7FFF, 0xC000_0100_0000_0000),
            "{cpu:?}"
        );
        assert_eq!(m.cpu.fpsr, CC_NAN | EXC_SNAN | AEXC_IOP, "{cpu:?} .S");

        // FMOVE.D (A0),FP0 with $7FF00000_00000001.
        let mut m = Machine::new(cpu);
        m.bus.write_long(DATA, 0x7FF0_0000);
        m.bus.write_long(DATA + 4, 0x0000_0001);
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x5400]);
        assert_eq!(
            bits(m.cpu.fpr[0]),
            (0x7FFF, 0xC000_0000_0000_0800),
            "{cpu:?}"
        );
        assert_eq!(m.cpu.fpsr, CC_NAN | EXC_SNAN | AEXC_IOP, "{cpu:?} .D");

        // A quiet single NaN converts without an exception.
        let mut m = Machine::new(cpu);
        m.bus.write_long(DATA, 0x7FC0_0000);
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x4400]);
        assert_eq!(m.cpu.fpsr, CC_NAN, "{cpu:?} quiet .S");
    }
}

#[test]
fn fmove_x_out_quiets_a_signaling_nan_and_rebuilds_the_exception_byte() {
    for cpu in ALL_FPUS {
        // FMOVE.X FP0,(A0) of an SNaN: SNAN set, quiet NaN stored, the
        // condition codes untouched.
        let mut m = Machine::new(cpu);
        m.cpu.fpr[0] = x80(SNAN_X.0, SNAN_X.1);
        m.cpu.fpsr = 0;
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x6800]);
        assert_eq!(m.bus.read_long(DATA), 0x7FFF_0000, "{cpu:?}");
        assert_eq!(m.bus.read_long(DATA + 4), 0xE000_0000, "{cpu:?}");
        assert_eq!(m.cpu.fpsr, EXC_SNAN | AEXC_IOP, "{cpu:?}");

        // A quiet NaN goes out unchanged; the previous instruction's
        // exception byte is cleared, the accrued byte and CC stay.
        let mut m = Machine::new(cpu);
        m.cpu.fpr[0] = x80(SNAN_X_QUIET.0, SNAN_X_QUIET.1);
        m.cpu.fpsr = CC_NAN | EXC_SNAN | AEXC_IOP;
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x6800]);
        assert_eq!(m.bus.read_long(DATA + 4), 0xE000_0000, "{cpu:?}");
        assert_eq!(m.cpu.fpsr, CC_NAN | AEXC_IOP, "{cpu:?}");
    }
}

#[test]
fn nan_operand_keeps_its_sign_and_wins_over_other_rules() {
    let zero = (0x0000, 0);
    let qnan = (0x7FFF, 0xC000_0000_0000_0000);
    let inf = (0x7FFF, 0);
    for cpu in ALL_FPUS {
        // FSUB: dst - src with a NaN src is that NaN, not its negation.
        let m = reg_op(cpu, 0x28, qnan, zero);
        assert_eq!(bits(m.cpu.fpr[0]), qnan, "{cpu:?} FSUB");
        assert_eq!(m.cpu.fpsr, CC_NAN, "{cpu:?} FSUB");
    }
    // FSCALE (6888x; the 040/060 leave it to their software packages): a
    // NaN destination is returned ahead of the infinite-count OPERR, and a
    // NaN count is returned too.
    let m = reg_op(CpuType::M68030, 0x26, inf, SNAN_X);
    assert_eq!(bits(m.cpu.fpr[0]), SNAN_X_QUIET);
    assert_eq!(m.cpu.fpsr, CC_NAN | EXC_SNAN | AEXC_IOP);
    let m = reg_op(CpuType::M68030, 0x26, SNAN_X, qnan);
    assert_eq!(bits(m.cpu.fpr[0]), qnan);
    assert_eq!(m.cpu.fpsr, CC_NAN | EXC_SNAN | AEXC_IOP);
    let m = reg_op(CpuType::M68030, 0x26, qnan, zero);
    assert_eq!(bits(m.cpu.fpr[0]), qnan);
    assert_eq!(m.cpu.fpsr, CC_NAN);
}

#[test]
fn fabs_and_fneg_round_the_result_not_the_operand() {
    // -sqrt(2) and +sqrt(2) in extended; rounded to single toward minus
    // infinity |-sqrt(2)| is $B504F3 (truncated), not $B504F4.
    let neg_root2 = (0xBFFF, 0xB504_F333_F9DE_6484);
    let pos_root2 = (0x3FFF, 0xB504_F333_F9DE_6484);
    let trunc = 0xB504_F300_0000_0000;
    for cpu in [CpuType::M68040, CpuType::M68060] {
        // FSABS / FSNEG under RM / RP.
        let m = reg_op_fpcr(cpu, 0x58, neg_root2, (0, 0), 0x20);
        assert_eq!(bits(m.cpu.fpr[0]), (0x3FFF, trunc), "{cpu:?} FSABS RM");
        assert_eq!(m.cpu.fpsr, EXC_INEX2 | AEXC_INEX, "{cpu:?} FSABS RM");
        let m = reg_op_fpcr(cpu, 0x5A, pos_root2, (0, 0), 0x30);
        assert_eq!(bits(m.cpu.fpr[0]), (0xBFFF, trunc), "{cpu:?} FSNEG RP");
    }
    // FABS / FNEG with FPCR single precision on every FPU.
    for cpu in ALL_FPUS {
        let m = reg_op_fpcr(cpu, 0x18, neg_root2, (0, 0), 0x60);
        assert_eq!(bits(m.cpu.fpr[0]), (0x3FFF, trunc), "{cpu:?} FABS RM.S");
        let m = reg_op_fpcr(cpu, 0x1A, pos_root2, (0, 0), 0x70);
        assert_eq!(bits(m.cpu.fpr[0]), (0xBFFF, trunc), "{cpu:?} FNEG RP.S");
    }
}

#[test]
fn fcmp_of_equal_infinities_sets_the_infinity_sign() {
    let pinf = (0x7FFF, 0);
    let ninf = (0xFFFF, 0);
    for cpu in ALL_FPUS {
        let m = reg_op(cpu, 0x38, ninf, ninf);
        assert_eq!(m.cpu.fpsr, CC_N | CC_Z, "{cpu:?} -inf vs -inf");
        let m = reg_op(cpu, 0x38, pinf, pinf);
        assert_eq!(m.cpu.fpsr, CC_Z, "{cpu:?} +inf vs +inf");
        let m = reg_op(cpu, 0x38, pinf, ninf);
        assert_eq!(m.cpu.fpsr, CC_N, "{cpu:?} -inf vs +inf");
        // The destination is not written.
        assert_eq!(bits(m.cpu.fpr[0]), ninf, "{cpu:?}");
    }
}

/// FMOVECR #offset,FP0 on a 6888x under FPCR `fpcr`.
fn fmovecr(offset: u16, fpcr: u32) -> Machine {
    let mut m = Machine::new(CpuType::M68030);
    m.cpu.fpcr = fpcr;
    m.cpu.fpsr = 0;
    m.run(&[0xF200, 0x5C00 | offset]);
    m
}

#[test]
fn fmovecr_rounds_the_rom_constant_and_reports_inexact() {
    let inex = EXC_INEX2 | AEXC_INEX;
    // pi under each rounding mode, then at single and double precision.
    for (fpcr, mant) in [
        (0x00, 0xC90F_DAA2_2168_C235),
        (0x10, 0xC90F_DAA2_2168_C234),
        (0x20, 0xC90F_DAA2_2168_C234),
        (0x30, 0xC90F_DAA2_2168_C235),
        (0x40, 0xC90F_DB00_0000_0000),
        (0x80, 0xC90F_DAA2_2168_C000),
    ] {
        let m = fmovecr(0x00, fpcr);
        assert_eq!(bits(m.cpu.fpr[0]), (0x4000, mant), "pi fpcr {fpcr:#x}");
        assert_eq!(m.cpu.fpsr, inex, "pi fpcr {fpcr:#x}");
    }
    // 10^256 in RN is the correctly rounded value, inexact.
    let m = fmovecr(0x3B, 0);
    assert_eq!(bits(m.cpu.fpr[0]), (0x4351, 0xAA7E_EBFB_9DF9_DE8E));
    assert_eq!(m.cpu.fpsr, inex);
    // 10^16 is exact in extended, but not in single.
    let m = fmovecr(0x37, 0);
    assert_eq!(bits(m.cpu.fpr[0]), (0x4034, 0x8E1B_C9BF_0400_0000));
    assert_eq!(m.cpu.fpsr, 0);
    let m = fmovecr(0x37, 0x40);
    assert_eq!(m.cpu.fpsr, inex);
    // e under RP is one ulp above its RN pattern.
    let m = fmovecr(0x0C, 0x30);
    assert_eq!(bits(m.cpu.fpr[0]), (0x4000, 0xADF8_5458_A2BB_4A9B));
    // An offset with no constant is an exact +0.0.
    let m = fmovecr(0x0F, 0);
    assert_eq!(bits(m.cpu.fpr[0]), (0, 0));
    assert_eq!(m.cpu.fpsr, CC_Z);
}

#[test]
fn packed_decimal_conversions_report_inexact() {
    // The 68040 runs packed decimal natively here only with its
    // unimplemented-instruction emulation on (the default).
    for cpu in [CpuType::M68030, CpuType::M68040] {
        // FMOVE.P (A0),FP0 of 1E-1: inexact decimal input, INEX1.
        let mut m = Machine::new(cpu);
        m.bus.write_long(DATA, 0x4001_0001);
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x4C00]);
        assert_eq!(
            bits(m.cpu.fpr[0]),
            (0x3FFB, 0xCCCC_CCCC_CCCC_CCCD),
            "{cpu:?}"
        );
        assert_eq!(m.cpu.fpsr, EXC_INEX1 | AEXC_INEX, "{cpu:?} in");

        // FMOVE.P FP0,(A0){#3} of 123456: 1.23E5, inexact, INEX2.
        let mut m = Machine::new(cpu);
        m.cpu.fpr[0] = x80(0x400F, 0xF120_0000_0000_0000);
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x6C03]);
        assert_eq!(m.bus.read_long(DATA), 0x0005_0001, "{cpu:?}");
        assert_eq!(m.bus.read_long(DATA + 4), 0x2300_0000, "{cpu:?}");
        assert_eq!(m.cpu.fpsr, EXC_INEX2 | AEXC_INEX, "{cpu:?} out");

        // An exact conversion raises nothing: 123456 with k = 6.
        let mut m = Machine::new(cpu);
        m.cpu.fpr[0] = x80(0x400F, 0xF120_0000_0000_0000);
        m.cpu.dar[8] = DATA;
        m.run(&[0xF210, 0x6C06]);
        assert_eq!(m.bus.read_long(DATA + 4), 0x2345_6000, "{cpu:?}");
        assert_eq!(m.cpu.fpsr, 0, "{cpu:?} exact out");
    }
}

/// A conditional instruction's encoding for a predicate.
type PredicateForm = (&'static str, fn(u16) -> Vec<u16>);

#[test]
fn ieee_nonaware_predicates_signal_bsun_on_unordered() {
    // FBcc.W *+4, FScc D0, FDBcc D0,*+4 and FTRAPcc for every predicate,
    // with each condition code an operation can produce.
    let forms: [PredicateForm; 4] = [
        ("FBcc", |p| vec![0xF280 | p, 0x0002]),
        ("FScc", |p| vec![0xF240, p]),
        ("FDBcc", |p| vec![0xF248, p, 0x0002]),
        ("FTRAPcc", |p| vec![0xF27C, p]),
    ];
    for cpu in ALL_FPUS {
        for (name, code) in forms {
            // FScc/FDBcc/FTRAPcc are left to the 68060 software package.
            if cpu == CpuType::M68060 && name != "FBcc" {
                continue;
            }
            for p in 0..32u16 {
                for cc in [0, CC_NAN, CC_Z, CC_N, CC_N | CC_NAN] {
                    let mut m = Machine::new(cpu);
                    m.cpu.fpsr = cc;
                    m.run(&code(p));
                    let want = if p & 0x10 != 0 && cc & CC_NAN != 0 {
                        cc | EXC_BSUN | AEXC_IOP
                    } else {
                        cc
                    };
                    assert_eq!(
                        m.cpu.fpsr, want,
                        "{cpu:?} {name} predicate {p:#x} cc {cc:#x}"
                    );
                }
            }
        }
    }
}

#[test]
fn fmod_and_frem_signal_a_signaling_nan_source_under_a_quiet_nan_destination() {
    // The destination NaN is the result, but the signaling NaN in the
    // source still raises SNAN (MC68881/MC68882UM 4.6.4). FMOD and FREM
    // are 6888x hardware instructions.
    let qnan = (0x7FFF, 0xC000_0000_0000_0000);
    for opmode in [0x21, 0x25] {
        let m = reg_op(CpuType::M68030, opmode, SNAN_X, qnan);
        assert_eq!(
            m.cpu.fpsr & (CC_NAN | EXC_SNAN | AEXC_IOP),
            CC_NAN | EXC_SNAN | AEXC_IOP,
            "opmode {opmode:#x}"
        );
        assert_eq!(bits(m.cpu.fpr[0]), qnan, "opmode {opmode:#x}");
    }
}

#[test]
fn fscale_truncates_its_scale_factor_without_reporting_inexact() {
    // FSCALE by 2.7 scales by 2: the truncation of the scale factor is
    // not an inexact result, and 1.0 * 2^2 is exact.
    let two_point_seven = (0x4000, 0xACCC_CCCC_CCCC_CCCD);
    let one = (0x3FFF, 0x8000_0000_0000_0000);
    let m = reg_op(CpuType::M68030, 0x26, two_point_seven, one);
    assert_eq!(bits(m.cpu.fpr[0]), (0x4001, 0x8000_0000_0000_0000));
    assert_eq!(m.cpu.fpsr, 0, "no INEX2, no accrued inexact");
}

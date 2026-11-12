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
const CC_NAN: u32 = 0x0100_0000;
/// FPSR exception-status bits.
const EXC_SNAN: u32 = 0x4000;
/// FPSR accrued-exception bits.
const AEXC_IOP: u32 = 0x0080;

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
    let mut m = Machine::new(cpu);
    m.cpu.fpr[1] = x80(src.0, src.1);
    m.cpu.fpr[0] = x80(dst.0, dst.1);
    m.cpu.fpsr = 0;
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

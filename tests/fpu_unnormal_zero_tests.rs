//! An extended-precision operand with a zero mantissa is a zero whatever its
//! exponent field holds (M68000PRM table 1-6 lists only the mantissa for
//! zero; the 6888x normalizes an unnormalized operand before using it, and a
//! zero mantissa normalizes to zero). FMOVEM loads registers without
//! conversion, so a guest can hand any such pattern to an arithmetic
//! instruction.

use m68k::core::memory::AddressBus;
use m68k::fpu::FloatX80;
use m68k::{CpuCore, CpuType, NoOpHleHandler};

struct TestBus {
    memory: Vec<u8>,
}

impl AddressBus for TestBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[(address as usize) & 0xFFFF]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let i = (address as usize) & 0xFFFF;
        u16::from_be_bytes([self.memory[i], self.memory[i + 1]])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let i = (address as usize) & 0xFFFF;
        u32::from_be_bytes(self.memory[i..i + 4].try_into().unwrap())
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[(address as usize) & 0xFFFF] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        let i = (address as usize) & 0xFFFF;
        self.memory[i..i + 2].copy_from_slice(&value.to_be_bytes());
    }

    fn write_long(&mut self, address: u32, value: u32) {
        let i = (address as usize) & 0xFFFF;
        self.memory[i..i + 4].copy_from_slice(&value.to_be_bytes());
    }
}

/// Run `FOP.X FP1,FP0` with the given operands and return the CPU.
fn run(cpu_type: CpuType, opmode: u16, fp1: FloatX80, fp0: FloatX80) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    let mut bus = TestBus {
        memory: vec![0; 0x10000],
    };
    bus.write_long(0x00, 0x1000);
    bus.write_long(0x04, 0x0200);
    cpu.reset(&mut bus);
    cpu.set_sr(0x2700);
    cpu.fpr[1] = fp1;
    cpu.fpr[0] = fp0;
    bus.write_word(0x0200, 0xF200);
    bus.write_word(0x0202, 0x0400 | opmode);
    cpu.pc = 0x0200;
    let mut hle = NoOpHleHandler;
    cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert_eq!(cpu.pc, 0x0204, "the instruction completes");
    cpu
}

const FPSR_Z: u32 = 0x0400_0000;
const FPSR_DZ: u32 = 0x0000_0400;

#[test]
fn a_zero_mantissa_is_zero_whatever_the_exponent() {
    let odd_zero = FloatX80::from_extended(0x3F81, 0);
    let three = FloatX80::from_f64(3.0);
    for cpu_type in [CpuType::M68030, CpuType::M68040] {
        // FDIV: a divide by zero, not a division by a zero mantissa.
        let cpu = run(cpu_type, 0x20, odd_zero, three);
        assert!(cpu.fpr[0].is_inf(), "{cpu_type:?}: 3 / 0");
        assert_eq!(cpu.fpsr & FPSR_DZ, FPSR_DZ, "{cpu_type:?}: DZ");

        // FTST: the zero condition code.
        let cpu = run(cpu_type, 0x3A, odd_zero, three);
        assert_eq!(cpu.fpsr & FPSR_Z, FPSR_Z, "{cpu_type:?}: Z");

        // FADD and FMUL.
        let cpu = run(cpu_type, 0x22, odd_zero, three);
        assert_eq!(cpu.fpr[0], three, "{cpu_type:?}: 3 + 0");
        let cpu = run(cpu_type, 0x23, odd_zero, three);
        assert_eq!(cpu.fpr[0], FloatX80::zero(false), "{cpu_type:?}: 3 * 0");

        // As the dividend.
        let cpu = run(cpu_type, 0x20, three, odd_zero);
        assert_eq!(cpu.fpr[0], FloatX80::zero(false), "{cpu_type:?}: 0 / 3");
    }
}

#[test]
fn a_zero_operand_yields_the_canonical_zero() {
    let odd_zero = FloatX80::from_extended(0xBF81, 0);
    let tiny = FloatX80::from_extended(0x0001, 0x8000_0000_0000_0000);
    for cpu_type in [CpuType::M68030, CpuType::M68040] {
        // FMOVE, FSQRT, FINT and FSCALE return the zero they were given.
        for opmode in [0x00, 0x04, 0x01] {
            let cpu = run(cpu_type, opmode, odd_zero, FloatX80::from_f64(3.0));
            assert_eq!(cpu.fpr[0], FloatX80::zero(true), "{cpu_type:?} {opmode:#x}");
        }
        let cpu = run(cpu_type, 0x26, FloatX80::from_f64(3.0), odd_zero);
        assert_eq!(cpu.fpr[0], FloatX80::zero(true), "{cpu_type:?} FSCALE");

        // FCMP orders it as a zero and leaves the register as it was.
        let cpu = run(cpu_type, 0x38, tiny, odd_zero);
        assert_eq!(
            cpu.fpsr & 0x0F00_0000,
            0x0800_0000,
            "{cpu_type:?}: 0 < tiny"
        );
        assert_eq!(cpu.fpr[0], odd_zero);
    }
}

#[test]
fn zero_classification_excludes_infinity() {
    assert!(FloatX80::from_extended(0x3F81, 0).is_zero());
    assert!(FloatX80::from_extended(0xBF81, 0).is_zero());
    assert!(FloatX80::zero(true).is_zero());
    assert!(!FloatX80::infinity(false).is_zero());
    assert!(!FloatX80::from_extended(0x0000, 1).is_zero());
}

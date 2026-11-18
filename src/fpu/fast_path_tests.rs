//! The common-case shortcuts of the general FP instructions: holding no
//! exception when FPCR enables none ([`CpuCore::fpu_may_hold_exception`]),
//! the direct dispatch of the dyadic arithmetic and the two-register undo
//! snapshot ([`super::operations::FprUndo`]) must leave exactly the state the
//! full exception path leaves.
use crate::core::cpu::CpuCore;
use crate::core::memory::LinearMemoryBus;
use crate::core::types::CpuType;
use crate::fpu::FloatX80;

const PC: u32 = 0x1000;

fn operands() -> Vec<FloatX80> {
    let x = FloatX80::from_extended;
    vec![
        FloatX80::from_f64(1.0),
        FloatX80::from_f64(-3.0),
        FloatX80::from_f64(0.1),
        FloatX80::zero(false),
        FloatX80::zero(true),
        x(0x7FFE, 0xFFFF_FFFF_FFFF_FFFF), // largest finite
        x(0x0001, 0x8000_0000_0000_0000), // smallest normal
        x(0x0000, 0x4000_0000_0000_0000), // denormal
        x(0x3FFF, 0x4000_0000_0000_0000), // unnormal
        x(0x7FFF, 0x0000_0000_0000_0000), // +inf
        x(0xFFFF, 0x0000_0000_0000_0000), // -inf
        x(0x7FFF, 0xC000_0000_0000_0001), // quiet NaN
        x(0x7FFF, 0x8000_0000_0000_0001), // signaling NaN
    ]
}

/// Register-to-register opmodes: the dyadic arithmetic (all precisions),
/// FSGLDIV/FSGLMUL, FMOVE/FSQRT/FABS/FNEG/FINT, FCMP/FTST and FSINCOS
/// (FP3 into FP1, cosine into FP2).
const OPMODES: [u16; 26] = [
    0x22, 0x62, 0x66, 0x23, 0x63, 0x67, 0x28, 0x68, 0x6C, 0x20, 0x60, 0x64, 0x24, 0x27, 0x00, 0x40,
    0x44, 0x04, 0x18, 0x1A, 0x01, 0x03, 0x38, 0x3A, 0x32, 0x5C,
];

#[derive(Clone, Copy, Debug)]
struct Model {
    cpu: CpuType,
    emulate: bool,
}

const MODELS: [Model; 5] = [
    Model {
        cpu: CpuType::M68030,
        emulate: true,
    },
    Model {
        cpu: CpuType::M68040,
        emulate: true,
    },
    Model {
        cpu: CpuType::M68040,
        emulate: false,
    },
    Model {
        cpu: CpuType::M68060,
        emulate: true,
    },
    Model {
        cpu: CpuType::M68060,
        emulate: false,
    },
];

/// Run `fop fp3,fp1` once with the given FPCR and return the CPU.
fn run(model: Model, opmode: u16, src: FloatX80, dst: FloatX80, fpcr: u32) -> CpuCore {
    let mut bus = LinearMemoryBus::new(0x10000);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(model.cpu);
    cpu.fpu_present = true;
    cpu.emulate_unimplemented_040 = model.emulate;
    cpu.emulate_unimplemented_060 = model.emulate;
    cpu.reset_soft();
    cpu.set_sr(0x2700);
    cpu.set_a(7, 0x8000);
    for (i, value) in [3.5, 1.0, 2.0, 0.0, -4.25, 6.0, 7.0, 8.0]
        .into_iter()
        .enumerate()
    {
        cpu.fpr[i] = FloatX80::from_f64(value);
    }
    cpu.fpr[3] = src;
    cpu.fpr[1] = dst;
    cpu.fpcr = fpcr;
    // Stale exception and accrued bytes: the instruction must rewrite the
    // first and add to the second.
    cpu.fpsr = 0x0000_FFA8;
    let w2 = (3 << 10) | (1 << 7) | opmode;
    bus.write_word_at(PC, 0xF200);
    bus.write_word_at(PC + 2, w2);
    bus.write_word_at(PC + 4, 0x4E71);
    cpu.pc = PC;
    cpu.step(&mut bus);
    cpu
}

fn state(cpu: &CpuCore) -> String {
    format!(
        "pc {:#x} sr {:#x} fpsr {:#010x} fpiar {:#x} fpr {:?} pending {} 040 {:?} 060 {:?}",
        cpu.pc,
        cpu.get_sr(),
        cpu.fpsr,
        cpu.fpiar,
        cpu.fpr,
        cpu.fpu_exception_pending,
        cpu.fpu_040,
        cpu.fpu_060_excp,
    )
}

#[test]
fn shortcuts_leave_the_state_of_the_full_exception_path() {
    // FPCR $8000 enables only BSUN, which no arithmetic raises: the
    // outcome is that of FPCR 0, but every hold goes through the full
    // priority search. The low byte varies rounding mode and precision.
    let mut flagged = 0;
    for model in MODELS {
        for opmode in OPMODES {
            for &src in &operands() {
                for &dst in &operands() {
                    for low in [0x00, 0x30, 0x40, 0x90] {
                        let fast = run(model, opmode, src, dst, low);
                        let full = run(model, opmode, src, dst, 0x8000 | low);
                        assert_eq!(
                            state(&fast),
                            state(&full),
                            "{model:?} opmode {opmode:#x} src {src:?} dst {dst:?} fpcr {low:#x}"
                        );
                        flagged += usize::from(fast.fpsr & 0xFF00 != 0);
                    }
                }
            }
        }
    }
    // The operand set exercises the exception bytes, not only exact results.
    assert!(flagged > 10_000, "{flagged}");
}

#[test]
fn an_enabled_operand_error_of_fsincos_restores_both_registers() {
    // FSINCOS of an infinity is an operand error. With OPERR enabled the
    // 68040 (its FPSP work done here) and the 6888x store neither result.
    let inf = FloatX80::from_extended(0x7FFF, 0);
    for model in [MODELS[0], MODELS[1]] {
        let cpu = run(model, 0x32, inf, FloatX80::from_f64(1.0), 0x2000);
        assert_eq!(
            cpu.fpr[1],
            FloatX80::from_f64(1.0),
            "{model:?} sine register"
        );
        assert_eq!(
            cpu.fpr[2],
            FloatX80::from_f64(2.0),
            "{model:?} cosine register"
        );
        assert_ne!(cpu.fpsr & 0x2000, 0, "{model:?} OPERR raised");
        // With OPERR disabled both results (NaNs) are stored.
        let cpu = run(model, 0x32, inf, FloatX80::from_f64(1.0), 0);
        assert!(cpu.fpr[1].is_nan() && cpu.fpr[2].is_nan(), "{model:?}");
    }
}

#[test]
fn an_enabled_exception_still_restores_the_destination() {
    // FDIV by zero with DZ enabled: the destination keeps its value and
    // the exception is held, on every model.
    for model in MODELS {
        let cpu = run(
            model,
            0x20,
            FloatX80::zero(false),
            FloatX80::from_f64(1.0),
            0x0400,
        );
        assert_eq!(cpu.fpr[1], FloatX80::from_f64(1.0), "{model:?}");
        assert_ne!(cpu.fpsr & 0x0400, 0, "{model:?} DZ raised");
        let held = cpu.fpu_exception_pending != 0 || cpu.fpu_040.pending;
        assert!(held, "{model:?} exception held");
    }
}

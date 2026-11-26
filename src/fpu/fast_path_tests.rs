//! The common-case shortcuts of the general FP instructions: holding no
//! exception when FPCR masks the raised flags ([`CpuCore::fpu_may_hold_exception`]),
//! the direct dispatch of the dyadic arithmetic and the two-register undo
//! snapshot ([`super::operations::FprUndo`]) must leave exactly the state the
//! full exception path leaves. So must the arithmetic the decoded batch loop
//! runs itself ([`CpuCore::fpu_general_op_decoded`]) leave the state of full
//! dispatch.
use crate::core::cpu::CpuCore;
use crate::core::memory::{AddressBus, LinearMemoryBus};
use crate::core::types::CpuType;
use crate::fpu::FloatX80;

std::thread_local! {
    pub(super) static FORCE_FULL_HOLD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Instructions the decoded batch loop ran without full dispatch.
    pub(super) static DECODED_OPS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

#[test]
fn unrelated_exception_enables_skip_the_hold_path() {
    let mut cpu = CpuCore::new();
    cpu.fpcr = 0x3400; // OPERR, OVFL and DZ enabled
    cpu.fpsr = 0x0208; // INEX2 raised and inexact accrued
    assert!(!cpu.fpu_may_hold_exception());
    cpu.fpcr |= 0x0200;
    assert!(cpu.fpu_may_hold_exception());
    cpu.fpcr = 0;
    for raised in [0x1000, 0x0800] {
        cpu.fpsr = raised;
        assert!(cpu.fpu_may_hold_exception());
    }
}

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
    // Compare identical FPCRs with and without the early exit.
    let mut flagged = 0;
    for model in MODELS {
        for opmode in OPMODES {
            for &src in &operands() {
                for &dst in &operands() {
                    for fpcr in [0x00, 0x3430, 0x0240, 0x8090] {
                        let fast = run(model, opmode, src, dst, fpcr);
                        FORCE_FULL_HOLD.set(true);
                        let full = run(model, opmode, src, dst, fpcr);
                        FORCE_FULL_HOLD.set(false);
                        assert_eq!(
                            state(&fast),
                            state(&full),
                            "{model:?} opmode {opmode:#x} src {src:?} dst {dst:?} fpcr {fpcr:#x}"
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

const DATA: u32 = 0x2000;

/// One instruction (or a short run of them) and the machine it starts on.
#[derive(Clone, Debug)]
struct Case {
    code: Vec<u16>,
    /// Bytes at [`DATA`].
    data: Vec<u8>,
    a0: u32,
    a7: u32,
    d1: u32,
    d2: u32,
    src: FloatX80,
    dst: FloatX80,
    fpcr: u32,
    instructions: u32,
}

impl Case {
    fn new(code: Vec<u16>) -> Self {
        Self {
            code,
            data: Vec::new(),
            a0: DATA,
            a7: 0x8000,
            d1: 2,
            d2: 0,
            src: FloatX80::from_f64(3.5),
            dst: FloatX80::from_f64(1.0),
            fpcr: 0,
            instructions: 1,
        }
    }

    fn start(&self, cpu: &mut CpuCore, model: Model) -> LinearMemoryBus {
        let mut bus = LinearMemoryBus::new(0x10000);
        if cpu.cpu_type != model.cpu {
            cpu.set_cpu_type(model.cpu);
        }
        cpu.fpu_present = true;
        cpu.emulate_unimplemented_040 = model.emulate;
        cpu.emulate_unimplemented_060 = model.emulate;
        cpu.reset_soft();
        cpu.set_sr(0x2700);
        for reg in 0..8 {
            cpu.set_d(reg, 0x1111_1111 * reg as u32);
            cpu.set_a(reg, 0x4000 + 0x100 * reg as u32);
        }
        cpu.set_a(0, self.a0);
        cpu.set_a(7, self.a7);
        cpu.set_d(1, self.d1);
        cpu.set_d(2, self.d2);
        for (i, value) in [3.5, 1.0, 2.0, 0.0, -4.25, 6.0, 7.0, 8.0]
            .into_iter()
            .enumerate()
        {
            cpu.fpr[i] = FloatX80::from_f64(value);
        }
        cpu.fpr[3] = self.src;
        cpu.fpr[1] = self.dst;
        cpu.fpcr = self.fpcr;
        cpu.fpsr = 0x0000_FFA8;
        for (i, &word) in self.code.iter().enumerate() {
            bus.write_word_at(PC + 2 * i as u32, word);
        }
        bus.write_word_at(PC + 2 * self.code.len() as u32, 0x4E71);
        for (i, &byte) in self.data.iter().enumerate() {
            bus.write_byte(DATA + i as u32, byte);
        }
        cpu.pc = PC;
        bus
    }

    /// The state full dispatch leaves, and the state the batch loop leaves
    /// with the number of instructions it ran itself.
    fn run(&self, cpus: &mut (CpuCore, CpuCore), model: Model) -> (String, String, u32) {
        let (stepped, batched) = cpus;
        let mut bus = self.start(stepped, model);
        for _ in 0..self.instructions {
            stepped.step(&mut bus);
        }
        let mut full = machine_state(stepped);
        let stepped_memory = bus;
        let mut bus = self.start(batched, model);
        DECODED_OPS.set(0);
        batched.run_batch(&mut bus, self.instructions, &[]);
        let mut batch = machine_state(batched);
        let (stepped_memory, memory) = (stepped_memory.as_slice(), bus.as_slice());
        if stepped_memory != memory
            && let Some(at) = (0..memory.len()).find(|&at| stepped_memory[at] != memory[at])
        {
            full += &format!(" memory {at:#x}: {:#04x}", stepped_memory[at]);
            batch += &format!(" memory {at:#x}: {:#04x}", memory[at]);
        }
        (full, batch, DECODED_OPS.get())
    }
}

fn machine_state(cpu: &CpuCore) -> String {
    format!("{} dar {:x?}", state(cpu), cpu.dar)
}

fn cpus() -> (CpuCore, CpuCore) {
    (CpuCore::new(), CpuCore::new())
}

#[test]
fn decoded_register_arithmetic_leaves_the_state_of_full_dispatch() {
    let mut cpus = cpus();
    let mut decoded = 0;
    for model in MODELS {
        for opmode in OPMODES {
            for &src in &operands() {
                for &dst in &operands() {
                    for fpcr in [0x00, 0x3430, 0xFF90] {
                        let case = Case {
                            src,
                            dst,
                            fpcr,
                            ..Case::new(vec![0xF200, (3 << 10) | (1 << 7) | opmode])
                        };
                        let (full, batch, ran) = case.run(&mut cpus, model);
                        assert_eq!(full, batch, "{model:?} {case:x?}");
                        decoded += ran;
                    }
                }
            }
        }
    }
    // Most of the cases run in the batch loop, not through its fallback.
    assert!(decoded > 30_000, "{decoded}");
}

#[test]
fn the_batch_loop_runs_only_hardware_arithmetic_itself() {
    let mut cpus = cpus();
    let register = |opmode: u16| Case::new(vec![0xF200, (3 << 10) | (1 << 7) | opmode]);
    for model in MODELS {
        // FADD, and FDADD where it exists.
        assert_eq!(register(0x22).run(&mut cpus, model).2, 1, "{model:?}");
        let rounded = u32::from(model.cpu != CpuType::M68030);
        assert_eq!(register(0x66).run(&mut cpus, model).2, rounded, "{model:?}");
        // FINT, FSIN and FSINCOS are not in every FPU; FMOVECR and FMOVE
        // to memory are not arithmetic on a fetched source.
        for opmode in [0x01, 0x0E, 0x32] {
            assert_eq!(register(opmode).run(&mut cpus, model).2, 0, "{model:?}");
        }
        let fmovecr = Case::new(vec![0xF200, 0x5C80]);
        assert_eq!(fmovecr.run(&mut cpus, model).2, 0, "{model:?}");
        let fmove_out = Case::new(vec![0xF210, 0x7480]);
        assert_eq!(fmove_out.run(&mut cpus, model).2, 0, "{model:?}");
        // A denormalized operand is the software package's when that
        // runs in the guest.
        let denormal = Case {
            src: FloatX80::from_extended(0, 0x4000_0000_0000_0000),
            ..register(0x22)
        };
        let hardware = u32::from(model.emulate || model.cpu == CpuType::M68030);
        assert_eq!(denormal.run(&mut cpus, model).2, hardware, "{model:?}");
    }
}

#[test]
fn a_held_exception_is_reported_by_full_dispatch() {
    // FDIV by zero with DZ enabled holds its exception; the FADD after it
    // takes it before executing, which only full dispatch does.
    let mut cpus = cpus();
    for model in MODELS {
        let case = Case {
            src: FloatX80::zero(false),
            fpcr: 0x0400,
            instructions: 2,
            ..Case::new(vec![0xF200, 0x0CA0, 0xF200, 0x0CA2])
        };
        let (full, batch, ran) = case.run(&mut cpus, model);
        assert_eq!(full, batch, "{model:?}");
        assert_eq!(ran, 1, "{model:?}");
    }
}

/// Source formats with operands in each: (format field, big-endian bytes).
fn sources() -> Vec<(u16, Vec<u8>)> {
    let mut sources = Vec::new();
    for long in [0u32, 1, 0xFFFF_FFFF, 0x7FFF_FFFF, 0x8000_0000] {
        sources.push((0, long.to_be_bytes().to_vec()));
    }
    for word in [0u16, 1, 0xFFFF, 0x8000] {
        sources.push((4, word.to_be_bytes().to_vec()));
    }
    for byte in [0u8, 1, 0xFF, 0x80] {
        sources.push((6, vec![byte]));
    }
    // 1.0, -0.1, zeros, a denormal, infinity, quiet and signaling NaNs and
    // the largest finite number.
    for single in [
        0x3F80_0000u32,
        0xBDCC_CCCD,
        0,
        0x8000_0000,
        0x0000_0001,
        0x7F80_0000,
        0x7FC0_0001,
        0x7F80_0001,
        0x7F7F_FFFF,
    ] {
        sources.push((1, single.to_be_bytes().to_vec()));
    }
    for double in [
        0x3FF0_0000_0000_0000u64,
        0xBFB9_9999_9999_999A,
        0,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x7FF0_0000_0000_0000,
        0x7FF8_0000_0000_0001,
        0x7FF0_0000_0000_0001,
        0x7FEF_FFFF_FFFF_FFFF,
    ] {
        sources.push((5, double.to_be_bytes().to_vec()));
    }
    for extended in operands() {
        let (exp_word, mantissa) = extended.to_extended();
        let mut bytes = (u32::from(exp_word) << 16).to_be_bytes().to_vec();
        bytes.extend(mantissa.to_be_bytes());
        sources.push((2, bytes));
    }
    sources
}

/// `fop.<fmt> <ea>,fp1` for each way of addressing `data`.
fn addressed(fmt: u16, opmode: u16, data: &[u8]) -> Vec<Case> {
    let w2 = 0x4000 | (fmt << 10) | (1 << 7) | opmode;
    let len = data.len() as u32;
    let memory = |ea: u16, ext: &[u16]| {
        let mut code = vec![0xF200 | ea, w2];
        code.extend_from_slice(ext);
        Case {
            data: data.to_vec(),
            ..Case::new(code)
        }
    };
    let mut cases = vec![
        memory(0x10, &[]),
        memory(0x18, &[]),
        Case {
            a0: DATA + len,
            ..memory(0x20, &[])
        },
        Case {
            a0: DATA - 16,
            ..memory(0x28, &[0x0010])
        },
        // (8,A0,D1.L*4) and the same through a full extension word.
        Case {
            a0: DATA - 16,
            ..memory(0x30, &[0x1C08])
        },
        Case {
            a0: DATA - 16,
            ..memory(0x30, &[0x1D20, 0x0008])
        },
        memory(0x38, &[DATA as u16]),
        memory(0x39, &[0, DATA as u16]),
        memory(0x3A, &[(DATA - PC - 4) as u16]),
        // (bd16,PC,D1.L*4): a brief extension word's byte cannot reach.
        memory(0x3B, &[0x1D20, (DATA - PC - 4 - 8) as u16]),
        // The stack pointer steps by two for a byte.
        Case {
            a7: DATA,
            ..memory(0x1F, &[])
        },
        Case {
            a7: DATA + len.max(2),
            ..memory(0x27, &[])
        },
        // Outside the window: full dispatch reads the bus.
        Case {
            a0: 0x0002_0000 + DATA,
            ..memory(0x10, &[])
        },
        // No such operand.
        memory(0x08, &[]),
    ];
    // Immediate: a byte in the low half of a word, the rest as stored.
    let mut immediate = data.to_vec();
    if immediate.len() == 1 {
        immediate.insert(0, 0x12);
    }
    let words: Vec<u16> = immediate
        .chunks(2)
        .map(|pair| u16::from_be_bytes([pair[0], pair[1]]))
        .collect();
    cases.push(memory(0x3C, &words));
    // A data register holds four bytes at most.
    let mut register = [0x12, 0x34, 0x56, 0x78];
    if len <= 4 {
        register[4 - data.len()..].copy_from_slice(data);
    }
    cases.push(Case {
        d2: u32::from_be_bytes(register),
        ..memory(0x02, &[])
    });
    cases
}

#[test]
fn decoded_source_arithmetic_leaves_the_state_of_full_dispatch() {
    // FADD, FMUL, FSUB, FDIV, FMOVE, FCMP, FTST, FSQRT, FDDIV, FSGLDIV, and
    // FINT and FSIN, which the batch loop leaves alone.
    const SOURCE_OPMODES: [u16; 12] = [
        0x22, 0x23, 0x28, 0x20, 0x00, 0x38, 0x3A, 0x04, 0x64, 0x24, 0x01, 0x0E,
    ];
    let x = FloatX80::from_extended;
    let destinations = [
        FloatX80::from_f64(1.0),
        x(0x0000, 0x4000_0000_0000_0000), // denormal
        x(0x7FFF, 0x0000_0000_0000_0000), // +inf
    ];
    let mut cpus = cpus();
    let mut decoded = 0;
    for model in MODELS {
        for (fmt, data) in sources() {
            for opmode in SOURCE_OPMODES {
                for case in addressed(fmt, opmode, &data) {
                    for dst in destinations {
                        for fpcr in [0x00, 0x3430, 0xFF90] {
                            let case = Case {
                                dst,
                                fpcr,
                                ..case.clone()
                            };
                            let (full, batch, ran) = case.run(&mut cpus, model);
                            assert_eq!(full, batch, "{model:?} {case:x?}");
                            decoded += ran;
                        }
                    }
                }
            }
        }
    }
    assert!(decoded > 200_000, "{decoded}");
}

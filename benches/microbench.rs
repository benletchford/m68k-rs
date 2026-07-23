use m68k::core::memory::{AddressBus, LinearMemoryBus};
use m68k::{CpuCore, CpuType};
use std::time::Instant;

trait BenchBus: AddressBus {
    fn new() -> Self;
    fn write_word_at(&mut self, address: u32, value: u16);

    fn filled(opcode: u16) -> Self
    where
        Self: Sized,
    {
        let mut bus = Self::new();
        for addr in (0..0x10000).step_by(2) {
            bus.write_word_at(addr as u32, opcode);
        }
        bus
    }
}

struct PlainBenchBus {
    memory: [u8; 0x10000],
}

impl BenchBus for PlainBenchBus {
    fn new() -> Self {
        Self {
            memory: [0; 0x10000],
        }
    }

    fn write_word_at(&mut self, address: u32, value: u16) {
        let addr = (address as usize) & 0xFFFF;
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[(addr + 1) & 0xFFFF] = bytes[1];
    }
}

impl AddressBus for PlainBenchBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[(address as usize) & 0xFFFF]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let addr = (address as usize) & 0xFFFF;
        u16::from_be_bytes([self.memory[addr], self.memory[(addr + 1) & 0xFFFF]])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let addr = (address as usize) & 0xFFFF;
        u32::from_be_bytes([
            self.memory[addr],
            self.memory[(addr + 1) & 0xFFFF],
            self.memory[(addr + 2) & 0xFFFF],
            self.memory[(addr + 3) & 0xFFFF],
        ])
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[(address as usize) & 0xFFFF] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        self.write_word_at(address, value);
    }

    fn write_long(&mut self, address: u32, value: u32) {
        let addr = (address as usize) & 0xFFFF;
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[(addr + 1) & 0xFFFF] = bytes[1];
        self.memory[(addr + 2) & 0xFFFF] = bytes[2];
        self.memory[(addr + 3) & 0xFFFF] = bytes[3];
    }
}

impl BenchBus for LinearMemoryBus {
    fn new() -> Self {
        LinearMemoryBus::new(0x10000)
    }

    fn write_word_at(&mut self, address: u32, value: u16) {
        LinearMemoryBus::write_word_at(self, address, value);
    }
}

fn cpu_at_zero() -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.set_sr(0x2700);
    cpu.pc = 0;
    cpu
}

fn bench_linear<B: BenchBus>(
    label: &str,
    name: &str,
    opcode: u16,
    cycles_per_instr: i32,
    instrs: u64,
) {
    let mut bus = B::filled(opcode);
    let mut cpu = cpu_at_zero();
    cpu.execute(&mut bus, 100_000 * cycles_per_instr);

    let mut cpu = cpu_at_zero();
    let cycles = (instrs as i32) * cycles_per_instr;
    let start = Instant::now();
    let used = cpu.execute(&mut bus, cycles);
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{label:9} {name:18} {:8.1} M instr/s  cycles={used}",
        instrs as f64 / elapsed / 1_000_000.0
    );
}

fn bench_loop<B: BenchBus>(
    label: &str,
    name: &str,
    words: &[u16],
    cycles_per_iter: i32,
    instrs_per_iter: u64,
    iters: u64,
) {
    let mut bus = B::new();
    for (i, word) in words.iter().enumerate() {
        bus.write_word_at((i * 2) as u32, *word);
    }

    let mut cpu = cpu_at_zero();
    cpu.set_d(0, 3);
    cpu.set_d(1, 2);
    cpu.execute(&mut bus, 10_000 * cycles_per_iter);

    let mut cpu = cpu_at_zero();
    cpu.set_d(0, 3);
    cpu.set_d(1, 2);
    let cycles = (iters as i32) * cycles_per_iter;
    let instrs = iters * instrs_per_iter;
    let start = Instant::now();
    let used = cpu.execute(&mut bus, cycles);
    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "{label:9} {name:18} {:8.1} M instr/s  cycles={used}",
        instrs as f64 / elapsed / 1_000_000.0
    );
}

fn bench_batch_loop(label: &str, name: &str, words: &[u16], instrs: u32) {
    bench_batch_loop_at(label, name, words, instrs, 0x100);
}

fn bench_batch_loop_at(label: &str, name: &str, words: &[u16], instrs: u32, code_base: usize) {
    let mut bus = LinearMemoryBus::new(0x10000);
    for (i, word) in words.iter().enumerate() {
        bus.write_word_at((code_base + i * 2) as u32, *word);
    }

    let prepare_cpu = || {
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = code_base as u32;
        cpu.set_a(5, 0x1000);
        cpu.set_a(7, 0x8000);
        cpu
    };

    let mut warm_cpu = prepare_cpu();
    // Systemless always watches PC 0 as its clean-exit sentinel. An
    // unrelated watched PC must not force a nonzero self-loop to execute
    // only one iteration per native trace call.
    let warm = warm_cpu.run_batch(&mut bus, 5_000_000, &[0]);
    assert_eq!(warm.instructions, 5_000_000);

    let mut cpu = prepare_cpu();
    let start = Instant::now();
    let result = cpu.run_batch(&mut bus, instrs, &[0]);
    let elapsed = start.elapsed().as_secs_f64();
    assert_eq!(result.instructions, instrs);
    println!(
        "{label:9} {name:18} {:8.1} M instr/s",
        instrs as f64 / elapsed / 1_000_000.0
    );
}

fn bench_one_shot_trace(head_ops: usize, instrs: u32) {
    assert!((2..=16).contains(&head_ops));
    let mut words = vec![0x5280; head_ops - 1]; // ADDQ.L #1,D0
    words.push(0x6002); // BRA.B over the padding word
    words.push(0x4E71); // padding, never executed
    words.push(0x5381); // SUBQ.L #1,D1 (interpreted return path)
    let bytes_after_back_branch = (words.len() + 1) * 2;
    let back_disp = -(bytes_after_back_branch as i16);
    assert!((-128..=-1).contains(&back_disp));
    words.push(0x6000 | (back_disp as u8 as u16));

    bench_batch_loop_at(
        "batch",
        &format!("one-shot {head_ops}"),
        &words,
        instrs,
        0x100 + head_ops * 0x40,
    );
}

fn bench_one_shot_a5_trace(head_ops: usize, instrs: u32) {
    assert!((2..=9).contains(&head_ops));
    let a5_ops: &[&[u16]] = &[
        &[0x4A2D, 0x0100],         // TST.B $0100(A5)
        &[0x526D, 0x0100],         // ADDQ.W #1,$0100(A5)
        &[0x322D, 0x0100],         // MOVE.W $0100(A5),D1
        &[0x422D, 0x0100],         // CLR.B $0100(A5)
        &[0x1B40, 0x0100],         // MOVE.B D0,$0100(A5)
        &[0x082D, 0x0003, 0x0100], // BTST #3,$0100(A5)
    ];
    let mut words = Vec::new();
    for i in 0..head_ops - 1 {
        words.extend_from_slice(a5_ops[i % a5_ops.len()]);
    }
    words.push(0x6002); // BRA.B over the padding word
    words.push(0x4E71); // padding, never executed
    words.push(0x5381); // SUBQ.L #1,D1 (interpreted return path)
    let bytes_after_back_branch = (words.len() + 1) * 2;
    let back_disp = -(bytes_after_back_branch as i16);
    assert!((-128..=-1).contains(&back_disp));
    words.push(0x6000 | (back_disp as u8 as u16));

    bench_batch_loop_at(
        "batch",
        &format!("A5 one-shot {head_ops}"),
        &words,
        instrs,
        0x800 + head_ops * 0x40,
    );
}

/// Measure decoded generic memory operations without allowing a backward
/// branch to turn the workload into a native JIT loop. Each pass walks the
/// same straight-line code, retaining the decoded-op cache while resetting
/// only the architectural state changed by the instruction stream.
fn bench_generic_memory(name: &str, words: &[u16], instrs_per_pattern: u32) {
    const CODE_BASE: usize = 0x100;
    const PATTERNS: u32 = 16_384;
    const PASSES: u32 = 512;
    let mut bus = LinearMemoryBus::new(0x10_0000);
    for pattern in 0..PATTERNS as usize {
        for (word, value) in words.iter().enumerate() {
            bus.write_word_at(
                (CODE_BASE + (pattern * words.len() + word) * 2) as u32,
                *value,
            );
        }
    }

    let instrs_per_pass = PATTERNS * instrs_per_pattern;
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68040);
    cpu.set_sr(0x2700);
    cpu.pc = CODE_BASE as u32;
    cpu.set_a(0, 0x80000);
    let warm = cpu.run_batch(&mut bus, instrs_per_pass, &[0]);
    assert_eq!(warm.instructions, instrs_per_pass);

    let start = Instant::now();
    for _ in 0..PASSES {
        cpu.pc = CODE_BASE as u32;
        cpu.set_a(0, 0x80000);
        let result = cpu.run_batch(&mut bus, instrs_per_pass, &[0]);
        assert_eq!(result.instructions, instrs_per_pass);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let instrs = u64::from(instrs_per_pass) * u64::from(PASSES);
    println!(
        "batch     {name:18} {:8.1} M instr/s",
        instrs as f64 / elapsed / 1_000_000.0
    );
}

fn bench_set<B: BenchBus>(label: &str) {
    bench_linear::<B>(label, "linear NOP", 0x4E71, 4, 40_000_000);
    bench_linear::<B>(label, "linear ADDQ", 0x5280, 4, 40_000_000);
    bench_linear::<B>(label, "linear MOVEQ", 0x7001, 4, 40_000_000);
    bench_loop::<B>(label, "loop ADDQ/BRA", &[0x5280, 0x60FC], 14, 2, 30_000_000);
    bench_loop::<B>(label, "loop TST/BNE", &[0x4A80, 0x66FC], 14, 2, 30_000_000);
    bench_loop::<B>(
        label,
        "loop TST/BNE.W",
        &[0x4A80, 0x6600, 0xFFFC],
        14,
        2,
        30_000_000,
    );
    bench_loop::<B>(
        label,
        "loop reg mix",
        &[0x2400, 0xD481, 0x5282, 0xB182, 0x4A82, 0x60F4],
        30,
        6,
        12_500_000,
    );
}

fn main() {
    println!("m68k microbench");
    let only = std::env::args().nth(1);
    if only.as_deref() == Some("trace-calls") {
        // Unlike a self-loop, each native trace here executes once before
        // returning to Rust. This isolates the call-boundary break-even
        // point seen in branch-heavy application code such as Lemmings.
        for head_ops in 2..=9 {
            bench_one_shot_trace(head_ops, 50_000_000);
        }
        return;
    }
    if only.as_deref() == Some("a5-trace-calls") {
        for head_ops in 2..=9 {
            bench_one_shot_a5_trace(head_ops, 50_000_000);
        }
        return;
    }
    if only.as_deref() == Some("generic-memory") {
        bench_generic_memory("TST.B (A0)", &[0x4A10], 1);
        bench_generic_memory("TST.B (A0)+", &[0x4A18], 1);
        bench_generic_memory("TST.B index", &[0x4A30, 0x0000], 1);
        bench_generic_memory("ADD.W (A0),D0", &[0xD050], 1);
        return;
    }
    if only.as_deref() == Some("region") {
        bench_batch_loop(
            "batch",
            "multi-block region",
            &[
                0x5280, // ADDQ.L #1,D0
                0x6602, // BNE.S skip
                0x4E71, // uncommon fallthrough
                0x5281, // skip: ADDQ.L #1,D1
                0x60F6, // BRA.S loop
            ],
            200_000_000,
        );
        return;
    }
    if only.as_deref() != Some("batch") {
        bench_set::<PlainBenchBus>("plain");
        bench_set::<LinearMemoryBus>("linearbus");
    }
    // A5-relative globals and stack temporaries dominate classic Mac code.
    // This loop mirrors the hottest Lemmings opcode forms found by runtime
    // profiling while remaining deterministic and self-contained.
    if only.as_deref() != Some("legacy") {
        bench_batch_loop(
            "batch",
            "classic Mac mix",
            &[
                0x4A2D, 0x0100, // TST.B $0100(A5)
                0x082D, 0x0003, 0x0100, // BTST #3,$0100(A5)
                0x1B40, 0x0100, // MOVE.B D0,$0100(A5)
                0x422D, 0x0100, // CLR.B $0100(A5)
                0x322D, 0x0100, // MOVE.W $0100(A5),D1
                0x526D, 0x0100, // ADDQ.W #1,$0100(A5)
                0x2F2D, 0x0100, // MOVE.L $0100(A5),-(A7)
                0x588F, // ADDQ.L #4,A7
                0x60DE, // BRA.B back to the first instruction
            ],
            200_000_000,
        );
    }
}

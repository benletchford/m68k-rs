use m68k::core::memory::{AddressBus, LinearMemoryBus};
use m68k::{CpuCore, CpuType, HleHandler, StepResult};
use std::collections::HashMap;
use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

const MEM_SIZE: usize = 0x10000;
const BENCH_PC: u32 = 0x0100;
const STACK_TOP: u32 = 0x8000;
const TICK_GLOBAL: u32 = 0x016A;
const PC_SAMPLE_INTERVAL: u64 = 1000;

trait BenchBus: AddressBus {
    fn new() -> Self;
    fn write_word_at(&mut self, address: u32, value: u16);

    fn write_long_at(&mut self, address: u32, value: u32) {
        self.write_long(address, value);
    }

    fn load_words(&mut self, address: u32, words: &[u16]) {
        for (i, word) in words.iter().enumerate() {
            self.write_word_at(address.wrapping_add((i * 2) as u32), *word);
        }
    }

    fn filled(opcode: u16) -> Self
    where
        Self: Sized,
    {
        let mut bus = Self::new();
        for addr in (0..MEM_SIZE).step_by(2) {
            bus.write_word_at(addr as u32, opcode);
        }
        bus
    }
}

struct PlainBenchBus {
    memory: [u8; MEM_SIZE],
}

impl BenchBus for PlainBenchBus {
    fn new() -> Self {
        Self {
            memory: [0; MEM_SIZE],
        }
    }

    fn write_word_at(&mut self, address: u32, value: u16) {
        let addr = (address as usize) & (MEM_SIZE - 1);
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[(addr + 1) & (MEM_SIZE - 1)] = bytes[1];
    }
}

impl AddressBus for PlainBenchBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[(address as usize) & (MEM_SIZE - 1)]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let addr = (address as usize) & (MEM_SIZE - 1);
        u16::from_be_bytes([self.memory[addr], self.memory[(addr + 1) & (MEM_SIZE - 1)]])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let addr = (address as usize) & (MEM_SIZE - 1);
        u32::from_be_bytes([
            self.memory[addr],
            self.memory[(addr + 1) & (MEM_SIZE - 1)],
            self.memory[(addr + 2) & (MEM_SIZE - 1)],
            self.memory[(addr + 3) & (MEM_SIZE - 1)],
        ])
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[(address as usize) & (MEM_SIZE - 1)] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        self.write_word_at(address, value);
    }

    fn write_long(&mut self, address: u32, value: u32) {
        let addr = (address as usize) & (MEM_SIZE - 1);
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[(addr + 1) & (MEM_SIZE - 1)] = bytes[1];
        self.memory[(addr + 2) & (MEM_SIZE - 1)] = bytes[2];
        self.memory[(addr + 3) & (MEM_SIZE - 1)] = bytes[3];
    }
}

struct MappedBenchBus {
    memory: Vec<u8>,
    mask: usize,
    slow_path_reads: u64,
}

impl MappedBenchBus {
    fn index(&mut self, address: u32) -> usize {
        let address = address & 0x00FF_FFFF;
        if address as usize >= self.memory.len() {
            self.slow_path_reads = self.slow_path_reads.wrapping_add(1);
        }
        (address as usize) & self.mask
    }
}

impl BenchBus for MappedBenchBus {
    fn new() -> Self {
        Self {
            memory: vec![0; 0x0040_0000],
            mask: 0x0040_0000 - 1,
            slow_path_reads: 0,
        }
    }

    fn write_word_at(&mut self, address: u32, value: u16) {
        let addr = self.index(address);
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[(addr + 1) & self.mask] = bytes[1];
    }
}

impl AddressBus for MappedBenchBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        let idx = self.index(address);
        self.memory[idx]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let idx = self.index(address);
        u16::from_be_bytes([self.memory[idx], self.memory[(idx + 1) & self.mask]])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let idx = self.index(address);
        u32::from_be_bytes([
            self.memory[idx],
            self.memory[(idx + 1) & self.mask],
            self.memory[(idx + 2) & self.mask],
            self.memory[(idx + 3) & self.mask],
        ])
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        let idx = self.index(address);
        self.memory[idx] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        self.write_word_at(address, value);
    }

    fn write_long(&mut self, address: u32, value: u32) {
        let idx = self.index(address);
        let bytes = value.to_be_bytes();
        self.memory[idx] = bytes[0];
        self.memory[(idx + 1) & self.mask] = bytes[1];
        self.memory[(idx + 2) & self.mask] = bytes[2];
        self.memory[(idx + 3) & self.mask] = bytes[3];
    }
}

impl BenchBus for LinearMemoryBus {
    fn new() -> Self {
        LinearMemoryBus::new(MEM_SIZE)
    }

    fn write_word_at(&mut self, address: u32, value: u16) {
        LinearMemoryBus::write_word_at(self, address, value);
    }

    fn write_long_at(&mut self, address: u32, value: u32) {
        LinearMemoryBus::write_long_at(self, address, value);
    }
}

#[derive(Clone)]
struct Config {
    scale: f64,
    filter: Option<String>,
}

impl Config {
    fn from_env() -> Self {
        let quick = env::args().any(|arg| arg == "--quick");
        let scale = env::var("M68K_BENCH_SCALE")
            .ok()
            .and_then(|value| value.parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0)
            .unwrap_or(if quick { 0.03 } else { 1.0 });
        let filter = env::var("M68K_BENCH_FILTER")
            .ok()
            .map(|value| value.to_ascii_lowercase())
            .filter(|value| !value.is_empty());

        Self { scale, filter }
    }

    fn iterations(&self, base: u64, minimum: u64) -> u64 {
        ((base as f64 * self.scale).round() as u64).max(minimum)
    }

    fn includes(&self, group: &str, name: &str) -> bool {
        match &self.filter {
            Some(filter) => {
                group.to_ascii_lowercase().contains(filter)
                    || name.to_ascii_lowercase().contains(filter)
            }
            None => true,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct Outcome {
    instructions: u64,
    cycles: i64,
    checksum: u64,
}

impl Outcome {
    fn with_cpu(mut self, cpu: &CpuCore) -> Self {
        self.checksum ^= cpu.pc as u64;
        self.checksum ^= (cpu.d(0) as u64).rotate_left(7);
        self.checksum ^= (cpu.d(1) as u64).rotate_left(13);
        self.checksum ^= (cpu.d(2) as u64).rotate_left(19);
        self.checksum ^= (cpu.a(7) as u64).rotate_left(29);
        self
    }
}

fn cpu_at(pc: u32, cpu_type: CpuType) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.set_sr(0x2700);
    cpu.pc = pc;
    cpu.set_d(0, 3);
    cpu.set_d(1, 2);
    cpu.set_d(2, 0);
    cpu.set_a(0, 0x2000);
    cpu.set_a(1, 0x3000);
    cpu.set_a(5, 0x5000);
    cpu.set_a(6, 0x7000);
    cpu.set_a(7, STACK_TOP);
    cpu
}

fn print_header(config: &Config) {
    println!("m68k local development benches");
    println!(
        "scale={:.3} filter={}",
        config.scale,
        config.filter.as_deref().unwrap_or("<none>")
    );
    println!(
        "{:<12} {:<24} {:>10} {:>10} {:>10} {:>12} {:>12}",
        "group", "bench", "M instr/s", "M cyc/s", "ns/inst", "instrs", "checksum"
    );
}

fn print_result(group: &str, name: &str, outcome: Outcome, elapsed: Duration) {
    let seconds = elapsed.as_secs_f64();
    let instrs = outcome.instructions.max(1);
    let mips = outcome.instructions as f64 / seconds / 1_000_000.0;
    let mcycles = outcome.cycles.max(0) as f64 / seconds / 1_000_000.0;
    let ns_per_instr = seconds * 1_000_000_000.0 / instrs as f64;
    println!(
        "{:<12} {:<24} {:>10.1} {:>10.1} {:>10.2} {:>12} {:>012X}",
        group,
        name,
        mips,
        mcycles,
        ns_per_instr,
        outcome.instructions,
        black_box(outcome.checksum)
    );
}

fn measure<F>(config: &Config, group: &str, name: &str, mut run: F)
where
    F: FnMut(u64) -> Outcome,
{
    if !config.includes(group, name) {
        return;
    }

    let _ = black_box(run(1));
    let start = Instant::now();
    let outcome = run(0);
    print_result(group, name, outcome, start.elapsed());
}

fn bench_execute_linear<B: BenchBus>(
    config: &Config,
    group: &str,
    name: &str,
    opcode: u16,
    cycles_per_instr: i32,
    base_instrs: u64,
) {
    let instrs = config.iterations(base_instrs, 50_000);
    measure(config, group, name, move |warmup| {
        let instrs = if warmup == 0 { instrs } else { 25_000 };
        let mut bus = B::filled(opcode);
        let mut cpu = cpu_at(0, CpuType::M68000);
        let budget = instrs.saturating_mul(cycles_per_instr as u64) as i32;
        let used = cpu.execute(&mut bus, budget);
        Outcome {
            instructions: (used / cycles_per_instr) as u64,
            cycles: used as i64,
            checksum: opcode as u64,
        }
        .with_cpu(&cpu)
    });
}

fn bench_execute_loop<B: BenchBus>(
    config: &Config,
    group: &str,
    name: &str,
    words: &'static [u16],
    cycles_per_iter: i32,
    instrs_per_iter: u64,
    base_iters: u64,
) {
    let iters = config.iterations(base_iters, 20_000);
    measure(config, group, name, move |warmup| {
        let iters = if warmup == 0 { iters } else { 10_000 };
        let mut bus = B::new();
        bus.load_words(0, words);
        let mut cpu = cpu_at(0, CpuType::M68000);
        let budget = iters.saturating_mul(cycles_per_iter as u64) as i32;
        let used = cpu.execute(&mut bus, budget);
        Outcome {
            instructions: iters * instrs_per_iter,
            cycles: used as i64,
            checksum: words.len() as u64,
        }
        .with_cpu(&cpu)
    });
}

fn bench_step_loop<B: BenchBus>(
    config: &Config,
    group: &str,
    name: &str,
    words: &'static [u16],
    base_steps: u64,
    bookkeeping: bool,
) {
    let steps = config.iterations(base_steps, 20_000);
    measure(config, group, name, move |warmup| {
        let steps = if warmup == 0 { steps } else { 10_000 };
        let mut bus = B::new();
        bus.write_long_at(TICK_GLOBAL, 600);
        bus.load_words(BENCH_PC, words);

        let mut cpu = cpu_at(BENCH_PC, CpuType::M68000);
        let mut cycles = 0i64;
        let mut instructions = 0u64;
        let mut tick_budget = 28_000i32;
        let mut ticks = 600u32;
        let mut opcode_histogram = if bookkeeping {
            Some(Box::new([0u64; 65536]))
        } else {
            None
        };
        let mut pc_histogram = HashMap::<u32, u64>::new();

        while instructions < steps {
            let pc = cpu.pc;
            tick_budget -= 1;
            if tick_budget <= 0 {
                ticks = ticks.wrapping_add(1);
                bus.write_long(TICK_GLOBAL, ticks);
                tick_budget += 28_000;
            }

            match cpu.step(&mut bus) {
                StepResult::Ok {
                    cycles: step_cycles,
                } => {
                    cycles += step_cycles as i64;
                    instructions += 1;
                    if let Some(histogram) = opcode_histogram.as_mut() {
                        histogram[cpu.ir as u16 as usize] =
                            histogram[cpu.ir as u16 as usize].saturating_add(1);
                        if instructions.is_multiple_of(PC_SAMPLE_INTERVAL) {
                            *pc_histogram.entry(pc).or_insert(0) += 1;
                        }
                    }
                }
                StepResult::Stopped => break,
                other => panic!("unexpected step result in {name}: {other:?}"),
            }
        }

        let mut outcome = Outcome {
            instructions,
            cycles,
            checksum: ticks as u64 ^ pc_histogram.len() as u64,
        }
        .with_cpu(&cpu);
        if let Some(histogram) = opcode_histogram {
            outcome.checksum ^= histogram[0x4A82].rotate_left(3);
            outcome.checksum ^= histogram[0x60F4].rotate_left(9);
        }
        outcome
    });
}

struct TrapHandler {
    tick_count: u32,
    trap_count: u64,
    histogram: [u64; 4096],
    modal_refire: bool,
}

impl TrapHandler {
    fn new(modal_refire: bool) -> Self {
        Self {
            tick_count: 600,
            trap_count: 0,
            histogram: [0; 4096],
            modal_refire,
        }
    }

    fn checksum(&self) -> u64 {
        self.trap_count
            ^ (self.histogram[0x975] << 7)
            ^ (self.histogram[0x991] << 13)
            ^ self.tick_count as u64
    }
}

impl HleHandler for TrapHandler {
    fn handle_aline(&mut self, cpu: &mut CpuCore, bus: &mut dyn AddressBus, opcode: u16) -> bool {
        self.trap_count += 1;
        self.histogram[(opcode & 0x0FFF) as usize] =
            self.histogram[(opcode & 0x0FFF) as usize].saturating_add(1);

        match opcode {
            0xA975 => {
                self.tick_count = self.tick_count.wrapping_add(1);
                bus.write_long(cpu.a(7), self.tick_count);
            }
            0xA991 if self.modal_refire => {
                cpu.pc = cpu.ppc;
            }
            _ => {}
        }

        true
    }
}

fn bench_hle_trap_loop<B: BenchBus>(
    config: &Config,
    group: &str,
    name: &str,
    words: &'static [u16],
    base_steps: u64,
    modal_refire: bool,
) {
    let steps = config.iterations(base_steps, 20_000);
    measure(config, group, name, move |warmup| {
        let steps = if warmup == 0 { steps } else { 10_000 };
        let mut bus = B::new();
        bus.load_words(BENCH_PC, words);
        let mut cpu = cpu_at(BENCH_PC, CpuType::M68000);
        let mut handler = TrapHandler::new(modal_refire);
        let mut cycles = 0i64;
        let mut instructions = 0u64;

        while instructions < steps {
            match cpu.step_with_hle_handler(&mut bus, &mut handler) {
                StepResult::Ok {
                    cycles: step_cycles,
                } => {
                    cycles += step_cycles as i64;
                    instructions += 1;
                }
                StepResult::Stopped => break,
                other => panic!("unexpected HLE step result in {name}: {other:?}"),
            }
        }

        Outcome {
            instructions,
            cycles,
            checksum: handler.checksum(),
        }
        .with_cpu(&cpu)
    });
}

fn bench_inline_trap_loop<B: BenchBus>(
    config: &Config,
    group: &str,
    name: &str,
    words: &'static [u16],
    base_steps: u64,
) {
    let steps = config.iterations(base_steps, 20_000);
    measure(config, group, name, move |warmup| {
        let steps = if warmup == 0 { steps } else { 10_000 };
        let mut bus = B::new();
        bus.load_words(BENCH_PC, words);
        let mut cpu = cpu_at(BENCH_PC, CpuType::M68000);
        let mut trap_count = 0u64;
        let mut inline_skipped = [0u64; 4096];
        let mut tick_count = 600u32;
        let mut cycles = 0i64;
        let mut instructions = 0u64;

        while instructions < steps {
            match cpu.step(&mut bus) {
                StepResult::Ok {
                    cycles: step_cycles,
                } => {
                    cycles += step_cycles as i64;
                    instructions += 1;
                }
                StepResult::AlineTrap { opcode } => {
                    instructions += 1;
                    trap_count += 1;
                    let idx = (opcode & 0x0FFF) as usize;
                    inline_skipped[idx] = inline_skipped[idx].saturating_add(1);
                    if opcode == 0xA975 {
                        tick_count = tick_count.wrapping_add(1);
                        bus.write_long(cpu.a(7), tick_count);
                    }
                }
                StepResult::Stopped => break,
                other => panic!("unexpected inline trap result in {name}: {other:?}"),
            }
        }

        Outcome {
            instructions,
            cycles,
            checksum: trap_count
                ^ (inline_skipped[0x975] << 5)
                ^ (inline_skipped[0x991] << 11)
                ^ tick_count as u64,
        }
        .with_cpu(&cpu)
    });
}

fn bench_decoded_batch_trap_loop<B: BenchBus>(
    config: &Config,
    group: &str,
    name: &str,
    words: &'static [u16],
    base_steps: u64,
) {
    let steps = config.iterations(base_steps, 20_000);
    let code_end = BENCH_PC + (words.len() as u32 * 2);
    measure(config, group, name, move |warmup| {
        let steps = if warmup == 0 { steps } else { 10_000 };
        let mut bus = B::new();
        bus.load_words(BENCH_PC, words);
        let mut cpu = cpu_at(BENCH_PC, CpuType::M68000);
        let mut handler = TrapHandler::new(false);
        let mut cycles = 0i64;
        let mut instructions = 0u64;
        let mut batches = 0u64;

        while instructions < steps {
            let room = (steps - instructions).min(64) as usize;
            let (batch_instructions, batch_cycles) =
                cpu.step_decoded_simple_batch_in_range(&mut bus, room, BENCH_PC, code_end);
            if batch_instructions > 0 {
                instructions += batch_instructions as u64;
                cycles += batch_cycles as i64;
                batches += 1;
                continue;
            }

            match cpu.step_with_hle_handler(&mut bus, &mut handler) {
                StepResult::Ok {
                    cycles: step_cycles,
                } => {
                    cycles += step_cycles as i64;
                    instructions += 1;
                }
                StepResult::Stopped => break,
                other => panic!("unexpected decoded batch result in {name}: {other:?}"),
            }
        }

        Outcome {
            instructions,
            cycles,
            checksum: handler.checksum() ^ batches.rotate_left(17),
        }
        .with_cpu(&cpu)
    });
}

fn bench_set<B: BenchBus>(config: &Config, label: &str) {
    bench_execute_linear::<B>(config, label, "execute linear NOP", 0x4E71, 4, 30_000_000);
    bench_execute_linear::<B>(config, label, "execute linear ADDQ", 0x5280, 4, 30_000_000);
    bench_execute_linear::<B>(config, label, "execute linear MOVEQ", 0x7001, 4, 30_000_000);

    bench_execute_loop::<B>(
        config,
        label,
        "execute ADDQ/BRA",
        &[0x5280, 0x60FC],
        14,
        2,
        20_000_000,
    );
    bench_execute_loop::<B>(
        config,
        label,
        "execute TST/BNE",
        &[0x4A80, 0x66FC],
        14,
        2,
        20_000_000,
    );
    bench_execute_loop::<B>(
        config,
        label,
        "execute reg mix",
        &[0x2400, 0xD481, 0x5282, 0xB182, 0x4A82, 0x60F4],
        30,
        6,
        8_000_000,
    );

    bench_step_loop::<B>(
        config,
        label,
        "runner step reg mix",
        &[0x2400, 0xD481, 0x5282, 0xB182, 0x4A82, 0x60F4],
        8_000_000,
        false,
    );
    bench_step_loop::<B>(
        config,
        label,
        "runner bookkeeping",
        &[0x2400, 0xD481, 0x5282, 0xB182, 0x4A82, 0x60F4],
        5_000_000,
        true,
    );
    bench_step_loop::<B>(
        config,
        label,
        "runner DBRA loop",
        &[0x323C, 0xFFFF, 0x5280, 0x51C9, 0xFFFC, 0x60F4],
        5_000_000,
        true,
    );

    bench_hle_trap_loop::<B>(
        config,
        label,
        "hle mixed traps",
        &[0x7001, 0xA975, 0x5280, 0xA991, 0x60F6],
        4_000_000,
        false,
    );
    bench_inline_trap_loop::<B>(
        config,
        label,
        "runner inline traps",
        &[0x7001, 0xA975, 0x5280, 0xA991, 0x60F6],
        4_000_000,
    );
    bench_hle_trap_loop::<B>(
        config,
        label,
        "hle span trap",
        &[
            0x7001, 0x5280, 0x2400, 0xD481, 0x5282, 0xB182, 0x4A82, 0x7202, 0x5281, 0x4A81, 0xA975,
            0x60E8,
        ],
        4_000_000,
        false,
    );
    bench_decoded_batch_trap_loop::<B>(
        config,
        label,
        "decoded batch span",
        &[
            0x7001, 0x5280, 0x2400, 0xD481, 0x5282, 0xB182, 0x4A82, 0x7202, 0x5281, 0x4A81, 0xA975,
            0x60E8,
        ],
        4_000_000,
    );
    bench_hle_trap_loop::<B>(
        config,
        label,
        "modal refire trap",
        &[0xA991],
        4_000_000,
        true,
    );
}

fn main() {
    let config = Config::from_env();
    print_header(&config);
    bench_set::<PlainBenchBus>(&config, "plain");
    bench_set::<LinearMemoryBus>(&config, "linearbus");
    bench_set::<MappedBenchBus>(&config, "mappedbus");
}

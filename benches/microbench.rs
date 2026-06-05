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
    bench_set::<PlainBenchBus>("plain");
    bench_set::<LinearMemoryBus>("linearbus");
}

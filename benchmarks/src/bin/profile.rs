//! Scratch profiling target for use under valgrind --tool=callgrind.
//!
//! Usage: profile <interp|batch> <memcpy|nop|addqbra>
//!
//! Runs one workload on one m68k-rs engine with a budget sized for
//! callgrind's ~50x slowdown.

use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus};

struct FlatBus {
    memory: Vec<u8>,
}

const MEM_SIZE: usize = 0x100_0000;
const MEM_MASK: usize = MEM_SIZE - 1;

impl AddressBus for FlatBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[address as usize & MEM_MASK]
    }
    fn read_word(&mut self, address: u32) -> u16 {
        let a = address as usize & MEM_MASK;
        ((self.memory[a] as u16) << 8) | self.memory[(a + 1) & MEM_MASK] as u16
    }
    fn read_long(&mut self, address: u32) -> u32 {
        let a = address as usize & MEM_MASK;
        ((self.memory[a] as u32) << 24)
            | ((self.memory[(a + 1) & MEM_MASK] as u32) << 16)
            | ((self.memory[(a + 2) & MEM_MASK] as u32) << 8)
            | self.memory[(a + 3) & MEM_MASK] as u32
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[address as usize & MEM_MASK] = value;
    }
    fn write_word(&mut self, address: u32, value: u16) {
        let a = address as usize & MEM_MASK;
        self.memory[a] = (value >> 8) as u8;
        self.memory[(a + 1) & MEM_MASK] = value as u8;
    }
    fn write_long(&mut self, address: u32, value: u32) {
        let a = address as usize & MEM_MASK;
        self.memory[a] = (value >> 24) as u8;
        self.memory[(a + 1) & MEM_MASK] = (value >> 16) as u8;
        self.memory[(a + 2) & MEM_MASK] = (value >> 8) as u8;
        self.memory[(a + 3) & MEM_MASK] = value as u8;
    }
}

fn build_memory(workload: &str) -> Vec<u8> {
    let mut memory = vec![0u8; MEM_SIZE];
    match workload {
        "callret" => {
            let words: &[u16] = &[0x5280, 0x6104, 0x60FA, 0x4E71, 0x4E75];
            for (i, w) in words.iter().enumerate() {
                let a = 0x1000 + i * 2;
                memory[a..a + 2].copy_from_slice(&w.to_be_bytes());
            }
        }
        "memcpy" => {
            let words: &[u16] = &[
                0x41F9, 0x0000, 0x8000, // LEA $8000.L,A0
                0x43F9, 0x0001, 0x0000, // LEA $10000.L,A1
                0x303C, 0x03FF, // MOVE.W #$03FF,D0
                0x22D8, // MOVE.L (A0)+,(A1)+
                0x51C8, 0xFFFC, // DBRA D0,-4
                0x5241, // ADDQ.W #1,D1
                0x60E6, // BRA.S -26
            ];
            for (i, w) in words.iter().enumerate() {
                let a = 0x1000 + i * 2;
                memory[a..a + 2].copy_from_slice(&w.to_be_bytes());
            }
        }
        "addqbra" => {
            let words: &[u16] = &[0x5280, 0x60FC];
            for (i, w) in words.iter().enumerate() {
                let a = 0x1000 + i * 2;
                memory[a..a + 2].copy_from_slice(&w.to_be_bytes());
            }
        }
        "nop" => {
            for chunk in memory.chunks_exact_mut(2) {
                chunk.copy_from_slice(&0x4E71u16.to_be_bytes());
            }
        }
        other => panic!("unknown workload {other}"),
    }
    memory
}

fn fresh_cpu() -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.set_sr(0x2700);
    cpu.pc = 0x1000;
    cpu.set_a(7, 0x0080_0000);
    cpu
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("interp");
    let workload = args.get(2).map(String::as_str).unwrap_or("memcpy");
    let memory = build_memory(workload);
    let mut cpu = fresh_cpu();

    match mode {
        "interp" => {
            let mut bus = FlatBus { memory };
            let used = cpu.execute(&mut bus, 15_000_000);
            println!(
                "interp {workload}: used={used} d0={} d1={}",
                cpu.d(0),
                cpu.d(1)
            );
        }
        "batch" => {
            let mut bus = LinearMemoryBus::from_vec(memory);
            let result = cpu.run_batch(&mut bus, 2_000_000, &[]);
            println!(
                "batch {workload}: retired={} d0={} d1={}",
                result.instructions,
                cpu.d(0),
                cpu.d(1)
            );
        }
        other => panic!("unknown mode {other}"),
    }
}

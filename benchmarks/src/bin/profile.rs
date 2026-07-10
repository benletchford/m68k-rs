//! Scratch profiling target: runs the memcpy workload on the m68k-rs
//! interpreter only, small budget, for use under callgrind.

use m68k::{AddressBus, CpuCore, CpuType};

struct FlatBus {
    memory: Vec<u8>,
}

const MEM_MASK: usize = 0xFF_FFFF;

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

fn main() {
    let words: &[u16] = &[
        0x41F9, 0x0000, 0x8000, // LEA $8000.L,A0
        0x43F9, 0x0001, 0x0000, // LEA $10000.L,A1
        0x303C, 0x03FF, // MOVE.W #$03FF,D0
        0x22D8, // MOVE.L (A0)+,(A1)+
        0x51C8, 0xFFFC, // DBRA D0,-4
        0x5241, // ADDQ.W #1,D1
        0x60E6, // BRA.S -26
    ];
    let mut bus = FlatBus {
        memory: vec![0; 0x100_0000],
    };
    for (i, w) in words.iter().enumerate() {
        let a = 0x1000 + i * 2;
        bus.memory[a..a + 2].copy_from_slice(&w.to_be_bytes());
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.set_sr(0x2700);
    cpu.pc = 0x1000;
    cpu.set_a(7, 0x0080_0000);
    let used = cpu.execute(&mut bus, 15_000_000);
    println!("used={used} d1={}", cpu.d(1));
}

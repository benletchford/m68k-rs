//! Host-reported instruction-fetch stalls in the 68040 timing model.
//!
//! The 68040 model prices instructions as if the whole fetch stream hit the
//! instruction cache. A host with a cache model reports the extra clocks its
//! misses cost through `AddressBus::instruction_fetch_stall_cycles`, and the
//! core adds them to that instruction's cycle count.

use std::cell::Cell;
use std::collections::HashSet;

use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, NoOpHleHandler, StepResult};

/// One instruction-cache line on the 68040.
const LINE: u32 = 16;
const MISS_PENALTY: u32 = 30;

/// A flat 64 KiB bus with an optional, never-evicting instruction cache.
/// Every instruction-stream word or long that touches an uncached line pays
/// `MISS_PENALTY` per line, so extension words crossing a line are charged.
struct CacheBus {
    memory: Vec<u8>,
    model_cache: bool,
    /// Report this many stall clocks for every instruction regardless of the
    /// cache, to check models that must ignore the hook.
    fixed_stall: Option<u32>,
    lines: HashSet<u32>,
    stall: u32,
    stall_reads: Cell<u32>,
}

impl CacheBus {
    fn new(model_cache: bool) -> Self {
        Self {
            memory: vec![0; 0x10000],
            model_cache,
            fixed_stall: None,
            lines: HashSet::new(),
            stall: 0,
            stall_reads: Cell::new(0),
        }
    }

    fn write_word_at(&mut self, addr: u32, value: u16) {
        self.memory[addr as usize..addr as usize + 2].copy_from_slice(&value.to_be_bytes());
    }

    fn write_long_at(&mut self, addr: u32, value: u32) {
        self.memory[addr as usize..addr as usize + 4].copy_from_slice(&value.to_be_bytes());
    }

    fn fetch(&mut self, address: u32, bytes: u32) {
        if !self.model_cache {
            return;
        }
        let first = address & !(LINE - 1);
        let last = (address + bytes - 1) & !(LINE - 1);
        let mut line = first;
        loop {
            if self.lines.insert(line) {
                self.stall += MISS_PENALTY;
            }
            if line == last {
                break;
            }
            line += LINE;
        }
    }
}

impl AddressBus for CacheBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[(address & 0xFFFF) as usize]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let a = (address & 0xFFFF) as usize;
        u16::from_be_bytes([self.memory[a], self.memory[a + 1]])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        (u32::from(self.read_word(address)) << 16) | u32::from(self.read_word(address + 2))
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[(address & 0xFFFF) as usize] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        self.write_word_at(address & 0xFFFF, value);
    }

    fn write_long(&mut self, address: u32, value: u32) {
        self.write_long_at(address & 0xFFFF, value);
    }

    fn read_immediate_word(&mut self, address: u32) -> u16 {
        self.fetch(address, 2);
        self.read_word(address)
    }

    fn read_immediate_long(&mut self, address: u32) -> u32 {
        self.fetch(address, 4);
        self.read_long(address)
    }

    fn begin_instruction_fetches(&mut self) {
        self.stall = 0;
    }

    fn instruction_fetch_stall_cycles(&self) -> u32 {
        self.stall_reads.set(self.stall_reads.get() + 1);
        self.fixed_stall.unwrap_or(self.stall)
    }
}

fn setup(cpu_type: CpuType, bus: &mut CacheBus) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0200);
    cpu.reset(bus);
    cpu.pc = 0x0200;
    cpu.set_sr(0x2700);
    // Reset vector reads are not instruction fetches; start with a cold cache.
    bus.lines.clear();
    cpu
}

fn step_cycles(cpu: &mut CpuCore, bus: &mut CacheBus) -> i32 {
    let mut hle = NoOpHleHandler;
    match cpu.step_with_hle_handler(bus, &mut hle) {
        StepResult::Ok { cycles } => cycles,
        other => panic!("unexpected step result: {other:?}"),
    }
}

/// Lay out a small program: MOVEQ at $200, MOVEQ at $202 (same line), and a
/// MOVE.L #imm,D2 at $20C whose immediate long crosses into the $210 line.
fn load_program(bus: &mut CacheBus) {
    bus.write_word_at(0x0200, 0x7001); // MOVEQ #1,D0
    bus.write_word_at(0x0202, 0x7202); // MOVEQ #2,D1
    for addr in (0x0204..0x020C).step_by(2) {
        bus.write_word_at(addr, 0x4E71); // NOP
    }
    bus.write_word_at(0x020C, 0x243C); // MOVE.L #$12345678,D2
    bus.write_long_at(0x020E, 0x1234_5678);
}

#[test]
fn default_bus_keeps_the_cached_68040_costs() {
    let mut cached = CacheBus::new(false);
    let mut cpu = setup(CpuType::M68040, &mut cached);
    load_program(&mut cached);
    assert_eq!(step_cycles(&mut cpu, &mut cached), 1, "MOVEQ is one clock");
}

#[test]
fn a_68040_miss_adds_the_reported_stall_once_per_line() {
    let mut cached = CacheBus::new(false);
    let mut cpu_cached = setup(CpuType::M68040, &mut cached);
    load_program(&mut cached);
    let mut cold = CacheBus::new(true);
    let mut cpu_cold = setup(CpuType::M68040, &mut cold);
    load_program(&mut cold);

    let penalty = MISS_PENALTY as i32;
    // The first MOVEQ fills the $200 line; the second hits it.
    let base = step_cycles(&mut cpu_cached, &mut cached);
    assert_eq!(step_cycles(&mut cpu_cold, &mut cold), base + penalty);
    let base = step_cycles(&mut cpu_cached, &mut cached);
    assert_eq!(
        step_cycles(&mut cpu_cold, &mut cold),
        base,
        "same line hits"
    );

    // NOPs at $204-$20A stay in the warm line. NOP's pipeline sync makes its
    // cost depend on history, so compare each step against the cached run.
    for _ in 0..4 {
        let base = step_cycles(&mut cpu_cached, &mut cached);
        assert_eq!(step_cycles(&mut cpu_cold, &mut cold), base);
    }

    // The immediate long crosses into the $210 line: only that line misses.
    let base = step_cycles(&mut cpu_cached, &mut cached);
    assert_eq!(step_cycles(&mut cpu_cold, &mut cold), base + penalty);
    assert_eq!(cpu_cold.dar[2], 0x1234_5678);
    assert_eq!(cpu_cold.pc, 0x0212);
}

#[test]
fn a_cold_instruction_spanning_two_lines_pays_for_both() {
    let mut cached = CacheBus::new(false);
    let mut cpu_cached = setup(CpuType::M68040, &mut cached);
    load_program(&mut cached);
    let mut cold = CacheBus::new(true);
    let mut cpu_cold = setup(CpuType::M68040, &mut cold);
    load_program(&mut cold);
    cpu_cached.pc = 0x020C;
    cpu_cold.pc = 0x020C;

    let base = step_cycles(&mut cpu_cached, &mut cached);
    assert_eq!(
        step_cycles(&mut cpu_cold, &mut cold),
        base + 2 * MISS_PENALTY as i32
    );
}

#[test]
fn execute_charges_the_stall_against_the_cycle_budget() {
    let mut cached = CacheBus::new(false);
    let mut cpu_cached = setup(CpuType::M68040, &mut cached);
    load_program(&mut cached);
    let mut cold = CacheBus::new(true);
    let mut cpu_cold = setup(CpuType::M68040, &mut cold);
    load_program(&mut cold);

    // Two MOVEQs (one line fill) on each bus, then compare the clocks used.
    let used_cached = cpu_cached.execute(&mut cached, 2);
    let used_cold = cpu_cold.execute(&mut cold, 2 + MISS_PENALTY as i32);
    assert_eq!(cpu_cached.pc, 0x0204);
    assert_eq!(cpu_cold.pc, 0x0204);
    assert_eq!(used_cold, used_cached + MISS_PENALTY as i32);
}

#[test]
fn other_models_ignore_the_stall_hook() {
    for cpu_type in [
        CpuType::M68000,
        CpuType::M68020,
        CpuType::M68030,
        CpuType::M68060,
    ] {
        let mut plain = CacheBus::new(false);
        let mut cpu_plain = setup(cpu_type, &mut plain);
        load_program(&mut plain);
        let mut stalling = CacheBus::new(false);
        stalling.fixed_stall = Some(1000);
        let mut cpu_stalling = setup(cpu_type, &mut stalling);
        load_program(&mut stalling);
        for _ in 0..7 {
            assert_eq!(
                step_cycles(&mut cpu_stalling, &mut stalling),
                step_cycles(&mut cpu_plain, &mut plain),
                "{cpu_type:?}"
            );
        }
    }
}

#[test]
fn a_huge_stall_saturates_instead_of_wrapping() {
    let mut bus = CacheBus::new(false);
    bus.fixed_stall = Some(u32::MAX);
    let mut cpu = setup(CpuType::M68040, &mut bus);
    load_program(&mut bus);
    assert_eq!(step_cycles(&mut cpu, &mut bus), i32::MAX);
}

#[test]
fn run_batch_does_not_read_the_stall() {
    let mut bus = CacheBus::new(true);
    let mut cpu = setup(CpuType::M68040, &mut bus);
    load_program(&mut bus);
    let result = cpu.run_batch(&mut bus, 7, &[]);
    assert_eq!(result.instructions, 7);
    assert_eq!(cpu.dar[2], 0x1234_5678);
    assert_eq!(bus.stall_reads.get(), 0);
}

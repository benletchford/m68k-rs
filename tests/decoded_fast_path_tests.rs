use m68k::core::memory::{AddressBus, InstructionCacheBus, LinearMemoryBus};
use m68k::{CpuCore, CpuType, StepResult};

struct TestBus {
    memory: [u8; 0x10000],
    instruction_version: u64,
    instruction_invalidations: usize,
    last_instruction_invalidation: Option<(u32, u32)>,
}

impl TestBus {
    fn new() -> Self {
        Self {
            memory: [0; 0x10000],
            instruction_version: 1,
            instruction_invalidations: 0,
            last_instruction_invalidation: None,
        }
    }

    fn reset_instruction_invalidations(&mut self) {
        self.instruction_invalidations = 0;
        self.last_instruction_invalidation = None;
    }

    fn write_word_at(&mut self, addr: u32, value: u16) {
        let bytes = value.to_be_bytes();
        let idx = addr as usize;
        self.memory[idx] = bytes[0];
        self.memory[idx + 1] = bytes[1];
    }

    fn write_long_at(&mut self, addr: u32, value: u32) {
        let bytes = value.to_be_bytes();
        let idx = addr as usize;
        self.memory[idx] = bytes[0];
        self.memory[idx + 1] = bytes[1];
        self.memory[idx + 2] = bytes[2];
        self.memory[idx + 3] = bytes[3];
    }
}

impl AddressBus for TestBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[(address as usize) & 0xFFFF]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let addr = (address as usize) & 0xFFFF;
        u16::from_be_bytes([self.memory[addr], self.memory[addr + 1]])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let addr = (address as usize) & 0xFFFF;
        u32::from_be_bytes([
            self.memory[addr],
            self.memory[addr + 1],
            self.memory[addr + 2],
            self.memory[addr + 3],
        ])
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[(address as usize) & 0xFFFF] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        let addr = (address as usize) & 0xFFFF;
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[addr + 1] = bytes[1];
    }

    fn write_long(&mut self, address: u32, value: u32) {
        let addr = (address as usize) & 0xFFFF;
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[addr + 1] = bytes[1];
        self.memory[addr + 2] = bytes[2];
        self.memory[addr + 3] = bytes[3];
    }
}

impl InstructionCacheBus for TestBus {
    fn instruction_cache_version(&mut self, _address: u32) -> Option<u64> {
        Some(self.instruction_version)
    }

    fn invalidate_instruction_cache(&mut self, address: u32, len: u32) {
        self.instruction_invalidations += 1;
        self.last_instruction_invalidation = Some((address, len));
        self.instruction_version = self.instruction_version.wrapping_add(1);
    }
}

fn boot_cpu(bus: &mut TestBus) -> CpuCore {
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0100);

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.reset(bus);
    cpu.set_sr(0x2700);
    cpu
}

#[test]
fn execute_decoded_register_run_matches_interpreter_semantics() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x2400); // MOVE.L D0,D2
    bus.write_word_at(0x0102, 0xD481); // ADD.L D1,D2
    bus.write_word_at(0x0104, 0x5282); // ADDQ.L #1,D2
    bus.write_word_at(0x0106, 0xB182); // EOR.L D0,D2
    bus.write_word_at(0x0108, 0x4A82); // TST.L D2
    bus.write_word_at(0x010A, 0x4E71); // NOP

    cpu.set_d(0, 3);
    cpu.set_d(1, 2);

    let cycles = cpu.execute(&mut bus, 36);

    assert_eq!(cycles, 36);
    assert_eq!(cpu.pc, 0x010C);
    assert_eq!(cpu.d(2), 5);
    assert!(!cpu.flag_z());
    assert!(!cpu.flag_n());
}

#[test]
fn move_long_postincrement_to_postincrement_copies_and_updates_flags() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x22D8); // MOVE.L (A0)+,(A1)+
    bus.write_long_at(0x2000, 0x8000_0001);
    cpu.set_a(0, 0x2000);
    cpu.set_a(1, 0x3000);

    let cycles = match cpu.step(&mut bus) {
        StepResult::Ok { cycles } => cycles,
        other => panic!("unexpected step result: {other:?}"),
    };

    assert_eq!(cycles, 20);
    assert_eq!(cpu.pc, 0x0102);
    assert_eq!(cpu.a(0), 0x2004);
    assert_eq!(cpu.a(1), 0x3004);
    assert_eq!(bus.read_long(0x3000), 0x8000_0001);
    assert!(cpu.flag_n());
    assert!(!cpu.flag_z());
    assert!(!cpu.flag_v());
    assert!(!cpu.flag_c());
}

#[test]
fn execute_decoded_short_branch_loop_runs_at_instruction_boundary() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x5280); // ADDQ.L #1,D0
    bus.write_word_at(0x0102, 0x60FC); // BRA.S -4

    let cycles = cpu.execute(&mut bus, 180);

    assert_eq!(cycles, 180);
    assert_eq!(cpu.d(0), 10);
    assert_eq!(cpu.pc, 0x0100);
}

#[test]
fn execute_decoded_short_branch_loop_observes_modified_opcode() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x5280); // ADDQ.L #1,D0
    bus.write_word_at(0x0102, 0x60FC); // BRA.S -4

    let cycles = cpu.execute(&mut bus, 18);
    assert_eq!(cycles, 18);
    assert_eq!(cpu.d(0), 1);
    assert_eq!(cpu.pc, 0x0100);

    bus.write_word_at(0x0100, 0x5380); // SUBQ.L #1,D0

    let cycles = cpu.execute(&mut bus, 18);
    assert_eq!(cycles, 18);
    assert_eq!(cpu.d(0), 0);
    assert_eq!(cpu.pc, 0x0100);
}

#[test]
fn execute_trace_jit_loop_observes_modified_opcode_after_warmup() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x5280); // ADDQ.L #1,D0
    bus.write_word_at(0x0102, 0x60FC); // BRA.S -4

    let cycles = cpu.execute(&mut bus, 180);
    assert_eq!(cycles, 180);
    assert_eq!(cpu.d(0), 10);
    assert_eq!(cpu.pc, 0x0100);

    bus.write_word_at(0x0100, 0x5380); // SUBQ.L #1,D0

    let cycles = cpu.execute(&mut bus, 18);
    assert_eq!(cycles, 18);
    assert_eq!(cpu.d(0), 9);
    assert_eq!(cpu.pc, 0x0100);
}

#[test]
fn execute_trace_jit_dbra_loop_runs_to_counter_expiration() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x5280); // ADDQ.L #1,D0
    bus.write_word_at(0x0102, 0x51C9); // DBRA D1,-4
    bus.write_word_at(0x0104, 0xFFFC);

    cpu.set_d(1, 4);

    let cycles = cpu.execute(&mut bus, 94);

    assert_eq!(cycles, 94);
    assert_eq!(cpu.d(0), 5);
    assert_eq!(cpu.d(1) & 0xFFFF, 0xFFFF);
    assert_eq!(cpu.pc, 0x0106);
}

#[test]
fn execute_trace_jit_conditional_branch_uses_jitted_word_flags() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x5340); // SUBQ.W #1,D0
    bus.write_word_at(0x0102, 0x66FC); // BNE.S -4

    cpu.set_d(0, 0x1234_0003);

    let cycles = cpu.execute(&mut bus, 40);

    assert_eq!(cycles, 40);
    assert_eq!(cpu.d(0), 0x1234_0000);
    assert!(cpu.flag_z());
    assert_eq!(cpu.pc, 0x0104);
}

#[test]
fn instruction_cache_bus_versions_can_be_invalidated() {
    let mut bus = TestBus::new();
    bus.reset_instruction_invalidations();

    assert_eq!(bus.instruction_cache_version(0x0100), Some(1));

    bus.invalidate_instruction_cache(0x0100, 2);
    assert_eq!(bus.instruction_invalidations, 1);
    assert_eq!(bus.last_instruction_invalidation, Some((0x0100, 2)));
    assert_eq!(bus.instruction_cache_version(0x0100), Some(2));

    bus.invalidate_instruction_cache(0x0103, 1);
    assert_eq!(bus.instruction_invalidations, 2);
    assert_eq!(bus.last_instruction_invalidation, Some((0x0103, 1)));

    bus.invalidate_instruction_cache(0x0104, 4);
    assert_eq!(bus.instruction_invalidations, 3);
    assert_eq!(bus.last_instruction_invalidation, Some((0x0104, 4)));
}

#[test]
fn linear_memory_bus_wraps_and_versions_instruction_memory() {
    let mut bus = LinearMemoryBus::new(8);

    assert_eq!(bus.instruction_cache_version(0), Some(1));

    bus.write_long(6, 0x1122_3344);
    assert_eq!(bus.read_byte(6), 0x11);
    assert_eq!(bus.read_byte(7), 0x22);
    assert_eq!(bus.read_byte(0), 0x33);
    assert_eq!(bus.read_byte(1), 0x44);
    assert_eq!(bus.read_long(6), 0x1122_3344);

    let version = bus.instruction_cache_version(0).unwrap();
    assert!(version > 1);

    bus.load(7, &[0xAA, 0xBB, 0xCC]);
    assert_eq!(bus.read_byte(7), 0xAA);
    assert_eq!(bus.read_byte(0), 0xBB);
    assert_eq!(bus.read_byte(1), 0xCC);
    assert!(bus.instruction_cache_version(0).unwrap() > version);
}

#[test]
fn execute_decoded_dynamic_bit_register_ops_match_interpreter_semantics() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x01C1); // BSET D0,D1
    bus.write_word_at(0x0102, 0x0101); // BTST D0,D1
    bus.write_word_at(0x0104, 0x0141); // BCHG D0,D1
    bus.write_word_at(0x0106, 0x0181); // BCLR D0,D1

    cpu.set_d(0, 3);
    cpu.set_d(1, 0);

    let cycles = cpu.execute(&mut bus, 26);

    assert_eq!(cycles, 26);
    assert_eq!(cpu.pc, 0x0108);
    assert_eq!(cpu.d(1), 0);
    assert!(cpu.flag_z());
}

#[test]
fn execute_decoded_bcd_register_ops_match_interpreter_semantics() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0xC101); // ABCD D1,D0
    bus.write_word_at(0x0102, 0x8102); // SBCD D2,D0

    cpu.set_d(0, 0xAA00_0012);
    cpu.set_d(1, 0x34);
    cpu.set_d(2, 0x11);

    let cycles = cpu.execute(&mut bus, 12);

    assert_eq!(cycles, 12);
    assert_eq!(cpu.pc, 0x0104);
    assert_eq!(cpu.d(0), 0xAA00_0035);
}

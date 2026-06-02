use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType};

struct TestBus {
    memory: [u8; 0x10000],
}

impl TestBus {
    fn new() -> Self {
        Self {
            memory: [0; 0x10000],
        }
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

    let cycles = cpu.execute(&mut bus, 28);

    assert_eq!(cycles, 28);
    assert_eq!(cpu.pc, 0x010C);
    assert_eq!(cpu.d(2), 5);
    assert!(!cpu.flag_z());
    assert!(!cpu.flag_n());
}

#[test]
fn execute_decoded_short_branch_loop_runs_at_instruction_boundary() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x5280); // ADDQ.L #1,D0
    bus.write_word_at(0x0102, 0x60FC); // BRA.S -4

    let cycles = cpu.execute(&mut bus, 140);

    assert_eq!(cycles, 140);
    assert_eq!(cpu.d(0), 10);
    assert_eq!(cpu.pc, 0x0100);
}

#[test]
fn execute_decoded_short_branch_loop_observes_modified_opcode() {
    let mut bus = TestBus::new();
    let mut cpu = boot_cpu(&mut bus);

    bus.write_word_at(0x0100, 0x5280); // ADDQ.L #1,D0
    bus.write_word_at(0x0102, 0x60FC); // BRA.S -4

    let cycles = cpu.execute(&mut bus, 14);
    assert_eq!(cycles, 14);
    assert_eq!(cpu.d(0), 1);
    assert_eq!(cpu.pc, 0x0100);

    bus.write_word_at(0x0100, 0x5380); // SUBQ.L #1,D0

    let cycles = cpu.execute(&mut bus, 14);
    assert_eq!(cycles, 14);
    assert_eq!(cpu.d(0), 0);
    assert_eq!(cpu.pc, 0x0100);
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

    let cycles = cpu.execute(&mut bus, 32);

    assert_eq!(cycles, 32);
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

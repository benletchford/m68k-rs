use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, StepResult};

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

#[test]
fn test_odd_pc_fetch_triggers_address_error_on_68010() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010);
    let mut bus = TestBus::new();

    // Vectors: SSP=0x1000, PC=0x0100
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0100);
    // Address error vector (vector 3) -> 0x0200
    bus.write_long_at(0x0C, 0x0200);

    // NOP at odd address (will be fetched if no address error)
    bus.write_word_at(0x0101, 0x4E71);

    cpu.reset(&mut bus);
    cpu.pc = 0x0101; // Force odd instruction fetch
    cpu.set_sr(0x2700);

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(
        cpu.pc, 0x0200,
        "68010 should take address error on odd instruction fetch"
    );
}

#[test]
fn test_odd_pc_fetch_triggers_address_error_on_68020() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68020);
    let mut bus = TestBus::new();

    // Vectors: SSP=0x1000, PC=0x0100
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0100);
    // Address error vector (vector 3) -> 0x0200
    bus.write_long_at(0x0C, 0x0200);

    // NOP at odd address (will be fetched if no address error)
    bus.write_word_at(0x0101, 0x4E71);

    cpu.reset(&mut bus);
    cpu.pc = 0x0101; // Force odd instruction fetch
    cpu.set_sr(0x2700);

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(
        cpu.pc, 0x0200,
        "68020 should take address error on odd instruction fetch"
    );
}

// Ported from upstream m68k-rs (post-fork regression test): a register-only
// instruction followed by an odd-PC opcode fetch must not restore a stale
// data/address rollback snapshot when the fetch faults. Confirms the vendored
// cycle-exact core never leaks pre-instruction register state through a
// fetch-fault path.
#[test]
fn test_odd_pc_fetch_after_register_only_instruction_does_not_restore_stale_snapshot() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010);
    let mut bus = TestBus::new();

    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0100);
    bus.write_long_at(0x0C, 0x0200);
    bus.write_word_at(0x0100, 0x7001); // MOVEQ #1,D0

    cpu.reset(&mut bus);
    cpu.pc = 0x0100;
    cpu.set_sr(0x2700);
    cpu.set_d(0, 0xDEAD_BEEF);

    let result = cpu.step(&mut bus);
    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.d(0), 1);

    cpu.pc = 0x0103;
    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.pc, 0x0200);
    assert_eq!(
        cpu.d(0),
        1,
        "opcode-fetch faults must not restore stale rollback state"
    );
}

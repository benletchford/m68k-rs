use m68k::{CpuCore, CpuType, NoOpHleHandler, StepResult};
use m68k::core::memory::AddressBus;

struct TestBus {
    memory: [u8; 0x10000],
}

impl TestBus {
    fn new() -> Self {
        Self { memory: [0; 0x10000] }
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
fn test_pmove_is_privileged_on_68030() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68030);
    let mut bus = TestBus::new();

    // Vectors: SSP=0x1000, PC=0x0200
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0200);
    // Privilege violation vector (vector 8) -> 0x0300
    bus.write_long_at(0x20, 0x0300);

    // PMOVE TC, (A0) - opcode 0xF010, extension 0x4200 (from fixture)
    bus.write_word_at(0x0200, 0xF010);
    bus.write_word_at(0x0202, 0x4200);

    cpu.reset(&mut bus);
    cpu.pc = 0x0200;
    cpu.set_sr(0x0000); // user mode

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(
        cpu.pc, 0x0300,
        "PMOVE in user mode should trap to privilege violation vector"
    );
}

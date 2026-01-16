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
fn test_move16_misaligned_triggers_address_error_68040() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68040);
    let mut bus = TestBus::new();

    // Vectors: SSP=0x1000, PC=0x0100
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0100);
    // Address error vector (vector 3) -> 0x0200
    bus.write_long_at(0x0C, 0x0200);

    // MOVE16 (Ax)+,(Ay)+ with misaligned addresses should trap.
    bus.write_word_at(0x0100, 0xF620);
    bus.write_word_at(0x0102, 0x9000); // dest A1

    cpu.reset(&mut bus);
    cpu.pc = 0x0100;
    cpu.set_sr(0x2700);
    cpu.set_a(0, 0x0101); // misaligned source
    cpu.set_a(1, 0x0201); // misaligned dest

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(
        cpu.pc, 0x0200,
        "MOVE16 with misaligned addresses should take address error on 68040"
    );
}

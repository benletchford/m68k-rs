use m68k::{CpuCore, CpuType, NoOpHleHandler};
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
fn test_cmp2_is_illegal_on_68000() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TestBus::new();

    // Vectors: SSP=0x1000, PC=0x0200
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0200);
    // Illegal instruction vector (vector 4) -> 0x0300
    bus.write_long_at(0x10, 0x0300);

    // CMP2.W (opcode 0x04D0: size=word, mode=010 reg=000), ext word 0x0000
    bus.write_word_at(0x0200, 0x04D0);
    bus.write_word_at(0x0202, 0x0000);

    cpu.reset(&mut bus);
    cpu.pc = 0x0200;
    cpu.set_sr(0x2700);

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);

    assert!(matches!(result, m68k::StepResult::Ok { .. }));
    assert_eq!(
        cpu.pc, 0x0300,
        "CMP2 on 68000 should trap to illegal instruction vector"
    );
}

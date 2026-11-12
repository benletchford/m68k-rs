use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, NoOpHleHandler};

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
fn test_rtm_is_illegal_on_68000() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TestBus::new();

    // Vectors: SSP=0x1000, PC=0x0200
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0200);
    // Illegal instruction vector (vector 4) -> 0x0300
    bus.write_long_at(0x10, 0x0300);

    // RTM D0 (0x06C0) - 68020 only
    bus.write_word_at(0x0200, 0x06C0);

    cpu.reset(&mut bus);
    cpu.pc = 0x0200;
    cpu.set_sr(0x2700);

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);

    assert!(matches!(result, m68k::StepResult::Ok { .. }));
    assert_eq!(
        cpu.pc, 0x0300,
        "RTM on 68000 should trap to illegal instruction vector"
    );
}

/// Run `words` at 0x0200 with the illegal-instruction vector at 0x0300 and
/// the line-F vector at 0x0340.
fn step_module_opcode(cpu_type: CpuType, words: &[u16]) -> (CpuCore, TestBus) {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    let mut bus = TestBus::new();
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0200);
    bus.write_long_at(4 * 4, 0x0300);
    bus.write_long_at(11 * 4, 0x0340);
    for (i, w) in words.iter().enumerate() {
        bus.write_word_at(0x0200 + 2 * i as u32, *w);
    }
    // A module descriptor at 0x0800 for the models that implement CALLM.
    bus.write_long_at(0x0804, 0x0400);

    cpu.reset(&mut bus);
    cpu.pc = 0x0200;
    cpu.set_sr(0x2700);
    cpu.set_a(0, 0x0800);

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(result, m68k::StepResult::Ok { .. }));
    (cpu, bus)
}

/// The module instructions were dropped after the 68020: on the 68030 and
/// later the encodings are unimplemented and take the illegal-instruction
/// exception with the opcode's address stacked, not the line-F exception.
#[test]
fn rtm_and_callm_are_illegal_after_the_68020() {
    for cpu_type in [
        CpuType::M68010,
        CpuType::M68EC030,
        CpuType::M68030,
        CpuType::M68EC040,
        CpuType::M68LC040,
        CpuType::M68040,
        CpuType::M68060,
    ] {
        // RTM D0, RTM A0, CALLM #0,(A0)
        for words in [&[0x06c0u16][..], &[0x06c8], &[0x06d0, 0x0000]] {
            let (cpu, mut bus) = step_module_opcode(cpu_type, words);
            assert_eq!(
                cpu.pc, 0x0300,
                "{cpu_type:?} {:04x}: illegal instruction vector",
                words[0]
            );
            assert_eq!(
                bus.read_long(cpu.sp() + 2),
                0x0200,
                "{cpu_type:?} {:04x}: stacked PC is the opcode",
                words[0]
            );
            assert_eq!(
                bus.read_word(cpu.sp() + 6),
                4 << 2,
                "{cpu_type:?} {:04x}: format $0, vector 4",
                words[0]
            );
        }
    }
}

#[test]
fn rtm_and_callm_execute_on_the_68020_and_68ec020() {
    for cpu_type in [CpuType::M68020, CpuType::M68EC020] {
        let (cpu, _) = step_module_opcode(cpu_type, &[0x06d0, 0x0000]);
        assert_eq!(cpu.pc, 0x0400, "{cpu_type:?}: CALLM enters the module");
        let (cpu, _) = step_module_opcode(cpu_type, &[0x06c8]);
        assert!(
            cpu.pc != 0x0300 && cpu.pc != 0x0340,
            "{cpu_type:?}: RTM is not an exception"
        );
    }
}

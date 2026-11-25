//! Where the MC68010 multiplier and divider place their one prefetch.
//!
//! vAmigaTS `CPU/Prefetch` (Dirk Hoffmann) uses the Blitter as a timer that
//! rewrites the word the instruction under test prefetches, and counts the
//! stripes in which the old word was read. Photographed on an A500 with an
//! MC68010 (`*_A500_ECS_68010.jpeg`): DIVU 15, DIVS 16, MULU 14, MULS 14
//! stripes. The README concludes that the 68010 multiplier prefetches last
//! and the divider first (the 68000 is the other way round). Execution times
//! are M68000UM Table 9-6: MULU 40, MULS 42 (maxima), DIVU 108, DIVS 122.
use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus, StepResult};

#[derive(Debug, PartialEq)]
enum Event {
    Read(u32),
    Internal(u32),
}

struct Bus {
    memory: LinearMemoryBus,
    events: Vec<Event>,
}

impl AddressBus for Bus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory.read_byte(address)
    }
    fn read_word(&mut self, address: u32) -> u16 {
        self.events.push(Event::Read(address));
        self.memory.read_word(address)
    }
    fn read_long(&mut self, address: u32) -> u32 {
        self.memory.read_long(address)
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory.write_byte(address, value);
    }
    fn write_word(&mut self, address: u32, value: u16) {
        self.memory.write_word(address, value);
    }
    fn write_long(&mut self, address: u32, value: u32) {
        self.memory.write_long(address, value);
    }
    fn sync(&mut self, clocks: u32) {
        self.events.push(Event::Internal(clocks));
    }
}

fn setup(opcode: u16, next: u16) -> (CpuCore, Bus) {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010);
    cpu.set_sr(0x2700);
    cpu.pc = 0x1000;
    // A steady-state queue, avoiding cold-start reads in the measured opcode.
    cpu.prefetch_queue = [opcode, next];
    cpu.prefetch_count = 2;
    assert!(cpu.prefetch_enabled());
    let mut memory = LinearMemoryBus::new(0x4000);
    memory.write_word(0x1004, 0x4e71);
    memory.write_word(0x1006, 0x4e71);
    (
        cpu,
        Bus {
            memory,
            events: Vec::new(),
        },
    )
}

fn step(cpu: &mut CpuCore, bus: &mut Bus) -> u32 {
    match cpu.step(bus) {
        StepResult::Ok { cycles } => cycles as u32,
        other => panic!("unexpected divide result: {other:?}"),
    }
}

#[test]
fn register_multiplies_compute_before_the_final_prefetch() {
    // MULU/MULS D1,D0 with the vAmigaTS operands.
    for (opcode, src, total) in [(0xc0c1u16, 0xffffu32, 40u32), (0xc1c1, 0x5555, 42)] {
        let (mut cpu, mut bus) = setup(opcode, 0x4e71);
        cpu.set_d(0, 0x1234);
        cpu.set_d(1, src);
        let cycles = step(&mut cpu, &mut bus);
        assert_eq!(cycles, total, "{opcode:04x}");
        assert_eq!(
            bus.events,
            [Event::Internal(cycles - 4), Event::Read(0x1004)],
            "{opcode:04x}"
        );
    }
}

#[test]
fn register_divides_prefetch_before_computing() {
    // DIVU/DIVS D1,D0: 100 / 7, as in vAmigaTS.
    for (opcode, total) in [(0x80c1u16, 108u32), (0x81c1, 122)] {
        let (mut cpu, mut bus) = setup(opcode, 0x4e71);
        cpu.set_d(0, 100);
        cpu.set_d(1, 7);
        let cycles = step(&mut cpu, &mut bus);
        assert_eq!(cycles, total, "{opcode:04x}");
        assert_eq!(
            bus.events,
            [Event::Read(0x1004), Event::Internal(cycles - 4)],
            "{opcode:04x}"
        );
        assert_eq!(cpu.d(0), (100 % 7) << 16 | (100 / 7), "{opcode:04x}");
    }
}

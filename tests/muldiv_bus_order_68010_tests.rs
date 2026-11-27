//! MC68010 MULU, MULS, DIVU and DIVS: how long they take and where they
//! place their one prefetch.
//!
//! The figures come from the Motorola manual and from photographs of an A500
//! with an MC68010 that ship with vAmigaTS (Dirk Hoffmann):
//!
//! - M68000UM Table 9-6 lists MULU 40, MULS 42, DIVU 108 and DIVS 122 clocks
//!   with a register operand, the multiplies as maxima.
//! - `CPU/Prefetch` uses the Blitter as a timer that rewrites the word the
//!   instruction prefetches, and counts the stripes in which the old word was
//!   read: DIVS 16, DIVU 15, MULU 14, MULS 14. The multiplier prefetches last
//!   and the divider early (the 68000 is the other way round): DIVS at once,
//!   DIVU after 16 clocks.
//! - `CPU/Timing/MUL` and `CPU/Timing/DIV` mark the end of the instruction.
//!   MULU takes 40 clocks whatever the multiplier, MULS 40 and 42 for a
//!   negative multiplier, DIVU 108 and DIVS 118 for a small non-negative
//!   dividend and divisor.
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

const MULU_D1_D0: u16 = 0xc0c1;
const MULS_D1_D0: u16 = 0xc1c1;
const DIVU_D1_D0: u16 = 0x80c1;
const DIVS_D1_D0: u16 = 0x81c1;
const MULU_A0_D0: u16 = 0xc0d0;
const MULS_A0_D0: u16 = 0xc1d0;
const DIVU_A0_D0: u16 = 0x80d0;
const DIVS_A0_D0: u16 = 0x81d0;

/// Run `opcode` with `dst` in D0 and `src` in D1 and in the word A0 points
/// at; return the clocks charged and the bus events.
fn run(opcode: u16, dst: u32, src: u16) -> (u32, Vec<Event>, u32) {
    let (mut cpu, mut bus) = setup(opcode, 0x4e71);
    cpu.set_d(0, dst);
    cpu.set_d(1, u32::from(src));
    cpu.set_a(0, 0x2000);
    bus.memory.write_word(0x2000, src);
    bus.events.clear();
    let cycles = step(&mut cpu, &mut bus);
    (cycles, bus.events, cpu.d(0))
}

#[test]
fn mulu_takes_40_clocks_whatever_the_multiplier() {
    // No bit pattern, however many ones it holds, changes the time: the
    // 68000 would take from 38 to 70 clocks over these.
    for src in [
        0x0000u16, 0x0001, 0x0003, 0x0100, 0x5555, 0x7fff, 0x8000, 0xaaaa, 0xffff,
    ] {
        for dst in [0u32, 1, 0x1234, 0xffff] {
            let (cycles, events, result) = run(MULU_D1_D0, dst, src);
            assert_eq!(cycles, 40, "{src:04x} * {dst:04x}");
            assert_eq!(events, [Event::Internal(36), Event::Read(0x1004)]);
            assert_eq!(result, u32::from(src) * dst);
        }
    }
}

#[test]
fn muls_takes_two_more_clocks_for_a_negative_multiplier() {
    for src in [
        0x0000u16, 0x0001, 0x0003, 0x0100, 0x5555, 0x7fff, 0x8000, 0xaaaa, 0xffff,
    ] {
        // The sign of the multiplicand does not matter.
        for dst in [0u32, 1, 0x1234, 0x8000, 0xffff] {
            let total = if src & 0x8000 != 0 { 42 } else { 40 };
            let (cycles, events, result) = run(MULS_D1_D0, dst, src);
            assert_eq!(cycles, total, "{src:04x} * {dst:04x}");
            assert_eq!(events, [Event::Internal(total - 4), Event::Read(0x1004)]);
            let product = i32::from(src as i16) * i32::from(dst as u16 as i16);
            assert_eq!(result, product as u32);
        }
    }
}

#[test]
fn divu_takes_108_clocks_and_prefetches_after_16() {
    // Quotients with no, few and many bits set, and an overflow: the 68000
    // would take anything from 10 to 140 clocks over these.
    for (dst, src) in [
        (0u32, 4u16),
        (99, 9),
        (0x0001_0000, 0xffff),
        (0xfffe_0001, 0xffff),
        (0x7fff_ffff, 0x8000),
        (0x0003_0000, 2),
    ] {
        let (cycles, events, result) = run(DIVU_D1_D0, dst, src);
        assert_eq!(cycles, 108, "{dst:08x} / {src:04x}");
        assert_eq!(
            events,
            [
                Event::Internal(16),
                Event::Read(0x1004),
                Event::Internal(88)
            ],
            "{dst:08x} / {src:04x}"
        );
        let quotient = dst / u32::from(src);
        if quotient <= 0xffff {
            assert_eq!(result, (dst % u32::from(src)) << 16 | quotient);
        } else {
            assert_eq!(result, dst, "an overflow leaves the dividend alone");
        }
    }
}

#[test]
fn divs_prefetches_at_once_and_takes_118_clocks_without_a_negative_operand() {
    for (dst, src, total) in [
        (0u32, 4u16, 118u32),
        (99, 9, 118),
        (0x3fff_0000, 0x7fff, 118),
        (0xffff_ff9d, 9, 122),      // -99 / 9
        (99, 0xfff7, 122),          // 99 / -9
        (0xffff_ff9d, 0xfff7, 122), // -99 / -9
    ] {
        let (cycles, events, result) = run(DIVS_D1_D0, dst, src);
        assert_eq!(cycles, total, "{dst:08x} / {src:04x}");
        assert_eq!(
            events,
            [Event::Read(0x1004), Event::Internal(total - 4)],
            "{dst:08x} / {src:04x}"
        );
        let (n, d) = (dst as i32, i32::from(src as i16));
        let expected = ((n % d) as u16 as u32) << 16 | u32::from((n / d) as u16);
        assert_eq!(result, expected, "{dst:08x} / {src:04x}");
    }
}

#[test]
fn a_memory_operand_is_read_first_and_adds_its_fetch() {
    // (A0) costs one more bus cycle, ahead of everything else.
    let operand = Event::Read(0x2000);
    let prefetch = Event::Read(0x1004);

    let (cycles, events, _) = run(MULU_A0_D0, 0x1234, 0xffff);
    assert_eq!(cycles, 44);
    assert_eq!(events, [operand, Event::Internal(36), prefetch]);

    let operand = Event::Read(0x2000);
    let prefetch = Event::Read(0x1004);
    let (cycles, events, _) = run(MULS_A0_D0, 0x1234, 0xffff);
    assert_eq!(cycles, 46);
    assert_eq!(events, [operand, Event::Internal(38), prefetch]);

    let operand = Event::Read(0x2000);
    let prefetch = Event::Read(0x1004);
    let (cycles, events, _) = run(DIVU_A0_D0, 99, 9);
    assert_eq!(cycles, 112);
    assert_eq!(
        events,
        [operand, Event::Internal(16), prefetch, Event::Internal(88)]
    );

    let operand = Event::Read(0x2000);
    let prefetch = Event::Read(0x1004);
    let (cycles, events, _) = run(DIVS_A0_D0, 99, 9);
    assert_eq!(cycles, 122);
    assert_eq!(events, [operand, prefetch, Event::Internal(114)]);
}

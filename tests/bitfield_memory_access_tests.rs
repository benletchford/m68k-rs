//! Memory-form bit-field instructions access exactly the bytes their field
//! spans, at the transfer size that span needs (MC68020UM 5.3.1), and never
//! touch a neighbouring byte. A field within four bytes is one operand cycle
//! and a five-byte span two (8.2.14); a real A1200 measures spans of one,
//! two, three and four bytes at identical cost with only the five-byte span
//! adding an access (Copperline timing-test/bfprobe.asm).
//!
//! Widening the operand to a long would read and write up to three bytes
//! outside the field, which is observable on memory-mapped registers and
//! moves the fault boundary past the end of a mapped region. These tests pin
//! the touched byte set, not just the access count, so that cannot come back.

use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus};

/// Wraps LinearMemoryBus and records every data access as (address, size).
/// Instruction fetches bypass the log so only operand traffic is recorded.
struct CountingBus {
    inner: LinearMemoryBus,
    reads: Vec<(u32, u32)>,
    writes: Vec<(u32, u32)>,
}

impl CountingBus {
    fn new() -> Self {
        Self {
            inner: LinearMemoryBus::new(0x10000),
            reads: Vec::new(),
            writes: Vec::new(),
        }
    }

    /// Every byte address the log says was read or written, in order and
    /// without duplicates.
    fn bytes_touched(&self) -> Vec<u32> {
        let mut out: Vec<u32> = self
            .reads
            .iter()
            .chain(self.writes.iter())
            .flat_map(|&(addr, size)| (0..size).map(move |i| addr + i))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

impl AddressBus for CountingBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.reads.push((address, 1));
        self.inner.read_byte(address)
    }
    fn read_word(&mut self, address: u32) -> u16 {
        self.reads.push((address, 2));
        self.inner.read_word(address)
    }
    fn read_long(&mut self, address: u32) -> u32 {
        self.reads.push((address, 4));
        self.inner.read_long(address)
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        self.writes.push((address, 1));
        self.inner.write_byte(address, value)
    }
    fn write_word(&mut self, address: u32, value: u16) {
        self.writes.push((address, 2));
        self.inner.write_word(address, value)
    }
    fn write_long(&mut self, address: u32, value: u32) {
        self.writes.push((address, 4));
        self.inner.write_long(address, value)
    }
    // Opcode and extension words are not operand traffic.
    fn read_immediate_word(&mut self, address: u32) -> u16 {
        self.inner.read_word(address)
    }
    fn read_immediate_long(&mut self, address: u32) -> u32 {
        self.inner.read_long(address)
    }
}

const CODE: u32 = 0x1000;
const DATA: u32 = 0x4000;
/// Written either side of every field so a stray access shows up as a value
/// change as well as a log entry.
const SENTINEL: u8 = 0xA5;

/// Assemble one bit-field opcode + extension at CODE and run it on a 68020
/// with A0 = DATA, D1 = the dynamic offset, and the data bytes preloaded.
/// DATA-8 .. DATA+16 is filled with sentinels first.
fn run_bitfield(opcode: u16, ext: u16, d1: u32, data: &[(u32, u8)]) -> (CpuCore, CountingBus) {
    let mut bus = CountingBus::new();
    for addr in (DATA - 8)..(DATA + 16) {
        bus.inner.load(addr, &[SENTINEL]);
    }
    bus.inner.load(CODE, &opcode.to_be_bytes());
    bus.inner.load(CODE + 2, &ext.to_be_bytes());
    bus.inner.load(CODE + 4, &0x4E71u16.to_be_bytes()); // NOP
    for &(addr, value) in data {
        bus.inner.load(addr, &[value]);
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68020);
    cpu.pc = CODE;
    cpu.set_sr(0x2700);
    cpu.set_a(7, 0x8000);
    cpu.set_a(0, DATA);
    cpu.set_d(1, d1);
    bus.reads.clear();
    bus.writes.clear();
    cpu.step(&mut bus);
    (cpu, bus)
}

/// The byte addresses a field starting at `start` and spanning `len` bytes
/// may legitimately touch, and nothing else.
fn span(start: u32, len: u32) -> Vec<u32> {
    (start..start + len).collect()
}

#[test]
fn one_byte_span_is_a_single_byte_transfer() {
    // BFSET (A0){D1:1} with D1 = 3: bit 3 (msb-first) of DATA.
    let (_, mut bus) = run_bitfield(0xEED0, 0x0841, 3, &[(DATA, 0x00)]);
    assert_eq!(bus.reads, vec![(DATA, 1)]);
    assert_eq!(bus.writes, vec![(DATA, 1)]);
    assert_eq!(bus.bytes_touched(), span(DATA, 1));
    assert_eq!(bus.inner.read_byte(DATA), 0x10);
    // A long operand would have read and rewritten these three.
    for i in 1..4 {
        assert_eq!(bus.inner.read_byte(DATA + i), SENTINEL, "byte +{i}");
    }
}

#[test]
fn two_byte_span_is_a_single_word_transfer() {
    // BFSET (A0){4:8}: bits 4..11 span DATA and DATA+1.
    let (_, mut bus) = run_bitfield(0xEED0, 0x0108, 0, &[(DATA, 0x00), (DATA + 1, 0x00)]);
    assert_eq!(bus.reads, vec![(DATA, 2)]);
    assert_eq!(bus.writes, vec![(DATA, 2)]);
    assert_eq!(bus.bytes_touched(), span(DATA, 2));
    assert_eq!(bus.inner.read_byte(DATA), 0x0F);
    assert_eq!(bus.inner.read_byte(DATA + 1), 0xF0);
    assert_eq!(bus.inner.read_byte(DATA + 2), SENTINEL);
    assert_eq!(bus.inner.read_byte(DATA + 3), SENTINEL);
}

#[test]
fn three_byte_span_never_reaches_the_fourth_byte() {
    // BFEXTU (A0){4:16},D3: bits 4..19 span DATA..DATA+2. AddressBus has no
    // three-byte transfer, so this composes a word and a byte - the right
    // bytes and only those. Rounding up to a long would touch DATA+3.
    let (cpu, mut bus) = run_bitfield(
        0xE9D0,
        0x3110,
        0,
        &[(DATA, 0x0A), (DATA + 1, 0xBC), (DATA + 2, 0xD0)],
    );
    assert_eq!(bus.reads, vec![(DATA, 2), (DATA + 2, 1)]);
    assert!(bus.writes.is_empty());
    assert_eq!(bus.bytes_touched(), span(DATA, 3));
    assert_eq!(cpu.d(3), 0xABCD);
    assert_eq!(bus.inner.read_byte(DATA + 3), SENTINEL);
}

#[test]
fn three_byte_modify_span_writes_back_only_its_three_bytes() {
    // BFSET (A0){4:16}: the modify form of the same span.
    let (_, mut bus) = run_bitfield(0xEED0, 0x0110, 0, &[]);
    assert_eq!(bus.reads, vec![(DATA, 2), (DATA + 2, 1)]);
    assert_eq!(bus.writes, vec![(DATA, 2), (DATA + 2, 1)]);
    assert_eq!(bus.bytes_touched(), span(DATA, 3));
    assert_eq!(bus.inner.read_byte(DATA + 3), SENTINEL);
}

#[test]
fn four_byte_span_is_a_single_long_transfer() {
    // BFSET (A0){0:32}: the field is exactly the four bytes.
    let (_, mut bus) = run_bitfield(0xEED0, 0x0000, 0, &[]);
    assert_eq!(bus.reads, vec![(DATA, 4)]);
    assert_eq!(bus.writes, vec![(DATA, 4)]);
    assert_eq!(bus.bytes_touched(), span(DATA, 4));
    assert_eq!(bus.inner.read_byte(DATA + 4), SENTINEL);
}

#[test]
fn five_byte_span_adds_one_byte_access_and_no_more() {
    // BFSET (A0){7:32}: bits 7..38 need the long plus the fifth byte, which
    // is the second operand cycle the MC68020UM bills the wide span.
    let (_, mut bus) = run_bitfield(0xEED0, 0x01C0, 0, &[]);
    assert_eq!(bus.reads, vec![(DATA, 4), (DATA + 4, 1)]);
    assert_eq!(bus.writes, vec![(DATA, 4), (DATA + 4, 1)]);
    assert_eq!(bus.bytes_touched(), span(DATA, 5));
    assert_eq!(bus.inner.read_byte(DATA + 5), SENTINEL);
}

#[test]
fn negative_dynamic_offset_reaches_back_exactly_one_byte() {
    // BFSET (A0){D1:1} with D1 = -1: the last bit of the byte before A0.
    let (_, mut bus) = run_bitfield(0xEED0, 0x0841, -1i32 as u32, &[(DATA - 1, 0x00)]);
    assert_eq!(bus.reads, vec![(DATA - 1, 1)]);
    assert_eq!(bus.writes, vec![(DATA - 1, 1)]);
    assert_eq!(bus.bytes_touched(), span(DATA - 1, 1));
    assert_eq!(bus.inner.read_byte(DATA - 1), 0x01);
    assert_eq!(bus.inner.read_byte(DATA - 2), SENTINEL);
    assert_eq!(bus.inner.read_byte(DATA), SENTINEL);
}

#[test]
fn read_only_forms_read_their_span_and_write_nothing() {
    // BFTST (A0){6:4}: bits 6..9 span two bytes; field = 0b1011.
    let (cpu, bus) = run_bitfield(0xE8D0, 0x0184, 0, &[(DATA, 0x02), (DATA + 1, 0xC0)]);
    assert_eq!(bus.reads, vec![(DATA, 2)]);
    assert!(bus.writes.is_empty());
    assert_eq!(bus.bytes_touched(), span(DATA, 2));
    // N set (msb of field), Z clear.
    let sr = cpu.get_sr();
    assert_ne!(sr & 0x0008, 0, "N should be set");
    assert_eq!(sr & 0x0004, 0, "Z should be clear");
}

#[test]
fn no_memory_form_touches_a_byte_outside_its_field() {
    // Every form, at every span, against the field the extension word names.
    // (opcode, extension, span in bytes)
    let cases: [(u16, u16, u32); 12] = [
        (0xE8D0, 0x0041, 1), // BFTST   (A0){1:1}
        (0xE9D0, 0x3110, 3), // BFEXTU  (A0){4:16},D3
        (0xEAD0, 0x0108, 2), // BFCHG   (A0){4:8}
        (0xEBD0, 0x3110, 3), // BFEXTS  (A0){4:16},D3
        (0xECD0, 0x0000, 4), // BFCLR   (A0){0:32}
        (0xEDD0, 0x3184, 2), // BFFFO   (A0){6:4},D3
        (0xEED0, 0x01C0, 5), // BFSET   (A0){7:32}
        (0xEFD0, 0x3041, 1), // BFINS   D3,(A0){1:1}
        (0xE8D0, 0x0000, 4), // BFTST   (A0){0:32}
        (0xEAD0, 0x01C0, 5), // BFCHG   (A0){7:32}
        (0xECD0, 0x0110, 3), // BFCLR   (A0){4:16}
        (0xEFD0, 0x3108, 2), // BFINS   D3,(A0){4:8}
    ];
    for (opcode, ext, len) in cases {
        let (_, bus) = run_bitfield(opcode, ext, 0, &[]);
        assert_eq!(
            bus.bytes_touched(),
            span(DATA, len),
            "opcode {opcode:04X} ext {ext:04X}"
        );
    }
}

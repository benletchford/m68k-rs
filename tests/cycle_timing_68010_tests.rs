//! MC68010 instruction execution times, checked against the tables of
//! M68000UM Section 9 ("MC68010 Instruction Execution Times").
//!
//! Every case runs one instruction from a steady prefetch queue and checks
//! two things: the time the core reports is the manual's, and the clocks it
//! spends on the bus (four per bus cycle, plus the internal clocks it hands
//! to `AddressBus::sync`) add up to it. A host that advances time from the
//! bus tops an instruction up to the reported time, but only once its last
//! bus cycle is over: a clock the core does not bill itself lands in the
//! wrong place, and is lost when that cycle was held off.
//!
//! Some entries follow measurements of the hardware where the manual prints
//! something else; they are marked "Measured" below and explained in
//! `src/core/timing_010.rs`.
//!
//! The multiply, divide and branch times have their own suites.
use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus, NoOpHleHandler, StepResult};

struct Bus {
    memory: LinearMemoryBus,
    clocks: u32,
}

impl AddressBus for Bus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.clocks += 4;
        self.memory.read_byte(address)
    }
    fn read_word(&mut self, address: u32) -> u16 {
        self.clocks += 4;
        self.memory.read_word(address)
    }
    fn read_long(&mut self, address: u32) -> u32 {
        self.clocks += 8;
        self.memory.read_long(address)
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        self.clocks += 4;
        self.memory.write_byte(address, value);
    }
    fn write_word(&mut self, address: u32, value: u16) {
        self.clocks += 4;
        self.memory.write_word(address, value);
    }
    fn write_long(&mut self, address: u32, value: u32) {
        self.clocks += 8;
        self.memory.write_long(address, value);
    }
    fn sync(&mut self, clocks: u32) {
        self.clocks += clocks;
    }
}

const CODE: u32 = 0x1000;
const DATA: u32 = 0x4000;
const STACK: u32 = 0x8000;

/// An addressing mode: its mode and register fields and extension words.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Ea {
    Dn,
    An,
    Ind,
    PostInc,
    PreDec,
    Disp,
    Index,
    AbsW,
    AbsL,
    PcDisp,
    PcIndex,
    Imm,
}

use Ea::*;

const DATA_ALTERABLE_MEMORY: [Ea; 7] = [Ind, PostInc, PreDec, Disp, Index, AbsW, AbsL];
const ALL_SOURCES: [Ea; 12] = [
    Dn, An, Ind, PostInc, PreDec, Disp, Index, AbsW, AbsL, PcDisp, PcIndex, Imm,
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Size {
    Byte,
    Word,
    Long,
}

impl Ea {
    /// Mode and register fields, using register `reg` where one is needed.
    fn field(self, reg: u16) -> u16 {
        match self {
            Dn => reg,
            An => 0o10 | reg,
            Ind => 0o20 | reg,
            PostInc => 0o30 | reg,
            PreDec => 0o40 | reg,
            Disp => 0o50 | reg,
            Index => 0o60 | reg,
            AbsW => 0o70,
            AbsL => 0o71,
            PcDisp => 0o72,
            PcIndex => 0o73,
            Imm => 0o74,
        }
    }

    fn extension(self, size: Size) -> Vec<u16> {
        match self {
            Dn | An | Ind | PostInc | PreDec => vec![],
            Disp => vec![0x0010],
            // (d8,An,D7.W) with D7 = 0
            Index => vec![0x7010],
            AbsW => vec![DATA as u16],
            AbsL => vec![0, DATA as u16],
            PcDisp => vec![0x0100],
            PcIndex => vec![0x707e],
            Imm => match size {
                Size::Long => vec![0, 2],
                _ => vec![2],
            },
        }
    }

    /// Table 9-1, "Fetch" columns: calculate the address and read the
    /// operand.
    fn fetch(self, size: Size) -> u32 {
        let word = match self {
            Dn | An => 0,
            Ind | PostInc | Imm => 4,
            PreDec => 6,
            Disp | AbsW | PcDisp => 8,
            Index | PcIndex => 10,
            AbsL => 12,
        };
        if size == Size::Long && !matches!(self, Dn | An) {
            word + 4
        } else {
            word
        }
    }

    /// Table 9-1, "No Fetch" columns: calculate the address only.
    fn no_fetch(self) -> u32 {
        match self {
            Ind => 2,
            PostInc | PreDec | Disp | AbsW => 4,
            Index | AbsL => 8,
            _ => panic!("{self:?} has no non-fetching time"),
        }
    }
}

struct Case {
    name: String,
    words: Vec<u16>,
    expected: u32,
    sr: u16,
    d0: u32,
    d1: u32,
    sp: u32,
}

fn case(name: impl Into<String>, words: Vec<u16>, expected: u32) -> Case {
    Case {
        name: name.into(),
        words,
        expected,
        sr: 0x2700,
        d0: 0x0000_0102,
        d1: 0x0000_0304,
        sp: STACK,
    }
}

fn with_ea(opcode: u16, immediate: &[u16], ea: Ea, size: Size) -> Vec<u16> {
    let mut words = vec![opcode];
    words.extend_from_slice(immediate);
    words.extend(ea.extension(size));
    words
}

/// Run one instruction; return the reported clocks and the bus clocks.
fn run(case: &Case) -> (u32, u32) {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010);
    let mut memory = LinearMemoryBus::new(0x10000);
    // Vectors lead to a handler full of NOPs.
    for vector in 0..256u32 {
        memory.write_long(vector * 4, 0x0000_3000);
    }
    for address in (0x3000..0x3100u32).step_by(2) {
        memory.write_word(address, 0x4e71);
    }
    // Operand area: small positive words, so that nothing overflows.
    for address in (0x3f00..0x4200u32).step_by(2) {
        memory.write_word(address, 0x0002);
    }
    // The PC-relative operands sit just past the code.
    for address in (0x1080..0x1200u32).step_by(2) {
        memory.write_word(address, 0x0002);
    }
    for (index, word) in case.words.iter().enumerate() {
        memory.write_word(CODE + 2 * index as u32, *word);
    }
    for index in case.words.len()..16 {
        memory.write_word(CODE + 2 * index as u32, 0x4e71);
    }
    // A return frame for RTE, RTR, RTS and RTD: format 0, SR, PC.
    memory.write_word(STACK, 0x2700);
    memory.write_long(STACK + 2, 0x0000_3000);
    memory.write_word(STACK + 6, 0x0000);
    cpu.set_sr(0x2700);
    cpu.pc = CODE;
    cpu.prefetch_queue = [case.words[0], memory.read_word(CODE + 2)];
    cpu.prefetch_count = 2;
    assert!(cpu.prefetch_enabled());
    for reg in 0..8 {
        cpu.set_d(reg, 0);
        cpu.set_a(reg, DATA + 0x20 * reg as u32);
    }
    cpu.set_d(0, case.d0);
    cpu.set_d(1, case.d1);
    cpu.set_a(7, case.sp);
    cpu.set_sr(case.sr);
    if case.sr & 0x2000 == 0 {
        cpu.set_a(7, STACK - 0x100);
    }
    let mut bus = Bus { memory, clocks: 0 };
    match cpu.step_with_hle_handler(&mut bus, &mut NoOpHleHandler) {
        StepResult::Ok { cycles } => (cycles as u32, bus.clocks),
        other => panic!("{}: {other:?}", case.name),
    }
}

fn check(cases: Vec<Case>) {
    let mut failures = Vec::new();
    for case in &cases {
        let (reported, bus) = run(case);
        // The core bills every clock itself, so that a host has none to add
        // after the last bus cycle. The exception is BCLR on memory: it is
        // two clocks longer than BSET and nothing measured says where they
        // fall, so they are left to the host.
        let host = if case.name.starts_with("bclr ") && !case.name.ends_with(",Dn") {
            2
        } else {
            0
        };
        if reported != case.expected || bus + host != case.expected {
            failures.push(format!(
                "{} ({:04x}): manual {} reported {} bus {}",
                case.name, case.words[0], case.expected, reported, bus
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n")
    );
}

const SIZES: [(Size, u16, &str); 3] = [
    (Size::Byte, 0, "b"),
    (Size::Word, 1, "w"),
    (Size::Long, 2, "l"),
];

fn long(size: Size) -> bool {
    size == Size::Long
}

// ---------------------------------------------------------------------------
// Tables 9-2 and 9-4: MOVE
// ---------------------------------------------------------------------------

/// The destination's share of a MOVE: Table 9-2 / 9-4 less the Dn column.
fn move_destination(ea: Ea, size: Size) -> u32 {
    let word = match ea {
        Dn | An => 0,
        Ind | PostInc | PreDec => 4,
        Disp | AbsW => 8,
        Index => 10,
        AbsL => 12,
        _ => unreachable!(),
    };
    if long(size) && !matches!(ea, Dn | An) {
        word + 4
    } else {
        word
    }
}

#[test]
fn move_times_follow_tables_9_2_and_9_4() {
    let mut cases = Vec::new();
    for (size, bits, tag) in [
        (Size::Byte, 0x1000u16, "b"),
        (Size::Word, 0x3000, "w"),
        (Size::Long, 0x2000, "l"),
    ] {
        for src in ALL_SOURCES {
            if src == An && size == Size::Byte {
                continue;
            }
            for dst in [Dn, An, Ind, PostInc, PreDec, Disp, Index, AbsW, AbsL] {
                if dst == An && size == Size::Byte {
                    continue;
                }
                let dst_field = dst.field(2);
                let opcode = bits | (dst_field & 7) << 9 | (dst_field >> 3) << 6 | src.field(1);
                let mut words = vec![opcode];
                words.extend(src.extension(size));
                words.extend(dst.extension(size));
                // Table 9-4 prints 14 for a long move from a register to
                // -(An); the hardware takes the 12 of the move to (An).
                let expected = 4 + src.fetch(size) + move_destination(dst, size);
                cases.push(case(format!("move.{tag} {src:?},{dst:?}"), words, expected));
            }
        }
    }
    check(cases);
}

// ---------------------------------------------------------------------------
// Table 9-6: standard instructions
// ---------------------------------------------------------------------------

#[test]
fn standard_instruction_times_follow_table_9_6() {
    let mut cases = Vec::new();
    // op <ea>,Dn: 4 byte and word, 6 long, plus the operand fetch.
    for (name, group, data_only) in [
        ("add", 0xd000u16, false),
        ("sub", 0x9000, false),
        ("and", 0xc000, true),
        ("or", 0x8000, true),
        ("cmp", 0xb000, false),
    ] {
        for (size, bits, tag) in SIZES {
            for src in ALL_SOURCES {
                if src == An && (data_only || size == Size::Byte) {
                    continue;
                }
                let opcode = group | 2 << 9 | bits << 6 | src.field(1);
                // Measured: a long compare with a memory operand takes 4
                // clocks plus the fetch, not the printed 6.
                let compare_from_memory =
                    name == "cmp" && long(size) && !matches!(src, Dn | An | Imm);
                let base = if long(size) && !compare_from_memory {
                    6
                } else {
                    4
                };
                let expected = base + src.fetch(size);
                cases.push(case(
                    format!("{name}.{tag} {src:?},Dn"),
                    with_ea(opcode, &[], src, size),
                    expected,
                ));
            }
        }
    }
    // op <ea>,An: ADDA and SUBA 8 word and 6 long, CMPA 6 both.
    for (name, group, word_base) in [
        ("adda", 0xd000u16, 8),
        ("suba", 0x9000, 8),
        ("cmpa", 0xb000, 6),
    ] {
        for (size, bits, tag) in [(Size::Word, 3u16, "w"), (Size::Long, 7, "l")] {
            for src in ALL_SOURCES {
                let opcode = group | 2 << 9 | bits << 6 | src.field(1);
                // Measured: ADDA.L and SUBA.L with a register or immediate
                // operand take 8 clocks plus the fetch, not the printed 6.
                let base = if !long(size) {
                    word_base
                } else if name != "cmpa" && matches!(src, Dn | An | Imm) {
                    8
                } else {
                    6
                };
                let expected = base + src.fetch(size);
                cases.push(case(
                    format!("{name}.{tag} {src:?},An"),
                    with_ea(opcode, &[], src, size),
                    expected,
                ));
            }
        }
    }
    // op Dn,<M>: 8 byte and word, 12 long, plus the operand fetch.
    for (name, group) in [
        ("add", 0xd000u16),
        ("sub", 0x9000),
        ("and", 0xc000),
        ("or", 0x8000),
        ("eor", 0xb000),
    ] {
        for (size, bits, tag) in SIZES {
            for dst in DATA_ALTERABLE_MEMORY {
                let opcode = group | 2 << 9 | (4 + bits) << 6 | dst.field(1);
                let expected = if long(size) { 12 } else { 8 } + dst.fetch(size);
                cases.push(case(
                    format!("{name}.{tag} Dn,{dst:?}"),
                    with_ea(opcode, &[], dst, size),
                    expected,
                ));
            }
        }
    }
    // EOR Dn,Dn: 4 byte and word, 6 long.
    for (size, bits, tag) in SIZES {
        let opcode = 0xb000 | 2 << 9 | (4 + bits) << 6 | Dn.field(1);
        cases.push(case(
            format!("eor.{tag} Dn,Dn"),
            vec![opcode],
            if long(size) { 6 } else { 4 },
        ));
    }
    check(cases);
}

// ---------------------------------------------------------------------------
// Table 9-8: immediate instructions
// ---------------------------------------------------------------------------

#[test]
fn immediate_instruction_times_follow_table_9_8() {
    let mut cases = Vec::new();
    for (name, group) in [
        ("ori", 0x0000u16),
        ("andi", 0x0200),
        ("subi", 0x0400),
        ("addi", 0x0600),
        ("eori", 0x0a00),
        ("cmpi", 0x0c00),
    ] {
        let compare = name == "cmpi";
        for (size, bits, tag) in SIZES {
            let immediate: &[u16] = if long(size) { &[0, 1] } else { &[1] };
            let register = match (compare, long(size)) {
                (_, false) => 8,
                (true, true) => 12,
                (false, true) => 14,
            };
            cases.push(case(
                format!("{name}.{tag} #,Dn"),
                with_ea(group | bits << 6 | Dn.field(1), immediate, Dn, size),
                register,
            ));
            for dst in DATA_ALTERABLE_MEMORY {
                let memory = match (compare, long(size)) {
                    (true, false) => 8,
                    (true, true) => 12,
                    (false, false) => 12,
                    (false, true) => 20,
                };
                cases.push(case(
                    format!("{name}.{tag} #,{dst:?}"),
                    with_ea(group | bits << 6 | dst.field(1), immediate, dst, size),
                    memory + dst.fetch(size),
                ));
            }
        }
    }
    for (name, group) in [("addq", 0x5000u16), ("subq", 0x5100)] {
        for (size, bits, tag) in SIZES {
            let opcode = group | 1 << 9 | bits << 6;
            cases.push(case(
                format!("{name}.{tag} #,Dn"),
                vec![opcode | Dn.field(1)],
                if long(size) { 8 } else { 4 },
            ));
            if size != Size::Byte {
                // Measured: 8 clocks for the word form too, where Table
                // 9-8 prints 4.
                cases.push(case(
                    format!("{name}.{tag} #,An"),
                    vec![opcode | An.field(1)],
                    8,
                ));
            }
            for dst in DATA_ALTERABLE_MEMORY {
                cases.push(case(
                    format!("{name}.{tag} #,{dst:?}"),
                    with_ea(opcode | dst.field(1), &[], dst, size),
                    if long(size) { 12 } else { 8 } + dst.fetch(size),
                ));
            }
        }
    }
    cases.push(case("moveq", vec![0x7401], 4));
    check(cases);
}

// ---------------------------------------------------------------------------
// Tables 9-9 and 9-10: single operand instructions and CLR
// ---------------------------------------------------------------------------

#[test]
fn single_operand_times_follow_tables_9_9_and_9_10() {
    let mut cases = Vec::new();
    for (name, group) in [("negx", 0x4000u16), ("neg", 0x4400), ("not", 0x4600)] {
        for (size, bits, tag) in SIZES {
            cases.push(case(
                format!("{name}.{tag} Dn"),
                vec![group | bits << 6 | Dn.field(1)],
                if long(size) { 6 } else { 4 },
            ));
            for dst in DATA_ALTERABLE_MEMORY {
                cases.push(case(
                    format!("{name}.{tag} {dst:?}"),
                    with_ea(group | bits << 6 | dst.field(1), &[], dst, size),
                    if long(size) { 12 } else { 8 } + dst.fetch(size),
                ));
            }
        }
    }
    for (size, bits, tag) in SIZES {
        cases.push(case(
            format!("tst.{tag} Dn"),
            vec![0x4a00 | bits << 6 | Dn.field(1)],
            4,
        ));
        for dst in DATA_ALTERABLE_MEMORY {
            cases.push(case(
                format!("tst.{tag} {dst:?}"),
                with_ea(0x4a00 | bits << 6 | dst.field(1), &[], dst, size),
                4 + dst.fetch(size),
            ));
        }
        // Table 9-10: CLR does not read its destination.
        cases.push(case(
            format!("clr.{tag} Dn"),
            vec![0x4200 | bits << 6 | Dn.field(1)],
            if long(size) { 6 } else { 4 },
        ));
        for (dst, word, long_time) in [
            (Ind, 8, 12),
            (PostInc, 8, 12),
            (PreDec, 10, 14),
            (Disp, 12, 16),
            (Index, 16, 20),
            (AbsW, 12, 16),
            (AbsL, 16, 20),
        ] {
            cases.push(case(
                format!("clr.{tag} {dst:?}"),
                with_ea(0x4200 | bits << 6 | dst.field(1), &[], dst, size),
                if long(size) { long_time } else { word },
            ));
        }
    }
    cases.push(case("nbcd Dn", vec![0x4800 | Dn.field(1)], 6));
    cases.push(case("tas Dn", vec![0x4ac0 | Dn.field(1)], 4));
    // Scc takes 4 clocks on a register whether or not the condition holds.
    cases.push(case("st Dn", vec![0x50c0 | Dn.field(1)], 4));
    cases.push(case("sf Dn", vec![0x51c0 | Dn.field(1)], 4));
    for dst in DATA_ALTERABLE_MEMORY {
        cases.push(case(
            format!("nbcd {dst:?}"),
            with_ea(0x4800 | dst.field(1), &[], dst, Size::Byte),
            8 + dst.fetch(Size::Byte),
        ));
        // Scc and TAS use the non-fetching address times.
        for (name, opcode) in [("st", 0x50c0u16), ("sf", 0x51c0)] {
            cases.push(case(
                format!("{name} {dst:?}"),
                with_ea(opcode | dst.field(1), &[], dst, Size::Byte),
                8 + dst.no_fetch(),
            ));
        }
        cases.push(case(
            format!("tas {dst:?}"),
            with_ea(0x4ac0 | dst.field(1), &[], dst, Size::Byte),
            14 + dst.no_fetch(),
        ));
    }
    check(cases);
}

// ---------------------------------------------------------------------------
// Tables 9-12 and 9-14: shifts, rotates and bit manipulation
// ---------------------------------------------------------------------------

#[test]
fn shift_and_rotate_times_follow_table_9_12() {
    let mut cases = Vec::new();
    for (kind, name) in [(0u16, "as"), (1, "ls"), (2, "rox"), (3, "ro")] {
        for direction in [0u16, 1] {
            for (size, bits, tag) in SIZES {
                // Immediate counts 1 to 8.
                for count in 1..=8u16 {
                    let opcode =
                        0xe000 | (count & 7) << 9 | direction << 8 | bits << 6 | kind << 3 | 1;
                    cases.push(case(
                        format!("{name}{direction}.{tag} #{count},Dn"),
                        vec![opcode],
                        if long(size) { 8 } else { 6 } + 2 * u32::from(count),
                    ));
                }
                // A count held in D0.
                for count in [0u32, 1, 17, 31] {
                    let opcode = 0xe020 | direction << 8 | bits << 6 | kind << 3 | 1;
                    let mut entry = case(
                        format!("{name}{direction}.{tag} D0={count},Dn"),
                        vec![opcode],
                        if long(size) { 8 } else { 6 } + 2 * count,
                    );
                    entry.d0 = count;
                    cases.push(entry);
                }
            }
            for dst in DATA_ALTERABLE_MEMORY {
                let opcode = 0xe0c0 | kind << 9 | direction << 8 | dst.field(1);
                cases.push(case(
                    format!("{name}{direction}.w {dst:?}"),
                    with_ea(opcode, &[], dst, Size::Word),
                    8 + dst.fetch(Size::Word),
                ));
            }
        }
    }
    check(cases);
}

#[test]
fn bit_manipulation_times_follow_table_9_14() {
    let mut cases = Vec::new();
    for (name, kind, dynamic_reg, dynamic_mem, static_reg, static_mem) in [
        ("btst", 0u16, 6, 4, 10, 8),
        ("bchg", 1, 8, 8, 12, 12),
        ("bclr", 2, 10, 10, 14, 14),
        ("bset", 3, 8, 8, 12, 12),
    ] {
        // The register times are maxima: reached with a bit number above 15.
        let mut entry = case(
            format!("{name} D0,Dn"),
            vec![0x0100 | kind << 6 | Dn.field(1)],
            dynamic_reg,
        );
        entry.d0 = 31;
        cases.push(entry);
        cases.push(case(
            format!("{name} #31,Dn"),
            vec![0x0800 | kind << 6 | Dn.field(1), 31],
            static_reg,
        ));
        for dst in DATA_ALTERABLE_MEMORY {
            cases.push(case(
                format!("{name} D0,{dst:?}"),
                with_ea(0x0100 | kind << 6 | dst.field(1), &[], dst, Size::Byte),
                dynamic_mem + dst.fetch(Size::Byte),
            ));
            cases.push(case(
                format!("{name} #1,{dst:?}"),
                with_ea(0x0800 | kind << 6 | dst.field(1), &[1], dst, Size::Byte),
                static_mem + dst.fetch(Size::Byte),
            ));
        }
    }
    check(cases);
}

// ---------------------------------------------------------------------------
// Table 9-16: JMP, JSR, LEA, PEA and MOVEM
// ---------------------------------------------------------------------------

#[test]
fn control_and_movem_times_follow_table_9_16() {
    let mut cases = Vec::new();
    for (ea, jmp, jsr, lea, pea) in [
        (Ind, 8, 16, 4, 12),
        (Disp, 10, 18, 8, 16),
        (Index, 14, 22, 12, 20),
        (AbsW, 10, 18, 8, 16),
        (AbsL, 12, 20, 12, 20),
        (PcDisp, 10, 18, 8, 16),
        (PcIndex, 14, 22, 12, 20),
    ] {
        cases.push(case(
            format!("jmp {ea:?}"),
            with_ea(0x4ec0 | ea.field(1), &[], ea, Size::Word),
            jmp,
        ));
        cases.push(case(
            format!("jsr {ea:?}"),
            with_ea(0x4e80 | ea.field(1), &[], ea, Size::Word),
            jsr,
        ));
        cases.push(case(
            format!("lea {ea:?}"),
            with_ea(0x41c0 | 2 << 9 | ea.field(1), &[], ea, Size::Word),
            lea,
        ));
        cases.push(case(
            format!("pea {ea:?}"),
            with_ea(0x4840 | ea.field(1), &[], ea, Size::Word),
            pea,
        ));
    }
    // MOVEM with the three registers D2, D3 and D4 (D4, D3, D2 mirrored for
    // the predecrement form).
    for (size, bit, per_register, tag) in
        [(Size::Word, 0u16, 4u32, "w"), (Size::Long, 0x40, 8, "l")]
    {
        for (ea, base) in [
            (Ind, 12),
            (PostInc, 12),
            (Disp, 16),
            (Index, 18),
            (AbsW, 16),
            (AbsL, 20),
            (PcDisp, 16),
            (PcIndex, 18),
        ] {
            cases.push(case(
                format!("movem.{tag} {ea:?},regs"),
                with_ea(0x4c80 | bit | ea.field(1), &[0x001c], ea, size),
                base + 3 * per_register,
            ));
        }
        for (ea, base, mask) in [
            (Ind, 8, 0x001cu16),
            (PreDec, 8, 0x3800),
            (Disp, 12, 0x001c),
            (Index, 14, 0x001c),
            (AbsW, 12, 0x001c),
            (AbsL, 16, 0x001c),
        ] {
            cases.push(case(
                format!("movem.{tag} regs,{ea:?}"),
                with_ea(0x4880 | bit | ea.field(1), &[mask], ea, size),
                base + 3 * per_register,
            ));
        }
    }
    check(cases);
}

// ---------------------------------------------------------------------------
// Tables 9-17 and 9-18: multiprecision and miscellaneous instructions
// ---------------------------------------------------------------------------

#[test]
fn multiprecision_times_follow_table_9_17() {
    let mut cases = Vec::new();
    for (name, group) in [("addx", 0xd100u16), ("subx", 0x9100)] {
        for (size, bits, tag) in SIZES {
            cases.push(case(
                format!("{name}.{tag} Dn,Dn"),
                vec![group | 2 << 9 | bits << 6 | 1],
                if long(size) { 6 } else { 4 },
            ));
            cases.push(case(
                format!("{name}.{tag} -(An),-(An)"),
                vec![group | 2 << 9 | bits << 6 | 8 | 1],
                if long(size) { 30 } else { 18 },
            ));
        }
    }
    for (size, bits, tag) in SIZES {
        cases.push(case(
            format!("cmpm.{tag}"),
            vec![0xb108 | 2 << 9 | bits << 6 | 1],
            if long(size) { 20 } else { 12 },
        ));
    }
    for (name, group) in [("abcd", 0xc100u16), ("sbcd", 0x8100)] {
        cases.push(case(format!("{name} Dn,Dn"), vec![group | 2 << 9 | 1], 6));
        cases.push(case(
            format!("{name} -(An),-(An)"),
            vec![group | 2 << 9 | 8 | 1],
            18,
        ));
    }
    check(cases);
}

#[test]
fn miscellaneous_times_follow_table_9_18() {
    let mut cases = vec![
        case("andi to ccr", vec![0x023c, 0x00ff], 16),
        case("andi to sr", vec![0x027c, 0xffff], 16),
        case("eori to ccr", vec![0x0a3c, 0x0000], 16),
        case("eori to sr", vec![0x0a7c, 0x0000], 16),
        case("ori to ccr", vec![0x003c, 0x0000], 16),
        case("ori to sr", vec![0x007c, 0x0000], 16),
        case("exg", vec![0xc141], 6),
        case("ext.w", vec![0x4881], 4),
        case("ext.l", vec![0x48c1], 4),
        case("link", vec![0x4e51, 0xfff8], 16),
        case("unlk", vec![0x4e59], 12),
        case("move from ccr Dn", vec![0x42c1], 4),
        case("move from sr Dn", vec![0x40c1], 4),
        case("move to ccr Dn", vec![0x44c1], 12),
        case("move to sr Dn", vec![0x46c1], 12),
        case("move usp,An", vec![0x4e69], 6),
        case("move An,usp", vec![0x4e61], 6),
        case("movec vbr,Dn", vec![0x4e7a, 0x1801], 12),
        case("movec Dn,vbr", vec![0x4e7b, 0x1801], 10),
        case("movep.w Dn,(d16,An)", vec![0x0189, 0x0010], 16),
        case("movep.l Dn,(d16,An)", vec![0x01c9, 0x0010], 24),
        case("movep.w (d16,An),Dn", vec![0x0109, 0x0010], 16),
        case("movep.l (d16,An),Dn", vec![0x0149, 0x0010], 24),
        case("nop", vec![0x4e71], 4),
        case("rte", vec![0x4e73], 24),
        case("rtr", vec![0x4e77], 20),
        case("swap", vec![0x4841], 4),
        case("trapv", vec![0x4e76], 4),
    ];
    // RTS and RTD pop a bare PC: start them at the frame's PC.
    for (name, words) in [("rts", vec![0x4e75]), ("rtd", vec![0x4e74, 0x0000])] {
        let mut entry = case(name, words, 16);
        entry.sp = STACK + 2;
        cases.push(entry);
    }
    for dst in DATA_ALTERABLE_MEMORY {
        // MOVE from SR and MOVE from CCR use the non-fetching address times.
        cases.push(case(
            format!("move from sr {dst:?}"),
            with_ea(0x40c0 | dst.field(1), &[], dst, Size::Word),
            8 + dst.no_fetch(),
        ));
        cases.push(case(
            format!("move from ccr {dst:?}"),
            with_ea(0x42c0 | dst.field(1), &[], dst, Size::Word),
            8 + dst.no_fetch(),
        ));
    }
    for src in [
        Ind, PostInc, PreDec, Disp, Index, AbsW, AbsL, PcDisp, PcIndex, Imm,
    ] {
        cases.push(case(
            format!("move {src:?},ccr"),
            with_ea(0x44c0 | src.field(1), &[], src, Size::Word),
            12 + src.fetch(Size::Word),
        ));
        // CHK with the bound above the register: no trap.
        let mut entry = case(
            format!("chk {src:?},Dn"),
            with_ea(0x4180 | src.field(1), &[], src, Size::Word),
            8 + src.fetch(Size::Word),
        );
        entry.d0 = 1;
        cases.push(entry);
    }
    let mut entry = case("chk Dn,Dn", vec![0x4181], 8);
    entry.d0 = 1;
    entry.d1 = 0x100;
    cases.push(entry);
    check(cases);
}

#[test]
fn the_indexed_move_from_ccr_lead_is_the_68010s_alone() {
    // MOVE CCR,(0,A0,D1.W): the 68010 spends the index calculation's two
    // clocks ahead of the write. The later models keep their own times.
    for (cpu_type, expected) in [
        (CpuType::M68010, 16),
        (CpuType::M68020, 9),
        (CpuType::M68040, 3),
        (CpuType::M68060, 3),
        (CpuType::SCC68070, 9),
    ] {
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(cpu_type);
        let mut memory = LinearMemoryBus::new(0x10000);
        memory.write_word(CODE, 0x42f0);
        memory.write_word(CODE + 2, 0x1000);
        memory.write_word(CODE + 4, 0x4e71);
        memory.write_word(CODE + 6, 0x4e71);
        cpu.set_sr(0x2715);
        cpu.pc = CODE;
        if cpu.prefetch_enabled() {
            cpu.prefetch_queue = [0x42f0, 0x1000];
            cpu.prefetch_count = 2;
        }
        cpu.set_a(0, 0x2000);
        cpu.set_d(1, 2);
        let result = cpu.step(&mut memory);
        assert_eq!(result, StepResult::Ok { cycles: expected }, "{cpu_type:?}");
        assert_eq!(memory.read_word(0x2002), 0x0015, "{cpu_type:?}");
    }
}

// ---------------------------------------------------------------------------
// Table 9-19: exception processing
// ---------------------------------------------------------------------------

#[test]
fn exception_times_follow_table_9_19() {
    let mut privilege = case("privilege violation", vec![0x4e72, 0x2000], 38);
    privilege.sr = 0x0000;
    // Measured: 40 clocks, where Table 9-19 prints 38.
    let mut overflow = case("trapv taken", vec![0x4e76], 40);
    overflow.sr = 0x2702;
    check(vec![
        case("illegal", vec![0x4afc], 38),
        case("trap", vec![0x4e41], 38),
        privilege,
        overflow,
    ]);
}

#[test]
fn movec_to_a_missing_control_register_takes_46_and_44_clocks() {
    // Table 9-19 prints 46 clocks for "MOVEC, Illegal Control Register";
    // the move to the register was measured two clocks shorter. The step
    // reports the exception's own time, so the bus clocks are checked: the
    // extension word, the internal clocks and the exception.
    for (opcode, expected) in [(0x4e7au16, 46), (0x4e7b, 44)] {
        let (_, bus) = run(&case("movec", vec![opcode, 0x1002], expected));
        assert_eq!(bus, expected, "{opcode:04x}");
    }
}

// ---------------------------------------------------------------------------
// Tables 9-3, 9-5, 9-7, 9-11, 9-13 and 9-17: loop mode
// ---------------------------------------------------------------------------

/// Run `body ; dbcc d4,body` for `count + 1` passes and return, for each
/// pass, the clocks reported and the bus clocks of the pair.
fn run_loop(body: u16, dbcc: u16, count: u32, zero_at: Option<u32>) -> Vec<(u32, u32)> {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010);
    let mut memory = LinearMemoryBus::new(0x10000);
    for address in (0x3f00..0x4400u32).step_by(2) {
        memory.write_word(address, 0x0001);
    }
    // A word of zero ends a DBEQ loop on a true condition.
    if let Some(index) = zero_at {
        memory.write_long(0x4200 + 2 * index, 0);
    }
    for (index, word) in [body, dbcc, 0xfffc, 0x4e71, 0x4e71, 0x4e71]
        .iter()
        .enumerate()
    {
        memory.write_word(CODE + 2 * index as u32, *word);
    }
    cpu.set_sr(0x2700);
    cpu.pc = CODE;
    cpu.prefetch_queue = [body, dbcc];
    cpu.prefetch_count = 2;
    for reg in 0..8 {
        cpu.set_d(reg, 1);
        cpu.set_a(reg, 0x4000);
    }
    cpu.set_d(4, count);
    cpu.set_a(4, 0x4100);
    cpu.set_a(5, 0x4200);
    cpu.set_a(7, STACK);
    let mut bus = Bus { memory, clocks: 0 };
    let mut passes = Vec::new();
    let mut pair = (0, 0);
    for _ in 0..64 {
        let pc = cpu.pc;
        if pc >= CODE + 6 {
            return passes;
        }
        let before = bus.clocks;
        let cycles = match cpu.step_with_hle_handler(&mut bus, &mut NoOpHleHandler) {
            StepResult::Ok { cycles } => cycles as u32,
            other => panic!("{body:04x}: {other:?}"),
        };
        pair.0 += cycles;
        pair.1 += bus.clocks - before;
        if pc == CODE + 2 {
            passes.push(pair);
            pair = (0, 0);
        }
    }
    panic!("{body:04x}: the loop did not end");
}

#[test]
fn loop_mode_times_follow_the_loop_tables() {
    // body, loop continued, loop ended by an expired count
    let rows: [(&str, u16, u32, u32); 33] = [
        ("move.w d0,(a5)", 0x3a80, 10, 16),
        ("move.w (a4),(a5)", 0x3a94, 14, 18),
        ("move.b (a4)+,(a5)+", 0x1adc, 14, 18),
        ("move.w (a4),-(a5)", 0x3b14, 16, 20),
        ("move.w -(a4),(a5)", 0x3aa4, 16, 20),
        ("move.w -(a4),-(a5)", 0x3b24, 18, 22),
        ("move.l d0,(a5)+", 0x2ac0, 14, 18),
        ("move.l (a4)+,(a5)+", 0x2adc, 22, 24),
        ("move.l -(a4),-(a5)", 0x2b24, 26, 28),
        ("clr.w (a5)", 0x4255, 10, 16),
        ("clr.w -(a5)", 0x4265, 12, 18),
        ("clr.l (a5)+", 0x429d, 14, 20),
        ("tst.w (a5)", 0x4a55, 12, 16),
        ("tst.l (a5)+", 0x4a9d, 18, 20),
        ("neg.w (a5)", 0x4455, 16, 20),
        ("not.l (a5)", 0x4695, 24, 28),
        ("negx.w -(a5)", 0x4065, 18, 22),
        ("nbcd (a5)", 0x4815, 18, 22),
        ("add.w (a5),d0", 0xd055, 16, 20),
        ("add.l (a5)+,d0", 0xd09d, 22, 26),
        ("add.w d0,(a5)", 0xd155, 16, 20),
        ("add.l d0,(a5)", 0xd195, 24, 28),
        ("add.w d0,-(a5)", 0xd165, 18, 22),
        ("adda.w (a5),a3", 0xd6d5, 18, 22),
        ("adda.l (a5),a3", 0xd7d5, 22, 26),
        ("cmp.w (a5),d0", 0xb055, 12, 16),
        ("cmp.l (a5),d0", 0xb095, 18, 20),
        ("eor.w d0,(a5)", 0xb155, 16, 20),
        ("asl.w (a5)", 0xe1d5, 18, 22),
        ("addx.w -(a4),-(a5)", 0xdb4c, 22, 26),
        ("addx.l -(a4),-(a5)", 0xdb8c, 32, 36),
        ("cmpm.w (a4)+,(a5)+", 0xbb4c, 14, 18),
        ("abcd -(a4),-(a5)", 0xcb0c, 24, 28),
    ];
    let mut failures = Vec::new();
    for (name, body, continued, expired) in rows {
        // Five passes: the first runs normally and its DBRA enters loop
        // mode, three continue the loop and the last one finds the count
        // expired.
        let passes = run_loop(body, 0x51cc, 4, None);
        let expected = [continued, continued, continued, expired];
        let got = &passes[1..];
        let reported: Vec<u32> = got.iter().map(|pass| pass.0).collect();
        // Every clock is billed by the core: nothing is left for the host
        // to add.
        let billed: Vec<u32> = got.iter().map(|pass| pass.1).collect();
        if passes.len() != 5 || reported != expected || billed != expected {
            failures.push(format!(
                "{name} ({body:04x}): manual {expected:?} got {passes:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn a_loop_ended_by_its_condition_takes_the_condition_true_time() {
    // DBEQ leaves on the fourth pass, when the body reads a zero.
    for (name, body, continued, condition_true) in [
        ("tst.w (a5)+", 0x4a5du16, 12u32, 18u32),
        ("move.w (a5)+,(a4)+", 0x38dd, 14, 20),
        ("move.l (a5)+,(a4)+", 0x28dd, 22, 28),
    ] {
        let zero_at = if body == 0x28dd { 6 } else { 3 };
        let passes = run_loop(body, 0x57cc, 9, Some(zero_at));
        let reported: Vec<u32> = passes[1..].iter().map(|pass| pass.0).collect();
        let billed: Vec<u32> = passes[1..].iter().map(|pass| pass.1).collect();
        assert_eq!(reported, [continued, continued, condition_true], "{name}");
        assert_eq!(billed, reported, "{name}");
    }
}

//! `run_batch`'s fastmem window with the PMMU enabled.
//!
//! The window is indexed by physical address, so with translation on every
//! access through it is first translated by the transparent-translation
//! registers and the ATC alone. Anything they cannot answer -- an ATC miss,
//! an access the cached translation would fault, an access spanning two
//! pages, a physical address outside the window -- runs through the
//! ordinary bus path, which searches the tables and raises any fault.
//!
//! Each test runs a program twice from the same state: once with `step` on a
//! bus that offers no window (the reference interpreter), once with
//! `run_batch` on a bus that offers one, and requires identical registers,
//! status register and memory. Bus-cycle counts then show the window (and
//! the traces compiled over it) actually served the translated accesses.

use m68k::core::memory::AddressBus;
use m68k::{BatchExit, CpuCore, CpuType, FastMem, StepResult, TrackedMem};

const RAM: usize = 0x4_0000;
/// 68030 page table: 4096 short page descriptors indexed by logical bits
/// 23:12 (`TC_030`).
const TABLE: u32 = 0x1_0000;
/// A second 68030 table, for root-pointer switches and the supervisor root.
const TABLE2: u32 = 0x1_4000;
/// The window: physical 0x20000..0x40000.
const WINDOW: u32 = 0x2_0000;
/// E, PS = 4 KB, IS = 8, TIA = 12.
const TC_030: u32 = 0x80C8_C000;
/// `TC_030` plus SRE: supervisor accesses use the SRP table.
const TC_030_SRE: u32 = TC_030 | 0x0200_0000;

/// Logical pages and the physical pages the tables give them. Everything
/// else below `RAM` is identity-mapped, so each remapped page also has an
/// identity alias at its own physical address.
const CODE: u32 = 0x8000; // -> 0x28000
const HANDLER: u32 = 0x9000; // -> 0x29000
const DATA: u32 = 0x5000; // -> 0x21000
const DATA2: u32 = 0x6000; // -> 0x23000, not the frame after DATA's
const STACK_TOP: u32 = 0x7F00; // page 0x7000 -> 0x27000
const USER_STACK_TOP: u32 = 0x7800;
const REMAPS: [(u32, u32); 5] = [
    (CODE, 0x2_8000),
    (HANDLER, 0x2_9000),
    (DATA, 0x2_1000),
    (DATA2, 0x2_3000),
    (0x7000, 0x2_7000),
];

/// 68040 tables: one root, pointer and page table covering logical
/// 0..0x40000 with 4 KB pages.
const ROOT_040: u32 = 0x1_0000;
const PTR_040: u32 = 0x1_0200;
const PAGE_040: u32 = 0x1_0400;
const TC_040: u32 = 0x0000_8000;

const ALINE: u16 = 0xA000;

struct Bus {
    mem: Vec<u8>,
    /// Offer the window: from `fast_mem`, or from `tracked_mem` if `tracked`.
    window: bool,
    tracked: bool,
    /// With `tracked`, the address of every bus write, and the number of
    /// memory-to-memory copy notifications.
    writes: Vec<u32>,
    copies: u32,
    /// Bus cycles outside the page tables.
    cycles: u64,
    /// Descriptor fetches (reads inside the tables).
    table_reads: u64,
}

impl Bus {
    fn new(window: bool) -> Self {
        let mut bus = Self {
            mem: vec![0; RAM],
            window,
            tracked: false,
            writes: Vec::new(),
            copies: 0,
            cycles: 0,
            table_reads: 0,
        };
        // Vector 2 (access fault) -> HANDLER, which stops the run.
        bus.poke_long(8, HANDLER);
        bus.poke_words(phys(HANDLER), &[ALINE]);
        bus
    }

    fn poke_long(&mut self, at: u32, value: u32) {
        let at = at as usize;
        self.mem[at..at + 4].copy_from_slice(&value.to_be_bytes());
    }

    fn peek_long(&self, at: u32) -> u32 {
        let at = at as usize;
        u32::from_be_bytes(self.mem[at..at + 4].try_into().unwrap())
    }

    fn poke_words(&mut self, at: u32, words: &[u16]) {
        for (i, w) in words.iter().enumerate() {
            let a = at as usize + 2 * i;
            self.mem[a..a + 2].copy_from_slice(&w.to_be_bytes());
        }
    }

    /// A 68030 short page descriptor for `logical`'s page in `table`.
    fn map_030(&mut self, table: u32, logical: u32, physical: u32, flags: u32) {
        self.poke_long(table + (logical >> 12) * 4, physical | flags | 1);
    }

    /// Identity-map all of RAM through `table`, then apply `REMAPS`.
    fn tables_030(&mut self, table: u32) {
        for page in 0..(RAM as u32 >> 12) {
            self.map_030(table, page << 12, page << 12, 0);
        }
        for (logical, physical) in REMAPS {
            self.map_030(table, logical, physical, 0);
        }
    }

    /// A 68040 page descriptor for `logical`'s page.
    fn map_040(&mut self, logical: u32, physical: u32, flags: u32) {
        self.poke_long(PAGE_040 + (logical >> 12) * 4, physical | flags | 1);
    }

    fn tables_040(&mut self) {
        self.poke_long(ROOT_040, PTR_040 | 2);
        self.poke_long(PTR_040, PAGE_040 | 2);
        for page in 0..(RAM as u32 >> 12) {
            self.map_040(page << 12, page << 12, 0);
        }
        for (logical, physical) in REMAPS {
            self.map_040(logical, physical, 0);
        }
    }

    fn note_write(&mut self, addr: u32) {
        self.cycles += 1;
        if self.tracked {
            self.writes.push(addr);
        }
    }

    fn note(&mut self, addr: u32) {
        if (TABLE..WINDOW).contains(&addr) {
            self.table_reads += 1;
        } else {
            self.cycles += 1;
        }
    }
}

/// The physical page the tables give a `REMAPS` logical address.
fn phys(logical: u32) -> u32 {
    let page = logical & !0xFFF;
    REMAPS
        .iter()
        .find(|(l, _)| *l == page)
        .map_or(logical, |(_, p)| p | (logical & 0xFFF))
}

impl AddressBus for Bus {
    fn read_byte(&mut self, addr: u32) -> u8 {
        self.note(addr);
        self.mem[addr as usize]
    }
    fn read_word(&mut self, addr: u32) -> u16 {
        self.note(addr);
        let a = addr as usize;
        u16::from_be_bytes([self.mem[a], self.mem[a + 1]])
    }
    fn read_long(&mut self, addr: u32) -> u32 {
        self.note(addr);
        self.peek_long(addr)
    }
    fn write_byte(&mut self, addr: u32, value: u8) {
        self.note_write(addr);
        self.mem[addr as usize] = value;
    }
    fn write_word(&mut self, addr: u32, value: u16) {
        self.note_write(addr);
        let a = addr as usize;
        self.mem[a..a + 2].copy_from_slice(&value.to_be_bytes());
    }
    fn write_long(&mut self, addr: u32, value: u32) {
        self.note_write(addr);
        self.poke_long(addr, value);
    }
    fn begin_memory_copy(&mut self, _source: u32, _bytes: u32) -> bool {
        self.copies += 1;
        false
    }
    fn fast_mem(&mut self) -> Option<FastMem> {
        (self.window && !self.tracked).then(|| FastMem {
            // SAFETY: WINDOW lies inside `mem`, which is never resized.
            ptr: unsafe { self.mem.as_mut_ptr().add(WINDOW as usize) },
            base: WINDOW,
            len: RAM as u32 - WINDOW,
        })
    }
    fn tracked_mem(&mut self) -> Option<TrackedMem> {
        (self.window && self.tracked).then(|| TrackedMem {
            // SAFETY: as for `fast_mem`.
            ptr: unsafe { self.mem.as_ptr().add(WINDOW as usize) },
            base: WINDOW,
            len: RAM as u32 - WINDOW,
        })
    }
}

#[derive(Clone, Copy)]
enum Mmu {
    Mc68030 { tc: u32 },
    Mc68040,
}

/// A supervisor-mode CPU with translation on, its tables built, the
/// program at `CODE` and `setup` applied.
fn machine(
    window: bool,
    mmu: Mmu,
    program: &[u16],
    setup: &dyn Fn(&mut CpuCore, &mut Bus),
) -> (CpuCore, Bus) {
    let mut bus = Bus::new(window);
    let mut cpu = CpuCore::new();
    match mmu {
        Mmu::Mc68030 { tc } => {
            bus.tables_030(TABLE);
            bus.tables_030(TABLE2);
            cpu.set_cpu_type(CpuType::M68030);
            cpu.mmu_crp_aptr = TABLE;
            cpu.mmu_crp_limit = 2;
            cpu.mmu_srp_aptr = TABLE2;
            cpu.mmu_srp_limit = 2;
            cpu.mmu_tc = tc;
        }
        Mmu::Mc68040 => {
            bus.tables_040();
            cpu.set_cpu_type(CpuType::M68040);
            cpu.mmu_crp_aptr = ROOT_040;
            cpu.mmu_srp_aptr = ROOT_040;
            cpu.mmu_tc = TC_040;
        }
    }
    cpu.pmmu_enabled = true;
    cpu.set_sr(0x2700);
    cpu.set_a(7, STACK_TOP);
    cpu.pc = CODE;
    bus.poke_words(phys(CODE), program);
    setup(&mut cpu, &mut bus);
    bus.cycles = 0;
    bus.table_reads = 0;
    (cpu, bus)
}

/// Run to the A-line sentinel one `step` at a time.
fn run_steps(cpu: &mut CpuCore, bus: &mut Bus) -> u64 {
    let mut steps = 0u64;
    loop {
        match cpu.step(bus) {
            StepResult::Ok { .. } => steps += 1,
            StepResult::AlineTrap { opcode: ALINE } => return steps,
            other => panic!("step: unexpected {other:?} at {:#x}", cpu.pc),
        }
        assert!(steps < 50_000_000, "step run diverged");
    }
}

/// Run to the A-line sentinel in batches.
fn run_batches(cpu: &mut CpuCore, bus: &mut Bus) -> u64 {
    let mut retired = 0u64;
    loop {
        let result = cpu.run_batch(bus, 4096, &[]);
        retired += u64::from(result.instructions);
        match result.exit {
            BatchExit::BudgetExhausted => {}
            BatchExit::AlineTrap { opcode: ALINE } => return retired,
            other => panic!("batch: unexpected {other:?} at {:#x}", cpu.pc),
        }
        assert!(retired < 50_000_000, "batch run diverged");
    }
}

fn assert_same(label: &str, a: &(CpuCore, Bus), b: &(CpuCore, Bus)) {
    let ((ca, ba), (cb, bb)) = (a, b);
    assert_eq!(ca.pc, cb.pc, "{label}: pc");
    assert_eq!(ca.get_sr(), cb.get_sr(), "{label}: sr");
    for i in 0..8 {
        assert_eq!(ca.d(i), cb.d(i), "{label}: D{i}");
        assert_eq!(ca.a(i), cb.a(i), "{label}: A{i}");
    }
    assert_eq!(ca.mmu_tc, cb.mmu_tc, "{label}: TC");
    assert_eq!(ca.pmmu_enabled, cb.pmmu_enabled, "{label}: PMMU enable");
    if ba.mem != bb.mem {
        let at = ba
            .mem
            .iter()
            .zip(&bb.mem)
            .position(|(x, y)| x != y)
            .unwrap();
        panic!("{label}: memory differs first at {at:#x}");
    }
}

/// Run `program` with `step` (no window) and with `run_batch` (window) and
/// require the same outcome. Returns both machines, reference first.
fn compare(
    label: &str,
    mmu: Mmu,
    program: &[u16],
    setup: &dyn Fn(&mut CpuCore, &mut Bus),
) -> ((CpuCore, Bus), (CpuCore, Bus)) {
    let mut reference = machine(false, mmu, program, setup);
    let steps = run_steps(&mut reference.0, &mut reference.1);
    let mut batched = machine(true, mmu, program, setup);
    let retired = run_batches(&mut batched.0, &mut batched.1);
    assert_eq!(steps, retired, "{label}: retired instructions");
    assert_same(label, &reference, &batched);
    (reference, batched)
}

/// `ADD.L D0,(A0) ; MOVE.L (A0),D1 ; ADDQ.L #1,D0 ; DBRA D7,loop`.
const RMW_LOOP: [u16; 5] = [0xD190, 0x2210, 0x5280, 0x51CF, 0xFFF8];

fn with_loop(prefix: &[u16], suffix: &[u16]) -> Vec<u16> {
    let mut program = prefix.to_vec();
    program.extend_from_slice(&RMW_LOOP);
    program.extend_from_slice(suffix);
    program.push(ALINE);
    program
}

fn loop_setup(cpu: &mut CpuCore, _bus: &mut Bus) {
    cpu.set_a(0, DATA);
    cpu.set_d(7, 19_999);
}

/// A remapped read-modify-write loop runs from the window: its code and data
/// pages translate to frames inside it, and after the first table search
/// for each page no access reaches the bus at all.
#[test]
fn translated_loop_runs_from_the_window_030() {
    let program = with_loop(&[], &[]);
    let (reference, batched) = compare(
        "030 loop",
        Mmu::Mc68030 { tc: TC_030 },
        &program,
        &loop_setup,
    );
    let expected: u32 = (0..20_000u32).sum();
    assert_eq!(
        batched.1.peek_long(phys(DATA)),
        expected,
        "the mapped frame"
    );
    assert_eq!(batched.1.peek_long(DATA), 0, "not the identity alias");
    assert!(
        reference.1.cycles > 80_000,
        "reference: {}",
        reference.1.cycles
    );
    assert!(
        batched.1.cycles < 16,
        "window run made {} bus cycles",
        batched.1.cycles
    );
    assert_eq!(
        batched.1.table_reads, reference.1.table_reads,
        "same table searches"
    );
}

#[test]
fn translated_loop_runs_from_the_window_040() {
    let program = with_loop(&[], &[]);
    let (reference, batched) = compare("040 loop", Mmu::Mc68040, &program, &loop_setup);
    assert_eq!(batched.1.peek_long(phys(DATA)), (0..20_000u32).sum::<u32>());
    assert!(reference.1.cycles > 80_000);
    assert!(
        batched.1.cycles < 16,
        "window run made {} bus cycles",
        batched.1.cycles
    );
}

/// An access through an invalid descriptor misses the ATC, takes the bus
/// path, and faults exactly as the interpreter does: same frame, same
/// handler entry, same registers.
#[test]
fn invalid_page_faults_like_the_interpreter() {
    for (label, mmu) in [("030", Mmu::Mc68030 { tc: TC_030 }), ("040", Mmu::Mc68040)] {
        // Warm the window up, then MOVE.L D0,(A1) to an invalid page.
        let program = with_loop(&[], &[0x2280]);
        let (_, batched) = compare(label, mmu, &program, &|cpu, bus| {
            loop_setup(cpu, bus);
            cpu.set_a(1, 0xA000);
            match mmu {
                Mmu::Mc68030 { .. } => bus.poke_long(TABLE + 0xA * 4, 0),
                Mmu::Mc68040 => bus.poke_long(PAGE_040 + 0xA * 4, 0),
            }
        });
        assert_eq!(batched.0.pc, HANDLER + 2, "{label}: stopped in the handler");
    }
}

/// A write-protected page is read through the window, and a write to it
/// faults as in the interpreter (on the 68030 a write misses an ATC entry
/// that a read hits; on the 68040 the entry carries the protection).
#[test]
fn write_protected_page_reads_from_the_window_and_faults_on_write() {
    for (label, mmu) in [("030", Mmu::Mc68030 { tc: TC_030 }), ("040", Mmu::Mc68040)] {
        // loop: MOVE.L (A2),D3 ; ADD.L D3,D4 ; DBRA D7,loop ; MOVE.L D4,(A2)
        let program = [0x2612, 0xD883, 0x51CF, 0xFFFA, 0x2484, ALINE];
        let (reference, batched) = compare(label, mmu, &program, &|cpu, bus| {
            cpu.set_a(2, DATA2);
            cpu.set_d(7, 9_999);
            bus.poke_long(phys(DATA2), 3);
            match mmu {
                Mmu::Mc68030 { .. } => bus.map_030(TABLE, DATA2, phys(DATA2), 0x4),
                Mmu::Mc68040 => bus.map_040(DATA2, phys(DATA2), 0x4),
            }
        });
        assert_eq!(batched.0.d(4), 30_000, "{label}: reads saw the frame");
        assert_eq!(batched.0.pc, HANDLER + 2, "{label}: the write faulted");
        assert_eq!(
            batched.1.peek_long(phys(DATA2)),
            3,
            "{label}: nothing written"
        );
        assert!(reference.1.cycles > 30_000);
        assert!(
            batched.1.cycles < 64,
            "{label}: {} bus cycles",
            batched.1.cycles
        );
    }
}

/// A user-mode access to a 68040 supervisor-only page faults although the
/// page's translation is cached for supervisor accesses; user accesses to
/// ordinary pages keep using the window.
#[test]
fn user_access_to_supervisor_page_faults_040() {
    // MOVE.L (A3),D5 (supervisor: fills the ATC) ; ANDI #$DFFF,SR ;
    // loop: MOVE.L (A0),D1 ; ADDQ.L #1,D1 ; MOVE.L D1,(A0) ; DBRA D7,loop ;
    // MOVE.L (A3),D6 (user: faults)
    let program = [
        0x2A13, 0x027C, 0xDFFF, 0x2210, 0x5281, 0x2081, 0x51CF, 0xFFF8, 0x2C13, ALINE,
    ];
    let (_, batched) = compare("040 S bit", Mmu::Mc68040, &program, &|cpu, bus| {
        cpu.set_a(0, DATA);
        cpu.set_a(3, DATA2);
        cpu.set_d(7, 4_999);
        cpu.set_usp(USER_STACK_TOP);
        bus.poke_long(phys(DATA2), 0x5555_AAAA);
        bus.map_040(DATA2, phys(DATA2), 0x80);
    });
    assert_eq!(batched.0.d(5), 0x5555_AAAA, "supervisor read");
    assert_eq!(batched.1.peek_long(phys(DATA)), 5_000);
    assert_eq!(batched.0.pc, HANDLER + 2, "user read faulted");
    assert_eq!(batched.0.d(6), 0);
}

/// The 68030 ATC tags entries with the access's function code: with SRE
/// the supervisor and user spaces map DATA to different frames, and the
/// window follows the privilege of each access.
#[test]
fn function_code_selects_the_translation_030() {
    // ADD.L D0,(A0) (supervisor) ; ANDI #$DFFF,SR ; then the user loop.
    let program = with_loop(&[0xD190, 0x027C, 0xDFFF], &[]);
    let (_, batched) = compare(
        "030 SRE",
        Mmu::Mc68030 { tc: TC_030_SRE },
        &program,
        &|cpu, bus| {
            loop_setup(cpu, bus);
            cpu.set_d(0, 0x100);
            cpu.set_usp(USER_STACK_TOP);
            bus.map_030(TABLE2, DATA, 0x2_2000, 0);
        },
    );
    assert_eq!(batched.1.peek_long(0x2_2000), 0x100, "supervisor frame");
    let user: u32 = (0x100..0x100 + 20_000u32).sum();
    assert_eq!(batched.1.peek_long(phys(DATA)), user, "user frame");
}

/// A transparent-translation match wins over a cached ATC entry for the
/// same page, through the window as in the interpreter.
#[test]
fn transparent_translation_wins_over_the_atc_030() {
    // MOVE.L (A0),D1 ; PMOVE (A1),TT0 ; loop: MOVE.L (A0),D2 ; ADD.L D2,D3 ;
    // DBRA D7,loop
    let program = [
        0x2210, 0xF011, 0x0800, 0x2410, 0xD682, 0x51CF, 0xFFFA, ALINE,
    ];
    let (reference, batched) = compare(
        "030 TT0",
        Mmu::Mc68030 { tc: TC_030 },
        &program,
        &|cpu, bus| {
            // Logical 0x24000 maps to 0x30000; TT0 makes supervisor data
            // in 0x00xxxxxx transparent, so it then reads 0x24000.
            cpu.set_a(0, 0x2_4000);
            cpu.set_a(1, 0x2_5000);
            cpu.set_d(7, 9_999);
            bus.map_030(TABLE, 0x2_4000, 0x3_0000, 0);
            bus.poke_long(0x3_0000, 0xBBBB);
            bus.poke_long(0x2_4000, 2);
            bus.poke_long(0x2_5000, 0x0000_8500); // E, FC = 5, FC mask 0
        },
    );
    assert_eq!(batched.0.d(1), 0xBBBB, "page-table translation first");
    assert_eq!(batched.0.d(3), 20_000, "then the transparent window");
    assert!(reference.1.cycles > 20_000);
    assert!(batched.1.cycles < 32, "{} bus cycles", batched.1.cycles);
}

/// Accesses that span two pages are split as the interpreter splits them,
/// each half translated on its own page, even when the two frames are not
/// adjacent.
#[test]
fn page_crossing_accesses_split_like_the_interpreter() {
    for (label, mmu) in [("030", Mmu::Mc68030 { tc: TC_030 }), ("040", Mmu::Mc68040)] {
        // loop: MOVE.L (A0),D1 ; ADD.L D1,D2 ; MOVE.L D2,(A0) ; DBRA D7,loop
        let program = [0x2210, 0xD481, 0x2082, 0x51CF, 0xFFF8, ALINE];
        let (_, batched) = compare(label, mmu, &program, &|cpu, bus| {
            cpu.set_a(0, DATA2 - 2);
            cpu.set_d(7, 999);
            bus.poke_long(phys(DATA2 - 2) & !3, 0x0000_0001);
        });
        let hi = u32::from_be_bytes([0, 0, batched.1.mem[0x2_1FFE], batched.1.mem[0x2_1FFF]]);
        let lo = u32::from_be_bytes([0, 0, batched.1.mem[0x2_3000], batched.1.mem[0x2_3001]]);
        assert_eq!(
            (hi << 16) | lo,
            batched.0.d(2),
            "{label}: halves in both frames"
        );
    }
}

/// Rewriting a descriptor takes effect for the window after PFLUSHA, and
/// not before: the ATC keeps the old translation exactly as the
/// interpreter's does.
#[test]
fn pflush_after_a_remap_moves_window_accesses_030() {
    for flush in [true, false] {
        // first loop ; MOVE.L D5,(A2) ; [PFLUSHA] ; MOVE.W #19999,D7 ; loop
        let mut program = with_loop(&[], &[0x2485]);
        program.pop();
        if flush {
            program.extend_from_slice(&[0xF000, 0x2400]);
        }
        program.extend_from_slice(&[0x3E3C, 19_999]);
        program.extend_from_slice(&RMW_LOOP);
        program.push(ALINE);
        let (_, batched) = compare(
            if flush { "PFLUSHA" } else { "no flush" },
            Mmu::Mc68030 { tc: TC_030 },
            &program,
            &|cpu, bus| {
                loop_setup(cpu, bus);
                cpu.set_a(2, TABLE + (DATA >> 12) * 4);
                cpu.set_d(5, 0x2_2001);
            },
        );
        let first: u32 = (0..20_000u32).sum();
        let second: u32 = (20_000..40_000u32).sum();
        if flush {
            assert_eq!(batched.1.peek_long(0x2_1000), first);
            assert_eq!(batched.1.peek_long(0x2_2000), second);
        } else {
            assert_eq!(batched.1.peek_long(0x2_1000), first.wrapping_add(second));
            assert_eq!(batched.1.peek_long(0x2_2000), 0);
        }
    }
}

/// PMOVE to CRP (without FD) flushes the ATC, so the window follows the
/// new root's tables.
#[test]
fn pmove_root_pointer_switches_the_window_030() {
    // first loop ; PMOVE (A3),CRP ; MOVE.W #19999,D7 ; loop
    let mut program = with_loop(&[], &[0xF013, 0x4C00, 0x3E3C, 19_999]);
    program.pop();
    program.extend_from_slice(&RMW_LOOP);
    program.push(ALINE);
    let (_, batched) = compare(
        "PMOVE CRP",
        Mmu::Mc68030 { tc: TC_030 },
        &program,
        &|cpu, bus| {
            loop_setup(cpu, bus);
            cpu.set_a(3, 0x2_6000);
            bus.poke_long(0x2_6000, 2); // limit/DT: short table
            bus.poke_long(0x2_6004, TABLE2);
            bus.map_030(TABLE2, DATA, 0x2_2000, 0);
        },
    );
    assert_eq!(batched.1.peek_long(0x2_1000), (0..20_000u32).sum::<u32>());
    assert_eq!(
        batched.1.peek_long(0x2_2000),
        (20_000..40_000u32).sum::<u32>()
    );
}

/// The 68040's MOVEC to URP/SRP and PFLUSHA take effect for the window
/// mid-batch.
#[test]
fn movec_root_pointer_and_pflusha_040() {
    // first loop ; MOVEC D6,SRP ; PFLUSHA ; MOVE.W #19999,D7 ; loop
    let mut program = with_loop(&[], &[0x4E7B, 0x6807, 0xF518, 0x3E3C, 19_999]);
    program.pop();
    program.extend_from_slice(&RMW_LOOP);
    program.push(ALINE);
    let (_, batched) = compare("040 MOVEC SRP", Mmu::Mc68040, &program, &|cpu, bus| {
        loop_setup(cpu, bus);
        // A second root whose page table maps DATA elsewhere.
        cpu.set_d(6, 0x1_1000);
        bus.poke_long(0x1_1000, 0x1_1200 | 2);
        bus.poke_long(0x1_1200, 0x1_1400 | 2);
        for page in 0..(RAM as u32 >> 12) {
            bus.poke_long(0x1_1400 + page * 4, (page << 12) | 1);
        }
        for (logical, physical) in REMAPS {
            bus.poke_long(0x1_1400 + (logical >> 12) * 4, physical | 1);
        }
        bus.poke_long(0x1_1400 + (DATA >> 12) * 4, 0x2_2000 | 1);
    });
    assert_eq!(batched.1.peek_long(0x2_1000), (0..20_000u32).sum::<u32>());
    assert_eq!(
        batched.1.peek_long(0x2_2000),
        (20_000..40_000u32).sum::<u32>()
    );
}

/// Translation switched on and off inside one batch (PMOVE to TC): the
/// window is used untranslated, then translated, then untranslated again.
#[test]
fn pmove_tc_switches_translation_within_a_batch_030() {
    // loop (MMU off) ; PMOVE (A4),TC ; MOVE.W #19999,D7 ; loop (on) ;
    // PMOVE (A5),TC ; MOVE.W #19999,D7 ; loop (off)
    let mut program = with_loop(&[], &[0xF014, 0x4000, 0x3E3C, 19_999]);
    program.pop();
    program.extend_from_slice(&RMW_LOOP);
    program.extend_from_slice(&[0xF015, 0x4000, 0x3E3C, 19_999]);
    program.extend_from_slice(&RMW_LOOP);
    program.push(ALINE);
    let (_, batched) = compare(
        "PMOVE TC",
        Mmu::Mc68030 { tc: TC_030 },
        &program,
        &|cpu, bus| {
            loop_setup(cpu, bus);
            // Start untranslated, from the code's physical frame (the PMOVE
            // that enables translation is the last instruction fetched
            // from it: CODE's frame continues at the same logical address).
            cpu.pmmu_enabled = false;
            cpu.mmu_tc = 0;
            cpu.pc = phys(CODE);
            cpu.set_a(0, phys(DATA2));
            cpu.set_a(4, 0x2_6000);
            cpu.set_a(5, 0x2_6004);
            bus.poke_long(0x2_6000, TC_030);
            bus.poke_long(0x2_6004, 0);
            // Logical 0x28000 (the code's frame, identity) keeps running;
            // the data address now translates to another frame.
            bus.map_030(TABLE, phys(DATA2), 0x3_1000, 0);
        },
    );
    let first: u32 = (0..20_000u32).sum();
    let second: u32 = (20_000..40_000u32).sum();
    let third: u32 = (40_000..60_000u32).sum();
    assert_eq!(batched.1.peek_long(phys(DATA2)), first.wrapping_add(third));
    assert_eq!(batched.1.peek_long(0x3_1000), second);
    assert!(!batched.0.pmmu_enabled);
}

/// Between batches the host may reset the CPU or rewrite the PMMU
/// registers directly; the next batch translates with the new state.
#[test]
fn host_pmmu_changes_take_effect_at_the_next_batch() {
    let program = with_loop(&[], &[]);
    let mut m = machine(true, Mmu::Mc68030 { tc: TC_030 }, &program, &loop_setup);
    run_batches(&mut m.0, &mut m.1);
    assert_eq!(m.1.peek_long(phys(DATA)), (0..20_000u32).sum::<u32>());

    // Host switch to translation off: addresses are physical again, so the
    // code runs from CODE's own address and the data lands at A0 itself.
    m.0.pmmu_enabled = false;
    m.0.pc = CODE;
    m.1.poke_words(CODE, &program);
    m.0.set_a(0, phys(DATA2));
    m.0.set_d(7, 19_999);
    run_batches(&mut m.0, &mut m.1);
    assert_eq!(m.1.peek_long(phys(DATA2)), (20_000..40_000u32).sum::<u32>());

    // Reset: translation off, as the host sees it. The ATC keeps its entries
    // (MC68030UM 9.7.5), which cannot be reached with translation off.
    m.1.poke_long(0, STACK_TOP);
    m.1.poke_long(4, phys(CODE));
    m.0.reset(&mut m.1);
    assert!(!m.0.pmmu_enabled);
    m.0.set_d(0, 40_000);
    m.0.set_a(0, 0x3_2000);
    m.0.set_d(7, 19_999);
    m.1.cycles = 0;
    run_batches(&mut m.0, &mut m.1);
    assert_eq!(m.1.peek_long(0x3_2000), (40_000..60_000u32).sum::<u32>());
    assert!(
        m.1.cycles < 16,
        "physical window run made {} bus cycles",
        m.1.cycles
    );
}

/// A trace compiled for a code page is rechecked through the translation on
/// every entry: once the page is remapped (and the ATC flushed) to a frame
/// holding different code, that code runs.
#[test]
fn compiled_traces_follow_a_code_page_remap() {
    // loop: ADDQ.L #step,D0 ; DBRA D7,loop
    let version = |step: u16| [0x5080 | (step << 9), 0x51CF, 0xFFFC, ALINE];
    for flush in [true, false] {
        let mut m = machine(true, Mmu::Mc68030 { tc: TC_030 }, &version(1), &|cpu, _| {
            cpu.set_d(7, 49_999);
        });
        m.1.poke_words(0x2_A000, &version(2));
        run_batches(&mut m.0, &mut m.1);
        assert_eq!(m.0.d(0), 50_000);

        m.1.map_030(TABLE, CODE, 0x2_A000, 0);
        if flush {
            m.0.atc_030.flush_all();
        }
        m.0.pc = CODE;
        m.0.set_d(7, 49_999);
        run_batches(&mut m.0, &mut m.1);
        let expected = if flush { 150_000 } else { 100_000 };
        assert_eq!(m.0.d(0), expected, "flush = {flush}");
    }
}

/// Two batch runs from the same state end in the same state, ATC included.
#[test]
fn translated_batches_are_deterministic() {
    let mut program = with_loop(&[], &[0x2485, 0xF000, 0x2400, 0x3E3C, 19_999]);
    program.pop();
    program.extend_from_slice(&RMW_LOOP);
    program.push(0x2280); // MOVE.L D0,(A1): faults
    program.push(ALINE);
    let setup = |cpu: &mut CpuCore, bus: &mut Bus| {
        loop_setup(cpu, bus);
        cpu.set_a(1, 0xA000);
        cpu.set_a(2, TABLE + (DATA >> 12) * 4);
        cpu.set_d(5, 0x2_2001);
        bus.poke_long(TABLE + 0xA * 4, 0);
    };
    let mut a = machine(true, Mmu::Mc68030 { tc: TC_030 }, &program, &setup);
    let mut b = machine(true, Mmu::Mc68030 { tc: TC_030 }, &program, &setup);
    run_batches(&mut a.0, &mut a.1);
    run_batches(&mut b.0, &mut b.1);
    assert_same("repeat", &a, &b);
    assert_eq!(
        a.0.atc_030, b.0.atc_030,
        "ATC contents and replacement order"
    );
    assert_eq!(a.0.pc, HANDLER + 2);
}

/// A tracked window under translation: the loop's stores reach the bus
/// write callbacks at their physical frames, and its memory-to-memory MOVE
/// sends no copy notification, as on the interpreter's translated path.
#[cfg(feature = "jit")]
#[test]
fn tracked_window_stores_reach_the_bus_at_physical_frames() {
    // loop: MOVE.L (A0),(A1) ; ADDQ.L #1,D0 ; ADD.L D0,(A0) ; DBRA D7,loop
    let program = [0x2290, 0x5280, 0xD190, 0x51CF, 0xFFF8, ALINE];
    let setup = |cpu: &mut CpuCore, bus: &mut Bus| {
        cpu.set_a(0, DATA);
        cpu.set_a(1, DATA2);
        cpu.set_d(7, 9_999);
        bus.tracked = true;
    };
    let mut reference = machine(false, Mmu::Mc68030 { tc: TC_030 }, &program, &setup);
    run_steps(&mut reference.0, &mut reference.1);
    let mut tracked = machine(true, Mmu::Mc68030 { tc: TC_030 }, &program, &setup);
    run_batches(&mut tracked.0, &mut tracked.1);
    assert_same("tracked", &reference, &tracked);
    assert_eq!(tracked.1.writes.len(), 20_000, "every store took the bus");
    assert!(
        tracked
            .1
            .writes
            .iter()
            .all(|&at| at == phys(DATA) || at == phys(DATA2)),
        "stores at the physical frames"
    );
    assert_eq!(reference.1.copies, 0, "no notifications under translation");
    assert_eq!(tracked.1.copies, 0, "none from the tracked window either");
}

/// Four A-line-terminated stages for the replacement-order tests:
/// 0: load DATA's then DATA2's ATC entry; 1: a hot read-modify-write loop on
/// DATA (traced once warm); 2: load 20 more pages (`A1` stepping by `A3`),
/// filling the 22-entry ATC and evicting; 3: read DATA2 then DATA again.
const REPLACEMENT_STAGES: [u16; 16] = [
    0x2210, 0x2212, ALINE, // MOVE.L (A0),D1 ; MOVE.L (A2),D1
    0xD190, 0x5280, 0x51CF, 0xFFFA, ALINE, // loop: ADD.L D0,(A0) ; ADDQ.L #1,D0 ; DBRA D7
    0x2411, 0xD3CB, 0x51CE, 0xFFFA, ALINE, // fill: MOVE.L (A1),D2 ; ADDA.L A3,A1 ; DBRA D6
    0x2612, 0x2210, ALINE, // MOVE.L (A2),D3 ; MOVE.L (A0),D1
];

fn replacement_setup(cpu: &mut CpuCore, bus: &mut Bus) {
    cpu.set_a(0, DATA);
    cpu.set_a(2, DATA2);
    cpu.set_d(7, 2_999);
    cpu.set_a(1, 0x2_C000);
    cpu.set_a(3, 0x1000);
    cpu.set_d(6, 19);
    bus.poke_long(phys(DATA2), 0xC0C0_0001);
    bus.poke_long(0x2_4000, 0xA1A1_0002);
    bus.poke_long(0x2_5000, 0xC1C1_0003);
}

/// Before the last stage the host remaps DATA and DATA2 to new frames
/// without a flush: a resident entry keeps the stale frame, an evicted one
/// searches the tables again.
fn replacement_remap(stage: usize, _cpu: &mut CpuCore, bus: &mut Bus) {
    if stage == 3 {
        bus.map_030(TABLE, DATA, 0x2_4000, 0);
        bus.map_030(TABLE, DATA2, 0x2_5000, 0);
    }
}

/// Run the replacement stages twice on one fresh thread: first with an
/// empty trace cache, then from the same start with the traces the first
/// run compiled. Returns (cold, warm).
fn cold_and_warm(setup: fn(&mut CpuCore, &mut Bus)) -> ((CpuCore, Bus), (CpuCore, Bus)) {
    std::thread::spawn(move || {
        let run = || {
            let mut m = machine(
                true,
                Mmu::Mc68030 { tc: TC_030 },
                &REPLACEMENT_STAGES,
                &setup,
            );
            for stage in 0..4 {
                replacement_remap(stage, &mut m.0, &mut m.1);
                run_batches(&mut m.0, &mut m.1);
            }
            m
        };
        let cold = run();
        let warm = run();
        (cold, warm)
    })
    .join()
    .unwrap()
}

/// The 68030 ATC's replacement order is saved state, so it must not depend
/// on which instructions ran from compiled traces, which access memory
/// through the window and fetch no instructions. While the window
/// translates, no hit records a use -- the interpreter's own bus-path hits
/// included -- so entries age by when they were loaded. A cold run (the
/// interpreter, then traces as they compile) and a warm one (traces from the
/// start) leave the same ATC, evict the same entries and keep the same
/// stale translation.
fn assert_replacement_order_ignores_the_trace_cache(
    label: &str,
    setup: fn(&mut CpuCore, &mut Bus),
) {
    let (cold, warm) = cold_and_warm(setup);
    assert_same(label, &cold, &warm);
    assert_eq!(cold.0.atc_030, warm.0.atc_030, "{label}: ATC");
    // Loads: code, DATA, DATA2, then 20 fill pages. The 20th fill page
    // evicts the code page, which the next fetch reloads in place of DATA:
    // DATA2 is still resident with its stale frame, DATA is searched anew.
    assert_eq!(
        warm.0.d(3),
        0xC0C0_0001,
        "{label}: DATA2 kept its stale frame"
    );
    assert_eq!(warm.0.d(1), 0xA1A1_0002, "{label}: DATA was evicted");
}

/// With a tracked window the interpreter runs every memory instruction
/// through full dispatch and `translate`, while a warm trace runs it
/// through the window.
#[cfg(feature = "jit")]
#[test]
fn atc_replacement_order_ignores_the_trace_cache_with_a_tracked_window() {
    assert_replacement_order_ignores_the_trace_cache("tracked", |cpu, bus| {
        replacement_setup(cpu, bus);
        bus.tracked = true;
    });
}

/// With the code outside the window the interpreter fetches every
/// instruction through `translate`, while a warm trace fetches none.
#[test]
fn atc_replacement_order_ignores_the_trace_cache_with_code_outside_the_window() {
    assert_replacement_order_ignores_the_trace_cache("code outside", |cpu, bus| {
        replacement_setup(cpu, bus);
        bus.poke_words(0xB000, &REPLACEMENT_STAGES);
        bus.map_030(TABLE, CODE, 0xB000, 0);
    });
}

/// The table search's history bits come out the same through the window:
/// a read loop loads the page's ATC entry with M clear, and the write that
/// follows it still searches the tables and sets M in the page descriptor.
#[test]
fn history_bits_match_the_interpreter_040() {
    // loop: MOVE.L (A2),D3 ; ADD.L D3,D4 ; DBRA D7,loop ; MOVE.L D4,(A2)
    let program = [0x2612, 0xD883, 0x51CF, 0xFFFA, 0x2484, ALINE];
    let (_, batched) = compare("040 history", Mmu::Mc68040, &program, &|cpu, bus| {
        cpu.set_a(2, DATA2);
        cpu.set_d(7, 999);
        bus.poke_long(phys(DATA2), 3);
    });
    let desc = batched.1.peek_long(PAGE_040 + (DATA2 >> 12) * 4);
    assert_eq!(desc & 0x18, 0x18, "U and M in {desc:#010x}");
    assert_eq!(batched.1.peek_long(phys(DATA2)), 3000);
}

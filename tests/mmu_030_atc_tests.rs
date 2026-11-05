//! 68030 address translation cache: a repeat access to a page is answered
//! without a table search, the cached translation outlives a descriptor edit
//! until software flushes it (PFLUSH, PFLUSHA, PMOVE to TC/SRP/CRP without
//! FD), PLOAD fills an entry, and write protection is still enforced on a hit.

use m68k::core::memory::AddressBus;
use m68k::mmu::{MmuFaultKind, translate_address};
use m68k::{CpuCore, CpuType, NoOpHleHandler, StepResult};

/// Flat test memory that counts reads inside the page table, which is how a
/// test tells a table search from an ATC hit.
struct TestBus {
    mem: Vec<u8>,
    table_reads: u32,
}

const TABLE: u32 = 0x1_0000;
/// Reads anywhere from here up count as table reads (the early-termination
/// test keeps its root table above the main one).
const TABLE_END: u32 = 0x2_0000;
const CODE: u32 = 0x1000;
/// E, PS = 4 KB, IS = 8, TIA = 12: one level of 4096 short descriptors
/// indexed by logical bits 23:12.
const TC: u32 = 0x80C8_C000;

impl TestBus {
    fn new() -> Self {
        let mut bus = Self {
            mem: vec![0; 0x2_0000],
            table_reads: 0,
        };
        // Identity-map the first 128 KB as 4 KB pages.
        for page in 0..32 {
            bus.map(page, page);
        }
        bus.write_long(0, 0x8000); // reset SSP
        bus.write_long(4, CODE); // reset PC
        bus.table_reads = 0;
        bus
    }

    fn map(&mut self, page: u32, phys_page: u32) {
        self.map_flags(page, phys_page, 0);
    }

    /// Page descriptor: physical page, DT = 1, plus extra flag bits (WP is
    /// bit 2).
    fn map_flags(&mut self, page: u32, phys_page: u32, flags: u32) {
        let at = (TABLE + page * 4) as usize;
        self.mem[at..at + 4].copy_from_slice(&((phys_page << 12) | flags | 1).to_be_bytes());
    }

    fn words(&mut self, at: u32, words: &[u16]) {
        for (i, w) in words.iter().enumerate() {
            let a = at as usize + i * 2;
            self.mem[a..a + 2].copy_from_slice(&w.to_be_bytes());
        }
    }

    fn note(&mut self, addr: u32) {
        if (TABLE..TABLE_END).contains(&addr) {
            self.table_reads += 1;
        }
    }
}

impl AddressBus for TestBus {
    fn read_byte(&mut self, addr: u32) -> u8 {
        self.note(addr);
        self.mem.get(addr as usize).copied().unwrap_or(0)
    }
    fn write_byte(&mut self, addr: u32, val: u8) {
        if let Some(m) = self.mem.get_mut(addr as usize) {
            *m = val;
        }
    }
    fn read_word(&mut self, addr: u32) -> u16 {
        self.note(addr);
        let a = addr as usize;
        u16::from_be_bytes([self.mem[a], self.mem[a + 1]])
    }
    fn write_word(&mut self, addr: u32, val: u16) {
        let a = addr as usize;
        self.mem[a..a + 2].copy_from_slice(&val.to_be_bytes());
    }
    fn read_long(&mut self, addr: u32) -> u32 {
        self.note(addr);
        let a = addr as usize;
        u32::from_be_bytes(self.mem[a..a + 4].try_into().unwrap())
    }
    fn write_long(&mut self, addr: u32, val: u32) {
        let a = addr as usize;
        self.mem[a..a + 4].copy_from_slice(&val.to_be_bytes());
    }
}

/// A supervisor-mode 68030 with the MMU on over the bus's table.
fn mmu_030(bus: &mut TestBus) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68030);
    cpu.reset(bus);
    cpu.set_sr(0x2700);
    cpu.mmu_crp_aptr = TABLE;
    cpu.mmu_crp_limit = 2; // DT = 2: short-format table
    cpu.mmu_tc = TC;
    cpu.pmmu_enabled = true;
    bus.table_reads = 0;
    cpu
}

fn data(cpu: &mut CpuCore, bus: &mut TestBus, logical: u32) -> u32 {
    translate_address(cpu, bus, logical, false, true, false).unwrap()
}

/// Run the one instruction at `CODE` (whatever `words` encode).
fn run(cpu: &mut CpuCore, bus: &mut TestBus, words: &[u16]) {
    bus.words(CODE, words);
    cpu.pc = CODE;
    let result = cpu.step_with_hle_handler(bus, &mut NoOpHleHandler);
    assert!(matches!(result, StepResult::Ok { .. }), "{result:?}");
    assert_eq!(cpu.pc, CODE + 2 * words.len() as u32, "instruction length");
}

#[test]
fn repeat_access_to_a_page_needs_no_table_search() {
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x3010), 0x3010);
    assert_eq!(bus.table_reads, 1, "first access searches the table");
    assert_eq!(data(&mut cpu, &mut bus, 0x3FFC), 0x3FFC);
    assert_eq!(bus.table_reads, 1, "same page: ATC hit");
    assert_eq!(data(&mut cpu, &mut bus, 0x4000), 0x4000);
    assert_eq!(bus.table_reads, 2, "next page: another search");
    // Each function code has its own entries.
    translate_address(&mut cpu, &mut bus, 0x3010, false, false, false).unwrap();
    assert_eq!(bus.table_reads, 3, "user data space");
    translate_address(&mut cpu, &mut bus, 0x3010, false, true, true).unwrap();
    assert_eq!(bus.table_reads, 4, "supervisor program space");
    translate_address(&mut cpu, &mut bus, 0x3020, false, false, false).unwrap();
    translate_address(&mut cpu, &mut bus, 0x3020, false, true, true).unwrap();
    assert_eq!(bus.table_reads, 4);
}

#[test]
fn early_termination_descriptor_is_cached_one_page_at_a_time() {
    // A two-level tree whose level-A descriptor terminates early, mapping a
    // whole 1 MB region at 0x0005_0000 + offset: each 4 KB page is its own
    // entry, and a hit reproduces the walk's address exactly.
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    let root = 0x1_8000u32;
    bus.write_long(root, 0x0005_0001);
    cpu.mmu_crp_aptr = root;
    // E, PS 4 KB, IS 8, TIA 4, TIB 8: TIA indexes logical bits 23:20, so all
    // three probes below, up to 0x000F_F000, use root entry 0.
    cpu.mmu_tc = 0x80C8_4800;
    for logical in [0x0000_1234u32, 0x0000_1FFF, 0x000F_F000] {
        let walked = 0x0005_0000 + (logical & 0x000F_FFFF);
        assert_eq!(data(&mut cpu, &mut bus, logical), walked);
        bus.table_reads = 0;
        assert_eq!(data(&mut cpu, &mut bus, logical ^ 0x10), walked ^ 0x10);
        assert_eq!(bus.table_reads, 0, "{logical:#x} hits");
    }
}

#[test]
fn tc_without_address_levels_translates_uncached() {
    // E and FCL only: the function-code level's page descriptor maps the
    // whole space, leaving no logical page bits to tag an entry with.
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    let root = 0x1_8000u32;
    bus.write_long(root + 5 * 4, 0x0000_2001); // FC 5: base 0x2000
    cpu.mmu_crp_aptr = root;
    cpu.mmu_tc = 0x8100_0000;
    assert_eq!(data(&mut cpu, &mut bus, 0x3010), 0x5010);
    assert_eq!(data(&mut cpu, &mut bus, 0x3010), 0x5010);
    assert_eq!(bus.table_reads, 2, "each access searches the table");
    assert_eq!(cpu.atc_030.valid_entries(), 0);
}

#[test]
fn stale_translation_lasts_until_pflush_of_its_page() {
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x5004);
    assert_eq!(data(&mut cpu, &mut bus, 0x6004), 0x6004);
    // Remap both pages without a flush: the ATC keeps the old answers.
    bus.map(5, 0x9);
    bus.map(6, 0xA);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x5004);

    // PFLUSH #5,#7,(A0) with A0 in page 5: only that page's supervisor
    // data entry goes.
    cpu.set_a(0, 0x5000);
    run(&mut cpu, &mut bus, &[0xF010, 0x38F5]);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x9004);
    assert_eq!(data(&mut cpu, &mut bus, 0x6004), 0x6004, "page 6 kept");

    // PFLUSH #4,#4: every supervisor entry.
    run(&mut cpu, &mut bus, &[0xF000, 0x3094]);
    assert_eq!(data(&mut cpu, &mut bus, 0x6004), 0xA004);
}

#[test]
fn pflusha_and_pmove_flush_but_pmovefd_does_not() {
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x5004);
    bus.map(5, 0x9);
    // PFLUSHA.
    run(&mut cpu, &mut bus, &[0xF000, 0x2400]);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x9004);

    // PMOVEFD (A0),TC rewrites TC with the same value and keeps the ATC.
    bus.write_long(0x7000, TC);
    cpu.set_a(0, 0x7000);
    bus.map(5, 0xB);
    run(&mut cpu, &mut bus, &[0xF010, 0x4100]);
    assert_eq!(cpu.mmu_tc, TC);
    assert_eq!(
        data(&mut cpu, &mut bus, 0x5004),
        0x9004,
        "FD kept the entry"
    );
    // PMOVE (A0),TC flushes it.
    run(&mut cpu, &mut bus, &[0xF010, 0x4000]);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0xB004);

    // So does a PMOVE to CRP.
    bus.write_long(0x7008, 2);
    bus.write_long(0x700C, TABLE);
    cpu.set_a(0, 0x7008);
    bus.map(5, 0xC);
    run(&mut cpu, &mut bus, &[0xF010, 0x4C00]);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0xC004);
}

#[test]
fn pmove_flush_follows_its_operand_fetch() {
    // A second table maps page 7, where the PMOVE (A0),CRP operand lives,
    // somewhere else. Fetching the operand through the old root caches
    // page 7's old translation; the flush after the write must drop it.
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    let new_root = 0x1_4000u32;
    for page in 0..32u32 {
        let phys = if page == 7 { 0xD } else { page };
        bus.write_long(new_root + page * 4, (phys << 12) | 1);
    }
    bus.write_long(0x7008, 2);
    bus.write_long(0x700C, new_root);
    cpu.set_a(0, 0x7008);
    run(&mut cpu, &mut bus, &[0xF010, 0x4C00]);
    assert_eq!(cpu.mmu_crp_aptr, new_root);
    assert_eq!(data(&mut cpu, &mut bus, 0x7004), 0xD004);
}

#[test]
fn pmove_operand_fetch_uses_the_cached_translation() {
    // Page 7 is cached, then its descriptor is edited without a flush: the
    // operand fetch still goes through the cached translation, as it does
    // on the part, so PMOVE (A0),TC reads the TC from the old page.
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x7004), 0x7004);
    bus.map(7, 0xE);
    bus.write_long(0x7000, TC);
    bus.write_long(0xE000, 0);
    cpu.set_a(0, 0x7000);
    run(&mut cpu, &mut bus, &[0xF010, 0x4000]);
    assert_eq!(cpu.mmu_tc, TC);
    assert!(cpu.pmmu_enabled);
    // The write flushed the stale entry: the edit is now in force.
    assert_eq!(data(&mut cpu, &mut bus, 0x7004), 0xE004);
}

#[test]
fn pload_of_an_invalidated_page_drops_its_old_entry() {
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x9004), 0x9004);
    // Invalidate the descriptor: the cached entry survives the edit ...
    bus.write_long(TABLE + 9 * 4, 0);
    assert_eq!(data(&mut cpu, &mut bus, 0x9004), 0x9004);
    // ... until PLOADR #5,(A0) searches the page and finds it invalid.
    cpu.set_a(0, 0x9000);
    run(&mut cpu, &mut bus, &[0xF010, 0x2215]);
    let err = translate_address(&mut cpu, &mut bus, 0x9004, false, true, false).unwrap_err();
    assert_eq!(err.kind, MmuFaultKind::AccessLevelViolation);
}

#[test]
fn pload_fills_the_entry_ahead_of_the_access() {
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    // PLOADR #5,(A0).
    cpu.set_a(0, 0x8000);
    run(&mut cpu, &mut bus, &[0xF010, 0x2215]);
    assert_eq!(cpu.atc_030.valid_entries(), 2, "program page + the PLOAD");
    bus.table_reads = 0;
    assert_eq!(data(&mut cpu, &mut bus, 0x8010), 0x8010);
    assert_eq!(bus.table_reads, 0, "loaded by PLOAD");
}

#[test]
fn write_protected_page_hits_for_reads_and_faults_writes() {
    let mut bus = TestBus::new();
    bus.map_flags(3, 3, 0x4); // WP
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x3000), 0x3000);
    let err = translate_address(&mut cpu, &mut bus, 0x3000, true, true, false).unwrap_err();
    assert_eq!(err.kind, MmuFaultKind::AccessLevelViolation);
    // Still faults on a second write, and reads still hit.
    assert!(translate_address(&mut cpu, &mut bus, 0x3004, true, true, false).is_err());
    let before = bus.table_reads;
    assert_eq!(data(&mut cpu, &mut bus, 0x3008), 0x3008);
    assert_eq!(bus.table_reads, before);
}

#[test]
fn reset_disables_translation_but_keeps_the_atc() {
    // MC68030UM 9.7.5: "The assertion of RESET disables translations by
    // clearing the E bits of the TC and TTx registers, but it does not flush
    // the ATC."
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x5004);
    cpu.mmu_tt0 = 0x0000_8000;
    cpu.mmu_tt1 = 0x0000_8000;
    cpu.reset(&mut bus);
    assert_eq!(cpu.mmu_tc & 0x8000_0000, 0, "TC E cleared");
    assert_eq!(cpu.mmu_tt0 & 0x8000, 0, "TT0 E cleared");
    assert_eq!(cpu.mmu_tt1 & 0x8000, 0, "TT1 E cleared");
    assert_eq!(cpu.atc_030.valid_entries(), 1, "entry survives RESET");
}

#[test]
fn pmove_to_tc_after_reset_decides_whether_stale_entries_survive() {
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x5004);
    cpu.reset(&mut bus);
    bus.map(5, 0x9);
    cpu.set_sr(0x2700);

    // With translation off the surviving entry is not consulted.
    assert!(!cpu.pmmu_enabled);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x5004);

    // PMOVEFD (A0),TC re-enables translation and keeps the stale entry.
    bus.write_long(0x7000, TC);
    cpu.set_a(0, 0x7000);
    run(&mut cpu, &mut bus, &[0xF010, 0x4100]);
    assert!(cpu.pmmu_enabled);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x5004, "FD kept it");

    // PMOVE (A0),TC flushes it, so the edited descriptor takes effect.
    run(&mut cpu, &mut bus, &[0xF010, 0x4000]);
    assert_eq!(data(&mut cpu, &mut bus, 0x5004), 0x9004);
}

#[cfg(feature = "serde")]
#[test]
fn atc_survives_a_serde_round_trip() {
    let mut bus = TestBus::new();
    let mut cpu = mmu_030(&mut bus);
    data(&mut cpu, &mut bus, 0x3000);
    data(&mut cpu, &mut bus, 0x4000);
    let json = serde_json::to_string(&cpu).unwrap();
    let restored: CpuCore = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.atc_030, cpu.atc_030);
    // A state from before the ATC existed restores with it empty.
    let mut value: serde_json::Value = serde_json::from_str(&json).unwrap();
    value.as_object_mut().unwrap().remove("atc_030");
    let old: CpuCore = serde_json::from_value(value).unwrap();
    assert_eq!(old.atc_030.valid_entries(), 0);
}

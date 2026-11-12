//! 68030 PMMU behaviour against a three-level table (TIA=8, TIB=6, TIC=6,
//! 4 KB pages; MC68030UM section 9): transparent translation through
//! TT0/TT1, the history bits the table search writes back, and the
//! root-pointer and long-descriptor limits.

use m68k::core::cpu::CpuCore;
use m68k::core::memory::AddressBus;
use m68k::core::types::CpuType;
use m68k::mmu::translate_address;

struct TestBus {
    mem: Vec<u8>,
}

impl AddressBus for TestBus {
    fn read_byte(&mut self, addr: u32) -> u8 {
        self.mem.get(addr as usize).copied().unwrap_or(0)
    }
    fn write_byte(&mut self, addr: u32, val: u8) {
        if let Some(m) = self.mem.get_mut(addr as usize) {
            *m = val;
        }
    }
    fn read_word(&mut self, addr: u32) -> u16 {
        ((self.read_byte(addr) as u16) << 8) | self.read_byte(addr.wrapping_add(1)) as u16
    }
    fn write_word(&mut self, addr: u32, val: u16) {
        self.write_byte(addr, (val >> 8) as u8);
        self.write_byte(addr.wrapping_add(1), val as u8);
    }
    fn read_long(&mut self, addr: u32) -> u32 {
        ((self.read_word(addr) as u32) << 16) | self.read_word(addr.wrapping_add(2)) as u32
    }
    fn write_long(&mut self, addr: u32, val: u32) {
        self.write_word(addr, (val >> 16) as u16);
        self.write_word(addr.wrapping_add(2), val as u16);
    }
}

/// E, PS = 4 KB, IS = 0, TIA = 8, TIB = 6, TIC = 6.
const TC: u32 = 0x80C0_8660;
/// Level A (256 short descriptors), level B and level C tables.
const TA: u32 = 0x2000;
const TB: u32 = 0x2400;
const TC_TABLE: u32 = 0x2500;
/// The logical page the tables remap, and where they send it.
const LOGICAL: u32 = 0x0001_0000;
const PHYS: u32 = 0x0001_8000;

/// Short-format tables: logical $00000000-$0003FFFF through level A entry
/// 0, level B entry 0 and 64 level C page descriptors (identity except
/// `LOGICAL`, which maps to `PHYS`); every other level A entry an
/// early-termination identity descriptor.
fn build() -> (CpuCore, TestBus) {
    let mut bus = TestBus {
        mem: vec![0; 0x4_0000],
    };
    bus.write_long(TA, TB | 2);
    for i in 1..256u32 {
        bus.write_long(TA + i * 4, (i << 24) | 1);
    }
    bus.write_long(TB, TC_TABLE | 2);
    for i in 0..64u32 {
        bus.write_long(TC_TABLE + i * 4, (i << 12) | 1);
    }
    bus.write_long(TC_TABLE + (LOGICAL >> 12) * 4, PHYS | 1);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68030);
    cpu.reset(&mut bus);
    cpu.set_sr(0x2700);
    cpu.mmu_crp_limit = 0x7FFF_0002; // upper limit $7FFF, short tables
    cpu.mmu_crp_aptr = TA;
    cpu.mmu_tc = TC;
    cpu.pmmu_enabled = true;
    (cpu, bus)
}

/// 68030 TT register: base, mask, E, R/W, RWM, FC base, FC mask
/// (MC68030UM 9.7.1).
fn tt(base: u32, mask: u32, rw: bool, rwm: bool, fcb: u32, fcm: u32) -> u32 {
    (base << 24)
        | (mask << 16)
        | 0x8000
        | if rw { 0x200 } else { 0 }
        | if rwm { 0x100 } else { 0 }
        | (fcb << 4)
        | fcm
}

/// Translate `LOGICAL + 0x500` for an access in function-code space `fc`.
fn xlate(cpu: &mut CpuCore, bus: &mut TestBus, fc: u8, write: bool) -> u32 {
    cpu.atc_030.flush_all();
    let supervisor = fc & 4 != 0;
    let instruction = fc & 3 == 2;
    cpu.set_sr(if supervisor { 0x2700 } else { 0x0700 });
    let r = translate_address(cpu, bus, LOGICAL + 0x500, write, supervisor, instruction);
    cpu.set_sr(0x2700);
    r.expect("translation")
}

const IDENT: u32 = LOGICAL + 0x500;
const TABLE: u32 = PHYS + 0x500;

#[test]
fn tt_fc_base_and_mask_select_the_function_codes() {
    let (mut cpu, mut bus) = build();
    // FC base 1, mask 0: user data only.
    cpu.mmu_tt0 = tt(0, 0, false, true, 1, 0);
    assert_eq!(xlate(&mut cpu, &mut bus, 1, false), IDENT);
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), TABLE);
    assert_eq!(xlate(&mut cpu, &mut bus, 2, false), TABLE);
    // TT1 behaves the same.
    cpu.mmu_tt0 = 0;
    cpu.mmu_tt1 = tt(0, 0, false, true, 1, 0);
    assert_eq!(xlate(&mut cpu, &mut bus, 1, true), IDENT);
    assert_eq!(xlate(&mut cpu, &mut bus, 5, true), TABLE);
    // FC mask 4 ignores FC2: user and supervisor data.
    cpu.mmu_tt1 = tt(0, 0, false, true, 1, 4);
    assert_eq!(xlate(&mut cpu, &mut bus, 1, false), IDENT);
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), IDENT);
    assert_eq!(xlate(&mut cpu, &mut bus, 6, false), TABLE);
    // FC base 6: supervisor program (instruction fetches).
    cpu.mmu_tt1 = tt(0, 0, false, true, 6, 0);
    assert_eq!(xlate(&mut cpu, &mut bus, 6, false), IDENT);
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), TABLE);
}

#[test]
fn tt_rw_selects_the_access_direction_unless_rwm_is_set() {
    let (mut cpu, mut bus) = build();
    cpu.mmu_tt0 = tt(0, 0, true, false, 5, 0); // reads only
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), IDENT);
    assert_eq!(xlate(&mut cpu, &mut bus, 5, true), TABLE);
    cpu.mmu_tt0 = tt(0, 0, false, false, 5, 0); // writes only
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), TABLE);
    assert_eq!(xlate(&mut cpu, &mut bus, 5, true), IDENT);
    cpu.mmu_tt0 = tt(0, 0, false, true, 5, 0); // RWM: both
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), IDENT);
    assert_eq!(xlate(&mut cpu, &mut bus, 5, true), IDENT);
}

#[test]
fn tt_address_mask_and_enable() {
    let (mut cpu, mut bus) = build();
    cpu.mmu_tt0 = tt(1, 1, false, true, 5, 0); // base $01, bit 24 masked
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), IDENT);
    cpu.mmu_tt0 = tt(1, 0, false, true, 5, 0); // $01 only
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), TABLE);
    cpu.mmu_tt0 = tt(0, 0, false, true, 5, 0) & !0x8000; // E = 0
    assert_eq!(xlate(&mut cpu, &mut bus, 5, false), TABLE);
}

const U: u32 = 0x08;
const M: u32 = 0x10;
const WP: u32 = 0x04;
const DESC: u32 = TC_TABLE + (LOGICAL >> 12) * 4;

fn read(cpu: &mut CpuCore, bus: &mut TestBus, addr: u32, supervisor: bool) -> Result<u32, ()> {
    cpu.set_sr(if supervisor { 0x2700 } else { 0x0700 });
    let r = translate_address(cpu, bus, addr, false, supervisor, false).map_err(|_| ());
    cpu.set_sr(0x2700);
    r
}

fn write(cpu: &mut CpuCore, bus: &mut TestBus, addr: u32, supervisor: bool) -> Result<u32, ()> {
    cpu.set_sr(if supervisor { 0x2700 } else { 0x0700 });
    let r = translate_address(cpu, bus, addr, true, supervisor, false).map_err(|_| ());
    cpu.set_sr(0x2700);
    r
}

/// Execute one PMMU instruction (`F010` = `(A0)`, plus its extension word)
/// at $1000 in supervisor mode with A0 = `a0`.
fn pmmu_op(cpu: &mut CpuCore, bus: &mut TestBus, ext: u16, a0: u32) {
    bus.write_word(0x1000, 0xF010);
    bus.write_word(0x1002, ext);
    bus.write_word(0x1004, 0x4E71);
    cpu.set_a(0, a0);
    cpu.pc = 0x1000;
    let mut hle = m68k::NoOpHleHandler;
    cpu.step_with_hle_handler(bus, &mut hle);
    assert_eq!(cpu.pc, 0x1004, "PMMU instruction executes");
}

#[test]
fn a_read_sets_used_in_every_level_and_leaves_modified_clear() {
    let (mut cpu, mut bus) = build();
    assert_eq!(read(&mut cpu, &mut bus, LOGICAL + 4, true), Ok(PHYS + 4));
    assert_eq!(bus.read_long(TA), TB | U | 2, "level A");
    assert_eq!(bus.read_long(TB), TC_TABLE | U | 2, "level B");
    assert_eq!(bus.read_long(DESC), PHYS | U | 1, "page: U, M clear");
}

#[test]
fn an_early_termination_descriptor_is_marked_used() {
    let (mut cpu, mut bus) = build();
    assert_eq!(read(&mut cpu, &mut bus, 0x0300_0010, true), Ok(0x0300_0010));
    assert_eq!(bus.read_long(TA + 3 * 4), 0x0300_0000 | U | 1);
    write(&mut cpu, &mut bus, 0x0400_0010, true).unwrap();
    assert_eq!(bus.read_long(TA + 4 * 4), 0x0400_0000 | U | M | 1);
}

#[test]
fn a_write_sets_used_and_modified() {
    let (mut cpu, mut bus) = build();
    assert_eq!(write(&mut cpu, &mut bus, LOGICAL + 8, true), Ok(PHYS + 8));
    assert_eq!(bus.read_long(DESC), PHYS | U | M | 1);
}

#[test]
fn a_write_after_a_read_sets_modified_through_the_atc_entry() {
    let (mut cpu, mut bus) = build();
    read(&mut cpu, &mut bus, LOGICAL + 4, true).unwrap();
    assert_eq!(bus.read_long(DESC) & M, 0);
    // The read's ATC entry has M clear: the write searches the tables.
    write(&mut cpu, &mut bus, LOGICAL + 8, true).unwrap();
    assert_eq!(bus.read_long(DESC), PHYS | U | M | 1);
    // Now the entry is modified and serves writes without a search.
    bus.write_long(DESC, PHYS | U | 1);
    write(&mut cpu, &mut bus, LOGICAL + 12, true).unwrap();
    assert_eq!(bus.read_long(DESC), PHYS | U | 1, "ATC hit");
}

#[test]
fn an_indirect_descriptor_is_left_unchanged_and_its_target_updated() {
    let (mut cpu, mut bus) = build();
    let target = 0x3000;
    bus.write_long(target, PHYS | 1);
    bus.write_long(DESC, target | 2); // DT=2 at the last level: indirect
    assert_eq!(
        read(&mut cpu, &mut bus, LOGICAL + 0x300, true),
        Ok(PHYS + 0x300)
    );
    assert_eq!(bus.read_long(DESC), target | 2);
    assert_eq!(bus.read_long(target), PHYS | U | 1);
}

#[test]
fn write_protection_sets_used_but_not_modified() {
    let (mut cpu, mut bus) = build();
    bus.write_long(DESC, PHYS | WP | 1);
    assert!(write(&mut cpu, &mut bus, LOGICAL, true).is_err());
    assert_eq!(bus.read_long(DESC), PHYS | WP | U | 1);

    // WP in a table descriptor protects the pages below it.
    let (mut cpu, mut bus) = build();
    bus.write_long(TB, TC_TABLE | WP | 2);
    assert!(write(&mut cpu, &mut bus, LOGICAL, true).is_err());
    assert_eq!(bus.read_long(TB), TC_TABLE | WP | U | 2);
    assert_eq!(bus.read_long(DESC), PHYS | U | 1);
}

#[test]
fn invalid_descriptors_are_not_updated() {
    let (mut cpu, mut bus) = build();
    bus.write_long(DESC, 0x494E_5600); // DT=0, OS-owned bits
    assert!(read(&mut cpu, &mut bus, LOGICAL, true).is_err());
    assert_eq!(bus.read_long(DESC), 0x494E_5600);
    assert_eq!(bus.read_long(TB), TC_TABLE | U | 2, "level B was used");
}

#[test]
fn long_format_descriptors_take_the_bits_in_their_first_long() {
    let (mut cpu, mut bus) = build();
    // Level B entry 0 as a long-format table: two long descriptors at $3000,
    // the second-level index 0 pointing (long) at a long page table at $3100.
    let tb_long = 0x3000;
    let tc_long = 0x3100;
    bus.write_long(TA, tb_long | 3);
    bus.write_long(tb_long, 0x7FFF_FC00 | 3);
    bus.write_long(tb_long + 4, tc_long);
    for i in 0..64u32 {
        bus.write_long(tc_long + i * 8, 0x7FFF_FC00 | 1);
        bus.write_long(tc_long + i * 8 + 4, i << 12);
    }
    let page = tc_long + (LOGICAL >> 12) * 8;
    bus.write_long(page + 4, PHYS);
    assert_eq!(write(&mut cpu, &mut bus, LOGICAL + 4, true), Ok(PHYS + 4));
    assert_eq!(bus.read_long(tb_long), 0x7FFF_FC00 | U | 3);
    assert_eq!(
        bus.read_long(tb_long + 4),
        tc_long,
        "address long untouched"
    );
    assert_eq!(bus.read_long(page), 0x7FFF_FC00 | U | M | 1);
    assert_eq!(bus.read_long(page + 4), PHYS);

    // A supervisor-only (S=1) page denies a user access: no update.
    let s_page = tc_long + 0x11 * 8;
    bus.write_long(s_page, 0x7FFF_FC00 | 0x100 | 1);
    assert!(read(&mut cpu, &mut bus, 0x0001_1000, false).is_err());
    assert_eq!(bus.read_long(s_page), 0x7FFF_FC00 | 0x100 | 1);
    assert!(read(&mut cpu, &mut bus, 0x0001_1000, true).is_ok());
    assert_eq!(bus.read_long(s_page), 0x7FFF_FC00 | 0x100 | U | 1);
}

#[test]
fn ptest_changes_no_descriptor() {
    let (mut cpu, mut bus) = build();
    pmmu_op(&mut cpu, &mut bus, 0x9E15, LOGICAL); // PTESTR #5,(A0),#7
    pmmu_op(&mut cpu, &mut bus, 0x9C15, LOGICAL); // PTESTW #5,(A0),#7
    // (Level A and B entry 0 also map the code page, whose instruction
    // fetches mark them used; the tested page's descriptor is its own.)
    assert_eq!(bus.read_long(DESC), PHYS | 1);
}

#[test]
fn pload_updates_history_as_its_direction() {
    let (mut cpu, mut bus) = build();
    pmmu_op(&mut cpu, &mut bus, 0x2215, LOGICAL); // PLOADR #5,(A0)
    assert_eq!(bus.read_long(DESC), PHYS | U | 1);
    let (mut cpu, mut bus) = build();
    pmmu_op(&mut cpu, &mut bus, 0x2015, LOGICAL); // PLOADW #5,(A0)
    assert_eq!(bus.read_long(DESC), PHYS | U | M | 1);
    // The PLOADW entry is modified: a write hits it without a search.
    bus.write_long(DESC, PHYS | 1);
    write(&mut cpu, &mut bus, LOGICAL, true).unwrap();
    assert_eq!(bus.read_long(DESC), PHYS | 1);
}

/// PTESTR #5,(A0),#7: the MMUSR.
fn ptest7(cpu: &mut CpuCore, bus: &mut TestBus, addr: u32) -> u32 {
    pmmu_op(cpu, bus, 0x9E15, addr);
    cpu.mmu_sr
}

const MMUSR_L: u32 = 0x4000;
const MMUSR_I: u32 = 0x0400;

#[test]
fn root_pointer_limit_bounds_the_level_a_index() {
    let (mut cpu, mut bus) = build();
    cpu.mmu_crp_limit = 2; // L/U = 0, LIMIT = 0: index 0 only
    assert_eq!(read(&mut cpu, &mut bus, LOGICAL, true), Ok(PHYS));
    assert!(read(&mut cpu, &mut bus, 0x0300_0010, true).is_err());
    assert_eq!(bus.read_long(TA + 3 * 4), 0x0300_0000 | 1, "not fetched");
    assert_eq!(
        ptest7(&mut cpu, &mut bus, 0x0300_0010) & 0xC400,
        MMUSR_L | MMUSR_I
    );
    assert_eq!(ptest7(&mut cpu, &mut bus, LOGICAL) & 0xC400, 0);

    // L/U = 1, LIMIT = 3: indexes below 3 are out of bounds.
    cpu.mmu_crp_limit = 0x8003_0002;
    cpu.atc_030.flush_all();
    assert!(read(&mut cpu, &mut bus, 0x0300_0010, true).is_ok());
    assert!(read(&mut cpu, &mut bus, 0x0200_0010, true).is_err());
}

/// Level A entry 0 points (short) at a long-format level B table whose
/// entry 0, with `b0` in its first long, points at a long page table.
/// Returns the level B descriptor's address.
fn long_tables(bus: &mut TestBus, b0: u32) -> u32 {
    let tb_long = 0x3000;
    let tc_long = 0x3100;
    bus.write_long(TA, tb_long | 3);
    bus.write_long(tb_long, b0);
    bus.write_long(tb_long + 4, tc_long);
    for i in 0..64u32 {
        bus.write_long(tc_long + i * 8, 0x7FFF_FC00 | 1);
        bus.write_long(tc_long + i * 8 + 4, i << 12);
    }
    tb_long
}

#[test]
fn long_table_descriptor_limit_bounds_the_next_index() {
    // Upper limit $10: level C indexes up to $10 translate.
    let (mut cpu, mut bus) = build();
    long_tables(&mut bus, (0x10 << 16) | 0xFC00 | 3);
    assert_eq!(read(&mut cpu, &mut bus, 0x0001_0010, true), Ok(0x0001_0010));
    assert!(read(&mut cpu, &mut bus, 0x0001_1010, true).is_err());
    assert_eq!(
        ptest7(&mut cpu, &mut bus, 0x0001_1010) & 0xC400,
        MMUSR_L | MMUSR_I
    );
    assert_eq!(ptest7(&mut cpu, &mut bus, 0x0001_0010) & 0xC400, 0);

    // A descriptor whose limit the walk violates is not marked used.
    let (mut cpu, mut bus) = build();
    let b = long_tables(&mut bus, (0x10 << 16) | 0xFC00 | 3);
    assert!(read(&mut cpu, &mut bus, 0x0003_0000, true).is_err());
    assert_eq!(bus.read_long(b) & U, 0);

    // Lower limit $10: indexes below $10 are out of bounds.
    let (mut cpu, mut bus) = build();
    long_tables(&mut bus, 0x8000_0000 | (0x10 << 16) | 0xFC00 | 3);
    assert!(read(&mut cpu, &mut bus, 0x0001_0010, true).is_ok());
    assert!(read(&mut cpu, &mut bus, 0x0000_F010, true).is_err());
}

#[test]
fn long_early_termination_limit_bounds_the_next_index() {
    let (mut cpu, mut bus) = build();
    // Level B entry 0: a long early-termination page descriptor for
    // $00000000-$0003FFFF with upper limit 31 on the level C index.
    let tb_long = 0x3000;
    bus.write_long(TA, tb_long | 3);
    bus.write_long(tb_long, (31 << 16) | 0xFC00 | 1);
    bus.write_long(tb_long + 4, 0);
    assert_eq!(read(&mut cpu, &mut bus, 0x0000_0100, true), Ok(0x0000_0100));
    assert!(read(&mut cpu, &mut bus, 0x0003_E100, true).is_err());
    assert_eq!(
        ptest7(&mut cpu, &mut bus, 0x0003_E100) & 0xC400,
        MMUSR_L | MMUSR_I
    );
}

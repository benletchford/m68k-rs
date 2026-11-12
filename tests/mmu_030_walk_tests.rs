//! 68030 PMMU behaviour against a three-level table (TIA=8, TIB=6, TIC=6,
//! 4 KB pages; MC68030UM section 9): transparent translation through
//! TT0/TT1.

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

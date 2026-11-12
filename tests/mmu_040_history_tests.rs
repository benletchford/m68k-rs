//! 68040/68060 table-search history bits: the search sets U in every root,
//! pointer and page descriptor it uses and M in the page descriptor on a
//! write, writing them back to the tables in memory (MC68040UM 3.2.2-3.2.3,
//! MC68060UM 4.2.2-4.2.3).

use m68k::core::cpu::CpuCore;
use m68k::core::memory::AddressBus;
use m68k::core::types::CpuType;
use m68k::mmu::translate_address;

struct TestBus {
    mem: Vec<u8>,
}

impl TestBus {
    fn new(size: usize) -> Self {
        Self { mem: vec![0; size] }
    }
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

const ROOT: u32 = 0x2000;
const PTR: u32 = 0x3000;
const PAGE: u32 = 0x4000;
const U: u32 = 0x08;
const M: u32 = 0x10;
const W: u32 = 0x04;
const S: u32 = 0x80;
/// A page whose descriptor the tests inspect, and where it maps.
const LOGICAL: u32 = 0x0001_0000;
const PHYS: u32 = 0x0001_8000;
const DESC: u32 = PAGE + (LOGICAL >> 12) * 4;
const MODELS: [CpuType; 2] = [CpuType::M68040, CpuType::M68060];

/// Tables identity-mapping the first 64 KB plus `LOGICAL` -> `PHYS`, with
/// `flags` in `LOGICAL`'s page descriptor.
fn build(cpu_type: CpuType, flags: u32) -> (CpuCore, TestBus) {
    let mut bus = TestBus::new(0x2_0000);
    bus.write_long(ROOT, PTR | 2);
    bus.write_long(PTR, PAGE | 2);
    for i in 0..16 {
        bus.write_long(PAGE + i * 4, (i << 12) | 1);
    }
    bus.write_long(DESC, PHYS | flags | 1);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    cpu.set_sr(0x2700);
    cpu.write_control_register(0x806, ROOT);
    cpu.write_control_register(0x807, ROOT);
    cpu.write_control_register(0x003, 0x0000_8000);
    (cpu, bus)
}

#[test]
fn a_read_sets_used_in_every_level_and_leaves_modified_clear() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model, 0);
        bus.write_long(PHYS + 4, 0x1234_5678);
        assert_eq!(cpu.read_32(&mut bus, LOGICAL + 4), 0x1234_5678);
        assert_eq!(bus.read_long(ROOT), PTR | 2 | U, "{model:?} root U");
        assert_eq!(bus.read_long(PTR), PAGE | 2 | U, "{model:?} pointer U");
        assert_eq!(
            bus.read_long(DESC),
            PHYS | U | 1,
            "{model:?} page U, M clear"
        );
    }
}

#[test]
fn a_write_sets_used_and_modified() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model, 0);
        cpu.write_32(&mut bus, LOGICAL + 8, 0xCAFE_F00D);
        assert_eq!(bus.read_long(PHYS + 8), 0xCAFE_F00D);
        assert_eq!(bus.read_long(DESC), PHYS | U | M | 1, "{model:?}");
    }
}

#[test]
fn a_write_after_a_read_sets_modified_through_the_atc_entry() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model, 0);
        cpu.read_32(&mut bus, LOGICAL + 4);
        assert_eq!(bus.read_long(DESC) & M, 0, "{model:?}: read leaves M clear");
        // The read loaded an ATC entry with M clear; the write must still
        // reach the descriptor.
        cpu.write_32(&mut bus, LOGICAL + 8, 2);
        assert_eq!(bus.read_long(DESC), PHYS | U | M | 1, "{model:?}");
        // Once M is set the entry serves writes: clearing M in memory
        // without a flush is not noticed.
        bus.write_long(DESC, PHYS | U | 1);
        cpu.write_32(&mut bus, LOGICAL + 12, 3);
        assert_eq!(bus.read_long(DESC), PHYS | U | 1, "{model:?}: ATC hit");
        assert_eq!(bus.read_long(PHYS + 12), 3);
    }
}

#[test]
fn an_indirect_page_descriptor_is_left_unchanged_and_its_target_updated() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model, 0);
        let target = 0x5000;
        bus.write_long(target, PHYS | 1);
        bus.write_long(DESC, target | 2); // PDT = 10: indirect
        bus.write_long(PHYS + 0x300, 0x5555_AAAA);
        assert_eq!(cpu.read_32(&mut bus, LOGICAL + 0x300), 0x5555_AAAA);
        assert_eq!(bus.read_long(DESC), target | 2, "{model:?}: indirect");
        assert_eq!(bus.read_long(target), PHYS | U | 1, "{model:?}: target");
    }
}

#[test]
fn a_protection_fault_sets_used_but_not_modified() {
    for model in MODELS {
        // Write-protected page: U is set before the protection check.
        let (mut cpu, mut bus) = build(model, W);
        let r = translate_address(&mut cpu, &mut bus, LOGICAL, true, true, false);
        assert!(r.is_err(), "{model:?}: write to a WP page faults");
        assert_eq!(bus.read_long(DESC), PHYS | W | U | 1, "{model:?}");

        // W in the pointer descriptor protects the page as well.
        let (mut cpu, mut bus) = build(model, 0);
        bus.write_long(PTR, PAGE | W | 2);
        assert!(translate_address(&mut cpu, &mut bus, LOGICAL, true, true, false).is_err());
        assert_eq!(bus.read_long(DESC), PHYS | U | 1, "{model:?}");

        // A user write to a supervisor-only page.
        let (mut cpu, mut bus) = build(model, S);
        cpu.set_sr(0x0700);
        assert!(translate_address(&mut cpu, &mut bus, LOGICAL, true, false, false).is_err());
        assert_eq!(bus.read_long(DESC), PHYS | S | U | 1, "{model:?}");
    }
}

#[test]
fn invalid_descriptors_are_not_updated() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model, 0);
        bus.write_long(DESC, 0x494E_5600); // PDT = 00, OS-owned bits
        assert!(translate_address(&mut cpu, &mut bus, LOGICAL, false, true, false).is_err());
        assert_eq!(bus.read_long(DESC), 0x494E_5600, "{model:?}");
        assert_eq!(bus.read_long(PTR), PAGE | 2 | U, "{model:?}: pointer used");

        let (mut cpu, mut bus) = build(model, 0);
        bus.write_long(PTR, 0x1234_5600); // UDT = 00
        assert!(translate_address(&mut cpu, &mut bus, LOGICAL, false, true, false).is_err());
        assert_eq!(bus.read_long(PTR), 0x1234_5600, "{model:?}");
        assert_eq!(bus.read_long(ROOT), PTR | 2 | U, "{model:?}: root used");
    }
}

#[test]
fn transparent_translation_does_not_touch_the_tables() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model, 0);
        cpu.write_control_register(0x006, 0x0000_C000); // DTT0: all, both
        cpu.write_32(&mut bus, LOGICAL, 1);
        assert_eq!(bus.read_long(ROOT), PTR | 2, "{model:?}");
        assert_eq!(bus.read_long(DESC), PHYS | 1, "{model:?}");
    }
}

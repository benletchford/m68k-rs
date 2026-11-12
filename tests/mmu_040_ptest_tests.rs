//! 68040 PTESTR/PTESTW (An): MMUSR reports the physical page, the page
//! descriptor's G, U1/U0, S, CM and M fields, W accumulated along the path
//! and R for a resident page; T and R for a transparent-translation hit;
//! zero for an invalid translation (MC68040UM 3.5, M68000PRM PTEST).

use m68k::core::cpu::CpuCore;
use m68k::core::memory::AddressBus;
use m68k::core::types::CpuType;

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

const G: u32 = 0x400;
const U0: u32 = 0x100;
const CM_CI: u32 = 0x40;

/// Run `PTESTR (A0)` (or `PTESTW`) with DFC = `fc` and return MMUSR.
fn ptest(cpu: &mut CpuCore, bus: &mut TestBus, addr: u32, fc: u32, write: bool) -> u32 {
    bus.write_word(0x1000, if write { 0xF548 } else { 0xF568 });
    bus.write_word(0x1002, 0x4E71);
    cpu.write_control_register(0x001, fc);
    cpu.dar[8] = addr;
    cpu.pc = 0x1000;
    let mut hle = m68k::NoOpHleHandler;
    cpu.step_with_hle_handler(bus, &mut hle);
    assert_eq!(cpu.pc, 0x1002, "PTEST executes");
    cpu.read_control_register(0x805)
}

#[test]
fn resident_page_reports_its_descriptor_fields() {
    let (mut cpu, mut bus) = build(CpuType::M68040, G | U0 | CM_CI);
    let mmusr = ptest(&mut cpu, &mut bus, LOGICAL + 0x44, 5, false);
    assert_eq!(mmusr, PHYS | G | U0 | CM_CI | 1);
    // The search is a table search: it marks the page used.
    assert_eq!(bus.read_long(DESC), PHYS | G | U0 | CM_CI | U | 1);
}

#[test]
fn write_protection_is_reported_not_faulted() {
    let (mut cpu, mut bus) = build(CpuType::M68040, W);
    let mmusr = ptest(&mut cpu, &mut bus, LOGICAL, 5, true);
    assert_eq!(mmusr, PHYS | W | 1, "PTESTW of a WP page: W and R");
    assert_eq!(
        bus.read_long(DESC) & M,
        0,
        "a protected page is not modified"
    );

    // W in a table descriptor is accumulated into MMUSR W.
    let (mut cpu, mut bus) = build(CpuType::M68040, 0);
    bus.write_long(PTR, PAGE | W | 2);
    assert_eq!(ptest(&mut cpu, &mut bus, LOGICAL, 5, false), PHYS | W | 1);
}

#[test]
fn supervisor_only_page_reports_s() {
    let (mut cpu, mut bus) = build(CpuType::M68040, S);
    assert_eq!(ptest(&mut cpu, &mut bus, LOGICAL, 5, false), PHYS | S | 1);
    // Tested from the user space the page is still resident; S says why a
    // user access would fault.
    assert_eq!(ptest(&mut cpu, &mut bus, LOGICAL, 1, false), PHYS | S | 1);
}

#[test]
fn modified_page_reports_m() {
    let (mut cpu, mut bus) = build(CpuType::M68040, 0);
    cpu.write_32(&mut bus, LOGICAL + 8, 5);
    assert_eq!(ptest(&mut cpu, &mut bus, LOGICAL, 5, false), PHYS | M | 1);
    // PTESTW searches as a write does and sets M itself.
    let (mut cpu, mut bus) = build(CpuType::M68040, 0);
    assert_eq!(ptest(&mut cpu, &mut bus, LOGICAL, 5, true), PHYS | M | 1);
    assert_eq!(bus.read_long(DESC), PHYS | U | M | 1);
}

#[test]
fn invalid_page_reports_zero() {
    let (mut cpu, mut bus) = build(CpuType::M68040, 0);
    bus.write_long(DESC, 0x494E_5600);
    assert_eq!(ptest(&mut cpu, &mut bus, LOGICAL, 5, false), 0);
    assert_eq!(bus.read_long(DESC), 0x494E_5600);
}

#[test]
fn transparent_translation_hit_reports_t_and_r() {
    let (mut cpu, mut bus) = build(CpuType::M68040, 0);
    cpu.write_control_register(0x006, 0x0000_8000); // DTT0: $00xxxxxx, user
    assert_eq!(ptest(&mut cpu, &mut bus, LOGICAL + 0x10, 1, false), 0x3);
    // DFC = 5 does not match the user-only register: the tables answer.
    assert_eq!(
        ptest(&mut cpu, &mut bus, LOGICAL + 0x10, 5, false),
        PHYS | 1
    );
}

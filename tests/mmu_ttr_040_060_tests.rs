//! 68040/68060 transparent translation: ITT0/ITT1 match instruction
//! fetches, DTT0/DTT1 data accesses, each by logical address bits 31-24
//! (with the mask) and the S field (00 user only, 01 supervisor only, 1x
//! both). A matching access is not translated (MC68040UM 3.1.3, MC68060UM
//! 4.1.3). Both parts use the same register layout, so every case runs on
//! both.

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
/// The logical page the tables remap.
const LOGICAL: u32 = 0x0001_0000;
/// Where the tables send it.
const PHYS: u32 = 0x0001_8000;

/// Tables (shared by URP and SRP) identity-mapping the first 64 KB and
/// sending `LOGICAL` to `PHYS`.
fn build(cpu_type: CpuType) -> (CpuCore, TestBus) {
    let mut bus = TestBus::new(0x2_0000);
    bus.write_long(ROOT, PTR | 2);
    bus.write_long(PTR, PAGE | 2);
    for i in 0..16 {
        bus.write_long(PAGE + i * 4, (i << 12) | 1);
    }
    bus.write_long(PAGE + (LOGICAL >> 12) * 4, PHYS | 1);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.reset(&mut bus);
    cpu.set_sr(0x2700);
    cpu.write_control_register(0x806, ROOT);
    cpu.write_control_register(0x807, ROOT);
    cpu.write_control_register(0x003, 0x0000_8000);
    (cpu, bus)
}

/// TTR value: base, mask, E, S field, W.
fn ttr(base: u32, mask: u32, s: u32, w: bool) -> u32 {
    (base << 24) | (mask << 16) | 0x8000 | (s << 13) | if w { 4 } else { 0 }
}

/// Translate one ordinary access of `LOGICAL + 0x500` in the given mode.
fn xlate(cpu: &mut CpuCore, bus: &mut TestBus, supervisor: bool, instruction: bool) -> u32 {
    cpu.set_sr(if supervisor { 0x2700 } else { 0x0700 });
    let r = translate_address(cpu, bus, LOGICAL + 0x500, false, supervisor, instruction);
    cpu.set_sr(0x2700);
    r.expect("translation")
}

const MODELS: [CpuType; 2] = [CpuType::M68040, CpuType::M68060];
const IDENT: u32 = LOGICAL + 0x500;
const TABLE: u32 = PHYS + 0x500;

#[test]
fn dtt_s_field_selects_user_supervisor_or_both() {
    for model in MODELS {
        for (reg, s, user, sup) in [
            (0x006, 0, IDENT, TABLE), // DTT0, user only
            (0x007, 0, IDENT, TABLE), // DTT1, user only
            (0x006, 1, TABLE, IDENT), // supervisor only
            (0x006, 2, IDENT, IDENT), // both
            (0x007, 3, IDENT, IDENT), // both (S=1x)
        ] {
            let (mut cpu, mut bus) = build(model);
            cpu.write_control_register(reg, ttr(0x00, 0, s, false));
            assert_eq!(
                xlate(&mut cpu, &mut bus, false, false),
                user,
                "{model:?} reg {reg:#x} S={s}: user data"
            );
            assert_eq!(
                xlate(&mut cpu, &mut bus, true, false),
                sup,
                "{model:?} reg {reg:#x} S={s}: supervisor data"
            );
        }
    }
}

#[test]
fn dtt_address_mask_ignores_the_masked_base_bits() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model);
        // Base $01 with bit 24 masked matches $00xxxxxx.
        cpu.write_control_register(0x006, ttr(0x01, 0x01, 2, false));
        assert_eq!(xlate(&mut cpu, &mut bus, true, false), IDENT, "{model:?}");
        // Unmasked, $01 does not match $00xxxxxx.
        cpu.write_control_register(0x006, ttr(0x01, 0x00, 2, false));
        assert_eq!(xlate(&mut cpu, &mut bus, true, false), TABLE, "{model:?}");
    }
}

#[test]
fn itt_matches_instruction_fetches_and_dtt_only_data() {
    for model in MODELS {
        for reg in [0x004, 0x005] {
            let (mut cpu, mut bus) = build(model);
            cpu.write_control_register(reg, ttr(0x00, 0, 2, false));
            assert_eq!(xlate(&mut cpu, &mut bus, true, true), IDENT, "{model:?}");
            assert_eq!(xlate(&mut cpu, &mut bus, true, false), TABLE, "{model:?}");
        }
        let (mut cpu, mut bus) = build(model);
        cpu.write_control_register(0x006, ttr(0x00, 0, 2, false));
        assert_eq!(xlate(&mut cpu, &mut bus, true, true), TABLE, "{model:?}");
    }
}

#[test]
fn plpar_of_a_dtt_matched_address_leaves_an_unchanged() {
    let (mut cpu, mut bus) = build(CpuType::M68060);
    cpu.write_control_register(0x006, ttr(0x00, 0, 0, false)); // user only
    cpu.write_control_register(0x001, 1); // DFC = user data
    bus.write_word(0x1000, 0xF5C8); // PLPAR (A0)
    bus.write_word(0x1002, 0x4E71);
    let mut hle = m68k::NoOpHleHandler;
    cpu.dar[8] = LOGICAL + 0x14;
    cpu.pc = 0x1000;
    cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert_eq!(cpu.pc, 0x1002);
    assert_eq!(cpu.dar[8], LOGICAL + 0x14, "transparent: no translation");

    // DFC = supervisor data does not match the user-only DTT0.
    cpu.write_control_register(0x001, 5);
    cpu.pc = 0x1000;
    cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert_eq!(cpu.dar[8], PHYS + 0x14, "translated through the tables");
}

/// Run `MOVE.L D1,(A0)` in user mode with A0 = `addr`; return the
/// supervisor stack pointer after the step (the frame, if one was taken).
fn user_write(cpu: &mut CpuCore, bus: &mut TestBus, addr: u32) -> u32 {
    bus.write_long(0x8, 0x1800); // vector 2: access fault handler
    bus.write_word(0x1000, 0x2081); // MOVE.L D1,(A0)
    bus.write_word(0x1002, 0x4E71);
    bus.write_word(0x1800, 0x4E71);
    cpu.dar[15] = 0x9000; // SSP
    cpu.set_sr(0x0000); // user mode
    cpu.dar[15] = 0xA000; // USP
    cpu.dar[8] = addr;
    cpu.dar[1] = 0x1122_3344;
    cpu.pc = 0x1000;
    let mut hle = m68k::NoOpHleHandler;
    cpu.step_with_hle_handler(bus, &mut hle);
    cpu.dar[15]
}

#[test]
fn write_through_a_write_protected_dtt_takes_an_access_fault() {
    for model in MODELS {
        let (mut cpu, mut bus) = build(model);
        bus.write_long(LOGICAL + 0x50c, 0xAAAA_AAAA);
        bus.write_long(PHYS + 0x50c, 0xBBBB_BBBB);
        // DTT0: user accesses, W=1. Supervisor stack pushes do not match.
        cpu.write_control_register(0x006, ttr(0x00, 0, 0, true));
        let sp = user_write(&mut cpu, &mut bus, LOGICAL + 0x50c);
        assert_eq!(cpu.pc, 0x1800, "{model:?}: vector 2 taken");
        assert_eq!(bus.read_long(LOGICAL + 0x50c), 0xAAAA_AAAA, "{model:?}");
        assert_eq!(bus.read_long(PHYS + 0x50c), 0xBBBB_BBBB, "{model:?}");
        if model == CpuType::M68060 {
            assert_eq!(bus.read_word(sp + 6), 0x4008, "060 format $4");
            assert_eq!(bus.read_long(sp + 8), LOGICAL + 0x50c, "fault address");
            let fslw = bus.read_long(sp + 12);
            // RW = write, SIZE = long, TM = user data, TTR hit.
            assert_eq!(fslw & !0x84, 0x0081_0008, "FSLW {fslw:#010x}");
        } else {
            assert_eq!(bus.read_word(sp + 6), 0x7008, "040 format 7");
            assert_eq!(bus.read_long(sp + 20), LOGICAL + 0x50c, "fault address");
            // SSW: write, long, TM = user data.
            assert_eq!(bus.read_word(sp + 12) & 0x017f, 0x0001, "040 SSW");
        }

        // A read through the same register is transparent and does not fault.
        cpu.set_sr(0x0700);
        let r = translate_address(&mut cpu, &mut bus, LOGICAL + 0x50c, false, false, false);
        assert_eq!(r, Ok(LOGICAL + 0x50c), "{model:?}");
        // W=0: the write goes through untranslated.
        cpu.set_sr(0x2700);
        cpu.write_control_register(0x006, ttr(0x00, 0, 0, false));
        user_write(&mut cpu, &mut bus, LOGICAL + 0x50c);
        assert_eq!(cpu.pc, 0x1002, "{model:?}: no fault");
        assert_eq!(bus.read_long(LOGICAL + 0x50c), 0x1122_3344, "{model:?}");
    }
}

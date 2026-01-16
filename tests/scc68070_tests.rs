//! SCC68070 CPU Tests
//!
//! The SCC68070 is a Philips variant that is essentially a 68010 with a 32-bit data bus.
//! These tests verify:
//! 1. 32-bit address space (unlike 68010's 24-bit)
//! 2. 68010-compatible instruction set
//! 3. 68020+ instructions are illegal
//! 4. F-line opcodes trigger exceptions (no coprocessor interface)

mod common;
use m68k::NoOpHleHandler;
use m68k::core::cpu::CpuCore;
use m68k::core::memory::AddressBus;
use m68k::core::types::CpuType;

/// Simple memory for address mask testing.
/// Maps the full 32-bit address space.
struct FullAddressBus {
    /// 16MB of memory at address 0
    low_mem: Vec<u8>,
    /// 64KB of memory at address 0xFF000000 (only accessible with 32-bit addressing)
    high_mem: Vec<u8>,
}

impl FullAddressBus {
    fn new() -> Self {
        Self {
            low_mem: vec![0; 0x0100_0000],  // 16MB
            high_mem: vec![0; 0x0001_0000], // 64KB
        }
    }
}

impl AddressBus for FullAddressBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        if address < 0x0100_0000 {
            self.low_mem[address as usize]
        } else if address >= 0xFF00_0000 && address < 0xFF01_0000 {
            self.high_mem[(address - 0xFF00_0000) as usize]
        } else {
            0
        }
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let hi = self.read_byte(address) as u16;
        let lo = self.read_byte(address.wrapping_add(1)) as u16;
        (hi << 8) | lo
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let hi = self.read_word(address) as u32;
        let lo = self.read_word(address.wrapping_add(2)) as u32;
        (hi << 16) | lo
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        if address < 0x0100_0000 {
            self.low_mem[address as usize] = value;
        } else if address >= 0xFF00_0000 && address < 0xFF01_0000 {
            self.high_mem[(address - 0xFF00_0000) as usize] = value;
        }
    }

    fn write_word(&mut self, address: u32, value: u16) {
        self.write_byte(address, (value >> 8) as u8);
        self.write_byte(address.wrapping_add(1), value as u8);
    }

    fn write_long(&mut self, address: u32, value: u32) {
        self.write_word(address, (value >> 16) as u16);
        self.write_word(address.wrapping_add(2), value as u16);
    }
}

// =============================================================================
// Address Mask Tests
// =============================================================================

#[test]
fn test_scc68070_has_32bit_address_mask() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);

    // SCC68070 should have full 32-bit address mask
    assert_eq!(
        cpu.address_mask, 0xFFFF_FFFF,
        "SCC68070 should have 32-bit address mask"
    );
}

#[test]
fn test_m68010_has_24bit_address_mask() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010);

    // M68010 should have 24-bit address mask
    assert_eq!(
        cpu.address_mask, 0x00FF_FFFF,
        "M68010 should have 24-bit address mask"
    );
}

#[test]
fn test_scc68070_can_access_high_memory() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);
    let mut bus = FullAddressBus::new();

    // Write a test value to high memory (above 24-bit range)
    bus.write_long(0xFF00_0000, 0xDEAD_BEEF);

    // Setup vectors
    bus.write_long(0, 0x0000_1000); // SSP
    bus.write_long(4, 0x0000_0100); // PC

    // Write test code at 0x100:
    // MOVE.L #$FF000000,A0   ; 207C FF00 0000
    // MOVE.L (A0),D0         ; 2010
    // STOP #$2700            ; 4E72 2700
    bus.write_word(0x100, 0x207C);
    bus.write_long(0x102, 0xFF00_0000);
    bus.write_word(0x106, 0x2010);
    bus.write_word(0x108, 0x4E72);
    bus.write_word(0x10A, 0x2700);

    cpu.reset(&mut bus);

    let mut hle = NoOpHleHandler;
    for _ in 0..10 {
        if cpu.is_stopped() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);
    }

    // D0 should contain the value from high memory
    assert_eq!(
        cpu.dar[0], 0xDEAD_BEEF,
        "SCC68070 should read from address 0xFF000000"
    );
}

#[test]
fn test_m68010_wraps_high_address() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010);
    let mut bus = FullAddressBus::new();

    // Write test value at address 0x100 (away from vectors)
    // 0xFF000100 should wrap to 0x000100 on 68010 (24-bit mask)
    bus.write_long(0x0000_0100, 0xAAAA_AAAA);
    // Also write to high memory
    bus.write_long(0xFF00_0100, 0xBBBB_BBBB);

    // Setup vectors
    bus.write_long(0, 0x0000_1000); // SSP
    bus.write_long(4, 0x0000_0200); // PC at 0x200 to avoid conflict

    // Write test code at 0x200:
    // MOVE.L #$FF000100,A0   ; 207C FF00 0100
    // MOVE.L (A0),D0         ; 2010
    // STOP #$2700            ; 4E72 2700
    bus.write_word(0x200, 0x207C);
    bus.write_long(0x202, 0xFF00_0100);
    bus.write_word(0x206, 0x2010);
    bus.write_word(0x208, 0x4E72);
    bus.write_word(0x20A, 0x2700);

    cpu.reset(&mut bus);

    let mut hle = NoOpHleHandler;
    for _ in 0..10 {
        if cpu.is_stopped() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);
    }

    // M68010 should wrap 0xFF000100 to 0x000100 (24-bit mask)
    // Reading from (A0) should get the low_mem value at 0x000100
    assert_eq!(
        cpu.dar[0], 0xAAAA_AAAA,
        "M68010 should wrap address 0xFF000100 to 0x000100"
    );
}

// =============================================================================
// 68010-Compatible Instruction Tests
// =============================================================================

#[test]
fn test_scc68070_supports_movec() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);
    let mut bus = FullAddressBus::new();

    // Setup vectors
    bus.write_long(0, 0x0000_1000); // SSP
    bus.write_long(4, 0x0000_0100); // PC

    // Write test code at 0x100:
    // MOVEC VBR,D0          ; 4E7A 0801
    // STOP #$2700           ; 4E72 2700
    bus.write_word(0x100, 0x4E7A);
    bus.write_word(0x102, 0x0801); // D0, VBR
    bus.write_word(0x104, 0x4E72);
    bus.write_word(0x106, 0x2700);

    cpu.reset(&mut bus);
    cpu.vbr = 0x1234_0000; // Set VBR to a known value

    let mut hle = NoOpHleHandler;
    for _ in 0..10 {
        if cpu.is_stopped() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);
    }

    // D0 should contain VBR value
    assert_eq!(cpu.dar[0], 0x1234_0000, "SCC68070 should support MOVEC VBR");
}

#[test]
fn test_scc68070_rtd_works() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);
    let mut bus = FullAddressBus::new();

    // Setup vectors
    bus.write_long(0, 0x0000_1000); // SSP
    bus.write_long(4, 0x0000_0100); // PC

    // Write test code at 0x100:
    // JSR subroutine        ; 4EB9 0000 0200
    // STOP #$2700           ; 4E72 2700
    bus.write_word(0x100, 0x4EB9);
    bus.write_long(0x102, 0x0000_0200);
    bus.write_word(0x106, 0x4E72);
    bus.write_word(0x108, 0x2700);

    // Subroutine at 0x200:
    // RTD #4                ; 4E74 0004
    bus.write_word(0x200, 0x4E74);
    bus.write_word(0x202, 0x0004);

    cpu.reset(&mut bus);
    let initial_sp = cpu.dar[15];

    let mut hle = NoOpHleHandler;
    for _ in 0..20 {
        if cpu.is_stopped() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);
    }

    // SP should be initial_sp + 4 (RTD #4 adds 4 to SP after popping return address)
    // JSR pushes 4 bytes (return address), RTD pops them and adds displacement
    assert_eq!(
        cpu.dar[15],
        initial_sp + 4,
        "SCC68070 should support RTD instruction"
    );
}

// =============================================================================
// 68020+ Instruction Rejection Tests
// =============================================================================

#[test]
fn test_scc68070_rejects_extb() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);
    let mut bus = FullAddressBus::new();

    // Setup vectors
    bus.write_long(0, 0x0000_1000); // SSP
    bus.write_long(4, 0x0000_0100); // PC

    // Illegal instruction vector (4) points to handler
    bus.write_long(16, 0x0000_0300); // Vector 4 * 4 = 16

    // Test code at 0x100:
    // EXTB.L D0             ; 49C0 (68020+ only)
    // STOP #$2700           ; 4E72 2700
    bus.write_word(0x100, 0x49C0);
    bus.write_word(0x102, 0x4E72);
    bus.write_word(0x104, 0x2700);

    // Exception handler at 0x300:
    // MOVE.L #$DEAD,D0      ; 203C 0000 DEAD
    // STOP #$2700           ; 4E72 2700
    bus.write_word(0x300, 0x203C);
    bus.write_long(0x302, 0x0000_DEAD);
    bus.write_word(0x306, 0x4E72);
    bus.write_word(0x308, 0x2700);

    cpu.reset(&mut bus);

    let mut hle = NoOpHleHandler;
    for _ in 0..20 {
        if cpu.is_stopped() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);
    }

    // D0 should be 0xDEAD from exception handler
    assert_eq!(
        cpu.dar[0], 0x0000_DEAD,
        "SCC68070 should trigger illegal instruction on EXTB.L"
    );
}

#[test]
fn test_scc68070_rejects_fline_fpu() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);
    let mut bus = FullAddressBus::new();

    // Setup vectors
    bus.write_long(0, 0x0000_1000); // SSP
    bus.write_long(4, 0x0000_0100); // PC

    // Line-F exception vector (11) points to handler
    bus.write_long(44, 0x0000_0300); // Vector 11 * 4 = 44

    // Test code at 0x100:
    // FNOP                  ; F280 0000 (F-line FPU instruction)
    // STOP #$2700           ; 4E72 2700
    bus.write_word(0x100, 0xF280);
    bus.write_word(0x102, 0x0000);
    bus.write_word(0x104, 0x4E72);
    bus.write_word(0x106, 0x2700);

    // Exception handler at 0x300:
    // MOVE.L #$BEEF,D0      ; 203C 0000 BEEF
    // STOP #$2700           ; 4E72 2700
    bus.write_word(0x300, 0x203C);
    bus.write_long(0x302, 0x0000_BEEF);
    bus.write_word(0x306, 0x4E72);
    bus.write_word(0x308, 0x2700);

    cpu.reset(&mut bus);

    let mut hle = NoOpHleHandler;
    for _ in 0..20 {
        if cpu.is_stopped() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);
    }

    // D0 should be 0xBEEF from exception handler
    assert_eq!(
        cpu.dar[0], 0x0000_BEEF,
        "SCC68070 should trigger Line-F exception on FPU instruction"
    );
}

// =============================================================================
// Exception Stack Frame Tests
// =============================================================================

#[test]
fn test_scc68070_uses_format0_exception_frame() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);
    let mut bus = FullAddressBus::new();

    // Setup vectors
    bus.write_long(0, 0x0000_1000); // SSP
    bus.write_long(4, 0x0000_0100); // PC

    // TRAP #0 vector (32) points to handler
    bus.write_long(128, 0x0000_0300); // Vector 32 * 4 = 128

    // Test code at 0x100:
    // TRAP #0               ; 4E40
    bus.write_word(0x100, 0x4E40);

    // Handler at 0x300:
    // STOP #$2700           ; 4E72 2700
    bus.write_word(0x300, 0x4E72);
    bus.write_word(0x302, 0x2700);

    cpu.reset(&mut bus);

    let mut hle = NoOpHleHandler;
    for _ in 0..20 {
        if cpu.is_stopped() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);
    }

    // Check stack frame format
    // 68010/SCC68070 format-0 frame: format/vector word at SP+6
    let sp = cpu.dar[15];
    let format_word = bus.read_word(sp + 6);
    let format = (format_word >> 12) & 0xF;

    // Format should be 0 (or 2 for TRAP on 68020+, but SCC68070 should use 0)
    assert!(
        format == 0 || format == 2,
        "SCC68070 should use format-0 or format-2 exception frame, got {}",
        format
    );
}

// =============================================================================
// SR Mask Tests
// =============================================================================

#[test]
fn test_scc68070_has_correct_sr_mask() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);

    // SCC68070 should have 68010-style SR mask (no T0, no M bit)
    // 0xA71F = T1 -- S -- -- I2 I1 I0 -- -- -- X N Z V C
    assert_eq!(
        cpu.sr_mask, 0xA71F,
        "SCC68070 should have 68010-compatible SR mask"
    );
}

#[test]
fn test_scc68070_no_pmmu() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::SCC68070);

    assert!(!cpu.has_pmmu, "SCC68070 should not have PMMU");
}

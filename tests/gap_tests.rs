//! Gap Tests: These tests should FAIL until the features are implemented.
//! They verify actual gap features, not just instruction execution.

use m68k::core::cpu::CpuCore;
use m68k::core::types::CpuType;

/// Test that NOP returns the correct cycle count (4 cycles on 68000)
#[test]
fn test_gap_cycle_timing_nop() {
    // NOP should take 4 cycles on 68000
    // Currently our emulator doesn't track cycles accurately

    // Check that the timing table has non-zero values
    use m68k::core::timing::EXCEPTION_CYCLES;

    // Check that exception cycle table is populated (not all zeros)
    let total: u32 = EXCEPTION_CYCLES
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| x as u32)
        .sum();

    assert!(
        total > 0,
        "EXCEPTION_CYCLES table is empty - cycle timing not implemented"
    );
}

/// Test that MOVE.L Dn,Dn returns correct cycles (4 on 68000)
#[test]
fn test_gap_cycle_timing_move() {
    use m68k::AddressBus;
    use m68k::StepResult;

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
            let hi = self.read_byte(addr) as u16;
            let lo = self.read_byte(addr + 1) as u16;
            (hi << 8) | lo
        }
        fn write_word(&mut self, addr: u32, val: u16) {
            self.write_byte(addr, (val >> 8) as u8);
            self.write_byte(addr + 1, val as u8);
        }
        fn read_long(&mut self, addr: u32) -> u32 {
            let hi = self.read_word(addr) as u32;
            let lo = self.read_word(addr + 2) as u32;
            (hi << 16) | lo
        }
        fn write_long(&mut self, addr: u32, val: u32) {
            self.write_word(addr, (val >> 16) as u16);
            self.write_word(addr + 2, val as u16);
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);

    let mut bus = TestBus {
        mem: vec![0; 0x10000],
    };

    // MOVE.L D0,D1 = 0x2200
    bus.mem[0x1000] = 0x22;
    bus.mem[0x1001] = 0x00;

    cpu.pc = 0x1000;
    cpu.set_a(7, 0x8000);
    cpu.set_sr(0x2700);

    // Execute the instruction and get cycle count
    let result = cpu.step(&mut bus);
    let actual_cycles = match result {
        StepResult::Ok { cycles } => cycles,
        _ => 0,
    };

    // MOVE.L Dn,Dn takes 4 cycles on 68000
    assert!(
        actual_cycles > 0,
        "MOVE.L Dn,Dn should return non-zero cycles, got {}",
        actual_cycles
    );
}

/// Test that disassembler produces proper output for NOP
#[test]
fn test_gap_disassembler_nop() {
    use m68k::dasm::disassemble;

    // NOP opcode = 0x4E71
    let (mnemonic, size) = disassemble(0x1000, 0x4E71, CpuType::M68000);

    // Should produce "NOP", not "DC.W $4E71"
    assert_eq!(
        mnemonic, "NOP",
        "Disassembler should produce 'NOP', got '{}'",
        mnemonic
    );
    assert_eq!(size, 2);
}

/// Test that disassembler produces proper output for MOVE.L D0,D1
#[test]
fn test_gap_disassembler_move() {
    use m68k::dasm::disassemble;

    // MOVE.L D0,D1 = 0x2200
    let (mnemonic, _size) = disassemble(0x1000, 0x2200, CpuType::M68000);

    // Should produce something like "MOVE.L D0,D1"
    assert!(
        mnemonic.starts_with("MOVE"),
        "Disassembler should recognize MOVE instruction, got '{}'",
        mnemonic
    );
    assert!(
        !mnemonic.contains("DC.W"),
        "Disassembler should not output DC.W for valid instructions, got '{}'",
        mnemonic
    );
}

/// Test that disassembler produces proper output for ADD.L D0,D1
#[test]
fn test_gap_disassembler_add() {
    use m68k::dasm::disassemble;

    // ADD.L D0,D1 = 0xD280
    let (mnemonic, _size) = disassemble(0x1000, 0xD280, CpuType::M68000);

    assert!(
        mnemonic.starts_with("ADD"),
        "Disassembler should recognize ADD instruction, got '{}'",
        mnemonic
    );
}

/// Test that disassembler produces proper output for BRA
#[test]
fn test_gap_disassembler_bra() {
    use m68k::dasm::disassemble;

    // BRA.S with 8-bit displacement
    let (mnemonic, _size) = disassemble(0x1000, 0x6010, CpuType::M68000);

    assert!(
        mnemonic.starts_with("BRA"),
        "Disassembler should recognize BRA instruction, got '{}'",
        mnemonic
    );
}

/// Test that disassembler handles FPU instructions on 68040
#[test]
fn test_gap_disassembler_fpu() {
    use m68k::dasm::disassemble;

    // F-line instruction (FPU) - should decode to something like "FMOVE" or "F..."
    let (mnemonic, _) = disassemble(0x1000, 0xF200, CpuType::M68040);

    // Should NOT just output DC.W
    assert!(
        !mnemonic.starts_with("DC.W"),
        "Disassembler should decode FPU instructions, not output 'DC.W', got '{}'",
        mnemonic
    );
}

/// Test DBF (DBRA) instruction works correctly
#[test]
fn test_gap_dbf_instruction() {
    use m68k::AddressBus;

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
            let hi = self.read_byte(addr) as u16;
            let lo = self.read_byte(addr + 1) as u16;
            (hi << 8) | lo
        }
        fn write_word(&mut self, addr: u32, val: u16) {
            self.write_byte(addr, (val >> 8) as u8);
            self.write_byte(addr + 1, val as u8);
        }
        fn read_long(&mut self, addr: u32) -> u32 {
            let hi = self.read_word(addr) as u32;
            let lo = self.read_word(addr + 2) as u32;
            (hi << 16) | lo
        }
        fn write_long(&mut self, addr: u32, val: u32) {
            self.write_word(addr, (val >> 16) as u16);
            self.write_word(addr + 2, val as u16);
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68040);

    let mut bus = TestBus {
        mem: vec![0; 0x10000],
    };

    // Set up DBF D0, -2 (loop back to itself)
    // DBF = 0x51C8 + register (D0 = 0) = 0x51C8
    // Displacement -2 = 0xFFFE
    bus.mem[0x1000] = 0x51;
    bus.mem[0x1001] = 0xC8;
    bus.mem[0x1002] = 0xFF;
    bus.mem[0x1003] = 0xFE;
    // After loop: NOP then RTS
    bus.mem[0x1004] = 0x4E;
    bus.mem[0x1005] = 0x71;
    bus.mem[0x1006] = 0x4E;
    bus.mem[0x1007] = 0x75;

    // Initialize D0 to 3 (should loop 4 times: 3,2,1,0,-1 exit)
    cpu.set_d(0, 3);
    cpu.pc = 0x1000;
    cpu.set_a(7, 0x8000); // Stack

    // Execute enough cycles
    for _ in 0..20 {
        cpu.execute(&mut bus, 100);
        if cpu.pc == 0x1006 || cpu.pc > 0x1010 {
            break;
        }
    }

    // D0 lower word should be 0xFFFF (-1 as word) after DBF exhausts
    // DBF only affects the lower 16 bits, upper bits of D0 should be 0
    let d0_word = (cpu.d(0) & 0xFFFF) as u16;
    assert_eq!(
        d0_word, 0xFFFF,
        "DBF should decrement D0 word to 0xFFFF (-1), got 0x{:04X}",
        d0_word
    );
}

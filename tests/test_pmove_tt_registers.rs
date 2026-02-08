use m68k::core::memory::AddressBus;
use m68k::{CpuCore, CpuType, NoOpHleHandler, StepResult};

struct TestBus {
    memory: [u8; 0x10000],
}

impl TestBus {
    fn new() -> Self {
        Self {
            memory: [0; 0x10000],
        }
    }

    fn write_word_at(&mut self, addr: u32, value: u16) {
        let bytes = value.to_be_bytes();
        let idx = addr as usize;
        self.memory[idx] = bytes[0];
        self.memory[idx + 1] = bytes[1];
    }

    fn write_long_at(&mut self, addr: u32, value: u32) {
        let bytes = value.to_be_bytes();
        let idx = addr as usize;
        self.memory[idx] = bytes[0];
        self.memory[idx + 1] = bytes[1];
        self.memory[idx + 2] = bytes[2];
        self.memory[idx + 3] = bytes[3];
    }

    fn read_long_at(&self, addr: u32) -> u32 {
        let idx = addr as usize;
        u32::from_be_bytes([
            self.memory[idx],
            self.memory[idx + 1],
            self.memory[idx + 2],
            self.memory[idx + 3],
        ])
    }
}

impl AddressBus for TestBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[(address as usize) & 0xFFFF]
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let addr = (address as usize) & 0xFFFF;
        u16::from_be_bytes([self.memory[addr], self.memory[addr + 1]])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let addr = (address as usize) & 0xFFFF;
        u32::from_be_bytes([
            self.memory[addr],
            self.memory[addr + 1],
            self.memory[addr + 2],
            self.memory[addr + 3],
        ])
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[(address as usize) & 0xFFFF] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        let addr = (address as usize) & 0xFFFF;
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[addr + 1] = bytes[1];
    }

    fn write_long(&mut self, address: u32, value: u32) {
        let addr = (address as usize) & 0xFFFF;
        let bytes = value.to_be_bytes();
        self.memory[addr] = bytes[0];
        self.memory[addr + 1] = bytes[1];
        self.memory[addr + 2] = bytes[2];
        self.memory[addr + 3] = bytes[3];
    }
}

/// Helper: set up a 68030 CPU in supervisor mode with vectors.
fn setup_cpu_and_bus() -> (CpuCore, TestBus) {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68030);
    let mut bus = TestBus::new();

    // Vectors: SSP=0x1000, PC=0x0200
    bus.write_long_at(0x00, 0x1000);
    bus.write_long_at(0x04, 0x0200);

    cpu.reset(&mut bus);
    cpu.pc = 0x0200;
    cpu.set_sr(0x2700); // supervisor mode

    (cpu, bus)
}

/// Test PMOVE (addr).L, TT0 — extension word 0x0800
///
/// The 68030 PMOVE format 2 uses bits 15-13 = 000, bits 12-10 = 010 for TT0.
/// The full 5-bit preg = 0x02 (TT0). With a 3-bit mask this is incorrectly
/// decoded as regsel=2 which is SRP (64-bit), causing 8 bytes to be read
/// instead of the correct 4 bytes.
#[test]
fn test_pmove_tt0_load_reads_32_bits_not_64() {
    let (mut cpu, mut bus) = setup_cpu_and_bus();

    // Place TT0 data at address 0x2000 (32-bit value).
    let tt0_value: u32 = 0x0000_8021;
    bus.write_long_at(0x2000, tt0_value);
    // Place a sentinel value right after (should NOT be read).
    bus.write_long_at(0x2004, 0xDEAD_BEEF);

    // A0 = 0x2000
    cpu.dar[8] = 0x2000;

    // PMOVE (A0), TT0:
    //   opcode = 0xF010  (coprocessor id=0, EA mode = (A0))
    //   extension = 0x0800  (format 2: bits 15-13=000, preg TT0: bits 12-10=010, R/W=0 -> to reg)
    bus.write_word_at(0x0200, 0xF010);
    bus.write_word_at(0x0202, 0x0800);

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(result, StepResult::Ok { .. }));

    // TT0 should be loaded with the 32-bit value.
    assert_eq!(
        cpu.mmu_tt0, tt0_value,
        "PMOVE (A0), TT0 should load 32-bit TT0 register, got 0x{:08X}",
        cpu.mmu_tt0
    );
    // SRP should NOT have been touched (it would be if regsel was decoded as 2 with 3-bit mask).
    assert_eq!(
        cpu.mmu_srp_limit, 0,
        "SRP limit should be untouched after PMOVE TT0, got 0x{:08X}",
        cpu.mmu_srp_limit
    );
    assert_eq!(
        cpu.mmu_srp_aptr, 0,
        "SRP aptr should be untouched after PMOVE TT0, got 0x{:08X}",
        cpu.mmu_srp_aptr
    );
}

/// Test PMOVE (addr).L, TT1 — extension word 0x0C00
///
/// Same issue: 5-bit preg = 0x03 (TT1), but 3-bit mask gives regsel=3 which
/// is CRP (64-bit), causing 8 bytes to be read instead of 4.
#[test]
fn test_pmove_tt1_load_reads_32_bits_not_64() {
    let (mut cpu, mut bus) = setup_cpu_and_bus();

    // Place TT1 data at address 0x2000 (32-bit value).
    let tt1_value: u32 = 0x0000_8043;
    bus.write_long_at(0x2000, tt1_value);
    // Sentinel after — should NOT be read.
    bus.write_long_at(0x2004, 0xCAFE_BABE);

    // A0 = 0x2000
    cpu.dar[8] = 0x2000;

    // PMOVE (A0), TT1:
    //   opcode = 0xF010
    //   extension = 0x0C00  (format 2: bits 15-13=000, preg TT1: bits 12-10=011, R/W=0)
    bus.write_word_at(0x0200, 0xF010);
    bus.write_word_at(0x0202, 0x0C00);

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(result, StepResult::Ok { .. }));

    assert_eq!(
        cpu.mmu_tt1, tt1_value,
        "PMOVE (A0), TT1 should load 32-bit TT1 register, got 0x{:08X}",
        cpu.mmu_tt1
    );
    // CRP should NOT have been touched.
    assert_eq!(
        cpu.mmu_crp_limit, 0,
        "CRP limit should be untouched after PMOVE TT1, got 0x{:08X}",
        cpu.mmu_crp_limit
    );
    assert_eq!(
        cpu.mmu_crp_aptr, 0,
        "CRP aptr should be untouched after PMOVE TT1, got 0x{:08X}",
        cpu.mmu_crp_aptr
    );
}

/// Test PMOVE TT0, (addr).L — extension word 0x0A00 (store TT0 to memory)
#[test]
fn test_pmove_tt0_store_writes_32_bits() {
    let (mut cpu, mut bus) = setup_cpu_and_bus();

    let tt0_value: u32 = 0x1234_5678;
    cpu.mmu_tt0 = tt0_value;

    // A0 = 0x2000
    cpu.dar[8] = 0x2000;
    // Clear destination.
    bus.write_long_at(0x2000, 0);
    bus.write_long_at(0x2004, 0);

    // PMOVE TT0, (A0):
    //   opcode = 0xF010
    //   extension = 0x0A00  (format 2, preg TT0, R/W=1 -> to EA)
    bus.write_word_at(0x0200, 0xF010);
    bus.write_word_at(0x0202, 0x0A00);

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(result, StepResult::Ok { .. }));

    assert_eq!(
        bus.read_long_at(0x2000),
        tt0_value,
        "PMOVE TT0, (A0) should store 32-bit TT0 to memory"
    );
    // Next 4 bytes should remain zero (not written as if 64-bit SRP).
    assert_eq!(
        bus.read_long_at(0x2004),
        0,
        "Only 4 bytes should be written for TT0, not 8"
    );
}

/// Test PMOVE TT1, (addr).L — extension word 0x0E00 (store TT1 to memory)
#[test]
fn test_pmove_tt1_store_writes_32_bits() {
    let (mut cpu, mut bus) = setup_cpu_and_bus();

    let tt1_value: u32 = 0xABCD_EF01;
    cpu.mmu_tt1 = tt1_value;

    // A0 = 0x2000
    cpu.dar[8] = 0x2000;
    bus.write_long_at(0x2000, 0);
    bus.write_long_at(0x2004, 0);

    // PMOVE TT1, (A0):
    //   opcode = 0xF010
    //   extension = 0x0E00  (format 2, preg TT1, R/W=1 -> to EA)
    bus.write_word_at(0x0200, 0xF010);
    bus.write_word_at(0x0202, 0x0E00);

    let mut hle = NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(result, StepResult::Ok { .. }));

    assert_eq!(
        bus.read_long_at(0x2000),
        tt1_value,
        "PMOVE TT1, (A0) should store 32-bit TT1 to memory"
    );
    assert_eq!(
        bus.read_long_at(0x2004),
        0,
        "Only 4 bytes should be written for TT1, not 8"
    );
}

/// Test the full EmuTOS-style MMU setup sequence:
///   PMOVE (crp_data), CRP   — 64-bit
///   PMOVE (tc_data), TC     — 32-bit
///   PMOVE (tt0_data), TT0   — 32-bit
///   PMOVE (tt1_data), TT1   — 32-bit
///
/// With the 3-bit bug, TT0 would be misidentified as SRP and TT1 as CRP,
/// corrupting the CRP register.
#[test]
fn test_pmove_full_mmu_setup_sequence() {
    let (mut cpu, mut bus) = setup_cpu_and_bus();

    // Data area at 0x2000:
    //   0x2000: CRP limit  (32 bits)
    //   0x2004: CRP aptr   (32 bits)
    //   0x2008: TC          (32 bits)
    //   0x200C: TT0         (32 bits)
    //   0x2010: TT1         (32 bits)
    let crp_limit: u32 = 0x7FFF_0002;
    let crp_aptr: u32 = 0x0070_0000;
    let tc_value: u32 = 0x0071_0507; // bit 31 clear: don't enable PMMU during test
    let tt0_value: u32 = 0x0000_8021;
    let tt1_value: u32 = 0x0000_8043;

    bus.write_long_at(0x2000, crp_limit);
    bus.write_long_at(0x2004, crp_aptr);
    bus.write_long_at(0x2008, tc_value);
    bus.write_long_at(0x200C, tt0_value);
    bus.write_long_at(0x2010, tt1_value);

    // Instruction stream at 0x0200:
    // 1. PMOVE (0x2000).L, CRP — opcode F039, ext 4C00
    let pc = 0x0200u32;
    bus.write_word_at(pc, 0xF039);      // coprocessor, EA = (xxx).L
    bus.write_word_at(pc + 2, 0x4C00);  // CRP, to reg
    bus.write_long_at(pc + 4, 0x2000);  // absolute long address

    // 2. PMOVE (0x2008).L, TC — opcode F039, ext 4000
    bus.write_word_at(pc + 8, 0xF039);
    bus.write_word_at(pc + 10, 0x4000); // TC, to reg
    bus.write_long_at(pc + 12, 0x2008);

    // 3. PMOVE (0x200C).L, TT0 — opcode F039, ext 0800
    bus.write_word_at(pc + 16, 0xF039);
    bus.write_word_at(pc + 18, 0x0800); // TT0, to reg
    bus.write_long_at(pc + 20, 0x200C);

    // 4. PMOVE (0x2010).L, TT1 — opcode F039, ext 0C00
    bus.write_word_at(pc + 24, 0xF039);
    bus.write_word_at(pc + 26, 0x0C00); // TT1, to reg
    bus.write_long_at(pc + 28, 0x2010);

    let mut hle = NoOpHleHandler;

    // Step 1: PMOVE CRP
    let r = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(r, StepResult::Ok { .. }), "CRP PMOVE failed");
    assert_eq!(cpu.mmu_crp_limit, crp_limit, "CRP limit mismatch");
    assert_eq!(cpu.mmu_crp_aptr, crp_aptr, "CRP aptr mismatch");

    // Step 2: PMOVE TC
    let r = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(r, StepResult::Ok { .. }), "TC PMOVE failed");
    assert_eq!(cpu.mmu_tc, tc_value, "TC mismatch");

    // Step 3: PMOVE TT0
    let r = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(r, StepResult::Ok { .. }), "TT0 PMOVE failed");
    assert_eq!(
        cpu.mmu_tt0, tt0_value,
        "TT0 should be 0x{:08X}, got 0x{:08X}",
        tt0_value, cpu.mmu_tt0
    );

    // Step 4: PMOVE TT1
    let r = cpu.step_with_hle_handler(&mut bus, &mut hle);
    assert!(matches!(r, StepResult::Ok { .. }), "TT1 PMOVE failed");
    assert_eq!(
        cpu.mmu_tt1, tt1_value,
        "TT1 should be 0x{:08X}, got 0x{:08X}",
        tt1_value, cpu.mmu_tt1
    );

    // CRP should NOT have been corrupted by the TT1 instruction.
    assert_eq!(
        cpu.mmu_crp_limit, crp_limit,
        "CRP limit was corrupted! Expected 0x{:08X}, got 0x{:08X}",
        crp_limit, cpu.mmu_crp_limit
    );
    assert_eq!(
        cpu.mmu_crp_aptr, crp_aptr,
        "CRP aptr was corrupted! Expected 0x{:08X}, got 0x{:08X}",
        crp_aptr, cpu.mmu_crp_aptr
    );

    // SRP should also be untouched (TT0 was misread as SRP with the bug).
    assert_eq!(
        cpu.mmu_srp_limit, 0,
        "SRP limit should be 0, got 0x{:08X}",
        cpu.mmu_srp_limit
    );
    assert_eq!(
        cpu.mmu_srp_aptr, 0,
        "SRP aptr should be 0, got 0x{:08X}",
        cpu.mmu_srp_aptr
    );
}

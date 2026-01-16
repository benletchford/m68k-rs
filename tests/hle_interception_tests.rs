//! Integration tests for trap interception for HLE using step_with_hle_handler().
//!
//! These tests verify that the CPU correctly intercepts various trap types
//! via the HleHandler callback mechanism:
//! - A-line (0xAxxx)   → handle_aline()
//! - F-line (0xFxxx)   → handle_fline() (68000/68010 only)
//! - TRAP #n           → handle_trap()
//! - BKPT #n           → handle_breakpoint()
//! - ILLEGAL (0x4AFC)  → handle_illegal()
//!
//! With the simplified API:
//! - step() auto-takes all exceptions (like original Musashi)
//! - step_with_hle_handler() invokes callbacks for HLE interception

use m68k::CpuCore;
use m68k::HleHandler;
use m68k::core::memory::AddressBus;
use m68k::core::types::{CpuType, StepResult};

/// Simple test bus for trap interception tests
struct TrapTestBus {
    memory: [u8; 0x10000],
}

impl TrapTestBus {
    fn new() -> Self {
        Self {
            memory: [0; 0x10000],
        }
    }

    fn write_word_at(&mut self, addr: u32, value: u16) {
        let bytes = value.to_be_bytes();
        self.memory[addr as usize] = bytes[0];
        self.memory[(addr + 1) as usize] = bytes[1];
    }

    fn write_long_at(&mut self, addr: u32, value: u32) {
        let bytes = value.to_be_bytes();
        self.memory[addr as usize] = bytes[0];
        self.memory[(addr + 1) as usize] = bytes[1];
        self.memory[(addr + 2) as usize] = bytes[2];
        self.memory[(addr + 3) as usize] = bytes[3];
    }
}

impl AddressBus for TrapTestBus {
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

// ============================================================================
// step_with_hle_handler() Tests
// ============================================================================

/// Test that A-line traps invoke handle_aline() callback.
#[test]
fn test_aline_trap_callback() {
    struct TestHandler {
        aline_called: bool,
        opcode_received: u16,
    }
    impl HleHandler for TestHandler {
        fn handle_aline(
            &mut self,
            _cpu: &mut CpuCore,
            _bus: &mut dyn AddressBus,
            opcode: u16,
        ) -> bool {
            self.aline_called = true;
            self.opcode_received = opcode;
            true // HLE handled
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    // Setup vectors
    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);

    // A-line trap at entry
    bus.write_word_at(0x100, 0xA9F0);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut handler = TestHandler {
        aline_called: false,
        opcode_received: 0,
    };

    let result = cpu.step_with_hle_handler(&mut bus, &mut handler);

    assert!(handler.aline_called, "handle_aline should be called");
    assert_eq!(handler.opcode_received, 0xA9F0, "opcode should match");
    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.pc, 0x102, "PC should advance past trap");
}

/// Test that unhandled A-line traps auto-take exception.
#[test]
fn test_aline_unhandled_takes_exception() {
    struct NoOpHandler;
    impl HleHandler for NoOpHandler {}

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    // Setup vectors
    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);
    bus.write_long_at(0x28, 0x200); // A-line vector

    // A-line trap at entry
    bus.write_word_at(0x100, 0xA9F0);
    // Handler code (NOP)
    bus.write_word_at(0x200, 0x4E71);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut handler = NoOpHandler;

    let _result = cpu.step_with_hle_handler(&mut bus, &mut handler);

    // With default handler returning false, exception should be taken
    assert_eq!(cpu.pc, 0x200, "PC should be at exception handler");
}

/// Test F-line trap callback on 68000.
#[test]
fn test_fline_trap_callback_68000() {
    struct TestHandler {
        fline_called: bool,
        opcode_received: u16,
    }
    impl HleHandler for TestHandler {
        fn handle_fline(
            &mut self,
            _cpu: &mut CpuCore,
            _bus: &mut dyn AddressBus,
            opcode: u16,
        ) -> bool {
            self.fline_called = true;
            self.opcode_received = opcode;
            true
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000); // F-line is only a trap on 68000/68010
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);

    // F-line instruction
    bus.write_word_at(0x100, 0xF000);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut handler = TestHandler {
        fline_called: false,
        opcode_received: 0,
    };

    cpu.step_with_hle_handler(&mut bus, &mut handler);

    assert!(handler.fline_called, "handle_fline should be called");
    assert_eq!(handler.opcode_received, 0xF000);
}

/// Test TRAP #n callback.
#[test]
fn test_trap_instruction_callback() {
    struct TestHandler {
        trap_called: bool,
        trap_num_received: u8,
    }
    impl HleHandler for TestHandler {
        fn handle_trap(
            &mut self,
            _cpu: &mut CpuCore,
            _bus: &mut dyn AddressBus,
            trap_num: u8,
        ) -> bool {
            self.trap_called = true;
            self.trap_num_received = trap_num;
            true
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);

    // TRAP #5 (0x4E45)
    bus.write_word_at(0x100, 0x4E45);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut handler = TestHandler {
        trap_called: false,
        trap_num_received: 0xFF,
    };

    cpu.step_with_hle_handler(&mut bus, &mut handler);

    assert!(handler.trap_called, "handle_trap should be called");
    assert_eq!(handler.trap_num_received, 5, "trap num should be 5");
}

/// Test BKPT #n callback.
#[test]
fn test_breakpoint_callback() {
    struct TestHandler {
        bp_called: bool,
        bp_num_received: u8,
    }
    impl HleHandler for TestHandler {
        fn handle_breakpoint(
            &mut self,
            _cpu: &mut CpuCore,
            _bus: &mut dyn AddressBus,
            bp_num: u8,
        ) -> bool {
            self.bp_called = true;
            self.bp_num_received = bp_num;
            true
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68010); // BKPT on 68010+
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);

    // BKPT #3 (0x484B)
    bus.write_word_at(0x100, 0x484B);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut handler = TestHandler {
        bp_called: false,
        bp_num_received: 0xFF,
    };

    cpu.step_with_hle_handler(&mut bus, &mut handler);

    assert!(handler.bp_called, "handle_breakpoint should be called");
    assert_eq!(handler.bp_num_received, 3, "bp num should be 3");
}

/// Test illegal instruction callback.
#[test]
fn test_illegal_instruction_callback() {
    struct TestHandler {
        illegal_called: bool,
        opcode_received: u16,
    }
    impl HleHandler for TestHandler {
        fn handle_illegal(
            &mut self,
            _cpu: &mut CpuCore,
            _bus: &mut dyn AddressBus,
            opcode: u16,
        ) -> bool {
            self.illegal_called = true;
            self.opcode_received = opcode;
            true
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);

    // ILLEGAL instruction (0x4AFC)
    bus.write_word_at(0x100, 0x4AFC);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut handler = TestHandler {
        illegal_called: false,
        opcode_received: 0,
    };

    cpu.step_with_hle_handler(&mut bus, &mut handler);

    assert!(handler.illegal_called, "handle_illegal should be called");
    assert_eq!(handler.opcode_received, 0x4AFC);
}

// ============================================================================
// step() Auto-Exception Tests
// ============================================================================

/// Test that step_with_hle_handler() auto-takes A-line exception without manual handling.
#[test]
fn test_step_auto_takes_aline_exception() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);
    bus.write_long_at(0x28, 0x200); // A-line vector

    bus.write_word_at(0x100, 0xA9F0);
    bus.write_word_at(0x200, 0x4E71); // NOP at handler

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut hle = m68k::NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);

    // step_with_hle_handler() should auto-take exception and return Ok with exception cycles
    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.pc, 0x200, "PC should be at exception handler");
}

/// Test that step_with_hle_handler() auto-takes TRAP exception.
#[test]
fn test_step_auto_takes_trap_exception() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);
    bus.write_long_at(0x80, 0x300); // TRAP #0 vector

    bus.write_word_at(0x100, 0x4E40); // TRAP #0
    bus.write_word_at(0x300, 0x4E71); // NOP at handler

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut hle = m68k::NoOpHleHandler;
    let result = cpu.step_with_hle_handler(&mut bus, &mut hle);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert_eq!(cpu.pc, 0x300, "PC should be at TRAP handler");
}

// ============================================================================
// Multiple Trap Sequence Test
// ============================================================================

/// Test multiple A-line traps in sequence using step_with_hle_handler().
#[test]
fn test_aline_trap_sequence() {
    struct CountingHandler {
        count: u32,
        last_opcode: u16,
    }
    impl HleHandler for CountingHandler {
        fn handle_aline(
            &mut self,
            _cpu: &mut CpuCore,
            _bus: &mut dyn AddressBus,
            opcode: u16,
        ) -> bool {
            self.count += 1;
            self.last_opcode = opcode;
            true
        }
    }

    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);

    bus.write_word_at(0x100, 0xA9F0);
    bus.write_word_at(0x102, 0xA9F1);
    bus.write_word_at(0x104, 0xA9F2);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let mut handler = CountingHandler {
        count: 0,
        last_opcode: 0,
    };

    cpu.step_with_hle_handler(&mut bus, &mut handler);
    assert_eq!(handler.count, 1);
    assert_eq!(handler.last_opcode, 0xA9F0);

    cpu.step_with_hle_handler(&mut bus, &mut handler);
    assert_eq!(handler.count, 2);
    assert_eq!(handler.last_opcode, 0xA9F1);

    cpu.step_with_hle_handler(&mut bus, &mut handler);
    assert_eq!(handler.count, 3);
    assert_eq!(handler.last_opcode, 0xA9F2);
}

/// Test that normal instructions return StepResult::Ok.
#[test]
fn test_normal_instruction_returns_ok() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    let mut bus = TrapTestBus::new();

    bus.write_long_at(0, 0x1000);
    bus.write_long_at(4, 0x100);

    // NOP instruction
    bus.write_word_at(0x100, 0x4E71);

    cpu.reset(&mut bus);
    cpu.pc = 0x100;
    cpu.set_sr(0x2700);

    let result = cpu.step(&mut bus);

    assert!(matches!(result, StepResult::Ok { .. }));
    assert!(result.cycles().unwrap() > 0);
}

/// Test StepResult helper methods.
#[test]
fn test_step_result_helpers() {
    let ok = StepResult::Ok { cycles: 10 };
    let stopped = StepResult::Stopped;

    assert_eq!(ok.cycles(), Some(10));
    assert_eq!(stopped.cycles(), None);

    assert!(!ok.is_stopped());
    assert!(stopped.is_stopped());
}

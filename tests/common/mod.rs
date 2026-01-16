//! Test harness for Musashi integration tests.

mod memory;
mod test_device;

pub use memory::{RamSlot, RomSlot};
pub use test_device::TestDevice;

use m68k::core::memory::AddressBus;
use m68k::core::memory::{BusFault, BusFaultKind};

/// Memory map slot size (64KB).
pub const SLOT_SIZE: u32 = 0x10000;

/// Test result from running a test binary.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct TestResult {
    pub pass_count: u32,
    pub fail_count: u32,
}

/// Complete test bus implementing the Musashi memory map.
pub struct TestBus {
    /// 0x000000-0x00FFFF: RAM (vector table, stack)
    pub ram: RamSlot,
    /// 0x010000-0x04FFFF: ROM (test code, 4 slots)
    pub rom: [RomSlot; 4],
    /// 0x100000-0x10FFFF: Test device
    pub test_device: TestDevice,
    /// 0x300000-0x30FFFF: Extra RAM
    pub extra_ram: RamSlot,
    /// If true, unmapped accesses raise a bus error instead of returning 0/ignoring writes.
    pub fault_on_unmapped: bool,
}

impl TestBus {
    pub fn new() -> Self {
        Self {
            ram: RamSlot::new(),
            rom: [
                RomSlot::new(),
                RomSlot::new(),
                RomSlot::new(),
                RomSlot::new(),
            ],
            test_device: TestDevice::new(),
            extra_ram: RamSlot::new(),
            fault_on_unmapped: false,
        }
    }

    /// Load a test binary into ROM slots.
    pub fn load_rom(&mut self, data: &[u8]) {
        for (i, chunk) in data.chunks(SLOT_SIZE as usize).enumerate() {
            if i < 4 {
                self.rom[i].load(chunk);
            }
        }
    }

    /// Setup boot vectors (SP and PC).
    pub fn setup_boot(&mut self) {
        // Match Musashi test harness: initial SSP at 0x3F0.
        self.write_long(0, 0x3F0);
        // Entry point at 0x10000 (ROM start)
        self.write_long(4, 0x10000);
    }

    /// Get test results.
    #[allow(dead_code)]
    pub fn result(&self) -> TestResult {
        TestResult {
            pass_count: self.test_device.pass_count,
            fail_count: self.test_device.fail_count,
        }
    }

    /// Map address to slot.
    fn map_address(&self, address: u32) -> (usize, u32) {
        let slot = (address / SLOT_SIZE) as usize;
        let offset = address & (SLOT_SIZE - 1);
        (slot, offset)
    }
}

impl Default for TestBus {
    fn default() -> Self {
        Self::new()
    }
}

impl AddressBus for TestBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        let (slot, offset) = self.map_address(address);
        match slot {
            0x00 => self.ram.read_byte(offset),
            0x01..=0x04 => self.rom[slot - 1].read_byte(offset),
            0x10 => self.test_device.read_byte(offset),
            0x30..=0x3F => self.extra_ram.read_byte(offset),
            _ => 0,
        }
    }

    fn read_word(&mut self, address: u32) -> u16 {
        let (slot, offset) = self.map_address(address);
        match slot {
            0x00 => self.ram.read_word(offset),
            0x01..=0x04 => self.rom[slot - 1].read_word(offset),
            0x10 => self.test_device.read_word(offset),
            0x30..=0x3F => self.extra_ram.read_word(offset),
            _ => 0,
        }
    }

    fn read_long(&mut self, address: u32) -> u32 {
        let (slot, offset) = self.map_address(address);
        match slot {
            0x00 => self.ram.read_long(offset),
            0x01..=0x04 => self.rom[slot - 1].read_long(offset),
            0x10 => self.test_device.read_long(offset),
            0x30..=0x3F => self.extra_ram.read_long(offset),
            _ => 0,
        }
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        let (slot, offset) = self.map_address(address);
        match slot {
            0x00 => self.ram.write_byte(offset, value),
            0x10 => self.test_device.write_byte(offset, value),
            0x30..=0x3F => self.extra_ram.write_byte(offset, value),
            _ => {} // ROM writes ignored
        }
    }

    fn write_word(&mut self, address: u32, value: u16) {
        let (slot, offset) = self.map_address(address);
        match slot {
            0x00 => self.ram.write_word(offset, value),
            0x10 => self.test_device.write_word(offset, value),
            0x30..=0x3F => self.extra_ram.write_word(offset, value),
            _ => {}
        }
    }

    fn write_long(&mut self, address: u32, value: u32) {
        let (slot, offset) = self.map_address(address);
        match slot {
            0x00 => self.ram.write_long(offset, value),
            0x10 => self.test_device.write_long(offset, value),
            0x30..=0x3F => self.extra_ram.write_long(offset, value),
            _ => {}
        }
    }

    fn try_read_byte(&mut self, address: u32) -> Result<u8, BusFault> {
        let (slot, _offset) = self.map_address(address);
        if self.fault_on_unmapped && !matches!(slot, 0x00 | 0x01..=0x04 | 0x10 | 0x30..=0x3F) {
            return Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            });
        }
        Ok(self.read_byte(address))
    }

    fn try_read_word(&mut self, address: u32) -> Result<u16, BusFault> {
        let (slot, _offset) = self.map_address(address);
        if self.fault_on_unmapped && !matches!(slot, 0x00 | 0x01..=0x04 | 0x10 | 0x30..=0x3F) {
            return Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            });
        }
        Ok(self.read_word(address))
    }

    fn try_read_long(&mut self, address: u32) -> Result<u32, BusFault> {
        let (slot, _offset) = self.map_address(address);
        if self.fault_on_unmapped && !matches!(slot, 0x00 | 0x01..=0x04 | 0x10 | 0x30..=0x3F) {
            return Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            });
        }
        Ok(self.read_long(address))
    }

    fn try_write_byte(&mut self, address: u32, value: u8) -> Result<(), BusFault> {
        let (slot, _offset) = self.map_address(address);
        if self.fault_on_unmapped && !matches!(slot, 0x00 | 0x01..=0x04 | 0x10 | 0x30..=0x3F) {
            return Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            });
        }
        self.write_byte(address, value);
        Ok(())
    }

    fn try_write_word(&mut self, address: u32, value: u16) -> Result<(), BusFault> {
        let (slot, _offset) = self.map_address(address);
        if self.fault_on_unmapped && !matches!(slot, 0x00 | 0x01..=0x04 | 0x10 | 0x30..=0x3F) {
            return Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            });
        }
        self.write_word(address, value);
        Ok(())
    }

    fn try_write_long(&mut self, address: u32, value: u32) -> Result<(), BusFault> {
        let (slot, _offset) = self.map_address(address);
        if self.fault_on_unmapped && !matches!(slot, 0x00 | 0x01..=0x04 | 0x10 | 0x30..=0x3F) {
            return Err(BusFault {
                kind: BusFaultKind::BusError,
                address,
            });
        }
        self.write_long(address, value);
        Ok(())
    }

    fn interrupt_acknowledge(&mut self, _level: u8) -> u32 {
        // Level-sensitive interrupt line: once the CPU acks, clear the device request so we
        // don't immediately re-enter the interrupt handler.
        self.test_device.interrupt_level = None;
        0xFFFF_FFFF // autovector
    }
}

/// Generic test fixture macro for all CPU types.
///
/// Usage:
/// ```
/// test_fixture!(test_name, CpuType::M68040, "fixtures/extra/m68040/bin/test.bin");
/// ```
#[macro_export]
macro_rules! test_fixture {
    ($name:ident, $cpu_type:expr, $fixture_path:literal) => {
        #[test]
        fn $name() {
            use m68k::{CpuCore, StepResult};
            use $crate::common::TestBus;

            let mut cpu = CpuCore::new();
            cpu.set_cpu_type($cpu_type);
            let mut bus = TestBus::new();

            let binary = include_bytes!($fixture_path);

            bus.load_rom(binary);
            bus.setup_boot();

            cpu.reset(&mut bus);
            cpu.pc = 0x10000;
            cpu.set_sr(0x2700);

            let mut hle = m68k::NoOpHleHandler;
            for _ in 0..100000 {
                if cpu.stopped != 0 {
                    break;
                }
                // Poll interrupt level from test device
                if let Some(level) = bus.test_device.interrupt_level {
                    cpu.int_level = level as u32;
                }

                // Execute one instruction. HLE handler falls back to exceptions.
                match cpu.step_with_hle_handler(&mut bus, &mut hle) {
                    StepResult::Ok { .. } => {}
                    StepResult::Stopped => break,
                    _ => {}
                }
            }

            let results = bus.result();
            println!(
                "Test {}: passes={}, fails={}",
                stringify!($name),
                results.pass_count,
                results.fail_count
            );

            if results.fail_count > 0 {
                panic!(
                    "Test failed: passes={}, fails={}",
                    results.pass_count, results.fail_count
                );
            }

            if results.pass_count == 0 {
                panic!(
                    "Test failed: passes={}, fails={}",
                    results.pass_count, results.fail_count
                );
            }
        }
    };
}

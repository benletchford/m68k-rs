//! Test device for pass/fail signaling.
//!
//! Memory-mapped device matching Musashi's test harness:
//! - 0x00: Write → increment fail count
//! - 0x04: Write → increment pass count
//! - 0x0C: Write → trigger interrupt (value = level)
//! - 0x14: Write byte → stdout

/// Test device that counts passes and failures.
#[derive(Debug, Default)]
pub struct TestDevice {
    pub pass_count: u32,
    pub fail_count: u32,
    pub interrupt_level: Option<u8>,
}

impl TestDevice {
    #[inline]
    fn verbose() -> bool {
        std::env::var("M68K_TEST_VERBOSE").ok().as_deref() == Some("1")
    }

    pub fn new() -> Self {
        Self::default()
    }

    pub fn read_byte(&self, _offset: u32) -> u8 {
        0
    }
    pub fn read_word(&self, _offset: u32) -> u16 {
        0
    }
    pub fn read_long(&self, _offset: u32) -> u32 {
        0
    }

    pub fn write_byte(&mut self, offset: u32, value: u8) {
        if offset == 0x14 {
            // stdout - print character
            print!("{}", value as char);
        }
    }

    pub fn write_word(&mut self, _offset: u32, _value: u16) {}

    pub fn write_long(&mut self, offset: u32, value: u32) {
        match offset {
            0x00 => {
                self.fail_count += 1;
                if Self::verbose() {
                    eprintln!(
                        "*** FAIL SIGNALED (value={:#X}, count={})",
                        value, self.fail_count
                    );
                }
            }
            0x04 => {
                self.pass_count += 1;
                if Self::verbose() {
                    eprintln!(
                        "*** PASS SIGNALED (value={:#X}, count={})",
                        value, self.pass_count
                    );
                }
            }
            0x0C => self.interrupt_level = Some((value & 0x7) as u8),
            _ => {}
        }
    }
}

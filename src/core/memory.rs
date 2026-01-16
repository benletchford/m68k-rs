//! Memory access trait.

/// Kind of bus-level fault during a memory access (distinct from 68000 address error).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusFaultKind {
    /// Generic bus error (unmapped address, device error, etc).
    BusError,
}

/// A bus-level fault that occurred during a memory access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BusFault {
    pub kind: BusFaultKind,
    pub address: u32,
}

pub trait AddressBus {
    fn read_byte(&mut self, address: u32) -> u8;
    fn read_word(&mut self, address: u32) -> u16;
    fn read_long(&mut self, address: u32) -> u32;
    fn write_byte(&mut self, address: u32, value: u8);
    fn write_word(&mut self, address: u32, value: u16);
    fn write_long(&mut self, address: u32, value: u32);

    /// Fallible read variants used to surface bus/MMU faults to the CPU core.
    ///
    /// Default implementations delegate to the infallible variants to preserve backwards
    /// compatibility for existing buses.
    #[inline]
    fn try_read_byte(&mut self, address: u32) -> Result<u8, BusFault> {
        Ok(self.read_byte(address))
    }
    #[inline]
    fn try_read_word(&mut self, address: u32) -> Result<u16, BusFault> {
        Ok(self.read_word(address))
    }
    #[inline]
    fn try_read_long(&mut self, address: u32) -> Result<u32, BusFault> {
        Ok(self.read_long(address))
    }
    #[inline]
    fn try_write_byte(&mut self, address: u32, value: u8) -> Result<(), BusFault> {
        self.write_byte(address, value);
        Ok(())
    }
    #[inline]
    fn try_write_word(&mut self, address: u32, value: u16) -> Result<(), BusFault> {
        self.write_word(address, value);
        Ok(())
    }
    #[inline]
    fn try_write_long(&mut self, address: u32, value: u32) -> Result<(), BusFault> {
        self.write_long(address, value);
        Ok(())
    }

    fn read_immediate_word(&mut self, address: u32) -> u16 {
        self.read_word(address)
    }
    fn read_immediate_long(&mut self, address: u32) -> u32 {
        self.read_long(address)
    }
    fn interrupt_acknowledge(&mut self, _level: u8) -> u32 {
        0xFFFF_FFFF
    }
    fn reset_devices(&mut self) {}
}

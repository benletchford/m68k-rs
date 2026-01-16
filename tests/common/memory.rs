//! Memory slot implementations (RAM and ROM).

use super::SLOT_SIZE;

/// 64KB RAM slot.
pub struct RamSlot {
    data: Box<[u8; SLOT_SIZE as usize]>,
}

impl RamSlot {
    pub fn new() -> Self {
        Self {
            data: Box::new([0u8; SLOT_SIZE as usize]),
        }
    }

    pub fn read_byte(&self, offset: u32) -> u8 {
        self.data[offset as usize]
    }

    pub fn read_word(&self, offset: u32) -> u16 {
        let o = offset as usize;
        ((self.data[o] as u16) << 8) | (self.data[o + 1] as u16)
    }

    pub fn read_long(&self, offset: u32) -> u32 {
        ((self.read_word(offset) as u32) << 16) | (self.read_word(offset + 2) as u32)
    }

    pub fn write_byte(&mut self, offset: u32, value: u8) {
        self.data[offset as usize] = value;
    }

    pub fn write_word(&mut self, offset: u32, value: u16) {
        let o = offset as usize;
        self.data[o] = (value >> 8) as u8;
        self.data[o + 1] = (value & 0xFF) as u8;
    }

    pub fn write_long(&mut self, offset: u32, value: u32) {
        self.write_word(offset, (value >> 16) as u16);
        self.write_word(offset + 2, (value & 0xFFFF) as u16);
    }
}

impl Default for RamSlot {
    fn default() -> Self {
        Self::new()
    }
}

/// 64KB ROM slot (read-only after loading).
pub struct RomSlot {
    data: Box<[u8; SLOT_SIZE as usize]>,
}

impl RomSlot {
    pub fn new() -> Self {
        Self {
            data: Box::new([0u8; SLOT_SIZE as usize]),
        }
    }

    pub fn load(&mut self, data: &[u8]) {
        let len = data.len().min(SLOT_SIZE as usize);
        self.data[..len].copy_from_slice(&data[..len]);
    }

    pub fn read_byte(&self, offset: u32) -> u8 {
        self.data[offset as usize]
    }

    pub fn read_word(&self, offset: u32) -> u16 {
        let o = offset as usize;
        ((self.data[o] as u16) << 8) | (self.data[o + 1] as u16)
    }

    pub fn read_long(&self, offset: u32) -> u32 {
        ((self.read_word(offset) as u32) << 16) | (self.read_word(offset + 2) as u32)
    }
}

impl Default for RomSlot {
    fn default() -> Self {
        Self::new()
    }
}

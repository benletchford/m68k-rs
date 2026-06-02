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

/// Optional companion trait for buses that can version instruction-visible memory.
///
/// This is intentionally separate from `AddressBus`: adding methods to the hot bus trait changes
/// code generation for opcode fetches in release builds. Future code caches/JIT paths can require
/// this trait without slowing existing interpreter users.
pub trait InstructionCacheBus: AddressBus {
    /// Stable version for instruction memory at `address`, when known.
    ///
    /// Returning `Some(version)` tells future code caches that fetches from this address may be
    /// reused until the version changes. Returning `None` is conservative.
    #[inline]
    fn instruction_cache_version(&mut self, _address: u32) -> Option<u64> {
        None
    }

    /// Notify the bus that bytes in a code-visible range were written by the CPU.
    ///
    /// Buses that implement `instruction_cache_version` should update the relevant version here.
    #[inline]
    fn invalidate_instruction_cache(&mut self, _address: u32, _len: u32) {}
}

/// Fast linear-memory bus for RAM-backed emulators and WebAssembly builds.
///
/// This keeps all normal memory accesses inside Rust/wasm linear memory instead of crossing into a
/// host callback for each byte/word/long. Addresses wrap within the backing buffer, with a fast mask
/// path for power-of-two sizes.
#[derive(Debug, Clone)]
pub struct LinearMemoryBus {
    memory: Vec<u8>,
    wrap_mask: usize,
    power_of_two_len: bool,
    instruction_version: u64,
}

impl LinearMemoryBus {
    /// Create a zero-filled bus with `size` bytes.
    pub fn new(size: usize) -> Self {
        Self::from_vec(vec![0; size])
    }

    /// Create a bus using an existing memory buffer.
    pub fn from_vec(memory: Vec<u8>) -> Self {
        assert!(
            !memory.is_empty(),
            "LinearMemoryBus requires non-empty memory"
        );
        let power_of_two_len = memory.len().is_power_of_two();
        let wrap_mask = memory.len().saturating_sub(1);
        Self {
            memory,
            wrap_mask,
            power_of_two_len,
            instruction_version: 1,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.memory
    }

    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.bump_instruction_version();
        &mut self.memory
    }

    /// Copy bytes into memory at `address`, wrapping at the end of the backing buffer.
    pub fn load(&mut self, address: u32, data: &[u8]) {
        if self.memory.is_empty() {
            return;
        }
        for (offset, value) in data.iter().copied().enumerate() {
            let idx = self.index(address.wrapping_add(offset as u32));
            self.memory[idx] = value;
        }
        self.bump_instruction_version();
    }

    #[inline]
    pub fn write_word_at(&mut self, address: u32, value: u16) {
        self.write_word(address, value);
    }

    #[inline]
    pub fn write_long_at(&mut self, address: u32, value: u32) {
        self.write_long(address, value);
    }

    #[inline]
    fn index(&self, address: u32) -> usize {
        debug_assert!(!self.memory.is_empty());
        if self.power_of_two_len {
            (address as usize) & self.wrap_mask
        } else {
            (address as usize) % self.memory.len()
        }
    }

    #[inline]
    fn read_index(&self, index: usize) -> u8 {
        debug_assert!(index < self.memory.len());
        // Indices are produced by `index`, which wraps into the backing buffer.
        unsafe { *self.memory.get_unchecked(index) }
    }

    #[inline]
    fn write_index(&mut self, index: usize, value: u8) {
        debug_assert!(index < self.memory.len());
        // Indices are produced by `index`, which wraps into the backing buffer.
        unsafe {
            *self.memory.get_unchecked_mut(index) = value;
        }
    }

    #[inline]
    fn bump_instruction_version(&mut self) {
        self.instruction_version = self.instruction_version.wrapping_add(1);
        if self.instruction_version == 0 {
            self.instruction_version = 1;
        }
    }
}

impl AddressBus for LinearMemoryBus {
    #[inline]
    fn read_byte(&mut self, address: u32) -> u8 {
        let idx = self.index(address);
        self.read_index(idx)
    }

    #[inline]
    fn read_word(&mut self, address: u32) -> u16 {
        let b0 = self.read_index(self.index(address));
        let b1 = self.read_index(self.index(address.wrapping_add(1)));
        ((b0 as u16) << 8) | b1 as u16
    }

    #[inline]
    fn read_long(&mut self, address: u32) -> u32 {
        let b0 = self.read_index(self.index(address));
        let b1 = self.read_index(self.index(address.wrapping_add(1)));
        let b2 = self.read_index(self.index(address.wrapping_add(2)));
        let b3 = self.read_index(self.index(address.wrapping_add(3)));
        ((b0 as u32) << 24) | ((b1 as u32) << 16) | ((b2 as u32) << 8) | b3 as u32
    }

    #[inline]
    fn write_byte(&mut self, address: u32, value: u8) {
        let idx = self.index(address);
        self.write_index(idx, value);
        self.bump_instruction_version();
    }

    #[inline]
    fn write_word(&mut self, address: u32, value: u16) {
        let idx0 = self.index(address);
        let idx1 = self.index(address.wrapping_add(1));
        self.write_index(idx0, (value >> 8) as u8);
        self.write_index(idx1, value as u8);
        self.bump_instruction_version();
    }

    #[inline]
    fn write_long(&mut self, address: u32, value: u32) {
        let idx0 = self.index(address);
        let idx1 = self.index(address.wrapping_add(1));
        let idx2 = self.index(address.wrapping_add(2));
        let idx3 = self.index(address.wrapping_add(3));
        self.write_index(idx0, (value >> 24) as u8);
        self.write_index(idx1, (value >> 16) as u8);
        self.write_index(idx2, (value >> 8) as u8);
        self.write_index(idx3, value as u8);
        self.bump_instruction_version();
    }

    #[inline]
    fn read_immediate_word(&mut self, address: u32) -> u16 {
        self.read_word(address)
    }

    #[inline]
    fn read_immediate_long(&mut self, address: u32) -> u32 {
        self.read_long(address)
    }
}

impl InstructionCacheBus for LinearMemoryBus {
    #[inline]
    fn instruction_cache_version(&mut self, _address: u32) -> Option<u64> {
        Some(self.instruction_version)
    }

    #[inline]
    fn invalidate_instruction_cache(&mut self, _address: u32, _len: u32) {
        self.bump_instruction_version();
    }
}

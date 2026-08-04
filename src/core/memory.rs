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
    /// Classification of the failed bus transaction.
    pub kind: BusFaultKind,
    /// Bus address at which the transaction failed.
    pub address: u32,
}

/// A contiguous guest-RAM window the CPU may access directly ("fastmem").
///
/// Returned by [`AddressBus::fast_mem`]. The window lets the batch
/// execution path ([`CpuCore::run_batch`](crate::CpuCore::run_batch))
/// fetch opcodes and execute memory-operand instructions without a bus
/// call per access.
///
/// # Contract
///
/// For every guest address `a` in `[base, base + len)`, and for the whole
/// duration of the `run_batch` call that captured the window:
///
/// - `ptr[a - base]` holds the same byte the bus would return from
///   `read_byte(a)`, stored big-endian for multi-byte values (i.e. the
///   window is the bus's actual backing RAM, not a copy);
/// - reads and writes through `ptr` have **no side effects** — no MMIO,
///   no watchpoints, no dirty tracking, no mirroring — and are fully
///   interchangeable with the `read_*`/`write_*` methods;
/// - the pointer stays valid and the backing storage is not moved or
///   resized, even across interleaved `AddressBus` method calls.
///
/// Buses with any interception (tracers, watchpoints, MMIO in range)
/// must return `None` from `fast_mem` while that interception is active.
/// `len` must be at least 4 bytes; smaller windows are ignored.
#[derive(Debug, Clone, Copy)]
pub struct FastMem {
    /// Host pointer to the byte backing guest address `base`.
    pub ptr: *mut u8,
    /// First guest address covered by the window.
    pub base: u32,
    /// Window length in bytes.
    pub len: u32,
}

/// Host-provided memory and device bus used by [`CpuCore`](crate::CpuCore).
///
/// Multi-byte values use the 68k's big-endian byte order. The six basic
/// read/write methods are required; the remaining hooks have conservative
/// defaults for functional emulators and can be overridden for precise
/// faults, timing, interrupts, caches, and batch acceleration.
pub trait AddressBus {
    /// Read one byte from `address`.
    fn read_byte(&mut self, address: u32) -> u8;
    /// Read one big-endian 16-bit word from `address`.
    fn read_word(&mut self, address: u32) -> u16;
    /// Read one big-endian 32-bit longword from `address`.
    fn read_long(&mut self, address: u32) -> u32;
    /// Write one byte to `address`.
    fn write_byte(&mut self, address: u32, value: u8);
    /// Write one big-endian 16-bit word to `address`.
    fn write_word(&mut self, address: u32, value: u16);
    /// Write one big-endian 32-bit longword to `address`.
    fn write_long(&mut self, address: u32, value: u32);

    /// Precise-timing callback (Part E.2): called immediately before each bus
    /// access with the number of CPU clocks of internal (non-bus) processing
    /// the core performed since its previous access. The access itself then
    /// takes the standard 4 CPU clocks of a 68000 bus cycle.
    ///
    /// Hosts that emulate surrounding hardware (DMA, video beam) advance it
    /// by `cpu_clocks` here so every access lands at the hardware-exact
    /// moment. The default is a no-op, so buses that only need functional
    /// emulation are unaffected.
    fn sync(&mut self, _cpu_clocks: u32) {}

    /// Fallible read variants used to surface bus/MMU faults to the CPU core.
    ///
    /// Default implementations delegate to the infallible variants to preserve backwards
    /// compatibility for existing buses.
    #[inline]
    fn try_read_byte(&mut self, address: u32) -> Result<u8, BusFault> {
        Ok(self.read_byte(address))
    }
    /// Fallible word read used for bus/MMU fault delivery.
    ///
    /// The default delegates to [`AddressBus::read_word`] and cannot fault.
    #[inline]
    fn try_read_word(&mut self, address: u32) -> Result<u16, BusFault> {
        Ok(self.read_word(address))
    }
    /// Fallible longword read used for bus/MMU fault delivery.
    ///
    /// The default delegates to [`AddressBus::read_long`] and cannot fault.
    #[inline]
    fn try_read_long(&mut self, address: u32) -> Result<u32, BusFault> {
        Ok(self.read_long(address))
    }
    /// Fallible byte write used for bus/MMU fault delivery.
    ///
    /// The default delegates to [`AddressBus::write_byte`] and cannot fault.
    #[inline]
    fn try_write_byte(&mut self, address: u32, value: u8) -> Result<(), BusFault> {
        self.write_byte(address, value);
        Ok(())
    }
    /// Fallible word write used for bus/MMU fault delivery.
    ///
    /// The default delegates to [`AddressBus::write_word`] and cannot fault.
    #[inline]
    fn try_write_word(&mut self, address: u32, value: u16) -> Result<(), BusFault> {
        self.write_word(address, value);
        Ok(())
    }
    /// Fallible longword write used for bus/MMU fault delivery.
    ///
    /// The default delegates to [`AddressBus::write_long`] and cannot fault.
    #[inline]
    fn try_write_long(&mut self, address: u32, value: u32) -> Result<(), BusFault> {
        self.write_long(address, value);
        Ok(())
    }

    /// Read a big-endian three-byte operand from `address`, returned in the
    /// low 24 bits.
    ///
    /// The 68020/68030 transfer an operand at the size it spans - byte, word,
    /// three-byte or long (MC68020UM 5.3.1) - so a bit field covering three
    /// bytes is a single operand transfer, not a word plus a byte. The
    /// default composes one anyway, which is correct for every bus that only
    /// moves data; hosts that bill bus cycles or count transactions should
    /// override it so the access costs what the hardware charges.
    ///
    /// The 68040/68060 bus has no three-byte encoding. A host modelling one
    /// may split the access however that processor does, provided it still
    /// touches only these three bytes.
    ///
    /// The core reaches this through [`AddressBus::try_read_three_bytes`];
    /// override that variant as well on a bus that reports faults.
    #[inline]
    fn read_three_bytes(&mut self, address: u32) -> u32 {
        let hi = self.read_word(address) as u32;
        let lo = self.read_byte(address.wrapping_add(2)) as u32;
        (hi << 8) | lo
    }

    /// Write the low 24 bits of `value` as a big-endian three-byte operand.
    ///
    /// See [`AddressBus::read_three_bytes`] for when the core uses this and
    /// why a host may want to override it.
    #[inline]
    fn write_three_bytes(&mut self, address: u32, value: u32) {
        self.write_word(address, (value >> 8) as u16);
        self.write_byte(address.wrapping_add(2), value as u8);
    }

    /// Fallible three-byte read used for bus/MMU fault delivery. **This is
    /// the variant the core calls**, so a host that bills bus cycles must
    /// override it (as well as [`AddressBus::read_three_bytes`]) for the
    /// billing to take effect.
    ///
    /// The default composes the fallible word and byte reads rather than
    /// delegating to the infallible variant, so an existing bus keeps
    /// reporting a fault on whichever half of the operand fails.
    #[inline]
    fn try_read_three_bytes(&mut self, address: u32) -> Result<u32, BusFault> {
        let hi = self.try_read_word(address)? as u32;
        let lo = self.try_read_byte(address.wrapping_add(2))? as u32;
        Ok((hi << 8) | lo)
    }

    /// Fallible three-byte write used for bus/MMU fault delivery.
    ///
    /// See [`AddressBus::try_read_three_bytes`]: this is the variant the core
    /// calls, and the default composes the fallible word and byte writes.
    #[inline]
    fn try_write_three_bytes(&mut self, address: u32, value: u32) -> Result<(), BusFault> {
        self.try_write_word(address, (value >> 8) as u16)?;
        self.try_write_byte(address.wrapping_add(2), value as u8)
    }

    /// Read an instruction-stream word.
    ///
    /// Override this when opcode/immediate fetches use a distinct bus path.
    fn read_immediate_word(&mut self, address: u32) -> u16 {
        self.read_word(address)
    }
    /// Read an instruction-stream longword.
    ///
    /// Override this when opcode/immediate fetches use a distinct bus path.
    fn read_immediate_long(&mut self, address: u32) -> u32 {
        self.read_long(address)
    }
    /// Instruction-stream reads with bus-fault reporting, used by the
    /// non-prefetch (68010+) opcode/immediate path so hosts can tell
    /// fetches from data reads (e.g. to model a 32-bit fetch path).
    #[inline]
    fn try_read_immediate_word(&mut self, address: u32) -> Result<u16, BusFault> {
        Ok(self.read_immediate_word(address))
    }
    /// Fallible instruction-stream longword read.
    ///
    /// The default delegates to [`AddressBus::read_immediate_long`] and
    /// cannot fault.
    #[inline]
    fn try_read_immediate_long(&mut self, address: u32) -> Result<u32, BusFault> {
        Ok(self.read_immediate_long(address))
    }
    /// Whether the most recent instruction-stream read was served from the
    /// CPU's instruction cache. The 68060 timing model gates superscalar
    /// pairing and branch folding on a cached fetch stream; plain test
    /// buses default to true (pair freely).
    fn last_fetch_was_cached(&self) -> bool {
        true
    }

    /// Start tracking cache residency for one complete instruction. The
    /// MC68020 timing tables define their cache case for an instruction that
    /// is in the cache, including extension and immediate words rather than
    /// only the opcode word. Hosts with an instruction-cache model use this
    /// hook to reset their per-instruction hit accumulator.
    fn begin_instruction_fetches(&mut self) {}

    /// Whether every instruction-stream access since
    /// `begin_instruction_fetches` hit the instruction cache. Functional test
    /// buses without a cache model default to the most recent fetch result.
    fn instruction_fetches_were_cached(&self) -> bool {
        self.last_fetch_was_cached()
    }

    /// Take a pending request to return from cycle-scheduled execution at
    /// the next completed instruction or interrupt-entry boundary.
    ///
    /// [`CpuCore::run_for_cycles`](crate::CpuCore::run_for_cycles) and
    /// [`CpuCore::run_for_cycles_with_hook`](crate::CpuCore::run_for_cycles_with_hook)
    /// and
    /// [`CpuCore::run_for_cycles_with_boundary_hook`](crate::CpuCore::run_for_cycles_with_boundary_hook)
    /// call this after an instruction completes normally or an entry interrupt
    /// is serviced, and before fetching or executing another instruction.
    /// Returning `true` makes the runner exit with
    /// [`CycleBatchExit::BoundaryRequested`](crate::CycleBatchExit::BoundaryRequested).
    /// Completed work is included in the result: an instruction contributes
    /// its cycles and retirement count, while interrupt entry contributes its
    /// cycles without incrementing the retirement count.
    ///
    /// Implementations should consume the request when returning `true`, so
    /// execution can resume without a separate acknowledgement call. A bus
    /// can retain any associated host work in its own queue until the caller
    /// processes the boundary exit. The default reports no request.
    #[inline]
    fn take_boundary_request(&mut self) -> bool {
        false
    }

    /// Perform an interrupt-acknowledge cycle for `level`.
    ///
    /// Return an explicit vector number in the low byte, or `u32::MAX` for
    /// the level's autovector. The default requests an autovector.
    fn interrupt_acknowledge(&mut self, _level: u8) -> u32 {
        0xFFFF_FFFF
    }

    /// IPL poll-point marker. The 68000/68010 sample their IPL pins at ONE
    /// microcode-determined point per instruction, and the take-interrupt
    /// decision at the next instruction boundary consumes that sample. A
    /// timing-accurate host latches the IPL level at the start of every bus
    /// access and, by default, lets the instruction's LAST access provide
    /// the boundary sample. For instructions whose poll point is NOT the
    /// last access (e.g. read-modify-write instructions poll during the
    /// final prefetch that precedes the writeback), the core calls this
    /// right after the polling access: the host must keep that access's
    /// sample and ignore later accesses until the boundary decision
    /// consumes it. Functional-only buses can ignore it.
    fn ipl_hold_sample(&mut self) {}

    /// Release an `ipl_hold_sample` poll-point hold before the instruction
    /// boundary consumes it. Called on exception dispatch: the vector jump's
    /// handler-entry prefetch is a fresh poll point on real silicon (Moira
    /// jumpToVector polls during the final refill read), so a hold placed
    /// earlier in the faulted instruction must not survive into the handler.
    fn ipl_release_sample(&mut self) {}

    /// Notify attached devices that the CPU asserted the external RESET line.
    fn reset_devices(&mut self) {}

    /// Expose a direct window into contiguous, side-effect-free guest RAM.
    ///
    /// See [`FastMem`] for the exact contract. Returning `Some` lets
    /// [`CpuCore::run_batch`](crate::CpuCore::run_batch) execute
    /// memory-operand instructions and opcode fetches without a bus call
    /// per access — typically a large speedup for memory-heavy guest
    /// code. The default returns `None` (no fast path); cycle-accurate
    /// entry points (`execute`/`step`) never use the window either way.
    #[inline]
    fn fast_mem(&mut self) -> Option<FastMem> {
        None
    }
}

/// Optional companion trait for buses that version instruction-visible memory.
///
/// This is intentionally separate from `AddressBus`: adding methods to the hot bus trait changes
/// code generation for opcode fetches in release builds. Embedders can use it
/// to coordinate external code caches without slowing buses that need only
/// [`AddressBus`].
pub trait InstructionCacheBus: AddressBus {
    /// Stable version for instruction memory at `address`, when known.
    ///
    /// Returning `Some(version)` tells an instruction cache that fetches from
    /// this address may be reused until the version changes. Returning `None`
    /// is conservative.
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
    /// Return the number of bytes in the backing buffer.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    #[inline]
    /// Return whether the backing buffer is empty.
    ///
    /// A constructed `LinearMemoryBus` is never empty because
    /// [`LinearMemoryBus::from_vec`] rejects an empty buffer.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    #[inline]
    /// Borrow the complete backing buffer.
    pub fn as_slice(&self) -> &[u8] {
        &self.memory
    }

    #[inline]
    /// Mutably borrow the complete backing buffer.
    ///
    /// Taking a mutable slice advances the instruction-memory version
    /// conservatively because the caller may change executable bytes.
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
    /// Write one big-endian word, wrapping within the backing buffer.
    pub fn write_word_at(&mut self, address: u32, value: u16) {
        self.write_word(address, value);
    }

    #[inline]
    /// Write one big-endian longword, wrapping within the backing buffer.
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

    /// The whole backing buffer is side-effect-free RAM, so expose it as a
    /// fastmem window starting at guest address 0. Accesses beyond `len`
    /// (which the bus methods wrap) simply fall back to the bus.
    #[inline]
    fn fast_mem(&mut self) -> Option<FastMem> {
        Some(FastMem {
            ptr: self.memory.as_mut_ptr(),
            base: 0,
            len: u32::try_from(self.memory.len()).unwrap_or(u32::MAX),
        })
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

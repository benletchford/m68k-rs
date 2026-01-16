//! Effective Address resolution.
//!
//! Implements all M68000 addressing modes.

use super::cpu::CpuCore;
use super::execute::RUN_MODE_BERR_AERR_RESET;
use super::memory::AddressBus;
use super::types::{CpuType, Size};

/// Addressing mode encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddressingMode {
    /// Data Register Direct: Dn
    DataDirect(u8),
    /// Address Register Direct: An
    AddressDirect(u8),
    /// Address Register Indirect: (An)
    AddressIndirect(u8),
    /// Address Register Indirect with Post-Increment: (An)+
    PostIncrement(u8),
    /// Address Register Indirect with Pre-Decrement: -(An)
    PreDecrement(u8),
    /// Address Register Indirect with Displacement: (d16,An)
    Displacement(u8),
    /// Address Register Indirect with Index: (d8,An,Xn)
    Index(u8),
    /// Absolute Short: (xxx).W
    AbsoluteShort,
    /// Absolute Long: (xxx).L  
    AbsoluteLong,
    /// PC with Displacement: (d16,PC)
    PcDisplacement,
    /// PC with Index: (d8,PC,Xn)
    PcIndex,
    /// Immediate: #<data>
    Immediate,
}

impl AddressingMode {
    /// Decode mode and register fields from opcode.
    pub fn decode(mode: u8, reg: u8) -> Option<Self> {
        match mode {
            0b000 => Some(Self::DataDirect(reg)),
            0b001 => Some(Self::AddressDirect(reg)),
            0b010 => Some(Self::AddressIndirect(reg)),
            0b011 => Some(Self::PostIncrement(reg)),
            0b100 => Some(Self::PreDecrement(reg)),
            0b101 => Some(Self::Displacement(reg)),
            0b110 => Some(Self::Index(reg)),
            0b111 => match reg {
                0b000 => Some(Self::AbsoluteShort),
                0b001 => Some(Self::AbsoluteLong),
                0b010 => Some(Self::PcDisplacement),
                0b011 => Some(Self::PcIndex),
                0b100 => Some(Self::Immediate),
                _ => None,
            },
            _ => None,
        }
    }

    /// Check if this mode is a register direct mode.
    pub fn is_register_direct(&self) -> bool {
        matches!(self, Self::DataDirect(_) | Self::AddressDirect(_))
    }
}

/// Result of effective address calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EaResult {
    /// Value is in a data register.
    DataReg(u8),
    /// Value is in an address register.
    AddrReg(u8),
    /// Value is at a memory address.
    Memory(u32),
    /// Immediate value (with address of immediate data in instruction stream).
    Immediate(u32),
}

impl CpuCore {
    /// Get increment/decrement size for address register.
    /// Stack pointer (A7) always uses word alignment minimum.
    #[inline]
    fn addr_inc(&self, reg: u8, size: Size) -> u32 {
        if reg == 7 && size == Size::Byte {
            2 // A7 always word-aligned
        } else {
            size.bytes()
        }
    }

    /// Resolve effective address.
    pub fn resolve_ea<B: AddressBus>(
        &mut self,
        bus: &mut B,
        mode: AddressingMode,
        size: Size,
    ) -> EaResult {
        // If we're in the middle of processing a bus/address error, avoid applying additional
        // EA side effects (postinc/predec) from the faulting instruction.
        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            return EaResult::Memory(0);
        }
        match mode {
            AddressingMode::DataDirect(reg) => EaResult::DataReg(reg),
            AddressingMode::AddressDirect(reg) => EaResult::AddrReg(reg),
            AddressingMode::AddressIndirect(reg) => EaResult::Memory(self.a(reg as usize)),
            AddressingMode::PostIncrement(reg) => {
                let addr = self.a(reg as usize);
                let inc = self.addr_inc(reg, size);
                self.set_a(reg as usize, addr.wrapping_add(inc));
                EaResult::Memory(addr)
            }
            AddressingMode::PreDecrement(reg) => {
                let dec = self.addr_inc(reg, size);
                let addr = self.a(reg as usize).wrapping_sub(dec);
                self.set_a(reg as usize, addr);
                EaResult::Memory(addr)
            }
            AddressingMode::Displacement(reg) => {
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (self.a(reg as usize) as i32).wrapping_add(disp) as u32;
                EaResult::Memory(addr)
            }
            AddressingMode::Index(reg) => {
                let ext = self.read_imm_16(bus);
                let addr = self.compute_index(self.a(reg as usize), ext, bus);
                EaResult::Memory(addr)
            }
            AddressingMode::AbsoluteShort => {
                let addr = self.read_imm_16(bus) as i16 as i32 as u32;
                EaResult::Memory(addr)
            }
            AddressingMode::AbsoluteLong => {
                let addr = self.read_imm_32(bus);
                EaResult::Memory(addr)
            }
            AddressingMode::PcDisplacement => {
                let pc = self.pc;
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (pc as i32).wrapping_add(disp) as u32;
                EaResult::Memory(addr)
            }
            AddressingMode::PcIndex => {
                let pc = self.pc;
                let ext = self.read_imm_16(bus);
                let addr = self.compute_index(pc, ext, bus);
                EaResult::Memory(addr)
            }
            AddressingMode::Immediate => {
                let addr = self.pc;
                match size {
                    Size::Byte => {
                        self.pc += 2;
                    }
                    Size::Word => {
                        self.pc += 2;
                    }
                    Size::Long => {
                        self.pc += 4;
                    }
                }
                EaResult::Immediate(addr)
            }
        }
    }

    /// Compute indexed address from extension word.
    fn compute_index<B: AddressBus>(&mut self, base: u32, ext: u16, bus: &mut B) -> u32 {
        let d8 = (ext & 0xFF) as i8 as i32;
        let idx_reg = ((ext >> 12) & 0xF) as usize;
        let idx_is_addr = (ext & 0x8000) != 0;
        let idx_is_long = (ext & 0x0800) != 0;
        // Scale is a 68020+ feature (brief extension word on 68000/68010 does not scale).
        // Some test generators may leave non-zero scale bits in the extension word; 68000 ignores them.
        let scale = if self.is_020_plus() {
            let scale_shift = ((ext >> 9) & 0x3) as u32; // 00=1,01=2,10=4,11=8
            1i32 << scale_shift
        } else {
            1i32
        };

        let idx_val = if idx_is_addr {
            self.dar[8 + (idx_reg & 7)]
        } else {
            self.dar[idx_reg & 7]
        };

        let idx_val = if idx_is_long {
            idx_val as i32
        } else {
            (idx_val as i16) as i32
        };
        let idx_val = idx_val.wrapping_mul(scale);

        // 68020+ full extension word format
        if (ext & 0x0100) != 0 && self.is_020_plus() {
            self.compute_full_index(base, ext, idx_val, bus)
        } else {
            // Brief format
            (base as i32).wrapping_add(d8).wrapping_add(idx_val) as u32
        }
    }

    /// 68020+ full extension word format.
    fn compute_full_index<B: AddressBus>(
        &mut self,
        base: u32,
        ext: u16,
        idx_val: i32,
        bus: &mut B,
    ) -> u32 {
        let bs = (ext & 0x0080) != 0; // Base suppress
        let is = (ext & 0x0040) != 0; // Index suppress
        let bd_size = (ext >> 4) & 0x03;
        let i_is = ext & 0x07;

        let base = if bs { 0 } else { base };
        let idx = if is { 0 } else { idx_val };

        let bd: i32 = match bd_size {
            0 | 1 => 0, // Reserved / Null
            2 => self.read_imm_16(bus) as i16 as i32,
            3 => self.read_imm_32(bus) as i32,
            _ => 0,
        };

        // Memory indirect modes
        if i_is != 0 {
            let outer_disp = match i_is & 0x03 {
                0 | 1 => 0i32,
                2 => self.read_imm_16(bus) as i16 as i32,
                3 => self.read_imm_32(bus) as i32,
                _ => 0,
            };

            if (i_is & 0x04) != 0 {
                // Post-indexed
                let intermediate = (base as i32).wrapping_add(bd) as u32;
                let indirect = self.read_32(bus, intermediate) as i32;
                indirect.wrapping_add(idx).wrapping_add(outer_disp) as u32
            } else {
                // Pre-indexed
                let intermediate = (base as i32).wrapping_add(bd).wrapping_add(idx) as u32;
                let indirect = self.read_32(bus, intermediate) as i32;
                indirect.wrapping_add(outer_disp) as u32
            }
        } else {
            // No memory indirect
            (base as i32).wrapping_add(bd).wrapping_add(idx) as u32
        }
    }

    /// Check if CPU is 68020 or later.
    #[inline]
    fn is_020_plus(&self) -> bool {
        matches!(
            self.cpu_type,
            CpuType::M68EC020
                | CpuType::M68020
                | CpuType::M68EC030
                | CpuType::M68030
                | CpuType::M68EC040
                | CpuType::M68LC040
                | CpuType::M68040
        )
    }

    /// Read immediate 16-bit value and advance PC.
    #[inline]
    pub fn read_imm_16<B: AddressBus>(&mut self, bus: &mut B) -> u16 {
        let addr = self.pc;
        if (addr & 1) != 0 {
            self.trigger_address_error(bus, addr, false, true);
            return 0;
        }
        let mut addr = self.address(addr);
        if self.has_pmmu && self.pmmu_enabled {
            match crate::mmu::translate_address(
                self,
                bus,
                addr,
                /*write=*/ false,
                self.is_supervisor(),
                /*instruction=*/ true,
            ) {
                Ok(p) => addr = self.address(p),
                Err(f) => {
                    self.handle_mmu_fault(
                        bus, f, /*write=*/ false, /*instruction=*/ true,
                    );
                    return 0;
                }
            }
        }
        match bus.try_read_word(addr) {
            Ok(v) => {
                self.pc = self.pc.wrapping_add(2);
                v
            }
            Err(_) => {
                self.trigger_bus_error(bus, addr, false, true);
                0
            }
        }
    }

    /// Read immediate 32-bit value and advance PC.
    #[inline]
    pub fn read_imm_32<B: AddressBus>(&mut self, bus: &mut B) -> u32 {
        let addr = self.pc;
        if (addr & 1) != 0 {
            self.trigger_address_error(bus, addr, false, true);
            return 0;
        }
        let mut addr = self.address(addr);
        if self.has_pmmu && self.pmmu_enabled {
            match crate::mmu::translate_address(
                self,
                bus,
                addr,
                /*write=*/ false,
                self.is_supervisor(),
                /*instruction=*/ true,
            ) {
                Ok(p) => addr = self.address(p),
                Err(f) => {
                    self.handle_mmu_fault(
                        bus, f, /*write=*/ false, /*instruction=*/ true,
                    );
                    return 0;
                }
            }
        }
        match bus.try_read_long(addr) {
            Ok(v) => {
                self.pc = self.pc.wrapping_add(4);
                v
            }
            Err(_) => {
                self.trigger_bus_error(bus, addr, false, true);
                0
            }
        }
    }

    /// Read immediate 8-bit value and advance PC (reads word, returns low byte).
    #[inline]
    pub fn read_imm_8<B: AddressBus>(&mut self, bus: &mut B) -> u8 {
        (self.read_imm_16(bus) & 0xFF) as u8
    }
}

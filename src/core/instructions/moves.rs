// MOVES - Move to/from Address Space (68010+)
//
// Opcode: 0000 1110 ssmm mrrr
// Extension word: 1aaa rrrd 0000 0000
//   a/rrr = register number (0-7 for Dn, A for An)
//   d = direction (0 = register to EA, 1 = EA to register)

use crate::core::cpu::CpuCore;
use crate::core::ea::AddressingMode;
use crate::core::memory::AddressBus;
use crate::core::types::Size;

impl CpuCore {
    /// MOVES - Move to/from address space using SFC/DFC.
    /// In emulation, we treat this as a normal memory access since we don't
    /// emulate separate address spaces.
    pub fn exec_moves<B: AddressBus>(&mut self, bus: &mut B, opcode: u16) -> i32 {
        // Check supervisor mode
        if self.s_flag == 0 {
            return self.take_exception(bus, 8); // Privilege violation
        }

        let size = match (opcode >> 6) & 3 {
            0 => Size::Byte,
            1 => Size::Word,
            2 => Size::Long,
            _ => return 0, // Invalid
        };

        let ea_mode = ((opcode >> 3) & 7) as u8;
        let ea_reg = (opcode & 7) as u8;

        let ext = self.read_imm_16(bus);
        let reg_num = ((ext >> 12) & 0xF) as usize;
        let is_areg = reg_num >= 8;
        let reg_idx = reg_num & 7;
        let direction = (ext >> 11) & 1; // 0 = EA to reg, 1 = reg to EA

        let mode = match AddressingMode::decode(ea_mode, ea_reg) {
            Some(m) => m,
            None => return 0,
        };

        let addr = self.get_ea_address(bus, mode, size);

        if direction == 1 {
            // Register to EA (write)
            let value = if is_areg {
                self.a(reg_idx)
            } else {
                self.d(reg_idx)
            };

            match size {
                Size::Byte => self.write_8(bus, addr, value as u8),
                Size::Word => self.write_16(bus, addr, value as u16),
                Size::Long => self.write_32(bus, addr, value),
            }
        } else {
            // EA to register (read)
            let value = match size {
                Size::Byte => self.read_8(bus, addr) as u32,
                Size::Word => self.read_16(bus, addr) as u32,
                Size::Long => self.read_32(bus, addr),
            };

            if is_areg {
                // Sign extend for address register
                let extended = match size {
                    Size::Byte => value as i8 as i32 as u32,
                    Size::Word => value as i16 as i32 as u32,
                    Size::Long => value,
                };
                self.set_a(reg_idx, extended);
            } else {
                match size {
                    Size::Byte => self.set_d(reg_idx, (self.d(reg_idx) & 0xFFFFFF00) | value),
                    Size::Word => self.set_d(reg_idx, (self.d(reg_idx) & 0xFFFF0000) | value),
                    Size::Long => self.set_d(reg_idx, value),
                }
            }
        }

        // Condition codes are not affected
        4
    }
}

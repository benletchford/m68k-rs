// MOVE16 - 16-byte Aligned Block Transfer (68040 only)
//
// Opcode (Ax)+,(Ay)+: 1111 0110 0010 0yyy with extension 1xxx 0000 0000 0000
// Transfers 16 bytes from source to destination, both addresses aligned to 16-byte boundary

use crate::core::cpu::CpuCore;
use crate::core::memory::AddressBus;

impl CpuCore {
    /// MOVE16 - 16-byte aligned block transfer (68040 only).
    /// Format: (Ax)+,(Ay)+
    /// Opcode: 1111 0110 0010 0yyy, extension: 1xxx 0000 0000 0000
    pub fn exec_move16<B: AddressBus>(&mut self, bus: &mut B, opcode: u16) -> i32 {
        // Read extension word to get destination register
        let ext = self.read_imm_16(bus);

        // Note: For MOVE16 (Ax)+,(Ay)+:
        // - Opcode bits 2:0 = source register Ax
        // - Extension bits 14:12 = destination register Ay
        // (This matches gas assembler output)
        let src_reg = (opcode & 7) as usize;
        let dst_reg = ((ext >> 12) & 7) as usize;

        // MOVE16 requires 16-byte alignment for both addresses.
        let src_raw = self.a(src_reg);
        if (src_raw & 0xF) != 0 {
            self.trigger_address_error(bus, src_raw, false, false);
            return 0;
        }
        let dst_raw = self.a(dst_reg);
        if (dst_raw & 0xF) != 0 {
            self.trigger_address_error(bus, dst_raw, true, false);
            return 0;
        }

        // Get source and destination addresses (already aligned)
        let src_addr = src_raw;
        let dst_addr = dst_raw;

        // Transfer 16 bytes (4 longwords) - use bus directly for transfers
        for i in 0u32..4 {
            let offset = i * 4;
            let value = bus.read_long(src_addr + offset);
            bus.write_long(dst_addr + offset, value);
        }

        // Increment both registers by 16
        self.set_a(src_reg, self.a(src_reg).wrapping_add(16));
        self.set_a(dst_reg, self.a(dst_reg).wrapping_add(16));

        // Condition codes are not affected
        4
    }
}

//! BCD (Binary Coded Decimal) instructions.
//!
//! ABCD, SBCD, NBCD

use crate::core::cpu::{CFLAG_SET, CpuCore, NFLAG_SET, XFLAG_SET};
use crate::core::ea::AddressingMode;
use crate::core::memory::AddressBus;
use crate::core::types::Size;

impl CpuCore {
    /// Execute ABCD register-to-register.
    ///
    /// ABCD Dy, Dx
    pub fn exec_abcd_rr(&mut self, src_reg: usize, dst_reg: usize) -> i32 {
        let src = self.d(src_reg) & 0xFF;
        let dst = self.d(dst_reg) & 0xFF;
        let result = self.bcd_add(src, dst);

        self.set_d(dst_reg, (self.d(dst_reg) & 0xFFFFFF00) | result);
        6
    }

    /// Execute ABCD memory-to-memory.
    ///
    /// ABCD -(Ay), -(Ax)
    pub fn exec_abcd_mm<B: AddressBus>(
        &mut self,
        bus: &mut B,
        src_reg: usize,
        dst_reg: usize,
    ) -> i32 {
        // Pre-decrement both
        let src_dec = if src_reg == 7 { 2 } else { 1 };
        let src_addr = self.a(src_reg).wrapping_sub(src_dec);
        self.set_a(src_reg, src_addr);
        let dst_dec = if dst_reg == 7 { 2 } else { 1 };
        let dst_addr = self.a(dst_reg).wrapping_sub(dst_dec);
        self.set_a(dst_reg, dst_addr);

        let src = self.read_8(bus, src_addr) as u32;
        let dst = self.read_8(bus, dst_addr) as u32;
        let result = self.bcd_add(src, dst);

        self.write_8(bus, dst_addr, result as u8);
        18
    }

    /// Execute SBCD register-to-register.
    ///
    /// SBCD Dy, Dx
    pub fn exec_sbcd_rr(&mut self, src_reg: usize, dst_reg: usize) -> i32 {
        let src = self.d(src_reg) & 0xFF;
        let dst = self.d(dst_reg) & 0xFF;
        let result = self.bcd_sub(src, dst);

        self.set_d(dst_reg, (self.d(dst_reg) & 0xFFFFFF00) | result);
        6
    }

    /// Execute SBCD memory-to-memory.
    ///
    /// SBCD -(Ay), -(Ax)
    pub fn exec_sbcd_mm<B: AddressBus>(
        &mut self,
        bus: &mut B,
        src_reg: usize,
        dst_reg: usize,
    ) -> i32 {
        // Pre-decrement both
        let src_dec = if src_reg == 7 { 2 } else { 1 };
        let src_addr = self.a(src_reg).wrapping_sub(src_dec);
        self.set_a(src_reg, src_addr);
        let dst_dec = if dst_reg == 7 { 2 } else { 1 };
        let dst_addr = self.a(dst_reg).wrapping_sub(dst_dec);
        self.set_a(dst_reg, dst_addr);

        let src = self.read_8(bus, src_addr) as u32;
        let dst = self.read_8(bus, dst_addr) as u32;
        let result = self.bcd_sub(src, dst);

        self.write_8(bus, dst_addr, result as u8);
        18
    }

    /// Execute NBCD (negate BCD).
    ///
    /// NBCD <ea>
    pub fn exec_nbcd<B: AddressBus>(&mut self, bus: &mut B, mode: AddressingMode) -> i32 {
        let is_reg = mode.is_register_direct();
        let ea = self.resolve_ea(bus, mode, Size::Byte);
        let dst = self.read_resolved_ea(bus, ea, Size::Byte);
        if self.sst_m68000_compat {
            // SingleStepTests/MAME fixtures treat NBCD as a BCD subtraction helper.
            let res = self.bcd_sub_sst(dst, 0);
            self.write_resolved_ea(bus, ea, Size::Byte, res);
            return if is_reg { 6 } else { 8 };
        }
        // Match Musashi's NBCD behavior.
        // See `tests/fixtures/Musashi/m68k_in.c` `M68KMAKE_OP(nbcd, 8, ...)`.
        let x = if self.x_flag != 0 { 1u32 } else { 0 };
        let dst8 = dst & 0xFF;
        let mut res = 0x9Au32.wrapping_sub(dst8).wrapping_sub(x) & 0xFF;

        let mut should_write = false;
        if res != 0x9A {
            self.v_flag = !res;

            if (res & 0x0F) == 0x0A {
                res = (res & 0xF0).wrapping_add(0x10);
            }
            res &= 0xFF;
            self.v_flag &= res;

            // Z is sticky.
            self.not_z_flag |= res;
            self.c_flag = CFLAG_SET;
            self.x_flag = XFLAG_SET;
            should_write = true;
        } else {
            self.v_flag = 0;
            self.c_flag = 0;
            self.x_flag = 0;
        }
        self.n_flag = if (res & 0x80) != 0 { NFLAG_SET } else { 0 };

        // Musashi uses res==0x9A as a sentinel for "no change" (this occurs only when dst==0 and X==0).
        if should_write {
            self.write_resolved_ea(bus, ea, Size::Byte, res);
        }

        if is_reg { 6 } else { 8 }
    }

    // ========== BCD Helpers ==========

    /// SingleStepTests/MAME-style ABCD behavior (including "invalid digit" cases).
    fn bcd_add_sst(&mut self, src: u32, dst: u32) -> u32 {
        let x = if self.x_flag != 0 { 1u32 } else { 0 };
        let src = src & 0xFF;
        let dst = dst & 0xFF;

        let lo = (src & 0x0F).wrapping_add(dst & 0x0F).wrapping_add(x);
        let mut res = src.wrapping_add(dst).wrapping_add(x);
        if lo > 9 {
            res = res.wrapping_add(0x06);
        }
        // SingleStepTests behavior differs from Musashi: carry detection threshold is 0x9F.
        let carry = res > 0x9F;
        if carry {
            res = res.wrapping_add(0x60);
        }

        let res8 = res & 0xFF;
        self.x_flag = if carry { XFLAG_SET } else { 0 };
        self.c_flag = if carry { CFLAG_SET } else { 0 };
        if res8 != 0 {
            self.not_z_flag = res8;
        }
        res8
    }

    /// SingleStepTests/MAME-style SBCD behavior (including "invalid digit" cases).
    fn bcd_sub_sst(&mut self, src: u32, dst: u32) -> u32 {
        let x = if self.x_flag != 0 { 1i32 } else { 0i32 };
        let src = (src & 0xFF) as i32;
        let dst = (dst & 0xFF) as i32;

        let base = dst - src - x;
        let low_borrow = ((dst & 0x0F) - (src & 0x0F) - x) < 0;
        let borrow = base < 0;

        let mut res = base;
        if low_borrow {
            res -= 6;
        }
        let xc = res < 0 || borrow;
        if borrow {
            res -= 0x60;
        }

        let res8 = (res as u32) & 0xFF;
        self.x_flag = if xc { XFLAG_SET } else { 0 };
        self.c_flag = if xc { CFLAG_SET } else { 0 };
        if res8 != 0 {
            self.not_z_flag = res8;
        }
        res8
    }

    /// Perform BCD addition: src + dst + X
    fn bcd_add(&mut self, src: u32, dst: u32) -> u32 {
        if self.sst_m68000_compat {
            return self.bcd_add_sst(src, dst);
        }
        // Match Musashi's ABCD behavior (including its deterministic-but-"undefined" N/V).
        // See `tests/fixtures/Musashi/m68k_in.c` `M68KMAKE_OP(abcd, 8, ...)`.
        let x = if self.x_flag != 0 { 1u32 } else { 0 };
        let src = src & 0xFF;
        let dst = dst & 0xFF;

        let mut res = (src & 0x0F).wrapping_add(dst & 0x0F).wrapping_add(x);
        self.v_flag = !res;

        if res > 9 {
            res = res.wrapping_add(6);
        }
        res = res.wrapping_add(src & 0xF0).wrapping_add(dst & 0xF0);

        let carry = res > 0x99;
        self.x_flag = if carry { XFLAG_SET } else { 0 };
        self.c_flag = if carry { CFLAG_SET } else { 0 };
        if carry {
            res = res.wrapping_sub(0xA0);
        }

        self.v_flag &= res;
        self.n_flag = if (res & 0x80) != 0 { NFLAG_SET } else { 0 };

        let res8 = res & 0xFF;
        self.not_z_flag |= res8;

        res8
    }

    /// Perform BCD subtraction: dst - src - X
    fn bcd_sub(&mut self, src: u32, dst: u32) -> u32 {
        if self.sst_m68000_compat {
            return self.bcd_sub_sst(src, dst);
        }
        // Match Musashi's SBCD behavior (including deterministic-but-"undefined" N/V).
        // See `tests/fixtures/Musashi/m68k_in.c` `M68KMAKE_OP(sbcd, 8, ...)`.
        let x = if self.x_flag != 0 { 1u32 } else { 0 };
        let src = src & 0xFF;
        let dst = dst & 0xFF;

        let mut res = (dst & 0x0F).wrapping_sub(src & 0x0F).wrapping_sub(x);
        self.v_flag = !res;

        // Note: in unsigned arithmetic, an underflow will produce a large value (>9).
        if res > 9 {
            res = res.wrapping_sub(6);
        }
        res = res.wrapping_add(dst & 0xF0).wrapping_sub(src & 0xF0);

        let carry = res > 0x99;
        self.x_flag = if carry { XFLAG_SET } else { 0 };
        self.c_flag = if carry { CFLAG_SET } else { 0 };
        if carry {
            res = res.wrapping_add(0xA0);
        }

        let res8 = res & 0xFF;
        self.v_flag &= res8;
        self.n_flag = if (res8 & 0x80) != 0 { NFLAG_SET } else { 0 };
        self.not_z_flag |= res8;

        res8
    }

    // ========== PACK/UNPK (68020+) ==========

    /// Execute PACK register-to-register (68020+).
    ///
    /// PACK Ds, Dd, #adj
    /// Result = ((src[11:8] << 4) | src[3:0]) + adj
    pub fn exec_pack_rr(&mut self, src_reg: usize, dst_reg: usize, adj: u16) -> i32 {
        let src = self.d(src_reg) & 0xFFFF;
        let packed = (((src >> 8) & 0xF) << 4) | (src & 0xF);
        let result = (packed + adj as u32) & 0xFF;
        self.set_d(dst_reg, (self.d(dst_reg) & 0xFFFFFF00) | result);
        6
    }

    /// Execute PACK memory-to-memory (68020+).
    ///
    /// PACK -(As), -(Ad), #adj
    pub fn exec_pack_mm<B: AddressBus>(
        &mut self,
        bus: &mut B,
        src_reg: usize,
        dst_reg: usize,
        adj: u16,
    ) -> i32 {
        // Read source word from predecrement
        let src_addr = self.a(src_reg).wrapping_sub(2);
        self.set_a(src_reg, src_addr);
        let src = self.read_16(bus, src_addr) as u32;

        let packed = (((src >> 8) & 0xF) << 4) | (src & 0xF);
        let result = ((packed + adj as u32) & 0xFF) as u8;

        // Write destination byte to predecrement
        let dst_addr = self.a(dst_reg).wrapping_sub(1);
        self.set_a(dst_reg, dst_addr);
        self.write_8(bus, dst_addr, result);
        13
    }

    /// Execute UNPK register-to-register (68020+).
    ///
    /// UNPK Ds, Dd, #adj
    /// Result = ((src[7:4] << 8) | src[3:0]) + adj
    pub fn exec_unpk_rr(&mut self, src_reg: usize, dst_reg: usize, adj: u16) -> i32 {
        let src = self.d(src_reg) & 0xFF;
        let unpacked = (((src >> 4) & 0xF) << 8) | (src & 0xF);
        let result = (unpacked + adj as u32) & 0xFFFF;
        self.set_d(dst_reg, (self.d(dst_reg) & 0xFFFF0000) | result);
        8
    }

    /// Execute UNPK memory-to-memory (68020+).
    ///
    /// UNPK -(As), -(Ad), #adj
    pub fn exec_unpk_mm<B: AddressBus>(
        &mut self,
        bus: &mut B,
        src_reg: usize,
        dst_reg: usize,
        adj: u16,
    ) -> i32 {
        // Read source byte from predecrement
        let src_addr = self.a(src_reg).wrapping_sub(1);
        self.set_a(src_reg, src_addr);
        let src = self.read_8(bus, src_addr) as u32;

        let unpacked = (((src >> 4) & 0xF) << 8) | (src & 0xF);
        let result = ((unpacked + adj as u32) & 0xFFFF) as u16;

        // Write destination word to predecrement
        let dst_addr = self.a(dst_reg).wrapping_sub(2);
        self.set_a(dst_reg, dst_addr);
        self.write_16(bus, dst_addr, result);
        13
    }
}

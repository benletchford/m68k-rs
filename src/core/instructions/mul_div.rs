//! Multiply and divide instructions.
//!
//! MULS, MULU, DIVS, DIVU

use crate::core::cpu::CpuCore;
use crate::core::ea::AddressingMode;
use crate::core::execute::RUN_MODE_BERR_AERR_RESET;
use crate::core::memory::AddressBus;
use crate::core::types::Size;

impl CpuCore {
    /// Execute MULU (unsigned 16x16 -> 32 multiply).
    ///
    /// MULU <ea>, Dn
    pub fn exec_mulu<B: AddressBus>(
        &mut self,
        bus: &mut B,
        mode: AddressingMode,
        dst_reg: usize,
    ) -> i32 {
        let src = self.read_ea(bus, mode, Size::Word) & 0xFFFF;
        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            // Address/bus error while reading the operand: exception has been taken.
            return 50;
        }
        let dst = self.d(dst_reg) & 0xFFFF;
        let result = src * dst;

        self.set_d(dst_reg, result);

        // Set flags
        self.not_z_flag = result;
        self.n_flag = if result & 0x80000000 != 0 { 0x80 } else { 0 };
        self.v_flag = 0;
        self.c_flag = 0;

        // Base cycle time + variable based on bits set in multiplier
        38
    }

    /// Execute MULS (signed 16x16 -> 32 multiply).
    ///
    /// MULS <ea>, Dn
    pub fn exec_muls<B: AddressBus>(
        &mut self,
        bus: &mut B,
        mode: AddressingMode,
        dst_reg: usize,
    ) -> i32 {
        let src = self.read_ea(bus, mode, Size::Word) as i16 as i32;
        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            // Address/bus error while reading the operand: exception has been taken.
            return 50;
        }
        let dst = self.d(dst_reg) as i16 as i32;
        let result = (src * dst) as u32;

        self.set_d(dst_reg, result);

        // Set flags
        self.not_z_flag = result;
        self.n_flag = if result & 0x80000000 != 0 { 0x80 } else { 0 };
        self.v_flag = 0;
        self.c_flag = 0;

        38
    }

    /// Execute DIVU (unsigned 32÷16 -> 16Q + 16R).
    ///
    /// DIVU <ea>, Dn
    /// Result: Dn[31:16] = remainder, Dn[15:0] = quotient
    pub fn exec_divu<B: AddressBus>(
        &mut self,
        bus: &mut B,
        mode: AddressingMode,
        dst_reg: usize,
    ) -> i32 {
        let src = self.read_ea(bus, mode, Size::Word) & 0xFFFF;
        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            // Address/bus error while reading the operand: exception has been taken.
            return 50;
        }
        let dst = self.d(dst_reg);

        if src == 0 {
            // Division by zero - trigger trap
            return self.exception_zero_divide(bus);
        }

        let quotient = dst / src;
        let remainder = dst % src;

        // Check for overflow (quotient must fit in 16 bits)
        if quotient >= 0x10000 {
            self.v_flag = 0x80;
            if self.sst_m68000_compat {
                // SingleStepTests/MAME fixtures expect deterministic N/Z on overflow.
                self.n_flag = 0x80;
                self.not_z_flag = 1; // Z=0
                self.c_flag = 0;
            }
            return 140; // Worst case timing
        }

        self.set_d(dst_reg, (remainder << 16) | (quotient & 0xFFFF));

        self.not_z_flag = quotient;
        self.n_flag = if quotient & 0x8000 != 0 { 0x80 } else { 0 };
        self.v_flag = 0;
        self.c_flag = 0;

        140
    }

    /// Execute DIVS (signed 32÷16 -> 16Q + 16R).
    ///
    /// DIVS <ea>, Dn
    /// Result: Dn[31:16] = remainder, Dn[15:0] = quotient
    pub fn exec_divs<B: AddressBus>(
        &mut self,
        bus: &mut B,
        mode: AddressingMode,
        dst_reg: usize,
    ) -> i32 {
        let src = self.read_ea(bus, mode, Size::Word) as i16 as i32;
        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            // Address/bus error while reading the operand: exception has been taken.
            return 50;
        }
        let dst = self.d(dst_reg) as i32;

        if src == 0 {
            // Division by zero - trigger trap
            return self.exception_zero_divide(bus);
        }

        // Special case: 0x80000000 / -1 = 0x80000000 (would overflow)
        // But Musashi returns quotient=0, remainder=0 for this
        if dst == i32::MIN && src == -1 {
            self.set_d(dst_reg, 0);
            self.not_z_flag = 0;
            self.n_flag = 0;
            self.v_flag = 0;
            self.c_flag = 0;
            return 158;
        }

        let quotient = dst / src;
        let remainder = dst % src;

        // Check for overflow (quotient must fit in signed 16 bits: -32768 to 32767)
        if !(-32768..=32767).contains(&quotient) {
            self.v_flag = 0x80;
            if self.sst_m68000_compat {
                // SingleStepTests/MAME fixtures expect deterministic N/Z on overflow.
                self.n_flag = 0x80;
                self.not_z_flag = 1; // Z=0
                self.c_flag = 0;
            }
            return 158;
        }

        let quotient_u16 = quotient as i16 as u16 as u32;
        let remainder_u16 = remainder as i16 as u16 as u32;
        self.set_d(dst_reg, (remainder_u16 << 16) | quotient_u16);

        self.not_z_flag = quotient_u16;
        self.n_flag = if quotient_u16 & 0x8000 != 0 { 0x80 } else { 0 };
        self.v_flag = 0;
        self.c_flag = 0;

        158
    }
}

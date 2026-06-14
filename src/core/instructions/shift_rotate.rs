//! Shift and rotate instructions.
//!
//! ASL, ASR, LSL, LSR, ROL, ROR, ROXL, ROXR

use crate::core::cpu::{CFLAG_SET, CpuCore};
use crate::core::types::{CpuType, Size};

impl CpuCore {
    /// Register shift/rotate base cost. On the 68000 a long operation needs two
    /// extra clocks (base 8 vs 6 for byte/word); the variable part is `2 * count`
    /// on top. Other sizes and CPU types keep the base of 6.
    #[inline]
    fn shift_rot_base(&self, size: Size) -> i32 {
        if matches!(
            self.cpu_type,
            CpuType::M68000 | CpuType::M68010 | CpuType::SCC68070
        ) && size == Size::Long
        {
            8
        } else {
            6
        }
    }

    /// Execute ASL (Arithmetic Shift Left).
    pub fn exec_asl(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let shift = shift & 63;
        if shift == 0 {
            self.c_flag = 0;
            self.set_logic_flags(value, size);
            return (value, self.shift_rot_base(size));
        }

        let mask = size.mask();
        let msb = size.msb_mask();
        let bits = size.bits();

        let mut result = value & mask;
        let mut last_bit = 0u32;
        let mut overflow = false;

        for _ in 0..shift.min(bits as u32) {
            last_bit = result & msb;
            let new_top = (result << 1) & msb;
            if new_top != last_bit {
                overflow = true;
            }
            result = (result << 1) & mask;
        }

        // Carry/X rules:
        // - If shift == bits: carry is the last bit shifted out (equivalent to original bit0).
        // - If shift > bits: result is 0 and carry is cleared.
        self.c_flag = if shift > bits as u32 {
            0
        } else if last_bit != 0 {
            CFLAG_SET
        } else {
            0
        };
        self.x_flag = self.c_flag;
        self.v_flag = if overflow { 0x80 } else { 0 };
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * shift as i32)
    }

    /// Execute ASR (Arithmetic Shift Right).
    pub fn exec_asr(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let shift = shift & 63;
        if shift == 0 {
            self.c_flag = 0;
            self.set_logic_flags(value, size);
            return (value, self.shift_rot_base(size));
        }

        let mask = size.mask();
        let msb = size.msb_mask();
        let bits = size.bits();

        let sign = value & msb;
        let value = value & mask;
        let result = if shift >= bits as u32 {
            if sign != 0 { mask } else { 0 }
        } else if sign != 0 {
            (value >> shift) | (mask << (bits as u32 - shift))
        } else {
            value >> shift
        } & mask;
        let last_bit = if shift >= bits as u32 {
            sign
        } else {
            value & (1u32 << (shift - 1))
        };

        self.c_flag = if last_bit != 0 { CFLAG_SET } else { 0 };
        self.x_flag = self.c_flag;
        self.v_flag = 0; // ASR never sets overflow
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * shift as i32)
    }

    /// Execute LSL (Logical Shift Left).
    pub fn exec_lsl(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let shift = shift & 63;
        if shift == 0 {
            self.c_flag = 0;
            self.set_logic_flags(value, size);
            return (value, self.shift_rot_base(size));
        }

        let mask = size.mask();
        let bits = size.bits();

        let result = if shift >= bits as u32 {
            self.c_flag = if shift == bits as u32 && (value & 1) != 0 {
                CFLAG_SET
            } else {
                0
            };
            0
        } else {
            let last_out = (value >> (bits as u32 - shift)) & 1;
            self.c_flag = if last_out != 0 { CFLAG_SET } else { 0 };
            (value << shift) & mask
        };

        self.x_flag = self.c_flag;
        self.v_flag = 0;
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * shift as i32)
    }

    /// Execute LSR (Logical Shift Right).
    pub fn exec_lsr(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let shift = shift & 63;
        if shift == 0 {
            self.c_flag = 0;
            self.set_logic_flags(value, size);
            return (value, self.shift_rot_base(size));
        }

        let mask = size.mask();
        let bits = size.bits();
        let value = value & mask;

        let result = if shift >= bits as u32 {
            self.c_flag = if shift == bits as u32 && (value & size.msb_mask()) != 0 {
                CFLAG_SET
            } else {
                0
            };
            0
        } else {
            let last_out = (value >> (shift - 1)) & 1;
            self.c_flag = if last_out != 0 { CFLAG_SET } else { 0 };
            value >> shift
        };

        self.x_flag = self.c_flag;
        self.v_flag = 0;
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * shift as i32)
    }

    /// Execute ROL (Rotate Left).
    pub fn exec_rol(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let bits = size.bits() as u32;
        let mask = size.mask();
        let cnt = shift & 63;

        if cnt == 0 {
            let result = value & mask;
            // No rotation occurs. On 68000, C is cleared; X is unchanged; V cleared; N/Z from result.
            self.c_flag = 0;
            self.v_flag = 0;
            self.set_logic_flags_nv(result, size);
            return (result, self.shift_rot_base(size));
        }

        // Counts that are multiples of operand size still perform a full cycle
        // (result unchanged but C reflects the last rotated-out bit).
        let mut steps = cnt % bits;
        if steps == 0 {
            steps = bits;
        }

        let value = value & mask;
        let carry = (value >> (bits - steps)) & 1;
        let result = if steps == bits {
            value
        } else {
            ((value << steps) | (value >> (bits - steps))) & mask
        };

        self.c_flag = if carry != 0 { CFLAG_SET } else { 0 };
        self.v_flag = 0;
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * cnt as i32)
    }

    /// Execute ROR (Rotate Right).
    pub fn exec_ror(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let bits = size.bits() as u32;
        let mask = size.mask();
        let cnt = shift & 63;

        if cnt == 0 {
            let result = value & mask;
            // No rotation occurs. On 68000, C is cleared; X is unchanged; V cleared; N/Z from result.
            self.c_flag = 0;
            self.v_flag = 0;
            self.set_logic_flags_nv(result, size);
            return (result, self.shift_rot_base(size));
        }

        let mut steps = cnt % bits;
        if steps == 0 {
            steps = bits;
        }

        let value = value & mask;
        let carry = (value >> (steps - 1)) & 1;
        let result = if steps == bits {
            value
        } else {
            ((value >> steps) | (value << (bits - steps))) & mask
        };

        self.c_flag = if carry != 0 { CFLAG_SET } else { 0 };
        self.v_flag = 0;
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * cnt as i32)
    }

    /// Execute ROXL (Rotate Left through X).
    pub fn exec_roxl(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let bits = size.bits() as u32;
        let mask = size.mask();
        // Timing counts the full shift count; the rotation itself repeats every
        // (bits + 1) positions because X participates as an extra bit.
        let steps = shift % (bits + 1);

        if steps == 0 {
            let result = value & mask;
            // No net rotation; X is unaffected. C mirrors X; V cleared; N/Z from result.
            self.c_flag = self.x_flag;
            self.v_flag = 0;
            self.set_logic_flags_nv(result, size);
            return (result, self.shift_rot_base(size) + 2 * shift as i32);
        }

        let mut result = value & mask;
        let mut x = if self.x_flag != 0 { 1u32 } else { 0 };

        for _ in 0..steps {
            let carry = (result >> (bits - 1)) & 1;
            result = ((result << 1) | x) & mask;
            x = carry;
        }

        self.c_flag = if x != 0 { CFLAG_SET } else { 0 };
        self.x_flag = self.c_flag;
        self.v_flag = 0;
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * shift as i32)
    }

    /// Execute ROXR (Rotate Right through X).
    pub fn exec_roxr(&mut self, size: Size, shift: u32, value: u32) -> (u32, i32) {
        let bits = size.bits() as u32;
        let mask = size.mask();
        let msb = size.msb_mask();
        // Timing counts the full shift count; the rotation itself repeats every
        // (bits + 1) positions because X participates as an extra bit.
        let steps = shift % (bits + 1);

        if steps == 0 {
            let result = value & mask;
            // No net rotation; X is unaffected. C mirrors X; V cleared; N/Z from result.
            self.c_flag = self.x_flag;
            self.v_flag = 0;
            self.set_logic_flags_nv(result, size);
            return (result, self.shift_rot_base(size) + 2 * shift as i32);
        }

        let mut result = value & mask;
        let mut x = if self.x_flag != 0 { 1u32 } else { 0 };

        for _ in 0..steps {
            let carry = result & 1;
            result = (result >> 1) | (if x != 0 { msb } else { 0 });
            x = carry;
        }

        self.c_flag = if x != 0 { CFLAG_SET } else { 0 };
        self.x_flag = self.c_flag;
        self.v_flag = 0;
        self.set_logic_flags_nv(result, size);

        (result, self.shift_rot_base(size) + 2 * shift as i32)
    }

    /// Helper: set N and Z flags only (V already set by caller).
    fn set_logic_flags_nv(&mut self, value: u32, size: Size) {
        let msb = size.msb_mask();
        self.n_flag = if value & msb != 0 { 0x80 } else { 0 };
        self.not_z_flag = value & size.mask();
    }
}

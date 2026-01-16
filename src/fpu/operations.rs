//! FPU operations (68040/68881-class).
//!
//! Note: This is currently a **minimal bring-up** focused on plumbing + a few
//! OS-critical operations. Expect expansion over time.

use crate::core::cpu::CpuCore;
use crate::core::memory::AddressBus;

impl CpuCore {
    /// 68040 FPU "op0" entrypoint (opcode pattern 0xF2xx in Musashi: `040fpu0`).
    ///
    /// For now this is a stub (future: ALU ops, FMOVE FP,<ea>, FMOVEM, FScc/FBcc, etc.).
    pub fn exec_fpu_op0<B: AddressBus>(&mut self, bus: &mut B, opcode: u16) -> i32 {
        use crate::core::types::CpuType;

        // LC040 and EC040 don't have integrated FPUs - must trap as Line-F
        if matches!(self.cpu_type, CpuType::M68LC040 | CpuType::M68EC040) {
            return 0;
        }

        // IMPORTANT:
        // - PC currently points at the first extension word (w2).
        // - We must NOT consume w2 (or any EA extension) unless we handle the instruction.

        let w2 = self.read_16(bus, self.pc);
        let subop = (w2 >> 13) & 0x7;

        match subop {
            0x2 => {
                // FPU ALU <ea>, FPn - includes FMOVE, FADD, FSUB, FMUL, FDIV, FCMP from memory
                let src_fmt = (w2 >> 10) & 0x7;
                let dst = ((w2 >> 7) & 7) as usize;
                let mut opmode = w2 & 0x7f;
                // Handle Musashi-style rounding modifiers embedded in opmode.
                if (opmode & 0x44) == 0x44 {
                    opmode &= !0x44;
                } else if (opmode & 0x40) != 0 {
                    opmode &= !0x40;
                }

                // Consume w2 now that we're committed.
                let _w2 = self.read_imm_16(bus);

                // Read source value based on format
                let src_value: Option<f64> = match src_fmt {
                    0 => {
                        // Long integer
                        let v = self.exec_fpu_read_ea_long(bus, opcode);
                        if v.is_none() {
                            return 0;
                        }
                        Some(v.unwrap() as f64)
                    }
                    1 => {
                        // Single precision float
                        let v = self.exec_fpu_read_ea_single(bus, opcode);
                        if v.is_none() {
                            return 0;
                        }
                        Some(v.unwrap() as f64)
                    }
                    2 => {
                        // Extended precision float (96-bit)
                        let v = self.exec_fpu_read_ea_extended(bus, opcode);
                        if v.is_none() {
                            return 0;
                        }
                        Some(v.unwrap())
                    }
                    5 => {
                        // Double precision float
                        let v = self.exec_fpu_read_ea_double(bus, opcode);
                        if v.is_none() {
                            return 0;
                        }
                        Some(v.unwrap())
                    }
                    7 => {
                        // FMOVECR - load constant from ROM
                        // The opmode field contains the ROM offset
                        let rom_offset = opmode as usize;
                        let constant = match rom_offset {
                            0x00 => std::f64::consts::PI,      // Pi
                            0x0B => std::f64::consts::LOG10_2, // log10(2)
                            0x0C => std::f64::consts::E,       // e
                            0x0D => std::f64::consts::LN_2,    // log_e(2) = ln(2)
                            0x0E => std::f64::consts::LN_10,   // log_e(10) = ln(10)
                            0x0F => 0.0,                       // Zero
                            0x30 => std::f64::consts::LN_2,    // ln(2)
                            0x31 => std::f64::consts::LN_10,   // ln(10)
                            0x32 => 1.0,                       // 1.0
                            0x33 => 10.0,                      // 10.0
                            0x34 => 100.0,                     // 10^2
                            0x35 => 1.0e4,                     // 10^4
                            0x36 => 1.0e8,                     // 10^8
                            0x37 => 1.0e16,                    // 10^16
                            0x38 => 1.0e32,                    // 10^32
                            0x39 => 1.0e64,                    // 10^64
                            0x3A => 1.0e128,                   // 10^128
                            0x3B => 1.0e256,                   // 10^256
                            // Higher powers would overflow, return infinity
                            0x3C..=0x3F => f64::INFINITY,
                            _ => 0.0, // Unknown constant, return 0
                        };
                        self.fpr[dst] = constant;
                        self.fpu_set_cc(self.fpr[dst]);
                        return 4;
                    }
                    _ => return 0, // Other formats not yet implemented
                };

                let src = src_value.unwrap();

                match opmode {
                    0x00 => {
                        // FMOVE <ea>, FPn
                        self.fpr[dst] = src;
                        self.fpu_set_cc(src);
                        4
                    }
                    0x20 => {
                        // FDIV <ea>, FPn
                        if src == 0.0 {
                            if self.fpr[dst] == 0.0 {
                                // 0/0 = NaN, set OPERR
                                self.fpr[dst] = f64::NAN;
                                self.fpsr |= 0x20; // OPERR
                            } else {
                                // x/0 = Inf, set DZ
                                self.fpr[dst] = if self.fpr[dst] < 0.0 {
                                    f64::NEG_INFINITY
                                } else {
                                    f64::INFINITY
                                };
                                self.fpsr |= 0x10; // DZ
                            }
                        } else {
                            self.fpr[dst] /= src;
                        }
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x22 => {
                        // FADD <ea>, FPn
                        self.fpr[dst] += src;
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x23 => {
                        // FMUL <ea>, FPn
                        self.fpr[dst] *= src;
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x28 => {
                        // FSUB <ea>, FPn
                        self.fpr[dst] -= src;
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x38 => {
                        // FCMP <ea>, FPn
                        let diff = self.fpr[dst] - src;
                        self.fpu_set_cc(diff);
                        4
                    }
                    _ => 0, // Unimplemented opmode
                }
            }
            0x3 => {
                // FMOVE FP, <ea> - move FP register to memory/integer register
                let dst_fmt = (w2 >> 10) & 0x7;
                let src = ((w2 >> 7) & 7) as usize;

                // Consume w2 now that we're committed.
                let _w2 = self.read_imm_16(bus);

                let ea = (opcode & 0x3f) as u8;
                let ea_mode = (ea >> 3) & 7;
                let ea_reg = (ea & 7) as usize;

                match dst_fmt {
                    0 => {
                        // Format 0: Long integer
                        let int_val = self.fpr[src] as i32;
                        if ea_mode == 0 {
                            // Dn
                            self.set_d(ea_reg, int_val as u32);
                        } else {
                            // Memory - get address and write
                            let addr = self.get_fpu_ea_address(bus, ea_mode, ea_reg);
                            if addr == 0 && ea_mode != 0 {
                                return 0;
                            }
                            self.write_32(bus, addr, int_val as u32);
                        }
                        4
                    }
                    1 => {
                        // Format 1: Single precision float
                        let single_val = (self.fpr[src] as f32).to_bits();
                        if ea_mode == 0 {
                            // Dn
                            self.set_d(ea_reg, single_val);
                        } else {
                            let addr = self.get_fpu_ea_address(bus, ea_mode, ea_reg);
                            if addr == 0 && ea_mode != 0 {
                                return 0;
                            }
                            self.write_32(bus, addr, single_val);
                        }
                        4
                    }
                    5 => {
                        // Format 5: Double precision float
                        let double_bits = self.fpr[src].to_bits();
                        let addr = self.get_fpu_ea_address(bus, ea_mode, ea_reg);
                        if addr == 0 && ea_mode != 0 {
                            return 0;
                        }
                        let hi = (double_bits >> 32) as u32;
                        let lo = (double_bits & 0xFFFF_FFFF) as u32;
                        self.write_32(bus, addr, hi);
                        self.write_32(bus, addr.wrapping_add(4), lo);
                        4
                    }
                    _ => 0, // Other formats not implemented
                }
            }
            0x0 => {
                // FP register-to-register operations (FMOVE FPm,FPn, FADD, FSUB, FMUL, FDIV, FCMP, etc.)
                let src = ((w2 >> 10) & 7) as usize;
                let dst = ((w2 >> 7) & 7) as usize;
                let opmode = w2 & 0x7f;

                // Consume w2
                let _ = self.read_imm_16(bus);

                match opmode {
                    0x00 => {
                        // FMOVE FPm, FPn
                        self.fpr[dst] = self.fpr[src];
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x01 => {
                        // FINT FPm, FPn - round to integer using FPCR rounding mode
                        let rounding_mode = (self.fpcr >> 4) & 0x3;
                        let val = self.fpr[src];
                        self.fpr[dst] = match rounding_mode {
                            0 => val.round(), // RN - Round to Nearest
                            1 => val.trunc(), // RZ - Round toward Zero
                            2 => val.floor(), // RM - Round toward Minus Infinity
                            3 => val.ceil(),  // RP - Round toward Plus Infinity
                            _ => val.round(),
                        };
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x03 => {
                        // FINTRZ FPm, FPn - round to integer toward zero
                        self.fpr[dst] = self.fpr[src].trunc();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x04 | 0x44 | 0x45 => {
                        // FSQRT FPm, FPn
                        self.fpr[dst] = self.fpr[src].sqrt();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x18 | 0x58 | 0x5C => {
                        // FABS FPm, FPn
                        self.fpr[dst] = self.fpr[src].abs();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x1A | 0x5A | 0x5E => {
                        // FNEG FPm, FPn
                        self.fpr[dst] = -self.fpr[src];
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x20 | 0x60 | 0x64 => {
                        // FDIV FPm, FPn (0x20), with rounding variants
                        if self.fpr[src] == 0.0 {
                            if self.fpr[dst] == 0.0 {
                                // 0/0 = NaN, set OPERR
                                self.fpr[dst] = f64::NAN;
                                self.fpsr |= 0x20; // OPERR
                            } else {
                                // x/0 = Inf, set DZ
                                self.fpr[dst] = if self.fpr[dst] < 0.0 {
                                    f64::NEG_INFINITY
                                } else {
                                    f64::INFINITY
                                };
                                self.fpsr |= 0x10; // DZ
                            }
                        } else {
                            self.fpr[dst] /= self.fpr[src];
                        }
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x22 | 0x62 | 0x66 => {
                        // FADD FPm, FPn with rounding variants
                        self.fpr[dst] += self.fpr[src];
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x23 | 0x63 | 0x67 => {
                        // FMUL FPm, FPn with rounding variants
                        self.fpr[dst] *= self.fpr[src];
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x28 | 0x68 | 0x6C => {
                        // FSUB FPm, FPn with rounding variants
                        self.fpr[dst] -= self.fpr[src];
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x38 => {
                        // FCMP FPm, FPn
                        let diff = self.fpr[dst] - self.fpr[src];
                        self.fpu_set_cc(diff);
                        4
                    }
                    0x3A => {
                        // FTST FPm - test and set condition codes (doesn't write dst)
                        self.fpu_set_cc(self.fpr[src]);
                        4
                    }
                    0x17 => {
                        // FMOVECR - load constant from ROM
                        // The src field contains the ROM offset
                        let rom_offset = src;
                        let constant = match rom_offset {
                            0x00 => std::f64::consts::PI,      // Pi
                            0x0B => std::f64::consts::LOG10_2, // log10(2)
                            0x0C => std::f64::consts::E,       // e
                            0x0D => std::f64::consts::LN_2,    // log_e(2) = ln(2)
                            0x0E => std::f64::consts::LN_10,   // log_e(10) = ln(10)
                            0x0F => 0.0,                       // Zero
                            0x30 => std::f64::consts::LN_2,    // ln(2)
                            0x31 => std::f64::consts::LN_10,   // ln(10)
                            0x32 => 1.0,                       // 1.0
                            0x33 => 10.0,                      // 10.0
                            0x34 => 100.0,                     // 10^2
                            0x35 => 1.0e4,                     // 10^4
                            0x36 => 1.0e8,                     // 10^8
                            0x37 => 1.0e16,                    // 10^16
                            0x38 => 1.0e32,                    // 10^32
                            0x39 => 1.0e64,                    // 10^64
                            0x3A => 1.0e128,                   // 10^128
                            0x3B => 1.0e256,                   // 10^256
                            // Higher powers would overflow, return infinity
                            0x3C..=0x3F => f64::INFINITY,
                            _ => 0.0, // Unknown constant, return 0
                        };
                        self.fpr[dst] = constant;
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    // ========== Transcendental Functions ==========
                    0x0E => {
                        // FSIN FPm, FPn
                        self.fpr[dst] = self.fpr[src].sin();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x1D => {
                        // FCOS FPm, FPn
                        self.fpr[dst] = self.fpr[src].cos();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x0F => {
                        // FTAN FPm, FPn
                        self.fpr[dst] = self.fpr[src].tan();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x0C => {
                        // FASIN FPm, FPn
                        self.fpr[dst] = self.fpr[src].asin();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x1C => {
                        // FACOS FPm, FPn
                        self.fpr[dst] = self.fpr[src].acos();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x0A => {
                        // FATAN FPm, FPn
                        self.fpr[dst] = self.fpr[src].atan();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x02 => {
                        // FSINH FPm, FPn
                        self.fpr[dst] = self.fpr[src].sinh();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x19 => {
                        // FCOSH FPm, FPn
                        self.fpr[dst] = self.fpr[src].cosh();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x09 => {
                        // FTANH FPm, FPn
                        self.fpr[dst] = self.fpr[src].tanh();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x0D => {
                        // FATANH FPm, FPn
                        self.fpr[dst] = self.fpr[src].atanh();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x10 => {
                        // FETOX FPm, FPn (e^x)
                        self.fpr[dst] = self.fpr[src].exp();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x08 => {
                        // FETOXM1 FPm, FPn (e^x - 1)
                        self.fpr[dst] = self.fpr[src].exp_m1();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x11 => {
                        // FTWOTOX FPm, FPn (2^x)
                        self.fpr[dst] = (2.0_f64).powf(self.fpr[src]);
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x12 => {
                        // FTENTOX FPm, FPn (10^x)
                        self.fpr[dst] = (10.0_f64).powf(self.fpr[src]);
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x14 => {
                        // FLOGN FPm, FPn (ln(x))
                        self.fpr[dst] = self.fpr[src].ln();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x06 => {
                        // FLOGNP1 FPm, FPn (ln(1+x))
                        self.fpr[dst] = self.fpr[src].ln_1p();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x15 => {
                        // FLOG10 FPm, FPn (log10(x))
                        self.fpr[dst] = self.fpr[src].log10();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x16 => {
                        // FLOG2 FPm, FPn (log2(x))
                        self.fpr[dst] = self.fpr[src].log2();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x1E => {
                        // FGETEXP FPm, FPn - extract exponent
                        let val = self.fpr[src];
                        if val == 0.0 || val.is_nan() || val.is_infinite() {
                            self.fpr[dst] = if val.is_nan() || val.is_infinite() {
                                f64::NAN
                            } else {
                                0.0
                            };
                        } else {
                            // IEEE 754 double: sign (1 bit) | exponent (11 bits) | mantissa (52 bits)
                            let bits = val.to_bits();
                            let biased_exp = ((bits >> 52) & 0x7FF) as i32;
                            let exp = biased_exp - 1023; // Remove bias
                            self.fpr[dst] = exp as f64;
                        }
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x1F => {
                        // FGETMAN FPm, FPn - extract mantissa as 1.xxx
                        let val = self.fpr[src];
                        if val == 0.0 {
                            self.fpr[dst] = 0.0;
                        } else if val.is_nan() || val.is_infinite() {
                            self.fpr[dst] = val; // Keep special values
                        } else {
                            // Extract mantissa and set exponent to 0 (bias 1023)
                            let bits = val.to_bits();
                            let sign = bits & (1 << 63);
                            let mantissa_bits = bits & 0x000F_FFFF_FFFF_FFFF;
                            // Construct 1.mantissa with exponent 0 (biased 1023)
                            let result_bits = sign | (1023_u64 << 52) | mantissa_bits;
                            self.fpr[dst] = f64::from_bits(result_bits);
                        }
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x21 => {
                        // FMOD FPm, FPn
                        self.fpr[dst] %= self.fpr[src];
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x25 => {
                        // FREM FPm, FPn (IEEE remainder)
                        let src_val = self.fpr[src];
                        let dst_val = self.fpr[dst];
                        // IEEE remainder: r = x - y*round(x/y)
                        if src_val != 0.0 {
                            let n = (dst_val / src_val).round();
                            self.fpr[dst] = dst_val - src_val * n;
                        }
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x26 => {
                        // FSCALE FPm, FPn - multiply by power of 2
                        // dst = dst * 2^src
                        let scale = self.fpr[src] as i32;
                        self.fpr[dst] *= (2.0_f64).powi(scale);
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    0x30..=0x37 => {
                        // FSINCOS FPm, FPc:FPs - compute sin and cos simultaneously
                        // Bottom 3 bits (opmode & 7) = cos destination register
                        let cos_dst = (opmode & 7) as usize;
                        let val = self.fpr[src];
                        self.fpr[dst] = val.sin();
                        self.fpr[cos_dst] = val.cos();
                        self.fpu_set_cc(self.fpr[dst]);
                        4
                    }
                    _ => 0, // Unimplemented opmode
                }
            }
            0x6 | 0x7 => {
                // FMOVEM - move multiple FP registers to/from memory
                // subop 0x6: memory to FP registers (restore)
                // subop 0x7: FP registers to memory (save)
                let direction = subop;
                let reg_list = w2 & 0xFF;
                let mode_bits = (w2 >> 11) & 0x3;

                // Consume w2
                let _w2 = self.read_imm_16(bus);

                let ea = (opcode & 0x3f) as u8;
                let ea_mode = (ea >> 3) & 7;
                let ea_reg = (ea & 7) as usize;

                // Get base address
                let mut addr = self.get_fpu_ea_address(bus, ea_mode, ea_reg);
                if addr == 0 && ea_mode != 2 && ea_mode != 3 && ea_mode != 4 {
                    return 0;
                }

                // Handle predecrement specially
                if ea_mode == 4 && direction == 0x7 {
                    // Saving to predecrement - count registers and pre-subtract
                    let reg_count = reg_list.count_ones();
                    addr = self.a(ea_reg).wrapping_sub(reg_count * 12);
                    self.set_a(ea_reg, addr);
                }

                // Extended precision is 12 bytes (96 bits): 16-bit exponent + 64-bit mantissa
                // For simplicity, we store as: 4-byte zero padding + 8-byte double
                if direction == 0x6 {
                    // Memory to FP registers
                    for i in 0..8 {
                        let bit = if mode_bits & 0x1 != 0 {
                            1 << i
                        } else {
                            1 << (7 - i)
                        };
                        if reg_list & bit != 0 {
                            // Skip padding, read double
                            let _pad = self.read_32(bus, addr);
                            let hi = self.read_32(bus, addr.wrapping_add(4)) as u64;
                            let lo = self.read_32(bus, addr.wrapping_add(8)) as u64;
                            self.fpr[i] = f64::from_bits((hi << 32) | lo);
                            addr = addr.wrapping_add(12);
                        }
                    }
                } else {
                    // FP registers to memory
                    for i in 0..8 {
                        let bit = if mode_bits & 0x1 != 0 {
                            1 << i
                        } else {
                            1 << (7 - i)
                        };
                        if reg_list & bit != 0 {
                            let val = self.fpr[i].to_bits();
                            // Write as: 4-byte zero padding + 8-byte double
                            self.write_32(bus, addr, 0);
                            self.write_32(bus, addr.wrapping_add(4), (val >> 32) as u32);
                            self.write_32(bus, addr.wrapping_add(8), (val & 0xFFFF_FFFF) as u32);
                            addr = addr.wrapping_add(12);
                        }
                    }
                }

                // Handle post-increment
                if ea_mode == 3 {
                    self.set_a(ea_reg, addr);
                }

                8
            }
            0x4 => {
                // FMOVE <ea>, control register (FPCR, FPSR, FPIAR)
                // or FMOVEM <ea>, control register list
                let ctrl_sel = (w2 >> 10) & 0x7;
                let _w2 = self.read_imm_16(bus);

                let ea = (opcode & 0x3f) as u8;
                let ea_mode = (ea >> 3) & 7;
                let ea_reg = (ea & 7) as usize;

                // Read source value
                let src: u32 = if ea_mode == 0 {
                    self.d(ea_reg)
                } else if ea_mode == 7 && ea_reg == 4 {
                    // Immediate mode
                    self.read_imm_32(bus)
                } else {
                    let addr = self.get_fpu_ea_address(bus, ea_mode, ea_reg);
                    self.read_32(bus, addr)
                };

                // Single register moves: ctrl_sel is 4 (FPCR), 2 (FPSR), or 1 (FPIAR)
                if ctrl_sel & 0x4 != 0 {
                    self.fpcr = src;
                }
                if ctrl_sel & 0x2 != 0 {
                    self.fpsr = src;
                }
                if ctrl_sel & 0x1 != 0 {
                    self.fpiar = src;
                }

                4
            }
            0x5 => {
                // FMOVE control register, <ea> (FPCR, FPSR, FPIAR)
                let ctrl_sel = (w2 >> 10) & 0x7;
                let _w2 = self.read_imm_16(bus);

                let ea = (opcode & 0x3f) as u8;
                let ea_mode = (ea >> 3) & 7;
                let ea_reg = (ea & 7) as usize;

                // For single register: ctrl_sel is 4 (FPCR), 2 (FPSR), or 1 (FPIAR)
                let value = if ctrl_sel == 4 {
                    self.fpcr
                } else if ctrl_sel == 2 {
                    self.fpsr
                } else if ctrl_sel == 1 {
                    self.fpiar
                } else {
                    self.fpcr
                };

                if ea_mode == 0 {
                    self.set_d(ea_reg, value);
                } else {
                    let addr = self.get_fpu_ea_address(bus, ea_mode, ea_reg);
                    // Handle FMOVEM with multiple control regs
                    let mut cur_addr = addr;
                    if ctrl_sel & 0x4 != 0 {
                        self.write_32(bus, cur_addr, self.fpcr);
                        cur_addr = cur_addr.wrapping_add(4);
                    }
                    if ctrl_sel & 0x2 != 0 {
                        self.write_32(bus, cur_addr, self.fpsr);
                        cur_addr = cur_addr.wrapping_add(4);
                    }
                    if ctrl_sel & 0x1 != 0 {
                        self.write_32(bus, cur_addr, self.fpiar);
                    }
                }

                4
            }
            _ => 0,
        }
    }

    /// FBcc - FPU conditional branch.
    ///
    /// Note: The PC has already been advanced past the displacement when this is called.
    pub fn exec_fbcc(&mut self, condition: u8, disp: i32) -> i32 {
        // Extract FPSR condition codes
        const FPCC_N: u32 = 0x0800_0000;
        const FPCC_Z: u32 = 0x0400_0000;
        const FPCC_I: u32 = 0x0200_0000;
        const FPCC_NAN: u32 = 0x0100_0000;

        let n = (self.fpsr & FPCC_N) != 0;
        let z = (self.fpsr & FPCC_Z) != 0;
        let _i = (self.fpsr & FPCC_I) != 0;
        let nan = (self.fpsr & FPCC_NAN) != 0;

        // Evaluate FPU condition (simplified for common cases)
        let take_branch = match condition {
            0x00 => false,            // FBF - never
            0x01 => z,                // FBEQ
            0x0E => !nan && !z,       // FBNE (ordered not equal)
            0x0F => true,             // FBT - always
            0x10 => false,            // FBSF
            0x11 => z,                // FBSEQ
            0x12 => !(nan || z || n), // FBGT
            0x13 => z || !(nan || n), // FBGE
            0x14 => n && !(nan || z), // FBLT
            0x15 => z || (n && !nan), // FBLE
            0x16 => !(nan || z),      // FBGL
            0x17 => !nan,             // FBGLE
            0x18 => nan,              // FBNGLE
            0x19 => nan || z,         // FBNGL
            0x1A => nan || !(n || z), // FBNLE
            0x1B => nan || z || !n,   // FBNLT
            0x1C => nan || (n && !z), // FBNGE
            0x1D => nan || n || z,    // FBNGT
            0x1E => !z,               // FBSNE
            0x1F => true,             // FBST
            _ => false,
        };

        if take_branch {
            self.change_of_flow = true;
            // PC was already advanced past displacement; adjust relative to that position
            // Compute target: (PC - disp_size) + disp
            // Since PC is after displacement, we compute: base_pc + disp
            // where base_pc is the address of the first extension word
            let base_pc = self.ppc.wrapping_add(2); // ppc is opcode, +2 is extension word
            self.pc = (base_pc as i32).wrapping_add(disp) as u32;
        }

        8
    }

    /// FScc - Set byte on FPU condition.
    pub fn exec_fscc<B: AddressBus>(
        &mut self,
        bus: &mut B,
        ea_mode: u8,
        ea_reg: usize,
        condition: u8,
    ) -> i32 {
        // Extract FPSR condition codes
        const FPCC_N: u32 = 0x0800_0000;
        const FPCC_Z: u32 = 0x0400_0000;
        const FPCC_I: u32 = 0x0200_0000;
        const FPCC_NAN: u32 = 0x0100_0000;

        let n = (self.fpsr & FPCC_N) != 0;
        let z = (self.fpsr & FPCC_Z) != 0;
        let _i = (self.fpsr & FPCC_I) != 0;
        let nan = (self.fpsr & FPCC_NAN) != 0;

        // Evaluate FPU condition
        let cond_true = match condition {
            0x00 => false,            // SF - never
            0x01 => z,                // SEQ
            0x0E => !nan && !z,       // SNE (ordered not equal)
            0x0F => true,             // ST - always
            0x10 => false,            // SF
            0x11 => z,                // SEQ
            0x12 => !(nan || z || n), // SGT
            0x13 => z || !(nan || n), // SGE
            0x14 => n && !(nan || z), // SLT
            0x15 => z || (n && !nan), // SLE
            0x16 => !(nan || z),      // SGL
            0x17 => !nan,             // SGLE
            0x18 => nan,              // SNGLE
            0x19 => nan || z,         // SNGL
            0x1A => nan || !(n || z), // SNLE
            0x1B => nan || z || !n,   // SNLT
            0x1C => nan || (n && !z), // SNGE
            0x1D => nan || n || z,    // SNGT
            0x1E => !z,               // SSNE
            0x1F => true,             // SST
            _ => false,
        };

        let value = if cond_true { 0xFFu8 } else { 0x00u8 };

        if ea_mode == 0 {
            // Data register Dn
            self.set_d(ea_reg, (self.d(ea_reg) & 0xFFFFFF00) | value as u32);
        } else {
            // Memory
            let addr = self.get_fpu_ea_address(bus, ea_mode, ea_reg);
            self.write_8(bus, addr, value);
        }

        4
    }

    /// 68040 FPU "op1" entrypoint (opcode pattern 0xF3xx in Musashi: `040fpu1`).
    ///
    /// Implements a minimal subset: `FSAVE <ea>` and `FRESTORE <ea>` for a NULL/IDLE frame.
    pub fn exec_fpu_op1<B: AddressBus>(&mut self, bus: &mut B, opcode: u16) -> i32 {
        let ea_mode = ((opcode >> 3) & 7) as u8;
        let ea_reg = (opcode & 7) as usize;
        let op = ((opcode >> 6) & 3) as u8;

        match op {
            0 => self.exec_fsave(bus, ea_mode, ea_reg),
            1 => self.exec_frestore(bus, ea_mode, ea_reg),
            _ => 0, // unsupported -> let caller raise LINE1111 without consuming extensions
        }
    }

    fn exec_fsave<B: AddressBus>(&mut self, bus: &mut B, ea_mode: u8, ea_reg: usize) -> i32 {
        // Musashi supports only (An)+ and -(An) here for 68040.
        match ea_mode {
            3 => {
                // (An)+
                let addr = self.a(ea_reg);
                self.set_a(ea_reg, addr.wrapping_add(4));

                if self.fpu_just_reset {
                    self.write_32(bus, addr, 0);
                } else {
                    // Total frame size is 7 longs (28 bytes). EA increment already did +4.
                    self.set_a(ea_reg, self.a(ea_reg).wrapping_add(6 * 4));
                    perform_fsave(bus, self, addr, true);
                }
                8
            }
            4 => {
                // -(An)
                let addr_hi = self.a(ea_reg).wrapping_sub(4);
                self.set_a(ea_reg, addr_hi);

                if self.fpu_just_reset {
                    self.write_32(bus, addr_hi, 0);
                } else {
                    // Total frame size is 28 bytes; one predecrement already happened (-4).
                    self.set_a(ea_reg, self.a(ea_reg).wrapping_sub(6 * 4));
                    perform_fsave(bus, self, addr_hi, false);
                }
                8
            }
            _ => 0,
        }
    }

    fn exec_frestore<B: AddressBus>(&mut self, bus: &mut B, ea_mode: u8, ea_reg: usize) -> i32 {
        match ea_mode {
            2 => {
                // (An)
                let addr = self.a(ea_reg);
                let header = self.read_32(bus, addr);
                if (header & 0xFF00_0000) == 0 {
                    self.do_frestore_null();
                } else {
                    self.fpu_just_reset = false;
                }
                8
            }
            3 => {
                // (An)+
                let addr = self.a(ea_reg);
                self.set_a(ea_reg, addr.wrapping_add(4));
                let header = self.read_32(bus, addr);

                if (header & 0xFF00_0000) == 0 {
                    self.do_frestore_null();
                } else {
                    self.fpu_just_reset = false;

                    // Musashi adjusts A-reg by additional bytes based on frame type.
                    // (EA macro already did +4.)
                    let kind = header & 0x00FF_0000;
                    let extra = match kind {
                        0x0018_0000 => 6 * 4,  // IDLE
                        0x0038_0000 => 14 * 4, // UNIMP
                        0x00B4_0000 => 45 * 4, // BUSY
                        _ => 0,
                    };
                    self.set_a(ea_reg, self.a(ea_reg).wrapping_add(extra));
                }
                8
            }
            _ => 0,
        }
    }

    fn do_frestore_null(&mut self) {
        self.fpcr = 0;
        self.fpsr = 0;
        self.fpiar = 0;
        self.fpr = [f64::NAN; 8];
        self.fpu_just_reset = true;
    }

    /// Set FPU condition codes based on a floating point value.
    fn fpu_set_cc(&mut self, value: f64) {
        const FPCC_N: u32 = 0x0800_0000;
        const FPCC_Z: u32 = 0x0400_0000;
        const FPCC_I: u32 = 0x0200_0000;
        const FPCC_NAN: u32 = 0x0100_0000;

        self.fpsr &= !(FPCC_N | FPCC_Z | FPCC_I | FPCC_NAN);
        if value.is_nan() {
            self.fpsr |= FPCC_NAN;
        } else if value.is_infinite() {
            self.fpsr |= FPCC_I;
            if value < 0.0 {
                self.fpsr |= FPCC_N;
            }
        } else if value == 0.0 {
            self.fpsr |= FPCC_Z;
            // Check for -0.0 by examining sign bit
            if value.to_bits() >> 63 != 0 {
                self.fpsr |= FPCC_N;
            }
        } else if value < 0.0 {
            self.fpsr |= FPCC_N;
        }
    }

    /// Get effective address for FPU memory operations.
    fn get_fpu_ea_address<B: AddressBus>(
        &mut self,
        bus: &mut B,
        ea_mode: u8,
        ea_reg: usize,
    ) -> u32 {
        match ea_mode {
            2 => self.a(ea_reg),
            3 => {
                // Note: caller must handle post-increment based on data size
                self.a(ea_reg)
            }
            4 => {
                // Note: caller must handle pre-decrement based on data size
                self.a(ea_reg)
            }
            5 => {
                let disp = self.read_imm_16(bus) as i16 as i32;
                (self.a(ea_reg) as i32).wrapping_add(disp) as u32
            }
            7 => match ea_reg {
                0 => self.read_imm_16(bus) as i16 as i32 as u32,
                1 => self.read_imm_32(bus),
                2 => {
                    let pc = self.pc;
                    let disp = self.read_imm_16(bus) as i16 as i32;
                    (pc as i32).wrapping_add(disp) as u32
                }
                _ => 0,
            },
            _ => 0,
        }
    }

    /// Read a 32-bit integer from FPU EA (for format 0).
    fn exec_fpu_read_ea_long<B: AddressBus>(&mut self, bus: &mut B, opcode: u16) -> Option<i32> {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        match ea_mode {
            0 => Some(self.d(ea_reg) as i32),
            2 => Some(self.read_32(bus, self.a(ea_reg)) as i32),
            3 => {
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(4));
                Some(self.read_32(bus, a) as i32)
            }
            4 => {
                let a = self.a(ea_reg).wrapping_sub(4);
                self.set_a(ea_reg, a);
                Some(self.read_32(bus, a) as i32)
            }
            5 => {
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (self.a(ea_reg) as i32).wrapping_add(disp) as u32;
                Some(self.read_32(bus, addr) as i32)
            }
            7 => match ea_reg {
                0 => {
                    let addr = self.read_imm_16(bus) as i16 as i32 as u32;
                    Some(self.read_32(bus, addr) as i32)
                }
                1 => {
                    let addr = self.read_imm_32(bus);
                    Some(self.read_32(bus, addr) as i32)
                }
                2 => {
                    let pc = self.pc;
                    let disp = self.read_imm_16(bus) as i16 as i32;
                    Some(self.read_32(bus, (pc as i32).wrapping_add(disp) as u32) as i32)
                }
                4 => Some(self.read_imm_32(bus) as i32),
                _ => None,
            },
            _ => None,
        }
    }

    /// Read a 96-bit extended precision float from FPU EA (for format 2).
    /// Extended format: 16-bit sign+exponent, 16-bit padding, 64-bit mantissa = 12 bytes
    fn exec_fpu_read_ea_extended<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
    ) -> Option<f64> {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        // Helper to convert 80-bit extended to f64
        fn extended_to_f64(exp_word: u16, mantissa: u64) -> f64 {
            let sign = (exp_word >> 15) & 1;
            let exp = (exp_word & 0x7FFF) as i32;

            if exp == 0 && mantissa == 0 {
                return if sign != 0 { -0.0 } else { 0.0 };
            }
            if exp == 0x7FFF {
                return if mantissa == 0 {
                    if sign != 0 {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                } else {
                    f64::NAN
                };
            }

            // Bias for 80-bit: 16383, for 64-bit: 1023
            let biased_exp = exp - 16383 + 1023;
            if biased_exp <= 0 || biased_exp >= 2047 {
                // Overflow/underflow - simplified handling
                return if biased_exp >= 2047 {
                    if sign != 0 {
                        f64::NEG_INFINITY
                    } else {
                        f64::INFINITY
                    }
                } else {
                    0.0
                };
            }

            // Extended has explicit integer bit, double doesn't
            // Take top 52 bits of mantissa (after the explicit 1)
            let frac = (mantissa << 1) >> 12;
            let bits = ((sign as u64) << 63) | ((biased_exp as u64) << 52) | frac;
            f64::from_bits(bits)
        }

        match ea_mode {
            2 => {
                let addr = self.a(ea_reg);
                let exp_word = self.read_16(bus, addr);
                // Skip 16-bit padding
                let hi = self.read_32(bus, addr.wrapping_add(4)) as u64;
                let lo = self.read_32(bus, addr.wrapping_add(8)) as u64;
                Some(extended_to_f64(exp_word, (hi << 32) | lo))
            }
            3 => {
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(12));
                let exp_word = self.read_16(bus, a);
                let hi = self.read_32(bus, a.wrapping_add(4)) as u64;
                let lo = self.read_32(bus, a.wrapping_add(8)) as u64;
                Some(extended_to_f64(exp_word, (hi << 32) | lo))
            }
            4 => {
                let a = self.a(ea_reg).wrapping_sub(12);
                self.set_a(ea_reg, a);
                let exp_word = self.read_16(bus, a);
                let hi = self.read_32(bus, a.wrapping_add(4)) as u64;
                let lo = self.read_32(bus, a.wrapping_add(8)) as u64;
                Some(extended_to_f64(exp_word, (hi << 32) | lo))
            }
            5 => {
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (self.a(ea_reg) as i32).wrapping_add(disp) as u32;
                let exp_word = self.read_16(bus, addr);
                let hi = self.read_32(bus, addr.wrapping_add(4)) as u64;
                let lo = self.read_32(bus, addr.wrapping_add(8)) as u64;
                Some(extended_to_f64(exp_word, (hi << 32) | lo))
            }
            7 => match ea_reg {
                0 => {
                    let addr = self.read_imm_16(bus) as i16 as i32 as u32;
                    let exp_word = self.read_16(bus, addr);
                    let hi = self.read_32(bus, addr.wrapping_add(4)) as u64;
                    let lo = self.read_32(bus, addr.wrapping_add(8)) as u64;
                    Some(extended_to_f64(exp_word, (hi << 32) | lo))
                }
                1 => {
                    let addr = self.read_imm_32(bus);
                    let exp_word = self.read_16(bus, addr);
                    let hi = self.read_32(bus, addr.wrapping_add(4)) as u64;
                    let lo = self.read_32(bus, addr.wrapping_add(8)) as u64;
                    Some(extended_to_f64(exp_word, (hi << 32) | lo))
                }
                2 => {
                    let pc = self.pc;
                    let disp = self.read_imm_16(bus) as i16 as i32;
                    let addr = (pc as i32).wrapping_add(disp) as u32;
                    let exp_word = self.read_16(bus, addr);
                    let hi = self.read_32(bus, addr.wrapping_add(4)) as u64;
                    let lo = self.read_32(bus, addr.wrapping_add(8)) as u64;
                    Some(extended_to_f64(exp_word, (hi << 32) | lo))
                }
                _ => None,
            },
            _ => None,
        }
    }

    /// Read a 32-bit single precision float from FPU EA (for format 1).
    fn exec_fpu_read_ea_single<B: AddressBus>(&mut self, bus: &mut B, opcode: u16) -> Option<f32> {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        let bits: Option<u32> = match ea_mode {
            0 => Some(self.d(ea_reg)),
            2 => Some(self.read_32(bus, self.a(ea_reg))),
            3 => {
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(4));
                Some(self.read_32(bus, a))
            }
            4 => {
                let a = self.a(ea_reg).wrapping_sub(4);
                self.set_a(ea_reg, a);
                Some(self.read_32(bus, a))
            }
            5 => {
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (self.a(ea_reg) as i32).wrapping_add(disp) as u32;
                Some(self.read_32(bus, addr))
            }
            7 => match ea_reg {
                0 => {
                    let addr = self.read_imm_16(bus) as i16 as i32 as u32;
                    Some(self.read_32(bus, addr))
                }
                1 => {
                    let addr = self.read_imm_32(bus);
                    Some(self.read_32(bus, addr))
                }
                2 => {
                    let pc = self.pc;
                    let disp = self.read_imm_16(bus) as i16 as i32;
                    Some(self.read_32(bus, (pc as i32).wrapping_add(disp) as u32))
                }
                4 => Some(self.read_imm_32(bus)),
                _ => None,
            },
            _ => None,
        };

        bits.map(f32::from_bits)
    }

    /// Read a 64-bit double precision float from FPU EA (for format 5).
    fn exec_fpu_read_ea_double<B: AddressBus>(&mut self, bus: &mut B, opcode: u16) -> Option<f64> {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        let bits: Option<u64> = match ea_mode {
            2 => {
                let addr = self.a(ea_reg);
                let hi = self.read_32(bus, addr) as u64;
                let lo = self.read_32(bus, addr.wrapping_add(4)) as u64;
                Some((hi << 32) | lo)
            }
            3 => {
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(8));
                let hi = self.read_32(bus, a) as u64;
                let lo = self.read_32(bus, a.wrapping_add(4)) as u64;
                Some((hi << 32) | lo)
            }
            4 => {
                let a = self.a(ea_reg).wrapping_sub(8);
                self.set_a(ea_reg, a);
                let hi = self.read_32(bus, a) as u64;
                let lo = self.read_32(bus, a.wrapping_add(4)) as u64;
                Some((hi << 32) | lo)
            }
            5 => {
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (self.a(ea_reg) as i32).wrapping_add(disp) as u32;
                let hi = self.read_32(bus, addr) as u64;
                let lo = self.read_32(bus, addr.wrapping_add(4)) as u64;
                Some((hi << 32) | lo)
            }
            7 => match ea_reg {
                0 => {
                    let addr = self.read_imm_16(bus) as i16 as i32 as u32;
                    let hi = self.read_32(bus, addr) as u64;
                    let lo = self.read_32(bus, addr.wrapping_add(4)) as u64;
                    Some((hi << 32) | lo)
                }
                1 => {
                    let addr = self.read_imm_32(bus);
                    let hi = self.read_32(bus, addr) as u64;
                    let lo = self.read_32(bus, addr.wrapping_add(4)) as u64;
                    Some((hi << 32) | lo)
                }
                2 => {
                    let pc = self.pc;
                    let disp = self.read_imm_16(bus) as i16 as i32;
                    let addr = (pc as i32).wrapping_add(disp) as u32;
                    let hi = self.read_32(bus, addr) as u64;
                    let lo = self.read_32(bus, addr.wrapping_add(4)) as u64;
                    Some((hi << 32) | lo)
                }
                4 => {
                    let hi = self.read_imm_32(bus) as u64;
                    let lo = self.read_imm_32(bus) as u64;
                    Some((hi << 32) | lo)
                }
                _ => None,
            },
            _ => None,
        };

        bits.map(f64::from_bits)
    }
}

fn perform_fsave<B: AddressBus>(bus: &mut B, cpu: &mut CpuCore, addr: u32, inc: bool) {
    // Generate a 68881-style "IDLE" frame as Musashi does for 68040 FSAVE.
    // This is sufficient for many OSes that only probe save/restore behavior.
    if inc {
        cpu.write_32(bus, addr, 0x1F18_0000);
        cpu.write_32(bus, addr.wrapping_add(4), 0);
        cpu.write_32(bus, addr.wrapping_add(8), 0);
        cpu.write_32(bus, addr.wrapping_add(12), 0);
        cpu.write_32(bus, addr.wrapping_add(16), 0);
        cpu.write_32(bus, addr.wrapping_add(20), 0);
        cpu.write_32(bus, addr.wrapping_add(24), 0x7000_0000);
    } else {
        cpu.write_32(bus, addr, 0x7000_0000);
        cpu.write_32(bus, addr.wrapping_sub(4), 0);
        cpu.write_32(bus, addr.wrapping_sub(8), 0);
        cpu.write_32(bus, addr.wrapping_sub(12), 0);
        cpu.write_32(bus, addr.wrapping_sub(16), 0);
        cpu.write_32(bus, addr.wrapping_sub(20), 0);
        cpu.write_32(bus, addr.wrapping_sub(24), 0x1F18_0000);
    }
}

// TODO: These functions are scaffolding for full FPU FMOVE implementation.
// They will be wired up when we complete FPU support.
#[allow(dead_code)]
impl CpuCore {
    /// FMOVE.L <ea>, FPn - move 32-bit integer to FP register
    fn exec_fmove_ea_long_to_fp<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> i32 {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        // dst fp reg: bits 9..7 of w2
        let dst = ((w2 >> 7) & 7) as usize;

        // Read 32-bit integer from source EA
        let int_value: i32 = match ea_mode {
            0 => {
                // Dn - data register direct
                self.d(ea_reg) as i32
            }
            2 => {
                // (An)
                let addr = self.a(ea_reg);
                self.read_32(bus, addr) as i32
            }
            3 => {
                // (An)+
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(4));
                self.read_32(bus, a) as i32
            }
            4 => {
                // -(An)
                let a = self.a(ea_reg).wrapping_sub(4);
                self.set_a(ea_reg, a);
                self.read_32(bus, a) as i32
            }
            5 => {
                // (d16,An)
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (self.a(ea_reg) as i32).wrapping_add(disp) as u32;
                self.read_32(bus, addr) as i32
            }
            7 => match ea_reg {
                0 => {
                    // (xxx).W
                    let addr = self.read_imm_16(bus) as i16 as i32 as u32;
                    self.read_32(bus, addr) as i32
                }
                1 => {
                    // (xxx).L
                    let addr = self.read_imm_32(bus);
                    self.read_32(bus, addr) as i32
                }
                4 => {
                    // #<data> - immediate
                    self.read_imm_32(bus) as i32
                }
                _ => return 0,
            },
            _ => return 0,
        };

        // Convert integer to f64 and store
        let value = int_value as f64;
        self.fpr[dst] = value;

        // Update FPSR condition codes
        const FPCC_N: u32 = 0x0800_0000;
        const FPCC_Z: u32 = 0x0400_0000;
        const FPCC_NAN: u32 = 0x0100_0000;
        self.fpsr &= !(FPCC_N | FPCC_Z | FPCC_NAN);
        if value == 0.0 {
            self.fpsr |= FPCC_Z;
        }
        if value < 0.0 {
            self.fpsr |= FPCC_N;
        }

        4
    }

    /// FMOVE.S <ea>, FPn - move 32-bit single precision float to FP register
    fn exec_fmove_ea_single_to_fp<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> i32 {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        // dst fp reg: bits 9..7 of w2
        let dst = ((w2 >> 7) & 7) as usize;

        // Read 32-bit single precision float from source EA
        let raw_bits: u32 = match ea_mode {
            0 => {
                // Dn - data register direct
                self.d(ea_reg)
            }
            2 => {
                // (An)
                let addr = self.a(ea_reg);
                self.read_32(bus, addr)
            }
            3 => {
                // (An)+
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(4));
                self.read_32(bus, a)
            }
            4 => {
                // -(An)
                let a = self.a(ea_reg).wrapping_sub(4);
                self.set_a(ea_reg, a);
                self.read_32(bus, a)
            }
            5 => {
                // (d16,An)
                let disp = self.read_imm_16(bus) as i16 as i32;
                let addr = (self.a(ea_reg) as i32).wrapping_add(disp) as u32;
                self.read_32(bus, addr)
            }
            7 => match ea_reg {
                0 => {
                    // (xxx).W
                    let addr = self.read_imm_16(bus) as i16 as i32 as u32;
                    self.read_32(bus, addr)
                }
                1 => {
                    // (xxx).L
                    let addr = self.read_imm_32(bus);
                    self.read_32(bus, addr)
                }
                2 => {
                    // (d16,PC)
                    let pc = self.pc;
                    let disp = self.read_imm_16(bus) as i16 as i32;
                    let addr = (pc as i32).wrapping_add(disp) as u32;
                    self.read_32(bus, addr)
                }
                4 => {
                    // #<data> - immediate
                    self.read_imm_32(bus)
                }
                _ => return 0,
            },
            _ => return 0,
        };

        // Convert single precision to f64 and store
        let single_value = f32::from_bits(raw_bits);
        let value = single_value as f64;
        self.fpr[dst] = value;

        // Update FPSR condition codes
        const FPCC_N: u32 = 0x0800_0000;
        const FPCC_Z: u32 = 0x0400_0000;
        const FPCC_NAN: u32 = 0x0100_0000;
        self.fpsr &= !(FPCC_N | FPCC_Z | FPCC_NAN);
        if value == 0.0 {
            self.fpsr |= FPCC_Z;
        }
        if value < 0.0 {
            self.fpsr |= FPCC_N;
        }
        if single_value.is_nan() {
            self.fpsr |= FPCC_NAN;
        }

        4
    }

    fn exec_fmove_ea_double_to_fp<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> i32 {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        // Musashi fpgen_rm_reg fields:
        // - dst fp reg: bits 9..7
        let dst = ((w2 >> 7) & 7) as usize;

        let addr = match ea_mode {
            2 => self.a(ea_reg), // (An)
            3 => {
                // (An)+
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(8));
                a
            }
            4 => {
                // -(An)
                let a = self.a(ea_reg).wrapping_sub(8);
                self.set_a(ea_reg, a);
                a
            }
            5 => {
                // (d16,An)
                let disp = self.read_imm_16(bus) as i16 as i32;
                (self.a(ea_reg) as i32).wrapping_add(disp) as u32
            }
            7 => match ea_reg {
                0 => {
                    // (xxx).W
                    self.read_imm_16(bus) as i16 as i32 as u32
                }
                1 => {
                    // (xxx).L
                    self.read_imm_32(bus)
                }
                _ => return 0,
            },
            _ => return 0,
        };

        let bits = read_u64_be(self, bus, addr);
        let value = f64::from_bits(bits);
        self.fpr[dst] = value;

        // Minimal FPSR condition code updates (not yet a full IEEE model):
        // mirror Musashi's FPCC flags layout (top nybble of FPSR).
        const FPCC_N: u32 = 0x0800_0000;
        const FPCC_Z: u32 = 0x0400_0000;
        const FPCC_NAN: u32 = 0x0100_0000;
        self.fpsr &= !(FPCC_N | FPCC_Z | FPCC_NAN);
        if value.is_nan() {
            self.fpsr |= FPCC_NAN;
        } else {
            if value == 0.0 {
                self.fpsr |= FPCC_Z;
            }
            if value.is_sign_negative() && value != 0.0 {
                self.fpsr |= FPCC_N;
            }
        }

        4
    }

    fn exec_fmove_fp_to_ea_double<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> i32 {
        let ea = (opcode & 0x3f) as u8;
        let ea_mode = (ea >> 3) & 7;
        let ea_reg = (ea & 7) as usize;

        // w2 fields (Musashi):
        // - src fp reg: bits 9..7
        let src = ((w2 >> 7) & 7) as usize;
        let value = self.fpr[src];
        let bits = value.to_bits();

        // Resolve destination address with correct 64-bit (8 byte) (An)+/-(An) semantics.
        let addr = match ea_mode {
            2 => self.a(ea_reg), // (An)
            3 => {
                // (An)+
                let a = self.a(ea_reg);
                self.set_a(ea_reg, a.wrapping_add(8));
                a
            }
            4 => {
                // -(An)
                let a = self.a(ea_reg).wrapping_sub(8);
                self.set_a(ea_reg, a);
                a
            }
            5 => {
                // (d16,An)
                let disp = self.read_imm_16(bus) as i16 as i32;
                (self.a(ea_reg) as i32).wrapping_add(disp) as u32
            }
            7 => match ea_reg {
                0 => {
                    // (xxx).W
                    self.read_imm_16(bus) as i16 as i32 as u32
                }
                1 => {
                    // (xxx).L
                    self.read_imm_32(bus)
                }
                _ => return 0,
            },
            _ => return 0,
        };

        write_u64_be(self, bus, addr, bits);
        12
    }
}

#[allow(dead_code)]
fn write_u64_be<B: AddressBus>(cpu: &mut CpuCore, bus: &mut B, addr: u32, value: u64) {
    let hi = (value >> 32) as u32;
    let lo = (value & 0xFFFF_FFFF) as u32;
    cpu.write_32(bus, addr, hi);
    cpu.write_32(bus, addr.wrapping_add(4), lo);
}

#[allow(dead_code)]
fn read_u64_be<B: AddressBus>(cpu: &mut CpuCore, bus: &mut B, addr: u32) -> u64 {
    let hi = cpu.read_32(bus, addr) as u64;
    let lo = cpu.read_32(bus, addr.wrapping_add(4)) as u64;
    (hi << 32) | lo
}

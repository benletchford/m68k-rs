//! MC68060 floating-point arithmetic exceptions and the EXCP state frame
//! FSAVE gives their handlers.
//!
//! `CpuCore::emulate_unimplemented_060` says where the work of the
//! M68060SP is done. Clear, which is the default, the package is guest
//! code and the 68060 behaves as the part does (M68060UM 6.6 and appendix
//! C.3.2.3): overflow and underflow are nonmaskable, the other exceptions
//! are taken when enabled, and in every case the destination register is
//! not written and the EXCP frame holds the source operand, from which the
//! package redoes the instruction. Set, that work is done here: the
//! instruction stores the result a 6888x stores, an exception is taken
//! only when it is enabled, and the frame is the one the package hands the
//! user's handler, with the exception operand of M68060UM 6.7.
//!
//! An exception of an instruction whose destination is a floating-point
//! register is held and taken when the next floating-point instruction is
//! reached, as a pre-instruction exception that restarts it. Every
//! floating-point instruction but FSAVE and FRESTORE reports it. FSAVE
//! writes the EXCP frame ($E0, the vector number less 48 in bits 2-0, the
//! operand in the upper word and the two longs that follow) and leaves the
//! FPU idle; FRESTORE of an EXCP frame puts the FPU back in that state,
//! which is how the package delivers an exception to the user's handler.
//!
//! Not modelled:
//! - The exceptions of an FMOVE to memory or to a data register, which
//!   the 68060 takes as post-instruction exceptions. Those moves complete.
//! - An overflow or underflow of the selected rounding precision that the
//!   extended format can still hold: the arithmetic keeps the extended
//!   exponent range, so it raises neither.
//! - The underflow the 68060 wrongly reports for a product that is a
//!   normalized number with a zero exponent (M68060UM 6.6.5).
//! - A longword integer source appears in the frame converted to
//!   extended, not as the raw integer the 68060 leaves there for some
//!   instructions; the package accepts both.
//! - The manual has an exception stay pending until FSAVE even after it
//!   has been taken, so that a handler which runs another floating-point
//!   instruction first takes it again. Here taking it ends the pending
//!   state; the frame stays for the handler's FSAVE.

use super::operations::RawSource;
use crate::core::cpu::CpuCore;
use crate::core::exceptions::vector;
use crate::fpu::FloatX80;

/// The 68060 FPU's EXCP state: what FSAVE writes as the exception frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Fpu060Excp {
    /// V2-V0 of the frame's status word: the vector number less 48, or 7
    /// for an unsupported data type.
    pub(crate) status: u8,
    /// The exception operand.
    pub(crate) operand: FloatX80,
}

/// V2-V0 of an unsupported-data-type EXCP frame.
const STATUS_UNSUPP: u8 = 7;

impl Fpu060Excp {
    /// The state an arithmetic exception through `vector` leaves.
    pub(crate) fn new(vector: u32, operand: FloatX80) -> Self {
        Self {
            status: (vector - vector::FP_BSUN) as u8,
            operand,
        }
    }

    /// The vector of the exception this state holds.
    pub(crate) fn vector(self) -> u32 {
        if self.status == STATUS_UNSUPP {
            vector::FP_UNSUPP_DATA_TYPE
        } else {
            vector::FP_BSUN + u32::from(self.status)
        }
    }

    /// The three longs of the EXCP frame: the operand's sign and exponent
    /// over the status word, then its mantissa (M68060UM figure 6-10).
    pub(crate) fn frame(self) -> [u32; 3] {
        let (exp_word, mantissa) = self.operand.to_extended();
        [
            (u32::from(exp_word) << 16) | 0xE000 | u32::from(self.status),
            (mantissa >> 32) as u32,
            mantissa as u32,
        ]
    }

    /// The state an EXCP frame describes.
    pub(crate) fn from_frame(longs: [u32; 3]) -> Self {
        Self {
            status: (longs[0] & 7) as u8,
            operand: FloatX80::from_extended(
                (longs[0] >> 16) as u16,
                (u64::from(longs[1]) << 32) | u64::from(longs[2]),
            ),
        }
    }
}

impl CpuCore {
    /// A fetched source operand as the EXCP frame holds it.
    ///
    /// The 68060 widens a single or double without regard to the special
    /// exponents: the fraction goes under an integer bit that is always
    /// set, and the exponent field is rebiased as it is, so a zero or a
    /// denormalized number has the exponent $3F80 or $3C00 and an infinity
    /// or a NaN $407F or $43FF (M68060UM C.3.2.3.2; the package's
    /// fix_skewed_ops puts them right). With the package's work done here
    /// the operand is the corrected one. The package gives an infinity or
    /// a NaN the exponent $7FFF over the fraction alone, the integer bit
    /// clear and a signaling NaN left signaling; 68060.library 46.x was
    /// observed to hand exactly that to the user's handler.
    pub(super) fn fpu_060_source_operand(&self, raw: RawSource, value: FloatX80) -> FloatX80 {
        let (sign, exp_field, fraction, rebias, max) = match raw {
            RawSource::Single(bits) => (
                bits >> 31 != 0,
                ((bits >> 23) & 0xFF) as u16,
                u64::from(bits & 0x007F_FFFF) << 40,
                0x3F80,
                0xFF,
            ),
            RawSource::Double(bits) => (
                bits >> 63 != 0,
                ((bits >> 52) & 0x7FF) as u16,
                (bits & 0x000F_FFFF_FFFF_FFFF) << 11,
                0x3C00,
                0x7FF,
            ),
            _ => return value,
        };
        let sign = u16::from(sign) << 15;
        if self.trap_unimpl_060() {
            FloatX80::from_extended(sign | (exp_field + rebias), (1 << 63) | fraction)
        } else if exp_field == max {
            FloatX80::from_extended(sign | 0x7FFF, fraction)
        } else {
            value
        }
    }

    /// Whether the operation that raised `vector` wrote its destination
    /// register. The 68060 does not write it for any exception it takes;
    /// the package stores the result of an overflow, an underflow or an
    /// inexact operation before the user's handler runs (M68060UM
    /// C.3.2.3), and so does the crate when it does the package's work.
    fn fpu_060_stores_result(&self, vector: u32) -> bool {
        !self.trap_unimpl_060()
            && !matches!(vector, vector::FP_SNAN | vector::FP_OPERR | vector::FP_DZ)
    }

    /// The exception operand of the frame for the operation that just
    /// completed. `operand` is the source as the frame holds it, `src`
    /// its value and `dst_before` the destination register before the
    /// operation.
    ///
    /// The 68060 leaves the source operand there whatever the exception.
    /// The package replaces it for an overflow or underflow with the
    /// intermediate result rounded to extended precision, its exponent
    /// moved by $6000 into range, or zero when even that does not fit
    /// (M68060UM 6.7). An instruction the 68060 does not execute itself
    /// has no intermediate result here and keeps the source operand.
    fn fpu_060_exception_operand(
        &self,
        w2: u16,
        operand: FloatX80,
        src: FloatX80,
        dst_before: FloatX80,
    ) -> FloatX80 {
        let raised = self.fpsr >> 8;
        if self.trap_unimpl_060() || raised & 0x18 == 0 {
            return operand;
        }
        // FMOVECR's low seven bits are a ROM offset, not an opmode.
        let opmode = if w2 & 0xFC00 == 0x5C00 {
            0x17
        } else {
            w2 & 0x7F
        };
        let Some(unrounded) = self.fpu_unrounded_result(opmode, src, dst_before) else {
            return operand;
        };
        let (exp, mantissa) = unrounded.rounded_extended();
        let exp = if raised & 0x10 != 0 {
            exp - 0x6000
        } else {
            exp + 0x6000
        };
        let exp = if (0..=0x7FFF).contains(&exp) { exp } else { 0 };
        FloatX80::from_extended((u16::from(unrounded.sign) << 15) | exp as u16, mantissa)
    }

    /// Hold the arithmetic exception of the operation that just completed
    /// (`cycles` != 0) for the next floating-point instruction, with the
    /// EXCP state its handler saves. `w2` is the command word, `operand`
    /// the source as the frame holds it, `src` its value and `registers`
    /// the FP registers before the operation.
    pub(super) fn fpu_060_hold_exception(
        &mut self,
        cycles: i32,
        w2: u16,
        operand: FloatX80,
        src: FloatX80,
        registers: &[FloatX80; 8],
    ) -> i32 {
        if cycles != 0
            && let Some(vector) = self.fpu_arith_vector(self.trap_unimpl_060())
        {
            let dst_before = registers[((w2 >> 7) & 7) as usize];
            let operand = self.fpu_060_exception_operand(w2, operand, src, dst_before);
            self.fpu_060_excp = Some(Fpu060Excp::new(vector, operand));
            self.fpu_exception_pending = vector as u8;
            if !self.fpu_060_stores_result(vector) {
                self.fpr = *registers;
            }
        }
        cycles
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::CpuType;

    fn cpu(emulate: bool) -> CpuCore {
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68060);
        cpu.emulate_unimplemented_060 = emulate;
        cpu
    }

    #[test]
    fn excp_frame_round_trips() {
        let operand = FloatX80::from_extended(0xC002, 0xA000_0000_0000_0001);
        let state = Fpu060Excp::new(vector::FP_OVFL, operand);
        assert_eq!(state.frame(), [0xC002_E005, 0xA000_0000, 0x0000_0001]);
        assert_eq!(Fpu060Excp::from_frame(state.frame()), state);
        assert_eq!(state.vector(), vector::FP_OVFL);
        assert_eq!(
            Fpu060Excp::from_frame([0x0000_E007, 0, 0]).vector(),
            vector::FP_UNSUPP_DATA_TYPE
        );
    }

    #[test]
    fn single_and_double_sources_are_widened_without_their_special_cases() {
        let hw = cpu(false);
        let sp = cpu(true);
        let x = |e, m| FloatX80::from_extended(e, m);
        // (raw, as the 68060 leaves it, as the package corrects it)
        let cases = [
            // 1.5: the same either way.
            (
                RawSource::Single(0x3FC0_0000),
                x(0x3FFF, 0xC000_0000_0000_0000),
                x(0x3FFF, 0xC000_0000_0000_0000),
            ),
            // Zero, and minus zero.
            (
                RawSource::Single(0),
                x(0x3F80, 1 << 63),
                FloatX80::zero(false),
            ),
            (
                RawSource::Double(1 << 63),
                x(0xBC00, 1 << 63),
                FloatX80::zero(true),
            ),
            // The largest single denormal: its fraction under a spurious
            // integer bit, where the value has one bit fewer.
            (
                RawSource::Single(0x007F_FFFF),
                x(0x3F80, 0xFFFF_FF00_0000_0000),
                x(0x3F80, 0xFFFF_FE00_0000_0000),
            ),
            // Infinity: corrected to the exponent alone.
            (
                RawSource::Single(0x7F80_0000),
                x(0x407F, 1 << 63),
                x(0x7FFF, 0),
            ),
            // A double signaling NaN stays signaling.
            (
                RawSource::Double(0x7FF0_0000_0000_0001),
                x(0x43FF, 0x8000_0000_0000_0800),
                x(0x7FFF, 0x0000_0000_0000_0800),
            ),
        ];
        for (raw, hardware, corrected) in cases {
            let value = raw.to_x80(&mut Default::default());
            assert_eq!(hw.fpu_060_source_operand(raw, value), hardware);
            assert_eq!(sp.fpu_060_source_operand(raw, value), corrected);
        }
        // Other formats are converted.
        let raw = RawSource::Long(0xFFFF_FFFE);
        let value = raw.to_x80(&mut Default::default());
        assert_eq!(hw.fpu_060_source_operand(raw, value), value);
    }
}

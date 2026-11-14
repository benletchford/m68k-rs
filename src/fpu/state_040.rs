//! The MC68040 floating-point unit's exceptional internal state: what FSAVE
//! writes as a 26-word or busy state frame, and what FRESTORE loads back.
//!
//! The M68040FPSP (on AmigaOS, 68040.library) reads and rewrites these frames
//! in its Line-F, unsupported-data-type and arithmetic exception handlers, so
//! the field positions below are the ones its `fpsp.h` equates name. They
//! were cross-checked against the displacements a shipped FPSP binary uses
//! (frame-end relative: ETEMP -12, FPTEMP -24, T_BYTE -27, E_BYTE -28, DTAG
//! -32, CMDREG1B -36, STAG -40, NMCEXC -43, NMNEXC -44, CMDREG3B -48,
//! CMDREG2B -52, FPSR_SHADOW -64, WBTEMP -76, CU_SAVEPC -92). That code
//! never touches FPIARCU; its position (-60) is the equate's.

use super::FloatX80;

/// Which state frame the FPU would save.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub enum Fpu040Frame {
    /// No exceptional state: FSAVE writes the four-byte idle frame.
    #[default]
    Idle,
    /// The 26-word frame, version $41, size $30: an unimplemented
    /// instruction, or an arithmetic exception the conversion unit found
    /// (E1), which did not involve the normalization unit.
    Unimp,
    /// Busy frame, version $41, size $60: an unsupported data type, or an
    /// arithmetic exception of the normalization unit (E3).
    Busy,
}

/// The 68040 FPU's exceptional state between an exception and the FSAVE that
/// collects it (or between an FRESTORE and the instruction that reports it).
///
/// Opaque: a host keeps it with the rest of the CPU state (it serializes
/// with [`CpuCore`](crate::CpuCore)) and can ask which frame FSAVE would
/// write. The contents are the FPU's internal latches, whose layout is
/// free to follow the model as it is refined.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Fpu040State {
    /// The frame FSAVE writes next.
    pub(crate) frame: Fpu040Frame,
    /// An exception this state carries has not been taken yet: the next
    /// floating-point instruction takes it as a pre-instruction exception.
    /// Set by FRESTORE of a frame with E1 or E3 set, and by an instruction
    /// an FRESTORE completed; clear while a handler is already running for
    /// the exception that produced the state.
    pub(crate) pending: bool,
    /// The state longs that follow the header long of a busy frame (frame
    /// offsets $04-$60). An unimplemented-instruction frame is the last
    /// twelve of them (busy offsets $34-$60).
    pub(crate) longs: [u32; 24],
}

/// Byte offsets from the start of a busy frame (header long included).
pub(crate) mod off {
    /// Micro-PC of the conversion unit (byte). $FE asks FRESTORE to run
    /// the instruction in CMDREG1B from the frame's operands.
    pub const CU_SAVEPC: usize = 0x08;
    /// Intermediate result awaiting write-back (extended, 12 bytes).
    pub const WBTEMP: usize = 0x18;
    /// FPSR of the instruction in the normalisation unit.
    pub const FPSR_SHADOW: usize = 0x24;
    /// Address of the instruction in the conversion unit.
    pub const FPIARCU: usize = 0x28;
    /// Command word of the instruction that took an E3 exception (word).
    pub const CMDREG3B: usize = 0x34;
    /// Non-maskable exception bits of the normalisation unit (byte).
    pub const NMNEXC: usize = 0x38;
    /// Non-maskable exception bits, both units (byte).
    pub const NMCEXC: usize = 0x39;
    /// Source tag in bits 7-5, ETEMP exponent bit 15 in bit 4, and the
    /// guard and round bits of WBTEMP in bits 1-0 (byte).
    pub const STAG: usize = 0x3C;
    /// The sticky bit of WBTEMP in bit 7 (byte).
    pub const SBIT: usize = 0x3D;
    /// Command word of the instruction that took an E1 exception (word).
    pub const CMDREG1B: usize = 0x40;
    /// Destination tag in bits 7-5, FPTEMP exponent bit 15 in bit 4.
    pub const DTAG: usize = 0x44;
    /// WBTEMP exponent bit 15 in bit 4 (byte).
    pub const WB_BYTE: usize = 0x45;
    /// E1 (bit 2), E3 (bit 1) and S (bit 0) flags (byte).
    pub const E_BYTE: usize = 0x48;
    /// X (bit 7), U (bit 5) and T (bit 4) flags (byte).
    pub const T_BYTE: usize = 0x49;
    /// Destination operand (extended, 12 bytes).
    pub const FPTEMP: usize = 0x4C;
    /// Source operand (extended, 12 bytes).
    pub const ETEMP: usize = 0x58;
    /// First byte of the part an unimplemented-instruction frame keeps.
    pub const UNIMP_START: usize = 0x34;
}

/// E_BYTE: exception detected by the conversion unit (operand stage).
pub(crate) const E1: u8 = 0x04;
/// E_BYTE: exception detected by the normalisation unit (result stage).
pub(crate) const E3: u8 = 0x02;
/// E_BYTE: the shadow FPSR holds exception status to report.
pub(crate) const SFLAG: u8 = 0x01;
/// T_BYTE: the frame is an unimplemented instruction's.
pub(crate) const UFLAG: u8 = 0x20;
/// T_BYTE: the instruction is an FMOVE to memory or to a data register.
pub(crate) const TFLAG: u8 = 0x10;
/// NMCEXC: conversion-unit unsupported data type.
pub(crate) const NMC_UNSUP: u8 = 0x10;

/// Operand tags of STAG and DTAG.
pub(crate) mod tag {
    pub const NORM: u8 = 0;
    pub const ZERO: u8 = 1;
    pub const INF: u8 = 2;
    pub const NAN: u8 = 3;
    /// Extended-precision denormalized or unnormalized number.
    pub const EXT_DENORM: u8 = 4;
    /// Single- or double-precision denormalized number.
    pub const SD_DENORM: u8 = 5;
}

/// The tag the conversion unit gives an extended-precision value. A zero
/// mantissa is a zero whatever the exponent; a clear integer bit with a
/// nonzero mantissa is a denormalized (exponent 0) or unnormalized number,
/// which the 68040 does not handle in hardware. A zero exponent with the
/// integer bit set is an ordinary normalized number in this format.
pub(crate) fn classify(v: FloatX80) -> u8 {
    if v.biased_exp() == 0x7FFF {
        if v.mantissa << 1 == 0 {
            tag::INF
        } else {
            tag::NAN
        }
    } else if v.mantissa == 0 {
        tag::ZERO
    } else if v.mantissa >> 63 == 0 {
        tag::EXT_DENORM
    } else {
        tag::NORM
    }
}

/// Whether a tag names a data type the 68040 leaves to software.
pub(crate) fn tag_unsupported(t: u8) -> bool {
    t >= tag::EXT_DENORM
}

impl Fpu040State {
    /// True when FSAVE would write the idle frame.
    #[inline]
    pub fn is_idle(&self) -> bool {
        self.frame == Fpu040Frame::Idle
    }

    /// The state frame FSAVE would write.
    #[inline]
    pub fn frame(&self) -> Fpu040Frame {
        self.frame
    }

    /// Drop any exceptional state.
    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn long(&self, offset: usize) -> u32 {
        self.longs[(offset - 4) / 4]
    }

    pub(crate) fn set_long(&mut self, offset: usize, value: u32) {
        self.longs[(offset - 4) / 4] = value;
    }

    pub(crate) fn byte(&self, offset: usize) -> u8 {
        (self.long(offset & !3) >> (24 - 8 * (offset & 3))) as u8
    }

    pub(crate) fn set_byte(&mut self, offset: usize, value: u8) {
        let shift = 24 - 8 * (offset & 3);
        let long = self.long(offset & !3) & !(0xFF << shift);
        self.set_long(offset & !3, long | (u32::from(value) << shift));
    }

    /// The word in the upper half of the long at `offset`.
    pub(crate) fn word(&self, offset: usize) -> u16 {
        (self.long(offset) >> 16) as u16
    }

    pub(crate) fn set_word(&mut self, offset: usize, value: u16) {
        let long = self.long(offset) & 0x0000_FFFF;
        self.set_long(offset, long | (u32::from(value) << 16));
    }

    /// Store an extended value in the 96-bit memory layout at `offset`.
    pub(crate) fn set_extended(&mut self, offset: usize, v: FloatX80) {
        let (exp_word, mantissa) = v.to_extended();
        self.set_long(offset, u32::from(exp_word) << 16);
        self.set_long(offset + 4, (mantissa >> 32) as u32);
        self.set_long(offset + 8, mantissa as u32);
    }

    /// The operand at `offset`, with its exponent extended to sixteen bits
    /// by `e15` (ETE15, FPTE15): the bit is set for the lower half of the
    /// range, so a set bit over a fifteen-bit exponent of $4000 or more
    /// names a negative exponent - how the FPSP hands back a denormalized
    /// operand it normalized. The value comes back in register form, the
    /// mantissa shifted down to the format's zero exponent.
    pub(crate) fn operand(&self, offset: usize, e15: bool) -> FloatX80 {
        let exp_word = (self.long(offset) >> 16) as u16;
        let mantissa = (u64::from(self.long(offset + 4)) << 32) | u64::from(self.long(offset + 8));
        let exp = exp_word & 0x7FFF;
        if !(e15 && exp >= 0x4000) {
            return FloatX80::from_extended(exp_word, mantissa);
        }
        let shift = 0x8000 - u32::from(exp);
        FloatX80::from_extended(
            exp_word & 0x8000,
            if shift > 63 { 0 } else { mantissa >> shift },
        )
    }

    /// Store an intermediate result at `offset` in the FPU's internal
    /// form: the sign, the low fifteen bits of an exponent that may be out
    /// of the extended range, and the mantissa. Returns the exponent's
    /// sixteenth bit as the tag bytes hold it: set for the lower half of
    /// the range, so set for an underflowed exponent and clear for an
    /// overflowed one.
    pub(crate) fn set_internal(
        &mut self,
        offset: usize,
        sign: bool,
        exp: i32,
        mantissa: u64,
    ) -> bool {
        self.set_long(
            offset,
            (u32::from(sign) << 31) | ((exp as u32 & 0x7FFF) << 16),
        );
        self.set_long(offset + 4, (mantissa >> 32) as u32);
        self.set_long(offset + 8, mantissa as u32);
        exp < 0x4000
    }

    /// The tag byte for `v`: the tag in bits 7-5 and exponent bit 15 in bit
    /// 4, which is set for exponents up to $3FFF.
    pub(crate) fn tag_byte(t: u8, v: FloatX80) -> u8 {
        (t << 5) | (u8::from(v.biased_exp() < 0x4000) << 4)
    }
}

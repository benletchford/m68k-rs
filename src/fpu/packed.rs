//! 68881/68882 packed-decimal real format (FMOVE formats 3 and 7).
//!
//! The 96-bit packed-decimal real is three big-endian longwords:
//!   L0: SM(31) SE(30) yy(29-28) e2(27-24) e1(23-20) e0(19-16) ... D0(3-0)
//!   L1: 8 BCD fraction digits (D1..D8), most significant in bits 31-28
//!   L2: 8 BCD fraction digits (D9..D16)
//! The value is  (-1)^SM * D0.D1D2...D16 * 10^((-1)^SE * e2e1e0).
//! SE, both y bits and the exponent field all ones encode an infinity (zero
//! fraction) or a NaN.
//!
//! Decimal-to-binary scaling goes through the floatx80 engine (powers of ten
//! built by exponentiation-by-squaring), so a decoded value can be an ulp
//! off the correctly rounded one. Binary-to-decimal is exact: see
//! [`to_packed`].

use std::cmp::Ordering;

use super::softfloat::{self, ExcFlags, RoundCtx, RoundMode};
use super::types::FloatX80;

/// Largest decimal exponent the 3-digit field can hold.
const MAX_DECIMAL_EXP: u32 = 999;

fn pow10(n: u32, f: &mut ExcFlags) -> FloatX80 {
    // Exponentiation by squaring with base 10.
    let mut result = softfloat::from_u64(1, false);
    let mut base = softfloat::from_u64(10, false);
    let mut e = n;
    while e != 0 {
        if e & 1 != 0 {
            result = softfloat::mul(result, base, RoundCtx::NEAREST_EXT, f);
        }
        e >>= 1;
        if e != 0 {
            base = softfloat::mul(base, base, RoundCtx::NEAREST_EXT, f);
        }
    }
    result
}

/// Multiply `x` by 10^p (p may be negative).
fn mul_pow10(x: FloatX80, p: i32, f: &mut ExcFlags) -> FloatX80 {
    if p == 0 {
        return x;
    }
    let tp = pow10(p.unsigned_abs(), f);
    if p > 0 {
        softfloat::mul(x, tp, RoundCtx::NEAREST_EXT, f)
    } else {
        softfloat::div(x, tp, RoundCtx::NEAREST_EXT, f)
    }
}

/// Decode a packed-decimal real (12 bytes, big-endian) into an extended value.
/// A conversion that is not exact raises INEX1, the inexact-decimal-input
/// exception (MC68881/MC68882UM 6.1.8, M68000PRM FMOVE): INEX2 is reserved
/// for the rounding of the operation's own result.
pub fn from_packed(bytes: [u8; 12], f: &mut ExcFlags) -> FloatX80 {
    let mut g = ExcFlags::default();
    let value = decode_packed(bytes, &mut g);
    if g.has(ExcFlags::INEX2) {
        f.raise(ExcFlags::INEX1);
    }
    value
}

fn decode_packed(bytes: [u8; 12], f: &mut ExcFlags) -> FloatX80 {
    let w0 = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let w1 = u32::from_be_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    let w2 = u32::from_be_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);

    let sm = (w0 >> 31) & 1 != 0;
    let se = (w0 >> 30) & 1 != 0;

    if (w0 >> 16) & 0x7FFF == 0x7FFF {
        // An infinity or a NaN: SE and both y bits set over the exponent
        // $FFF. The sixteen fraction digits are moved bit for bit into
        // the mantissa, with no conversion, so a NaN keeps its payload
        // and a clear second bit makes it a signaling one; the integer
        // digit plays no part (MC68881/MC68882UM table 3-4, note 1).
        let mantissa = (u64::from(w1) << 32) | u64::from(w2);
        return FloatX80::from_extended((u16::from(sm) << 15) | 0x7FFF, mantissa);
    }

    let e2 = (w0 >> 24) & 0xF;
    let e1 = (w0 >> 20) & 0xF;
    let e0 = (w0 >> 16) & 0xF;
    let exp = (e2 * 100 + e1 * 10 + e0) as i32;
    let exp = if se { -exp } else { exp };

    // Assemble the 17-digit integer D0 D1 ... D16 (fits in u64: < 10^17).
    let mut m: u64 = (w0 & 0xF) as u64;
    for shift in (0..32).step_by(4).rev() {
        m = m * 10 + ((w1 >> shift) & 0xF) as u64;
    }
    for shift in (0..32).step_by(4).rev() {
        m = m * 10 + ((w2 >> shift) & 0xF) as u64;
    }

    if m == 0 {
        return FloatX80::zero(sm);
    }
    // value = m * 10^(exp - 16) (m carries 16 fraction digits).
    let base = softfloat::from_u64(m, sm);
    mul_pow10(base, exp - 16, f)
}

/// An unsigned integer of any size, least significant limb first and
/// without leading zero limbs: the exact intermediate values of the
/// binary-to-decimal conversion.
#[derive(Clone)]
struct Big(Vec<u32>);

impl Big {
    fn from_u64(v: u64) -> Self {
        let mut big = Big(vec![v as u32, (v >> 32) as u32]);
        big.trim();
        big
    }

    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }

    fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    fn mul_small(&mut self, m: u32) {
        let mut carry = 0u64;
        for limb in &mut self.0 {
            let t = u64::from(*limb) * u64::from(m) + carry;
            *limb = t as u32;
            carry = t >> 32;
        }
        if carry != 0 {
            self.0.push(carry as u32);
        }
    }

    fn mul_pow10(&mut self, mut n: u32) {
        while n >= 9 {
            self.mul_small(1_000_000_000);
            n -= 9;
        }
        if n != 0 {
            self.mul_small(10u32.pow(n));
        }
    }

    fn shl(&mut self, bits: u32) {
        if self.is_zero() {
            return;
        }
        let bit = bits % 32;
        if bit != 0 {
            let mut carry = 0u32;
            for limb in &mut self.0 {
                let t = (u64::from(*limb) << bit) | u64::from(carry);
                *limb = t as u32;
                carry = (t >> 32) as u32;
            }
            if carry != 0 {
                self.0.push(carry);
            }
        }
        let limbs = (bits / 32) as usize;
        self.0.splice(0..0, std::iter::repeat_n(0, limbs));
    }

    fn shr1(&mut self) {
        let mut carry = 0u32;
        for limb in self.0.iter_mut().rev() {
            let low = *limb & 1;
            *limb = (*limb >> 1) | (carry << 31);
            carry = low;
        }
        self.trim();
    }

    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .len()
            .cmp(&other.0.len())
            .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev()))
    }

    /// `self -= other`, for `self >= other`.
    fn sub(&mut self, other: &Self) {
        let mut borrow = false;
        for (i, limb) in self.0.iter_mut().enumerate() {
            let rhs = other.0.get(i).copied().unwrap_or(0);
            let (t, b1) = limb.overflowing_sub(rhs);
            let (t, b2) = t.overflowing_sub(u32::from(borrow));
            *limb = t;
            borrow = b1 || b2;
        }
        self.trim();
    }
}

/// What a division left over, against half of the divisor.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Remainder {
    None,
    BelowHalf,
    Half,
    AboveHalf,
}

/// `mant * 2^shift / 10^p10`, exactly: the integer part and what is left.
/// The integer part has to fit 64 bits; every caller scales its value
/// below 10^18.
fn scaled(mant: u64, shift: i32, p10: i32) -> (u64, Remainder) {
    let mut n = Big::from_u64(mant);
    let mut d = Big::from_u64(1);
    if shift >= 0 {
        n.shl(shift as u32);
    } else {
        d.shl(shift.unsigned_abs());
    }
    if p10 >= 0 {
        d.mul_pow10(p10 as u32);
    } else {
        n.mul_pow10(p10.unsigned_abs());
    }

    // Restoring division, one quotient bit at a time.
    let mut step = d.clone();
    step.shl(63);
    let mut q = 0u64;
    for bit in (0..64).rev() {
        if n.cmp(&step) != Ordering::Less {
            n.sub(&step);
            q |= 1 << bit;
        }
        step.shr1();
    }
    let rem = if n.is_zero() {
        Remainder::None
    } else {
        n.shl(1);
        match n.cmp(&d) {
            Ordering::Less => Remainder::BelowHalf,
            Ordering::Equal => Remainder::Half,
            Ordering::Greater => Remainder::AboveHalf,
        }
    };
    (q, rem)
}

/// floor(log10(mant * 2^shift)) for a normalized `mant` (bit 63 set).
fn decimal_exponent(mant: u64, shift: i32) -> i32 {
    // A floating-point estimate, then the exact test of 1 <= v/10^e < 10.
    let log2 = f64::from(shift + 63) + (mant as f64 / 9_223_372_036_854_775_808.0).log2();
    let mut e = (log2 * std::f64::consts::LOG10_2).floor() as i32;
    loop {
        match scaled(mant, shift, e).0 {
            0 => e -= 1,
            1..=9 => return e,
            _ => e += 1,
        }
    }
}

/// The three longs as the twelve bytes of the memory format.
fn longs(l0: u32, l1: u32, l2: u32) -> [u8; 12] {
    let mut out = [0u8; 12];
    out[0..4].copy_from_slice(&l0.to_be_bytes());
    out[4..8].copy_from_slice(&l1.to_be_bytes());
    out[8..12].copy_from_slice(&l2.to_be_bytes());
    out
}

/// The decimal exponent the conversion gives a denormalized number before
/// it looks at the digits: 2^-16383 is 3.4E-4932.
const DENORMAL_DECIMAL_EXP: i32 = -4933;

/// Encode an extended value as a packed-decimal real (`FMOVE.P FPn,<ea>`).
///
/// `k` is the k-factor: k > 0 asks for k significant digits, k <= 0 for
/// -k digits to the right of the decimal point. The value is cut to that
/// many digits (at least one, at most seventeen) by the FPCR rounding
/// `mode`, as an integer conversion would round it. The arithmetic here
/// is exact, so the string is the correctly rounded one.
///
/// The rest follows Motorola's own conversion routine (bindec in the
/// M68040FPSP and M68060SP, which the 68040 and 68060 run in place of the
/// MC68881/MC68882's microcode), with the corner cases as the M68060SP
/// has them:
///
/// - An infinity or a NaN is stored as its extended image, which is the
///   packed encoding of one (MC68881/MC68882UM table 3-6: SE and both y
///   bits set, exponent $FFF). A signaling NaN raises SNAN and is stored
///   quiet.
/// - A decimal exponent of four digits has its thousands digit in bits
///   15-12 of the first long and raises OPERR; so does a k-factor above
///   17, which gives seventeen digits (MC68881/MC68882UM 4.5, FMOVE).
/// - A denormalized number is converted with the fixed decimal exponent
///   -4933, its digits not normalized: 2^-16390 is 0.13...E-4933.
/// - With k <= 0 the digit position asked for can lie above the whole
///   value, which then rounds to zero (or to one unit of that position in
///   a directed mode). The zero has the exponent the routine was working
///   with: none for k = 0, -1 for a normalized number, k for a
///   denormalized one.
/// - Digits lost raise INEX2 (INEX1 is for decimal input only).
pub fn to_packed(x: FloatX80, k: i8, mode: RoundMode, f: &mut ExcFlags) -> [u8; 12] {
    let sign = x.sign();
    if x.biased_exp() == 0x7FFF {
        let x = if x.is_signaling_nan() {
            f.raise(ExcFlags::SNAN);
            x.quiet()
        } else {
            x
        };
        return longs(
            u32::from(x.sign_exp) << 16,
            (x.mantissa >> 32) as u32,
            x.mantissa as u32,
        );
    }
    if x.mantissa == 0 {
        return longs(u32::from(sign) << 31, 0, 0);
    }

    // value = mant * 2^shift with bit 63 of mant set. An exponent field of
    // zero has the weight of the field's minimum, integer bit included.
    let lz = x.mantissa.leading_zeros() as i32;
    let mant = x.mantissa << lz;
    let shift = i32::from(x.biased_exp()) - 16383 - 63 - lz;
    let denormal = shift + 63 < -16383;

    let k = i32::from(k);
    let mut ilog = if denormal {
        DENORMAL_DECIMAL_EXP
    } else {
        decimal_exponent(mant, shift)
    };
    let len = if k > 0 {
        if k > 17 {
            f.raise(ExcFlags::OPERR);
        }
        k.min(17)
    } else {
        (ilog + 1 - k).clamp(1, 17)
    };
    if k <= 0 && ilog < k {
        // The position asked for is above the leading digit.
        ilog = k;
    }

    let (mut digits, rem) = scaled(mant, shift, ilog + 1 - len);
    if rem != Remainder::None {
        f.raise(ExcFlags::INEX2);
    }
    let round_up = match mode {
        RoundMode::Nearest => {
            rem == Remainder::AboveHalf || (rem == Remainder::Half && digits & 1 != 0)
        }
        RoundMode::Zero => false,
        RoundMode::PosInf => !sign && rem != Remainder::None,
        RoundMode::NegInf => sign && rem != Remainder::None,
    };
    digits += u64::from(round_up);
    if digits == 10u64.pow(len as u32) {
        // Rounded up to the next power of ten.
        digits /= 10;
        ilog += 1;
    }

    let (exp, exp_negative) = if digits != 0 {
        (ilog.unsigned_abs(), ilog < 0)
    } else if k == 0 {
        (0, false)
    } else if !denormal {
        (1, true)
    } else if k < 0 {
        (k.unsigned_abs(), true)
    } else {
        (DENORMAL_DECIMAL_EXP.unsigned_abs(), true)
    };
    if exp > MAX_DECIMAL_EXP {
        f.raise(ExcFlags::OPERR);
    }

    // `len` digits, the first one the integer digit, then zeros.
    let mut bcd = [0u8; 17];
    let mut t = digits;
    for d in bcd[..len as usize].iter_mut().rev() {
        *d = (t % 10) as u8;
        t /= 10;
    }
    let nibbles = |ds: &[u8]| ds.iter().fold(0u32, |acc, &d| (acc << 4) | u32::from(d));
    let l0 = (u32::from(sign) << 31)
        | (u32::from(exp_negative) << 30)
        | (exp / 100 % 10) << 24
        | (exp / 10 % 10) << 20
        | (exp % 10) << 16
        | (exp / 1000) << 12
        | u32::from(bcd[0]);
    longs(l0, nibbles(&bcd[1..9]), nibbles(&bcd[9..17]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const RN: RoundMode = RoundMode::Nearest;

    fn roundtrip_f64(v: f64) -> f64 {
        let mut f = ExcFlags::default();
        let x = FloatX80::from_f64(v);
        let packed = to_packed(x, 17, RN, &mut f);
        from_packed(packed, &mut f).to_f64()
    }

    /// The three longs `FMOVE.P` stores and the exceptions it raises.
    fn out(x: FloatX80, k: i8, mode: RoundMode) -> ([u32; 3], u8) {
        let mut f = ExcFlags::default();
        let b = to_packed(x, k, mode, &mut f);
        let long = |i: usize| u32::from_be_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]]);
        ([long(0), long(4), long(8)], f.0)
    }

    fn x80(sign_exp: u16, mantissa: u64) -> FloatX80 {
        FloatX80::from_extended(sign_exp, mantissa)
    }

    const TENTH: FloatX80 = FloatX80 {
        sign_exp: 0x3FFB,
        mantissa: 0xCCCC_CCCC_CCCC_CCCD,
    };
    /// 2^-16384, a denormalized number: 8.4052578577802337657E-4933.
    const DENORMAL: FloatX80 = FloatX80 {
        sign_exp: 0,
        mantissa: 0x4000_0000_0000_0000,
    };
    /// The largest finite number: 1.18973149535723176502E+4932.
    const HUGE: FloatX80 = FloatX80 {
        sign_exp: 0x7FFE,
        mantissa: 0xFFFF_FFFF_FFFF_FFFF,
    };

    #[test]
    fn packed_roundtrip_exact_for_short_decimals() {
        // Few-digit values survive the 17-digit decimal round-trip exactly.
        for v in [0.0_f64, 1.0, -1.0, 2.5, -3.75, 100.0, 0.5, -9999.0, 1e10] {
            assert_eq!(roundtrip_f64(v), v, "round-trip {v}");
        }
    }

    #[test]
    fn packed_roundtrip_near_exact_for_long_decimals() {
        // The decimal-to-binary scaling is not bit-exact, so allow a tiny
        // relative error.
        for v in [
            123.456_f64,
            1e-10,
            std::f64::consts::PI,
            6.022e23,
            -std::f64::consts::E,
        ] {
            let r = roundtrip_f64(v);
            let rel = ((r - v) / v).abs();
            assert!(rel < 1e-13, "round-trip {v} -> {r} (rel {rel:e})");
        }
    }

    #[test]
    fn from_packed_known_value() {
        // +1.0 x 10^0: D0=1, all fractions 0, exp 0.
        let mut f = ExcFlags::default();
        let mut bytes = [0u8; 12];
        bytes[3] = 0x01; // D0 = 1
        assert_eq!(from_packed(bytes, &mut f).to_f64(), 1.0);

        // +1.5 x 10^1 = 15: D0=1, D1=5, exp=1.
        let mut bytes = [0u8; 12];
        bytes[2] = 0x00; // e0 nibble already 0; set exp=1 below
        bytes[3] = 0x01; // D0 = 1
        bytes[1] = 0x01; // e0 = 1 (w0 bits 19-16)
        bytes[4] = 0x50; // D1 = 5 (w1 top nibble)
        assert_eq!(from_packed(bytes, &mut f).to_f64(), 15.0);
    }

    #[test]
    fn infinities_and_nans_are_stored_as_their_extended_image() {
        // SE and both y bits set over the exponent $FFF.
        assert_eq!(
            out(FloatX80::infinity(false), 17, RN),
            ([0x7FFF_0000, 0, 0], 0)
        );
        assert_eq!(
            out(FloatX80::infinity(true), 3, RN),
            ([0xFFFF_0000, 0, 0], 0)
        );
        assert_eq!(
            out(x80(0x7FFF, 0xC000_0000_0000_1234), -5, RN),
            ([0x7FFF_0000, 0xC000_0000, 0x0000_1234], 0)
        );
        // A signaling NaN is stored quiet and raises SNAN.
        assert_eq!(
            out(x80(0xFFFF, 0x8000_0000_0000_0001), 17, RN),
            ([0xFFFF_0000, 0xC000_0000, 0x0000_0001], ExcFlags::SNAN)
        );
        // They read back as what they were.
        let mut f = ExcFlags::default();
        let inf = to_packed(FloatX80::infinity(true), 17, RN, &mut f);
        let r = from_packed(inf, &mut f);
        assert!(r.is_inf() && r.sign());
        let nan = to_packed(FloatX80::default_nan(), 17, RN, &mut f);
        assert!(from_packed(nan, &mut f).is_nan());
    }

    #[test]
    fn an_infinity_or_a_nan_is_read_bit_for_bit() {
        let read = |longs: [u32; 3]| {
            let mut bytes = [0u8; 12];
            for (i, long) in longs.iter().enumerate() {
                bytes[4 * i..4 * i + 4].copy_from_slice(&long.to_be_bytes());
            }
            from_packed(bytes, &mut ExcFlags::default())
        };
        // The fraction digits are the mantissa: a NaN keeps its payload.
        let nan = read([0x7FFF_0000, 0xC000_0000, 0x0000_0001]);
        assert_eq!(nan.to_extended(), (0x7FFF, 0xC000_0000_0000_0001));
        // One with the second bit clear is signaling, for the operation
        // that uses it to report.
        let snan = read([0xFFFF_0000, 0x8000_1234, 0x0000_0000]);
        assert_eq!(snan.to_extended(), (0xFFFF, 0x8000_1234_0000_0000));
        assert!(snan.is_signaling_nan());
        // The integer digit is no part of it: this is an infinity.
        let inf = read([0xFFFF_0009, 0, 0]);
        assert_eq!(inf.to_extended(), (0xFFFF, 0));
        // Without SE and both y bits, $FFF is the three digits of an
        // exponent, each read as fifteen: 1E1665 and 1E-1665.
        for (first, exp) in [(0x0FFF_0001, 0x559A), (0x4FFF_0001, 0x2A63)] {
            let x = read([first, 0, 0]);
            assert!(!x.is_nan() && !x.is_inf(), "{first:#x}");
            assert_eq!(x.to_extended().0, exp, "{first:#x}");
        }
    }

    #[test]
    fn zeros_keep_their_sign_and_nothing_else() {
        assert_eq!(out(FloatX80::zero(false), 17, RN), ([0, 0, 0], 0));
        assert_eq!(out(FloatX80::zero(true), -3, RN), ([0x8000_0000, 0, 0], 0));
        // A zero mantissa under any exponent is a zero.
        assert_eq!(out(x80(0x4000, 0), 5, RN), ([0, 0, 0], 0));
    }

    #[test]
    fn a_positive_k_factor_is_the_number_of_significant_digits() {
        // 123.456 with four digits: 1.235E2.
        let (l, f) = out(FloatX80::from_f64(123.456), 4, RN);
        assert_eq!((l, f), ([0x0002_0001, 0x2350_0000, 0], ExcFlags::INEX2));
        // Exact values raise nothing, whatever the digit count.
        for k in [1, 3, 17] {
            assert_eq!(out(FloatX80::from_f64(1.0), k, RN), ([1, 0, 0], 0));
            assert_eq!(
                out(FloatX80::from_f64(-7.0), k, RN),
                ([0x8000_0007, 0, 0], 0)
            );
        }
        // The nearest extended to 0.1 is not 0.1.
        assert_eq!(out(TENTH, 3, RN), ([0x4001_0001, 0, 0], ExcFlags::INEX2));
        assert_eq!(out(TENTH, 17, RN), ([0x4001_0001, 0, 0], ExcFlags::INEX2));
        // Seventeen digits of the largest number, correctly rounded.
        assert_eq!(out(HUGE, 17, RN).0, [0x0932_4001, 0x1897_3149, 0x5357_2318]);
    }

    #[test]
    fn a_k_factor_above_seventeen_is_an_operand_error() {
        assert_eq!(
            out(FloatX80::from_f64(1.0), 18, RN),
            ([1, 0, 0], ExcFlags::OPERR)
        );
        assert_eq!(
            out(TENTH, 63, RN),
            ([0x4001_0001, 0, 0], ExcFlags::OPERR | ExcFlags::INEX2)
        );
    }

    #[test]
    fn a_k_factor_of_zero_or_less_counts_digits_after_the_point() {
        // MC68881/MC68882UM, FMOVE: 12345.678765 with k = -5, -3, -1, 0.
        let v = FloatX80::from_f64(12345.678765);
        assert_eq!(out(v, -5, RN).0, [0x0004_0001, 0x2345_6787, 0x7000_0000]);
        assert_eq!(out(v, -3, RN).0, [0x0004_0001, 0x2345_6790, 0]);
        assert_eq!(out(v, -1, RN).0, [0x0004_0001, 0x2345_7000, 0]);
        assert_eq!(out(v, 0, RN).0, [0x0004_0001, 0x2346_0000, 0]);
        // More places than seventeen digits hold: seventeen, no OPERR.
        assert_eq!(out(TENTH, -64, RN), ([0x4001_0001, 0, 0], ExcFlags::INEX2));
        // Rounding into the next decade keeps the value.
        assert_eq!(
            out(FloatX80::from_f64(9.96), -1, RN),
            ([0x0001_0001, 0, 0], ExcFlags::INEX2)
        );
        assert_eq!(
            out(FloatX80::from_f64(99.5), 0, RN),
            ([0x0002_0001, 0, 0], ExcFlags::INEX2)
        );
    }

    #[test]
    fn a_value_below_the_last_place_rounds_to_zero_or_one_unit() {
        // 0.1 to no decimal places is 0; the zero has no exponent.
        assert_eq!(out(TENTH, 0, RN), ([0, 0, 0], ExcFlags::INEX2));
        assert_eq!(
            out(softfloat::neg(TENTH), 0, RN),
            ([0x8000_0000, 0, 0], ExcFlags::INEX2)
        );
        // Toward plus infinity it is one unit of that place.
        assert_eq!(
            out(TENTH, 0, RoundMode::PosInf),
            ([0x0000_0001, 0, 0], ExcFlags::INEX2)
        );
        // 0.5 is a tie and goes to the even 0; 0.6 goes to 1.
        assert_eq!(out(FloatX80::from_f64(0.5), 0, RN).0, [0, 0, 0]);
        assert_eq!(out(FloatX80::from_f64(0.6), 0, RN).0, [1, 0, 0]);
        // With places asked for, a normalized number's zero has the
        // exponent -1, and a unit is at the place: 1E-10 to five places.
        let small = FloatX80::from_f64(1e-10);
        assert_eq!(out(small, -5, RN), ([0x4001_0000, 0, 0], ExcFlags::INEX2));
        assert_eq!(
            out(small, -5, RoundMode::PosInf),
            ([0x4005_0001, 0, 0], ExcFlags::INEX2)
        );
    }

    #[test]
    fn the_rounding_mode_cuts_the_digits() {
        // 123456 to three digits.
        let v = FloatX80::from_f64(123456.0);
        let digits = |mode| out(v, 3, mode).0[1];
        assert_eq!(digits(RN), 0x2300_0000);
        assert_eq!(digits(RoundMode::Zero), 0x2300_0000);
        assert_eq!(digits(RoundMode::NegInf), 0x2300_0000);
        assert_eq!(digits(RoundMode::PosInf), 0x2400_0000);
        // A negative value rounds the other way.
        let v = FloatX80::from_f64(-123456.0);
        let digits = |mode| out(v, 3, mode).0[1];
        assert_eq!(digits(RoundMode::NegInf), 0x2400_0000);
        assert_eq!(digits(RoundMode::PosInf), 0x2300_0000);
        // A tie goes to the even digit: 125 and 135 to two digits.
        assert_eq!(out(FloatX80::from_f64(125.0), 2, RN).0[1], 0x2000_0000);
        assert_eq!(out(FloatX80::from_f64(135.0), 2, RN).0[1], 0x4000_0000);
        // 1/3 to five digits toward plus infinity: 3.3334E-1.
        let third = x80(0x3FFD, 0xAAAA_AAAA_AAAA_AAAB);
        assert_eq!(
            out(third, 5, RoundMode::PosInf),
            ([0x4001_0003, 0x3334_0000, 0], ExcFlags::INEX2)
        );
        // 9.99 to two digits carries into the exponent.
        assert_eq!(out(FloatX80::from_f64(9.99), 2, RN).0, [0x0001_0001, 0, 0]);
    }

    #[test]
    fn a_four_digit_exponent_is_an_operand_error() {
        // The thousands digit goes in bits 15-12 of the first long.
        assert_eq!(
            out(HUGE, 3, RN),
            (
                [0x0932_4001, 0x1900_0000, 0],
                ExcFlags::OPERR | ExcFlags::INEX2
            )
        );
        // 1E999 fits; 1E1000 does not.
        let mut f = ExcFlags::default();
        let e999 = mul_pow10(softfloat::from_u64(1, false), 999, &mut f);
        assert_eq!(out(e999, 1, RN).0, [0x0999_0001, 0, 0]);
        assert_eq!(out(e999, 1, RN).1 & ExcFlags::OPERR, 0);
        let e1000 = mul_pow10(softfloat::from_u64(1, false), 1000, &mut f);
        assert_eq!(out(e1000, 1, RN).0, [0x0000_1001, 0, 0]);
        assert_ne!(out(e1000, 1, RN).1 & ExcFlags::OPERR, 0);
    }

    #[test]
    fn a_denormalized_number_has_the_exponent_minus_4933() {
        let flags = ExcFlags::OPERR | ExcFlags::INEX2;
        assert_eq!(out(DENORMAL, 3, RN), ([0x4933_4008, 0x4100_0000, 0], flags));
        assert_eq!(
            out(DENORMAL, 17, RN),
            ([0x4933_4008, 0x4052_5785, 0x7780_2338], flags)
        );
        // A smaller one is not normalized: 2^-16390 is 0.13133...E-4933.
        assert_eq!(
            out(x80(0, 0x0100_0000_0000_0000), 5, RN),
            ([0x4933_4000, 0x1313_0000, 0], flags)
        );
        // The smallest has no digit left in seventeen.
        assert_eq!(out(x80(0, 1), 17, RN), ([0x4933_4000, 0, 0], flags));
        // To a number of places it is zero, with that exponent.
        assert_eq!(
            out(DENORMAL, -5, RN),
            ([0x4005_0000, 0, 0], ExcFlags::INEX2)
        );
        assert_eq!(
            out(DENORMAL, -64, RN),
            ([0x4064_0000, 0, 0], ExcFlags::INEX2)
        );
        assert_eq!(out(DENORMAL, 0, RN), ([0, 0, 0], ExcFlags::INEX2));
        // An unnormalized number is converted as the value it holds.
        assert_eq!(
            out(x80(0x4002, 0x3800_0000_0000_0000), 3, RN),
            out(FloatX80::from_f64(3.5), 3, RN)
        );
    }

    #[test]
    fn inexact_decimal_input_raises_inex1_only() {
        // 1E-1 (D0 = 1, SE, exponent 1) has no exact binary value.
        let mut f = ExcFlags::default();
        let mut bytes = [0u8; 12];
        bytes[0] = 0x40; // SE
        bytes[1] = 0x01; // e0 = 1
        bytes[3] = 0x01; // D0 = 1
        let v = from_packed(bytes, &mut f);
        assert_eq!((v.sign_exp, v.mantissa), (0x3FFB, 0xCCCC_CCCC_CCCC_CCCD));
        assert_eq!(f.0, ExcFlags::INEX1);
        // 1.5E1 is exact.
        let mut f = ExcFlags::default();
        let mut bytes = [0u8; 12];
        bytes[1] = 0x01;
        bytes[3] = 0x01;
        bytes[4] = 0x50;
        from_packed(bytes, &mut f);
        assert_eq!(f, ExcFlags::default());
    }
}

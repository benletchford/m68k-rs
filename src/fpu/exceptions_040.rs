//! MC68040 floating-point exceptions: the unimplemented-instruction and
//! unsupported-data-type traps the M68040FPSP services, the arithmetic
//! exceptions, the state frames FSAVE gives their handlers, and what
//! FRESTORE does with a frame a handler hands back.
//!
//! `CpuCore::emulate_unimplemented_040` says where the FPSP's work is done.
//! Clear, the FPSP is guest code and the 68040 behaves as the part does:
//! unimplemented instructions and unsupported data types trap, and
//! overflow and underflow are taken whether or not they are enabled, with
//! the destination register left for the handler to fill (M68040UM 9.7.4.2
//! and 9.7.5.2). Set, which is the default, that work is done here: those
//! instructions and operands complete, an overflow or underflow stores
//! the result a 6888x stores, and an arithmetic exception is taken only
//! when enabled, as the FPSP would pass it on to the user's handler.
//! Either way a taken exception leaves the state frame described below,
//! so a handler (the FPSP's or the user's) finds the same thing.
//!
//! The conversion unit (CU) of the 68040 FPU fetches an instruction's
//! operands, converts them to extended precision and tags them. It raises
//! the "E1" exceptions: unsupported data types, SNAN, OPERR and DZ on the
//! operands, and the rounding exceptions of FMOVE, FABS and FNEG, which
//! go no further. The normalization unit (NU) computes and rounds FADD,
//! FSUB, FMUL, FDIV and FSQRT; it raises the "E3" exceptions found on
//! their results: OVFL, UNFL and INEX. FSAVE exposes both units' latches
//! as the busy frame; what never entered the NU saves the 26-word frame.
//!
//! Sources. The exceptions and their stack frames are M68040UM 9.6.1
//! (unimplemented instructions), 9.6.2 (unsupported data types) and 9.7
//! (arithmetic exceptions); the state frames and what each exception
//! leaves in them are 9.8 and table 9-16. What FRESTORE must do with a
//! frame follows the FPSP, which is the consumer: its get_op (operands,
//! tags, the packed layout), res_func (CU_SAVEPC = $FE restarts the
//! instruction; FSQRT latched as $05), gen_except (E1/E3 frames it
//! restores for the 68040 to report; the shadow FPSR) and the overflow,
//! underflow and operand-error handlers (CMDREG3B, the T flag, WBTEMP's
//! wrapped exponent). The field positions were checked against the
//! displacements a shipped FPSP binary uses, and 68040.library 37.30
//! runs its handlers on these frames.
//!
//! Timing model: the 68040 holds an exception of an instruction whose
//! destination is a floating-point register pending and takes it when the
//! next floating-point instruction is reached, stacking that instruction's
//! address. Arithmetic exceptions are held that way here, with their
//! frame, and so is the exception of a frame loaded by FRESTORE; FSAVE
//! collects a held exception into the frame it writes. Every
//! floating-point instruction but FSAVE and FRESTORE reports it, FMOVEM
//! and the control-register moves included (M68040UM 9.8: any instruction
//! before the handler's FSAVE reports the exception again). The
//! unsupported-data-type exception is still taken at the boundary of the
//! instruction that raised it, stacking the address of the instruction
//! that follows.
//!
//! Not modelled:
//! - The exceptions of an FMOVE to memory or to a data register. Those
//!   moves complete with the result computed here and no trap: neither the
//!   nonmaskable operand error and signaling NaN of a move to an integer
//!   nor an overflow, underflow or enabled exception of a move to single,
//!   double or extended.
//! - An overflow or underflow of the selected rounding precision that the
//!   extended format can still hold. The arithmetic keeps the extended
//!   exponent range at every precision, so it raises neither.
//! - The frame as the FPSP leaves it for the user's handler when its work
//!   is done here: the frame is the one the 68040 saves, without the
//!   FPSP's own marks (the S and X flags, WBTEMP overwritten).
//! - WBTM66, CU_SAVEPC, CMDREG2B and the register dirty bits of a saved
//!   busy frame, which are zero.

use super::operations::RawSource;
use super::softfloat::ExcFlags;
use super::state_040::{
    E1, E3, Fpu040Frame, Fpu040State, NMC_UNSUP, SFLAG, TFLAG, UFLAG, classify, off, tag,
    tag_unsupported,
};
use crate::core::cpu::CpuCore;
use crate::core::exceptions::vector;
use crate::core::memory::AddressBus;
use crate::fpu::FloatX80;

/// The FP opmodes the 68040 executes in hardware (M68040UM 9.4): FMOVE,
/// FSQRT, FABS, FNEG, FDIV, FADD, FMUL, FSGLDIV, FSGLMUL, FSUB, FCMP, FTST
/// and the single- and double-rounding variants.
fn hardware_opmode(opmode: u16) -> bool {
    matches!(
        opmode,
        0x00 | 0x04
            | 0x18
            | 0x1A
            | 0x20
            | 0x22
            | 0x23
            | 0x24
            | 0x27
            | 0x28
            | 0x38
            | 0x3A
            | 0x40
            | 0x41
            | 0x44
            | 0x45
            | 0x58
            | 0x5A
            | 0x5C
            | 0x5E
            | 0x60
            | 0x62
            | 0x63
            | 0x64
            | 0x66
            | 0x67
            | 0x68
            | 0x6C
    )
}

/// The hardware opmodes whose result the normalization unit rounds: FSQRT,
/// FDIV, FADD, FMUL, FSUB and their single- and double-rounding variants,
/// FSGLDIV and FSGLMUL. An overflow, underflow or inexact result of these
/// is an E3 exception; of the others (FMOVE, FABS, FNEG) it is found by
/// the conversion unit and is an E1 exception (M68040UM 9.8).
fn nu_opmode(opmode: u16) -> bool {
    hardware_opmode(opmode)
        && !matches!(
            opmode,
            0x00 | 0x18 | 0x1A | 0x38 | 0x3A | 0x40 | 0x44 | 0x58 | 0x5A | 0x5C | 0x5E
        )
}

/// The exceptions the conversion unit finds on an instruction's operands
/// before anything is computed: SNAN, OPERR and DZ. The destination
/// register is not written.
fn operand_exception(vector: u32) -> bool {
    matches!(vector, vector::FP_SNAN | vector::FP_OPERR | vector::FP_DZ)
}

/// Whether an instruction reads its destination register as an operand.
/// This is the test the FPSP's chk_dy_mo applies to CMDREG1B: opmodes
/// $20-$2F and $60-$6F, and FCMP ($38).
fn dyadic(opmode: u16) -> bool {
    opmode & 0x30 == 0x20 || opmode & 0x7F == 0x38
}

/// A command word as the 68040 latches it: FSQRT ($04) is held as $05.
fn latched_command(w2: u16) -> u16 {
    if w2 & 0xE07F == 0x0004 || w2 & 0xE07F == 0x4004 {
        w2 | 1
    } else {
        w2
    }
}

/// CMDREG3B, the NU's copy of a command word: the register and bits 6, 1
/// and 0 stay, bit 2 moves to bit 5 and bits 5-3 move down one place.
fn cmdreg3b(w2: u16) -> u16 {
    (w2 & 0x03C3) | ((w2 & 0x0038) >> 1) | ((w2 & 0x0004) << 3)
}

/// A source operand as the CU latches it in ETEMP.
#[derive(Clone, Copy)]
struct Latched {
    /// ETEMP: the operand in extended form.
    value: FloatX80,
    tag: u8,
    /// The three longs of a packed decimal source.
    packed: Option<[u32; 3]>,
}

impl Latched {
    fn register(value: FloatX80) -> Self {
        Self {
            value,
            tag: classify(value),
            packed: None,
        }
    }

    /// The image of a fetched source. Integers and normalized singles and
    /// doubles convert exactly. A single or double denormal is tagged as
    /// such and keeps its fraction below a clear integer bit, under the
    /// exponent its zero exponent field converts to ($3F80, $3C00); the
    /// FPSP's get_op replaces that with $3F81 or $3C01 and normalizes.
    fn source(raw: RawSource) -> Self {
        match raw {
            RawSource::Single(bits) if bits & 0x7F80_0000 == 0 && bits & 0x007F_FFFF != 0 => Self {
                value: FloatX80::from_extended(
                    (((bits >> 31) as u16) << 15) | 0x3F80,
                    u64::from(bits & 0x007F_FFFF) << 40,
                ),
                tag: tag::SD_DENORM,
                packed: None,
            },
            RawSource::Double(bits)
                if bits & 0x7FF0_0000_0000_0000 == 0 && bits & 0x000F_FFFF_FFFF_FFFF != 0 =>
            {
                Self {
                    value: FloatX80::from_extended(
                        (((bits >> 63) as u16) << 15) | 0x3C00,
                        (bits & 0x000F_FFFF_FFFF_FFFF) << 11,
                    ),
                    tag: tag::SD_DENORM,
                    packed: None,
                }
            }
            // A NaN is widened as it is: a signaling NaN stays signaling
            // in the frame (the value conversion below would quiet it).
            RawSource::Single(bits)
                if bits & 0x7F80_0000 == 0x7F80_0000 && bits & 0x007F_FFFF != 0 =>
            {
                Self {
                    value: FloatX80::from_extended(
                        (((bits >> 31) as u16) << 15) | 0x7FFF,
                        (1 << 63) | (u64::from(bits & 0x007F_FFFF) << 40),
                    ),
                    tag: tag::NAN,
                    packed: None,
                }
            }
            RawSource::Double(bits)
                if bits & 0x7FF0_0000_0000_0000 == 0x7FF0_0000_0000_0000
                    && bits & 0x000F_FFFF_FFFF_FFFF != 0 =>
            {
                Self {
                    value: FloatX80::from_extended(
                        (((bits >> 63) as u16) << 15) | 0x7FFF,
                        (1 << 63) | ((bits & 0x000F_FFFF_FFFF_FFFF) << 11),
                    ),
                    tag: tag::NAN,
                    packed: None,
                }
            }
            RawSource::Packed(longs) => Self {
                value: FloatX80::from_extended(
                    (longs[0] >> 16) as u16,
                    (u64::from(longs[1]) << 32) | u64::from(longs[2]),
                ),
                tag: tag::NORM,
                packed: Some(longs),
            },
            other => Self::register(other.to_x80(&mut ExcFlags::default())),
        }
    }
}

impl CpuCore {
    /// Latch an instruction's operands into the state frame: the command
    /// word, the source in ETEMP with STAG, and the destination register in
    /// FPTEMP with DTAG.
    fn fpu_040_latch(&mut self, frame: Fpu040Frame, w2: u16, src: Latched, dst: FloatX80) {
        let mut st = Fpu040State {
            frame,
            ..Fpu040State::default()
        };
        st.set_word(off::CMDREG1B, latched_command(w2));
        st.set_word(off::CMDREG3B, cmdreg3b(w2));
        st.set_extended(off::ETEMP, src.value);
        st.set_byte(off::STAG, Fpu040State::tag_byte(src.tag, src.value));
        st.set_extended(off::FPTEMP, dst);
        st.set_byte(off::DTAG, Fpu040State::tag_byte(classify(dst), dst));
        if let Some(longs) = src.packed {
            // A packed source is not converted: its first long sits in
            // FPTEMP's last (the FPSP's get_op moves it into place before
            // decoding the twelve bytes at ETEMP).
            st.set_long(off::FPTEMP + 4, longs[1]);
            st.set_long(off::FPTEMP + 8, longs[0]);
        }
        st.set_long(off::FPIARCU, self.ppc);
        self.fpu_040 = st;
    }

    /// Record the state of an unimplemented instruction: the short frame
    /// with the U flag, and E1 when the source is packed decimal.
    fn fpu_040_unimp_state(&mut self, w2: u16, src: Latched, dst: FloatX80) {
        self.fpu_040_latch(Fpu040Frame::Unimp, w2, src, dst);
        self.fpu_040.set_byte(off::T_BYTE, UFLAG);
        if src.packed.is_some() {
            self.fpu_040.set_byte(off::E_BYTE, E1);
        }
    }

    /// Record the state of an unsupported data type: a busy frame with the
    /// CU exception flagged, and the T flag for an FMOVE out.
    fn fpu_040_unsupp_state(&mut self, w2: u16, src: Latched, dst: FloatX80) {
        self.fpu_040_latch(Fpu040Frame::Busy, w2, src, dst);
        self.fpu_040.set_byte(off::E_BYTE, E1);
        self.fpu_040.set_byte(off::NMCEXC, NMC_UNSUP);
        if w2 & 0xE000 == 0x6000 {
            self.fpu_040.set_byte(off::T_BYTE, TFLAG);
        }
    }

    /// FMOVECR on a 68040 with traps: the frame holds only the command.
    pub(super) fn fpu_040_fmovecr_trap<B: AddressBus>(&mut self, bus: &mut B, w2: u16) -> i32 {
        let none = Latched::register(FloatX80::default());
        let dst = self.fpr[((w2 >> 7) & 7) as usize];
        self.fpu_040_unimp_state(w2, none, dst);
        self.take_fp_unimp_060(bus, 0)
    }

    /// The pre-instruction unsupported-data-type exception of an
    /// instruction whose destination is a floating-point register: vector
    /// 55 with the four-word frame. The stacked PC is past the instruction
    /// because the handler finishes it by restoring the repaired frame.
    fn fpu_040_take_unsupp<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.fpiar = self.ppc;
        self.take_fp_exception_past_instruction(bus, vector::FP_UNSUPP_DATA_TYPE)
    }

    /// A general instruction with a memory, data-register or immediate
    /// source (the command word already consumed). `None` leaves an opmode
    /// that is neither in hardware nor in the FPSP to the caller.
    pub(super) fn fpu_040_ea_source<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> Option<i32> {
        let src_fmt = (w2 >> 10) & 0x7;
        let dst = ((w2 >> 7) & 7) as usize;
        let opmode = w2 & 0x7F;
        let unimplemented = super::operations::fpu_040_traps_opmode(opmode);
        if !unimplemented && !hardware_opmode(opmode) {
            return None;
        }
        let dst_value = self.fpr[dst];

        if unimplemented {
            // Takes precedence over the operand's data type (M68040UM
            // 9.6.1): the handler finds that in the frame. The operand has
            // been fetched; the format $2 frame carries its address.
            let (src, ea) = match self.fpu_fetch_source(bus, opcode, src_fmt) {
                Some((raw, ea)) => (Latched::source(raw), ea),
                None => (Latched::register(FloatX80::default()), 0),
            };
            self.fpu_040_unimp_state(w2, src, dst_value);
            return Some(self.take_fp_unimp_060(bus, ea));
        }

        let (raw, _) = self.fpu_fetch_source(bus, opcode, src_fmt)?;
        let src = Latched::source(raw);
        if src.packed.is_some()
            || tag_unsupported(src.tag)
            || (dyadic(opmode) && tag_unsupported(classify(dst_value)))
        {
            self.fpu_040_unsupp_state(w2, src, dst_value);
            return Some(self.fpu_040_take_unsupp(bus));
        }

        self.fpiar = self.ppc;
        let mut flags = ExcFlags::default();
        let value = raw.to_x80(&mut flags);
        let registers = self.fpr;
        let cycles = self.fpu_apply_op(opmode, dst, value, flags);
        Some(self.fpu_040_hold_exception(cycles, w2, src.value, &registers))
    }

    /// A general instruction with a register source (the command word
    /// already consumed and the opmode known to exist).
    pub(super) fn fpu_040_register_source<B: AddressBus>(
        &mut self,
        bus: &mut B,
        w2: u16,
    ) -> Option<i32> {
        let src = Latched::register(self.fpr[((w2 >> 10) & 7) as usize]);
        let dst = ((w2 >> 7) & 7) as usize;
        let opmode = w2 & 0x7F;
        let dst_value = self.fpr[dst];

        if super::operations::fpu_040_traps_opmode(opmode) {
            self.fpu_040_unimp_state(w2, src, dst_value);
            return Some(self.take_fp_unimp_060(bus, 0));
        }
        if !hardware_opmode(opmode) {
            return None;
        }
        if tag_unsupported(src.tag) || (dyadic(opmode) && tag_unsupported(classify(dst_value))) {
            self.fpu_040_unsupp_state(w2, src, dst_value);
            return Some(self.fpu_040_take_unsupp(bus));
        }

        self.fpiar = self.ppc;
        let registers = self.fpr;
        let cycles = self.fpu_apply_op(opmode, dst, src.value, ExcFlags::default());
        Some(self.fpu_040_hold_exception(cycles, w2, src.value, &registers))
    }

    /// `FMOVE FPn,<ea>` of a value or to a format the 68040 leaves to
    /// software: packed decimal, or a denormalized or unnormalized source
    /// in any format. A post-instruction exception through vector 55 with
    /// the format $3 frame (M68040UM 9.6.2); nothing is stored. `None` =
    /// the move is done in hardware. The command word is not yet consumed.
    pub(super) fn fpu_040_move_out<B: AddressBus>(
        &mut self,
        bus: &mut B,
        opcode: u16,
        w2: u16,
    ) -> Option<i32> {
        let dst_fmt = (w2 >> 10) & 0x7;
        let packed = dst_fmt == 3 || dst_fmt == 7;
        let value = self.fpr[((w2 >> 7) & 7) as usize];
        let src = Latched::register(value);
        if !packed && !tag_unsupported(src.tag) {
            return None;
        }
        let _w2 = self.read_imm_16(bus);
        let ea = self.fpu_060_trap_ea(bus, opcode, if packed { 3 } else { dst_fmt });
        // The handler reads the value to move from ETEMP; the exceptional
        // operand of a move out is in FPTEMP as well.
        self.fpu_040_unsupp_state(w2, src, value);
        Some(self.take_fp_unsupp_post_040(bus, ea))
    }

    /// The vector of the arithmetic exception the operation that just
    /// completed takes, from the FPSR exception byte it left.
    ///
    /// With the FPSP's work done here (`emulate_unimplemented_040` set)
    /// that is the highest-priority enabled exception. With the FPSP in
    /// the guest it is what the 68040 itself takes: overflow and underflow
    /// are nonmaskable (M68040UM 9.7.4.2 and 9.7.5.2), because the result
    /// a 6888x would store is the FPSP's to supply. Its handler stores
    /// it, then goes on to the user's handler for whichever exception is
    /// enabled, or returns.
    fn fpu_040_arith_vector(&self) -> Option<u32> {
        self.fpu_arith_vector(self.trap_unimpl_040())
    }

    /// Whether the operation that raised `vector` wrote its destination
    /// register. An operand exception (SNAN, OPERR, DZ) stops it first.
    /// The 68040 does not store an overflowed or underflowed result
    /// either: with the FPSP in the guest the register is left for its
    /// handler to fill; with the FPSP's work done here it is stored, as
    /// that handler would have done before the user's runs (M68040UM
    /// 9.7.4 and 9.7.5). An inexact result is always stored.
    fn fpu_040_stores_result(&self, vector: u32) -> bool {
        let left_to_the_fpsp =
            self.trap_unimpl_040() && matches!(vector, vector::FP_OVFL | vector::FP_UNFL);
        !(operand_exception(vector) || left_to_the_fpsp)
    }

    /// Hold the arithmetic exception of the operation that just completed
    /// (`cycles` != 0) for the next floating-point instruction, with the
    /// state frame its handler saves. `w2` is the command word, `src` the
    /// source operand as the conversion unit latched it and `registers`
    /// the FP registers before the operation. FSAVE before the next
    /// floating-point instruction collects the frame, and FRESTORE of it
    /// holds the exception again.
    pub(super) fn fpu_040_hold_exception(
        &mut self,
        cycles: i32,
        w2: u16,
        src: FloatX80,
        registers: &[FloatX80; 8],
    ) -> i32 {
        if cycles != 0
            && let Some(vector) = self.fpu_040_arith_vector()
        {
            self.fpu_040_arith_state(vector, w2, src, registers[((w2 >> 7) & 7) as usize]);
            if !self.fpu_040_stores_result(vector) {
                self.fpr = *registers;
            }
            self.fpu_040.pending = true;
        }
        cycles
    }

    /// The state frame of an arithmetic exception through `vector`, as
    /// M68040UM table 9-16 gives it for each exception:
    ///
    /// - SNAN, OPERR, DZ: E1 in the 26-word frame, the source in ETEMP and
    ///   the destination register in FPTEMP.
    /// - OVFL, UNFL, INEX of an instruction the normalization unit rounds
    ///   ([`nu_opmode`]): E3 in the busy frame, the intermediate result in
    ///   WBTEMP with its exponent's sixteenth bit in WBTE15. An overflow
    ///   holds the mantissa rounded to the precision; an underflow or an
    ///   inexact result holds the 64 bits before rounding, with the guard,
    ///   round and sticky bits beside STAG.
    /// - OVFL, UNFL, INEX of FMOVE, FABS and FNEG: E1 in the 26-word
    ///   frame, the intermediate result in FPTEMP the same way.
    ///
    /// An instruction the 68040 does not execute itself (run here with
    /// `emulate_unimplemented_040` set, or FMOVECR) has no intermediate
    /// result to show: its OVFL, UNFL or INEX frame is the E3 one with
    /// the stored result in WBTEMP.
    fn fpu_040_arith_state(&mut self, vector: u32, w2: u16, src: FloatX80, dst_before: FloatX80) {
        // FMOVECR's low seven bits are a ROM offset, not an opmode.
        let opmode = if w2 & 0xFC00 == 0x5C00 {
            0x17
        } else {
            w2 & 0x7F
        };
        let result = self.fpr[((w2 >> 7) & 7) as usize];
        if operand_exception(vector) {
            self.fpu_040_latch(Fpu040Frame::Unimp, w2, Latched::register(src), dst_before);
            self.fpu_040.set_byte(off::E_BYTE, E1);
            return;
        }
        // (sign, exponent, mantissa, guard/round/sticky) of the image.
        let image = self.fpu_unrounded_result(opmode, src, dst_before).map(|u| {
            if vector == vector::FP_OVFL {
                let (exp, mantissa) = u.rounded();
                (u.sign, exp, mantissa, 0)
            } else {
                let (mantissa, grs) = u.truncated();
                (u.sign, u.exp, mantissa, grs)
            }
        });
        if hardware_opmode(opmode) && !nu_opmode(opmode) {
            self.fpu_040_latch(Fpu040Frame::Unimp, w2, Latched::register(src), dst_before);
            self.fpu_040.set_byte(off::E_BYTE, E1);
            if let Some((sign, exp, mantissa, _)) = image {
                let e15 = self.fpu_040.set_internal(off::FPTEMP, sign, exp, mantissa);
                self.fpu_040
                    .set_byte(off::DTAG, (tag::NORM << 5) | (u8::from(e15) << 4));
            }
            return;
        }
        self.fpu_040_latch(Fpu040Frame::Busy, w2, Latched::register(src), dst_before);
        self.fpu_040.set_byte(off::E_BYTE, E3);
        match image {
            Some((sign, exp, mantissa, grs)) => {
                let e15 = self.fpu_040.set_internal(off::WBTEMP, sign, exp, mantissa);
                self.fpu_040.set_byte(off::WB_BYTE, u8::from(e15) << 4);
                let stag = self.fpu_040.byte(off::STAG);
                self.fpu_040.set_byte(off::STAG, stag | (grs >> 1));
                self.fpu_040.set_byte(off::SBIT, (grs & 1) << 7);
            }
            None => {
                self.fpu_040.set_extended(off::WBTEMP, result);
                self.fpu_040
                    .set_byte(off::WB_BYTE, u8::from(result.biased_exp() < 0x4000) << 4);
            }
        }
        self.fpu_040.set_long(off::FPSR_SHADOW, self.fpsr);
    }

    /// Before a floating-point instruction that reports exceptions: take
    /// the exception the FPU's state still carries, as a pre-instruction
    /// exception whose stacked PC restarts this instruction. With nothing
    /// to take, the instruction goes ahead and the old state is gone.
    pub(crate) fn fpu_040_pre_instruction<B: AddressBus>(&mut self, bus: &mut B) -> Option<i32> {
        if self.fpu_040.is_idle() {
            return None;
        }
        let reported = self.fpu_040_report_pending(bus);
        if reported.is_none() {
            self.fpu_040.clear();
        }
        reported
    }

    /// Take the exception the FPU's state still carries, if any, as a
    /// pre-instruction exception of the instruction now starting. The
    /// state stays for the handler's FSAVE.
    pub(crate) fn fpu_040_report_pending<B: AddressBus>(&mut self, bus: &mut B) -> Option<i32> {
        if self.fpu_040.pending
            && let Some(vector) = self.fpu_040_pending_vector()
        {
            self.fpu_040.pending = false;
            return Some(self.take_exception(bus, vector));
        }
        None
    }

    /// The vector of the exception the FPU's state carries, if any. E1 or
    /// E3 must be set. Enabled exceptions of the FPSR exception byte (the
    /// shadow FPSR's too for E3) are taken by priority, and an unsupported
    /// data type that is still flagged goes back through vector 55.
    ///
    /// Overflow and underflow depend on who supplies their result. With
    /// `emulate_unimplemented_040` clear they are taken whether enabled or
    /// not, as the 68040 takes them (M68040UM 9.7.4.2 and 9.7.5.2): its
    /// vectors lead to the FPSP, which stores the result and goes on to
    /// the user's handler for the exception that is enabled, so a
    /// disabled overflow with INEX enabled goes through vector 53. With
    /// it set there is no FPSP in between, and what is taken is what the
    /// FPSP would pass on: the enabled exception, the inexact one for an
    /// overflow that is disabled while INEX2 is enabled (M68040UM 9.7.7).
    fn fpu_040_pending_vector(&self) -> Option<u32> {
        let st = &self.fpu_040;
        let e = st.byte(off::E_BYTE);
        if e & (E1 | E3) == 0 {
            return None;
        }
        let busy = st.frame == Fpu040Frame::Busy;
        let mut exc = (self.fpsr >> 8) & 0xFF;
        let mut nmc = 0;
        if busy {
            if e & E3 != 0 {
                exc |= (st.long(off::FPSR_SHADOW) >> 8) & 0xFF;
            }
            nmc = st.byte(off::NMCEXC);
            // NMCEXC bits 1-0 and 6-5: overflow and underflow of either unit.
            exc |= u32::from(((nmc & 0x03) | ((nmc >> 5) & 0x03)) << 3);
        }
        let enabled = self.fpcr >> 8;
        let mut reportable = exc & enabled & 0x7F;
        if self.trap_unimpl_040() {
            reportable |= exc & 0x18;
        } else {
            reportable |= (exc & 0x10) >> 3 & enabled;
        }
        [
            (0x40, vector::FP_SNAN),
            (0x20, vector::FP_OPERR),
            (0x10, vector::FP_OVFL),
            (0x08, vector::FP_UNFL),
            (0x04, vector::FP_DZ),
            (0x03, vector::FP_INEX),
        ]
        .into_iter()
        .find(|&(bit, _)| reportable & bit != 0)
        .map(|(_, v)| v)
        .or_else(|| {
            (busy && (nmc & NMC_UNSUP != 0 || st.byte(off::NMNEXC) & 0x02 != 0))
                .then_some(vector::FP_UNSUPP_DATA_TYPE)
        })
    }

    /// The longs FSAVE writes for the exceptional state, header first.
    fn fpu_040_frame_longs(&self) -> ([u32; 25], usize) {
        let mut longs = [0u32; 25];
        match self.fpu_040.frame {
            Fpu040Frame::Idle => {
                longs[0] = 0x4100_0000;
                (longs, 1)
            }
            Fpu040Frame::Unimp => {
                longs[0] = 0x4130_0000;
                let first = (off::UNIMP_START - 4) / 4;
                longs[1..13].copy_from_slice(&self.fpu_040.longs[first..]);
                (longs, 13)
            }
            Fpu040Frame::Busy => {
                longs[0] = 0x4160_0000;
                longs[1..25].copy_from_slice(&self.fpu_040.longs);
                (longs, 25)
            }
        }
    }

    /// FSAVE with exceptional state in the FPU: write the 26-word or busy
    /// frame (M68040UM 9.8) and leave the FPU idle.
    pub(super) fn fpu_040_fsave<B: AddressBus>(
        &mut self,
        bus: &mut B,
        ea_mode: u8,
        ea_reg: usize,
    ) -> i32 {
        let (longs, count) = self.fpu_040_frame_longs();
        let bytes = 4 * count as u32;
        let addr = match ea_mode {
            3 => {
                let addr = self.a(ea_reg);
                self.set_a(ea_reg, addr.wrapping_add(bytes));
                addr
            }
            4 => {
                let addr = self.a(ea_reg).wrapping_sub(bytes);
                self.set_a(ea_reg, addr);
                addr
            }
            _ => match self.fpu_frame_control_ea(bus, ea_mode, ea_reg, false) {
                Some(addr) => addr,
                None => return 0,
            },
        };
        for (i, long) in longs.iter().take(count).enumerate() {
            self.write_32(bus, addr.wrapping_add(4 * i as u32), *long);
        }
        self.fpu_040.clear();
        8
    }

    /// FRESTORE of a valid non-null frame at `addr`: load the FPU's
    /// exceptional state from it.
    ///
    /// - An idle frame leaves nothing pending.
    /// - A busy frame whose CU_SAVEPC is $FE restarts the instruction in
    ///   CMDREG1B from ETEMP and FPTEMP: this is how the FPSP's
    ///   unsupported-data-type handler has the 68040 finish an instruction
    ///   once it has normalized the operands.
    /// - A frame with E1 or E3 set holds an exception that the next
    ///   floating-point instruction takes.
    /// - A busy frame with neither has nothing left to do; an
    ///   unimplemented-instruction frame is kept as restored.
    pub(super) fn fpu_040_frestore<B: AddressBus>(&mut self, bus: &mut B, addr: u32, header: u32) {
        let size = ((header >> 16) & 0xFF) as usize;
        let mut st = Fpu040State::default();
        match size {
            0x60 => st.frame = Fpu040Frame::Busy,
            0x30 | 0x28 => st.frame = Fpu040Frame::Unimp,
            _ => {
                self.fpu_040.clear();
                return;
            }
        }
        // The frame's last long always lands in the last state long; the
        // version $40 unimplemented frame ($28) lacks the two longs before
        // STAG.
        let first = st.longs.len() - size / 4;
        for i in 0..size / 4 {
            st.longs[first + i] = self.read_32(bus, addr.wrapping_add(4 + 4 * i as u32));
        }
        let e = st.byte(off::E_BYTE);
        if st.frame == Fpu040Frame::Busy && st.byte(off::CU_SAVEPC) == 0xFE {
            self.fpu_040.clear();
            self.fpu_040_resume(&st);
        } else if e & (E1 | E3) != 0 {
            st.pending = true;
            self.fpu_040 = st;
        } else if st.frame == Fpu040Frame::Unimp {
            self.fpu_040 = st;
        } else {
            self.fpu_040.clear();
        }
    }

    /// Run the register-destination instruction a restored busy frame
    /// describes, from the frame's operands. An enabled exception of the
    /// result is held for the next floating-point instruction.
    fn fpu_040_resume(&mut self, st: &Fpu040State) {
        let w2 = st.word(off::CMDREG1B);
        if w2 & 0xA000 != 0 {
            // Only register-destination instructions are restarted.
            return;
        }
        let opmode = match w2 & 0x7F {
            0x05 => 0x04, // FSQRT as the 68040 latches it
            other => other,
        };
        if !hardware_opmode(opmode) {
            return;
        }
        let dst = ((w2 >> 7) & 7) as usize;
        let src = st.operand(off::ETEMP, st.byte(off::STAG) & 0x10 != 0);
        let register = self.fpr[dst];
        if dyadic(opmode) {
            self.fpr[dst] = st.operand(off::FPTEMP, st.byte(off::DTAG) & 0x10 != 0);
        }
        let dst_before = self.fpr[dst];
        // With the S flag the handler passes exception status of its own
        // in the shadow FPSR (the FPSP's INEX1 for a packed source it
        // converted): the instruction reports it along with its own.
        let carried = if st.byte(off::E_BYTE) & SFLAG != 0 {
            ExcFlags::from_bits((st.long(off::FPSR_SHADOW) >> 8) as u8)
        } else {
            ExcFlags::default()
        };
        self.fpu_apply_op(opmode, dst, src, carried);
        if matches!(opmode, 0x38 | 0x3A) {
            // FCMP and FTST only set the condition codes.
            self.fpr[dst] = register;
        }
        if let Some(vector) = self.fpu_040_arith_vector() {
            let fpiarcu = st.long(off::FPIARCU);
            self.fpu_040_arith_state(vector, (w2 & !0x7F) | opmode, src, dst_before);
            self.fpu_040.set_long(off::FPIARCU, fpiarcu);
            self.fpu_040.pending = true;
            if !self.fpu_040_stores_result(vector) {
                self.fpr[dst] = register;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn command_words_are_reencoded_for_the_frame() {
        assert_eq!(latched_command(0x0404), 0x0405, "FSQRT FP1,FP0");
        assert_eq!(latched_command(0x4804), 0x4805, "FSQRT.X <ea>,FP0");
        assert_eq!(latched_command(0x0441), 0x0441, "FSSQRT keeps its opmode");
        assert_eq!(latched_command(0x6804), 0x6804, "an FMOVE out k-factor");
        // The FPSP's overflow handler finds FSGLMUL and FSGLDIV as $33 and
        // $30 in CMDREG3B, and the forced precisions as bits 6-5.
        assert_eq!(cmdreg3b(0x0027) & 0x7F, 0x33);
        assert_eq!(cmdreg3b(0x0024) & 0x7F, 0x30);
        assert_eq!(cmdreg3b(0x0062) & 0x60, 0x40);
        assert_eq!(cmdreg3b(0x0066) & 0x60, 0x60);
        assert_eq!(cmdreg3b(0x0380) & 0x0380, 0x0380, "register field kept");
    }

    #[test]
    fn the_intermediate_result_rounds_to_what_the_instruction_stores() {
        // The frame's intermediate result is computed apart from the
        // instruction itself; in range, rounding it must give the register.
        use crate::core::types::CpuType;
        let a = FloatX80::from_extended(0x4001, 0xC90F_DAA2_2168_C235);
        let b = FloatX80::from_extended(0xBFFD, 0xAAAA_AAAA_AAAA_AAAB);
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        let mut seen = 0;
        for opmode in 0..0x80u16 {
            if !hardware_opmode(opmode) || matches!(opmode, 0x38 | 0x3A) {
                continue;
            }
            for fpcr in [0x00, 0x10, 0x20, 0x30, 0x40, 0x80, 0xB0] {
                for (src, dst) in [(a, b), (b, a), (a, a)] {
                    cpu.fpcr = fpcr;
                    cpu.fpr[0] = dst;
                    cpu.fpu_apply_op(opmode, 0, src, ExcFlags::default());
                    let result = cpu.fpr[0];
                    if result.is_nan() || result.is_zero() {
                        continue; // FSQRT of a negative number; a - a
                    }
                    let u = cpu
                        .fpu_unrounded_result(opmode, src, dst)
                        .unwrap_or_else(|| panic!("{opmode:#x}: no intermediate"));
                    let (exp, mantissa) = u.rounded();
                    assert_eq!(
                        FloatX80::from_extended((u16::from(u.sign) << 15) | exp as u16, mantissa),
                        result,
                        "{opmode:#x} fpcr {fpcr:#x}"
                    );
                    seen += 1;
                }
            }
        }
        assert!(seen > 300, "{seen}");
        assert_eq!(cpu.fpu_unrounded_result(0x0E, a, b), None, "FSIN");
        assert_eq!(cpu.fpu_unrounded_result(0x38, a, b), None, "FCMP");
    }

    #[test]
    fn overflow_and_inexact_of_a_move_are_conversion_unit_exceptions() {
        for opmode in [0x00, 0x18, 0x1A, 0x40, 0x44, 0x58, 0x5A, 0x5C, 0x5E] {
            assert!(hardware_opmode(opmode) && !nu_opmode(opmode), "{opmode:#x}");
        }
        for opmode in [
            0x04, 0x41, 0x45, 0x20, 0x60, 0x64, 0x22, 0x62, 0x66, 0x23, 0x63, 0x67, 0x24, 0x27,
            0x28, 0x68, 0x6C,
        ] {
            assert!(nu_opmode(opmode), "{opmode:#x}");
        }
        assert!(!nu_opmode(0x0E), "FSIN is not a hardware instruction");
    }

    #[test]
    fn dyadic_matches_the_fpsp_classification() {
        for opmode in [
            0x20, 0x22, 0x23, 0x24, 0x27, 0x28, 0x38, 0x60, 0x67, 0x6C, 0x21,
        ] {
            assert!(dyadic(opmode), "{opmode:#x}");
        }
        for opmode in [0x00, 0x04, 0x18, 0x1A, 0x3A, 0x40, 0x45, 0x5E, 0x0E, 0x30] {
            assert!(!dyadic(opmode), "{opmode:#x}");
        }
    }

    #[test]
    fn frame_operands_take_exponent_bit_15() {
        let mut st = Fpu040State::default();
        // A denormal normalized by one place: exponent -1 in fifteen bits.
        st.set_extended(off::ETEMP, FloatX80::from_extended(0xFFFF, 1 << 63));
        assert_eq!(
            st.operand(off::ETEMP, true),
            FloatX80::from_extended(0x8000, 1 << 62)
        );
        // The same bit over an exponent below $4000 changes nothing.
        st.set_extended(off::ETEMP, FloatX80::from_extended(0x3FFE, 1 << 63));
        assert_eq!(
            st.operand(off::ETEMP, true),
            FloatX80::from_extended(0x3FFE, 1 << 63)
        );
        assert_eq!(
            st.operand(off::ETEMP, false),
            FloatX80::from_extended(0x3FFE, 1 << 63)
        );
        // Shifted out entirely.
        st.set_extended(off::ETEMP, FloatX80::from_extended(0x7F00, 1 << 63));
        assert_eq!(st.operand(off::ETEMP, true), FloatX80::zero(false));
    }

    #[test]
    fn state_bytes_and_words_address_big_endian_longs() {
        let mut st = Fpu040State::default();
        st.set_byte(off::E_BYTE, 0x04);
        st.set_byte(off::T_BYTE, 0x20);
        assert_eq!(st.long(0x48), 0x0420_0000);
        st.set_word(off::CMDREG1B, 0x4822);
        assert_eq!(st.long(0x40), 0x4822_0000);
        assert_eq!(st.word(off::CMDREG1B), 0x4822);
        assert_eq!(st.byte(off::T_BYTE), 0x20);
        st.set_byte(off::NMCEXC, 0x10);
        assert_eq!(st.long(0x38), 0x0010_0000);
    }
}

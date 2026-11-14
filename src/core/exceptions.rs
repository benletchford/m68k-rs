//! Exception handling.
//!
//! Defines exception vectors and processing.

use super::cpu::{CpuCore, SFLAG_SET};
use super::execute::RUN_MODE_BERR_AERR_RESET;
use super::memory::AddressBus;
use super::types::CpuType;

/// Exception vector numbers.
pub mod vector {
    /// Reset vector-table entry containing the initial supervisor stack pointer.
    pub const RESET_SSP: u32 = 0;
    /// Reset vector-table entry containing the initial program counter.
    pub const RESET_PC: u32 = 1;
    /// Bus-error exception vector.
    pub const BUS_ERROR: u32 = 2;
    /// Address-error exception vector.
    pub const ADDRESS_ERROR: u32 = 3;
    /// Illegal-instruction exception vector.
    pub const ILLEGAL_INSTRUCTION: u32 = 4;
    /// Integer divide-by-zero exception vector.
    pub const ZERO_DIVIDE: u32 = 5;
    /// CHK/CHK2 bounds exception vector.
    pub const CHK: u32 = 6;
    /// TRAPV/TRAPcc exception vector.
    pub const TRAPV: u32 = 7;
    /// Privilege-violation exception vector.
    pub const PRIVILEGE_VIOLATION: u32 = 8;
    /// Instruction-trace exception vector.
    pub const TRACE: u32 = 9;
    /// Line-A emulator exception vector.
    pub const LINE_1010: u32 = 10;
    /// Line-F emulator and coprocessor exception vector.
    pub const LINE_1111: u32 = 11;
    /// Invalid exception-frame format vector (68010+).
    pub const FORMAT_ERROR: u32 = 14;
    /// Uninitialized interrupt-vector exception.
    pub const UNINITIALIZED_INTERRUPT: u32 = 15;
    /// Spurious-interrupt vector.
    pub const SPURIOUS_INTERRUPT: u32 = 24;
    /// Base vector for `TRAP #0` through `TRAP #15`.
    pub const TRAP_BASE: u32 = 32;

    /// 68060: integer instructions removed from silicon (MOVEP, CHK2/CMP2,
    /// CAS2, misaligned CAS, 64-bit MUL/DIV) trap here for the OS-side
    /// 68060 software package to emulate.
    pub const UNIMPLEMENTED_INTEGER: u32 = 61;
    /// 68060: FP operand data types the FPU no longer handles in hardware
    /// (packed decimal, denormals) - pre-instruction, format $0.
    pub const FP_UNSUPP_DATA_TYPE: u32 = 55;
    /// 68060: FP addressing forms dropped from silicon (dynamic-list
    /// FMOVEM, immediate packed operands, multi-register control-list
    /// FMOVEM.L #imm) - pre-instruction, format $0.
    pub const FP_UNIMPLEMENTED_EA: u32 = 60;

    /// FPU exception vectors 48-54 (MC68881/MC68882UM 6.1, M68040UM 9.5,
    /// M68060UM 6.5): branch/set on unordered, inexact (INEX1 or INEX2),
    /// divide by zero, underflow, operand error, overflow, signaling NaN.
    pub const FP_BSUN: u32 = 48;
    /// FPU inexact-result exception (INEX1 or INEX2).
    pub const FP_INEX: u32 = 49;
    /// FPU divide-by-zero exception.
    pub const FP_DZ: u32 = 50;
    /// FPU underflow exception.
    pub const FP_UNFL: u32 = 51;
    /// FPU operand-error exception.
    pub const FP_OPERR: u32 = 52;
    /// FPU overflow exception.
    pub const FP_OVFL: u32 = 53;
    /// FPU signaling-NaN exception.
    pub const FP_SNAN: u32 = 54;

    /// 68020+ MMU configuration-error vector.
    pub const MMU_CONFIGURATION_ERROR: u32 = 56;
    /// 68020+ MMU illegal-operation vector.
    pub const MMU_ILLEGAL_OPERATION_ERROR: u32 = 57;
    /// 68020+ MMU access-level-violation vector.
    pub const MMU_ACCESS_LEVEL_VIOLATION_ERROR: u32 = 58;
}

/// 68060 fault status long word (FSLW) bits, as consumed by OS fault
/// handlers (bit names per MC68060UM Table 8-9 / Linux asm/traps.h).
pub mod fslw {
    /// Misaligned access: the fault is on the second or a later part of an
    /// operand split into several bus cycles.
    pub const MA: u32 = 0x0800_0000;
    /// Read access.
    pub const RW_R: u32 = 0x0100_0000;
    /// Write access.
    pub const RW_W: u32 = 0x0080_0000;
    /// Transfer size shift (bits 22-21): 00 long, 01 byte, 10 word.
    pub const SIZE_SHIFT: u32 = 21;
    /// Transfer modifier shift (bits 18-16): holds the function code.
    pub const TM_SHIFT: u32 = 16;
    /// Instruction (1) or operand (0) access.
    pub const IO: u32 = 0x0000_8000;
    /// Invalid descriptor in the root (level A) table.
    pub const PTA: u32 = 0x0000_1000;
    /// Invalid descriptor in the pointer (level B) table.
    pub const PTB: u32 = 0x0000_0800;
    /// Invalid indirect page descriptor.
    pub const IL: u32 = 0x0000_0400;
    /// Page fault (invalid page descriptor).
    pub const PF: u32 = 0x0000_0200;
    /// Supervisor protection violation.
    pub const SP: u32 = 0x0000_0100;
    /// Write protection violation.
    pub const WP: u32 = 0x0000_0080;
    /// Bus error on table search.
    pub const TWE: u32 = 0x0000_0040;
    /// Bus error on read.
    pub const RE: u32 = 0x0000_0020;
    /// Bus error on write.
    pub const WE: u32 = 0x0000_0010;
    /// Transparent translation register hit: a write through a
    /// write-protected TTR.
    pub const TTR: u32 = 0x0000_0008;
}

/// Compose the 68060 FSLW for an access error.
fn fslw_060(
    write: bool,
    instruction: bool,
    size: u32,
    fc: u16,
    cause: Option<crate::mmu::MmuFaultCause>,
    transparent: bool,
) -> u32 {
    use crate::mmu::MmuFaultCause;
    let mut w = if write { fslw::RW_W } else { fslw::RW_R };
    // As on the 68040, there is no three-byte size encoding (0b11 is a line
    // transfer), so a three-byte operand reports as a long.
    w |= match size {
        1 => 0b01 << fslw::SIZE_SHIFT,
        2 => 0b10 << fslw::SIZE_SHIFT,
        _ => 0, // long
    };
    w |= u32::from(fc & 7) << fslw::TM_SHIFT;
    if instruction {
        w |= fslw::IO;
    }
    w |= match cause {
        Some(MmuFaultCause::PointerA) => fslw::PTA,
        Some(MmuFaultCause::PointerB) => fslw::PTB,
        Some(MmuFaultCause::Indirect) => fslw::IL,
        Some(MmuFaultCause::PageFault) => fslw::PF,
        // A write-protected transparent translation register: MC68060UM
        // 8.4.1 names a TTR bit for this fault; whether WP is also set is
        // not stated, so only TTR is reported.
        Some(MmuFaultCause::WriteProtect) if transparent => fslw::TTR,
        Some(MmuFaultCause::WriteProtect) => fslw::WP,
        Some(MmuFaultCause::SupervisorProtect) => fslw::SP,
        Some(MmuFaultCause::TableWalkBusError) => fslw::TWE,
        // A physical bus error on the access itself.
        Some(MmuFaultCause::AccessBusError) | None => {
            if write {
                fslw::WE
            } else {
                fslw::RE
            }
        }
    };
    w
}

/// Compose the 68010 special status word for a faulted bus cycle
/// (M68010UM figure 5-8): IF (bit 13) marks an instruction fetch, DF (bit
/// 12) a data read, RW (bit 8) a read cycle, BY (bit 9) a byte transfer,
/// and bits 2-0 hold the function code. A write sets neither fetch bit.
fn ssw_010(fc: u16, write: bool, instruction: bool, size: u32) -> u16 {
    let mut ssw = fc & 7;
    if !write {
        ssw |= 0x0100;
        ssw |= if instruction { 0x2000 } else { 0x1000 };
    }
    if size == 1 {
        ssw |= 0x0200;
    }
    ssw
}

/// Whether a 68000 instruction's memory operand is a long (as opposed to
/// a word); byte operands cannot raise an address error.
fn operand_is_long_68000(opcode: u16) -> bool {
    match opcode >> 12 {
        1 | 3 => false,
        2 => true,
        // ADDA/SUBA/CMPA select the size with bit 8.
        0x9 | 0xB | 0xD if opcode & 0x00C0 == 0x00C0 => opcode & 0x0100 != 0,
        // CHK, MOVE to/from SR/CCR, DIVU/DIVS, MULU/MULS: word operands.
        _ if opcode & 0xF1C0 == 0x4180 || opcode & 0x00C0 == 0x00C0 => false,
        _ => opcode & 0x00C0 == 0x0080,
    }
}

/// Offset from the next unread instruction word to the PC a 68000 group 0
/// frame stacks for a faulted data access of `opcode`.
fn group0_pc_adjust_68000(opcode: u16, write: bool) -> i32 {
    let mode = (opcode >> 3) & 7;
    let line = opcode >> 12;
    // MOVEM (not EXT, which shares the encoding with a register mode) and
    // UNLK prefetch the next word before the first transfer.
    if (opcode & 0xFB80 == 0x4880 && mode != 0) || opcode & 0xFFF8 == 0x4E58 {
        return 2;
    }
    // ADDX/SUBX -(Ay),-(Ax) and CMPM (Ay)+,(Ax)+.
    if matches!(opcode & 0xF138, 0x9108 | 0xB108 | 0xD108) && opcode & 0x00C0 != 0x00C0 {
        return 2;
    }
    if (1..=3).contains(&line) && write {
        // MOVE to memory: the destination mode decides.
        let dst_mode = (opcode >> 6) & 7;
        let dst_reg = (opcode >> 9) & 7;
        let src_is_memory = mode >= 2 && !(mode == 7 && (opcode & 7) == 4);
        return match (dst_mode, dst_reg) {
            (2..=4, _) => 2,
            // An absolute long destination after a memory source has not
            // fetched its second address word yet.
            (7, 1) if src_is_memory => -2,
            _ => 0,
        };
    }
    match mode {
        // -(An): a word operand is fetched after the next prefetch.
        4 if !operand_is_long_68000(opcode) => 2,
        // d16(An), d8(An,Xn) and their PC-relative forms: the extension
        // word is still the prefetch address when the operand cycle starts.
        5 | 6 => -2,
        7 if matches!(opcode & 7, 2 | 3) => -2,
        _ => 0,
    }
}

/// Function code bits for exception stack frames.
pub mod fc {
    /// User-data address space.
    pub const USER_DATA: u16 = 1;
    /// User-program address space.
    pub const USER_PROGRAM: u16 = 2;
    /// Supervisor-data address space.
    pub const SUPERVISOR_DATA: u16 = 5;
    /// Supervisor-program address space.
    pub const SUPERVISOR_PROGRAM: u16 = 6;
}

impl CpuCore {
    #[inline]
    fn push_16_raw<B: AddressBus>(&mut self, bus: &mut B, value: u16) {
        self.dar[15] = self.dar[15].wrapping_sub(2);
        bus.write_word(self.address(self.dar[15]), value);
    }

    #[inline]
    fn push_32_raw<B: AddressBus>(&mut self, bus: &mut B, value: u32) {
        self.dar[15] = self.dar[15].wrapping_sub(4);
        bus.write_long(self.address(self.dar[15]), value);
    }

    #[inline]
    fn fake_push_16_raw(&mut self) {
        self.dar[15] = self.dar[15].wrapping_sub(2);
    }

    #[inline]
    fn fake_push_32_raw(&mut self) {
        self.dar[15] = self.dar[15].wrapping_sub(4);
    }

    /// Push the 68000's 3-word exception frame (SR + PC) in the bus order the
    /// hardware uses: PC low word first, then SR, then PC high word.
    pub(crate) fn push_exception_frame_68000<B: AddressBus>(
        &mut self,
        bus: &mut B,
        stacked_pc: u32,
        sr: u16,
    ) {
        let sp = self.dar[15].wrapping_sub(6);
        self.dar[15] = sp;
        self.write_16(bus, sp.wrapping_add(4), (stacked_pc & 0xFFFF) as u16);
        self.write_16(bus, sp, sr);
        self.write_16(bus, sp.wrapping_add(2), (stacked_pc >> 16) as u16);
    }

    /// Push the 68000's seven-word group 0 (bus error / address error) frame.
    ///
    /// Layout from the new stack pointer upwards (M68000UM figure 6-7):
    /// status word, access address, instruction register, status register,
    /// program counter. The status word holds the function code in bits
    /// 2-0, I/N in bit 3 (0 = the fault interrupted an instruction, 1 = it
    /// interrupted exception processing) and R/W in bit 4 (1 = read); its
    /// upper bits are undefined and carry the instruction register on real
    /// parts. The bus order matches the hardware: the SR/PC part is written
    /// first in the same order as the three-word frame, then IR, the access
    /// address and the status word.
    ///
    /// The writes are raw bus cycles (no alignment check): a misaligned
    /// supervisor stack must not raise a second address error from here.
    #[allow(clippy::too_many_arguments)]
    fn push_group0_frame_68000<B: AddressBus>(
        &mut self,
        bus: &mut B,
        fc: u16,
        write: bool,
        in_exception: bool,
        address: u32,
        old_sr: u16,
        stacked_pc: u32,
    ) {
        let ir = self.ir as u16;
        let mut status = (ir & !0x001F) | (fc & 7);
        if !write {
            status |= 0x0010;
        }
        if in_exception {
            status |= 0x0008;
        }
        let sp = self.dar[15].wrapping_sub(14);
        self.dar[15] = sp;
        let mut put = |offset: u32, value: u16| {
            bus.write_word(self.address(sp.wrapping_add(offset)), value);
        };
        put(12, (stacked_pc & 0xFFFF) as u16);
        put(8, old_sr);
        put(10, (stacked_pc >> 16) as u16);
        put(6, ir);
        put(4, (address & 0xFFFF) as u16);
        put(0, status);
        put(2, (address >> 16) as u16);
    }

    /// The PC a 68000 group 0 frame stacks: the processor's prefetch
    /// address at the time of the fault. M68000UM 6.3.9.1 only promises
    /// "2 to 10 bytes" past the opcode; where it lands depends on how far
    /// the instruction's microcode had advanced the prefetch when the bus
    /// cycle was attempted, which differs per addressing mode. The offsets
    /// below are the ones the SingleStepTests 68000 traces record.
    fn group0_stacked_pc_68000(&self, write: bool, instruction: bool) -> u32 {
        if instruction {
            // A faulted instruction fetch stacks the address it fetched from.
            return self.pc;
        }
        let lowest = self.ppc.wrapping_add(2);
        let pc = if self.pc.wrapping_sub(lowest) <= 8 {
            self.pc
        } else {
            lowest
        };
        pc.wrapping_add_signed(group0_pc_adjust_68000(self.ir as u16, write))
    }

    /// Push the 68010 format $8 bus/address error frame (29 words). Only
    /// the fields a handler can use are filled in: the special status word
    /// and the fault address. The data and instruction input buffers are
    /// zero and the sixteen internal words are not written, so RTE restarts
    /// the instruction from the stacked PC instead of continuing it.
    ///
    /// Layout from the new stack pointer upwards: SR, PC, format/vector,
    /// special status word (+$08), fault address (+$0A), unused, data
    /// output buffer (+$10), unused, data input buffer (+$14), unused,
    /// instruction input buffer (+$18), internal (+$1A..+$39).
    fn push_fault_frame_010<B: AddressBus>(
        &mut self,
        bus: &mut B,
        vector: u32,
        old_sr: u16,
        ssw: u16,
        address: u32,
    ) {
        for _ in 0..8 {
            self.fake_push_32_raw();
        }
        self.push_16_raw(bus, 0); // instruction input buffer
        self.fake_push_16_raw();
        self.push_16_raw(bus, 0); // data input buffer
        self.fake_push_16_raw();
        self.push_16_raw(bus, 0); // data output buffer
        self.fake_push_16_raw();
        self.push_32_raw(bus, address); // fault address
        self.push_16_raw(bus, ssw); // special status word
        self.push_16_raw(bus, 0x8000 | ((vector as u16) << 2));
        self.push_32_raw(bus, self.ppc);
        self.push_16_raw(bus, old_sr);
    }

    /// Push the 68020/68030 long bus-cycle fault frame (format $B) for a
    /// bus error or an address error.
    #[allow(clippy::too_many_arguments)]
    fn push_fault_frame_030<B: AddressBus>(
        &mut self,
        bus: &mut B,
        vector: u32,
        old_sr: u16,
        fc: u16,
        address: u32,
        write: bool,
        instruction: bool,
        size: u32,
        fault_wdata: u32,
    ) {
        // 68020/68030 long bus-cycle fault frame (format $B, 46
        // words, M68030UM 8.2). Real silicon dumps pipeline state
        // here and RTE *continues* the faulted instruction, with the
        // handler able to rerun or complete the data cycle through
        // the DF bit and the data input/output buffers. This core
        // rolls the faulting instruction back instead and stacks its
        // PC, so RTE restarts it from scratch: the pipeline dump and
        // writeback buffers are left zero, and a handler that
        // clears DF to suppress the rerun is not honoured (the
        // restart re-issues the access; harmless for memory, the
        // documented gap for side-effecting hardware registers).
        //
        // The special status word at +$0A is what a page-fault
        // handler (mmu.library, VMM, Enforcer) actually parses:
        // DF (bit 8) marks a faulted data cycle with its address in
        // the long at +$10, RW (bit 6) the direction, SIZ (bits 5:4,
        // 01 byte / 10 word / 00 long) the width, and FC2:0 the
        // address space. An instruction-fetch fault instead reports
        // a stage-B rerun (FB|RB, bits 14/12) with the fetch address
        // in the stage B address long at +$24, the shape real
        // handlers use to demand-page code.
        let data_fault = !instruction;
        let mut ssw: u16 = fc & 0x7;
        if data_fault {
            ssw |= 0x0100; // DF: rerun data cycle
            if !write {
                ssw |= 0x0040; // RW: read
            }
            // SIZ 11 is the three-byte transfer the 68020/68030 bus
            // performs for a dynamically sized access or a bit-field
            // operand spanning three bytes.
            ssw |= match size {
                1 => 0x0010, // SIZ 01: byte
                2 => 0x0020, // SIZ 10: word
                3 => 0x0030, // SIZ 11: three byte
                _ => 0x0000, // SIZ 00: long
            };
        } else {
            ssw |= 0x4000 | 0x1000; // FB|RB: rerun instruction stage B
        }
        let fmt_vec = 0xB000 | ((vector as u16) << 2);
        // Pushed high address -> low: internal words +$38..$5A (18
        // words), version word +$36, internal +$30..$34 (3 words),
        // data input buffer +$2C, internal +$28/$2A, stage B address
        // +$24, internal +$1C..$22 (4 words), data output buffer
        // +$18, internal +$14/$16, data fault address +$10, stage B
        // and C pipe words +$0E/$0C, SSW +$0A, internal +$08.
        for _ in 0..9 {
            self.push_32(bus, 0); // +$38..$5A
        }
        self.push_16(bus, 0); // +$36 version/internal
        for _ in 0..3 {
            self.push_16(bus, 0); // +$30..$34
        }
        self.push_32(bus, 0); // +$2C data input buffer
        self.push_32(bus, 0); // +$28/$2A internal
        self.push_32(bus, if data_fault { 0 } else { address }); // +$24 stage B address
        self.push_32(bus, 0); // +$20/$22 internal
        self.push_32(bus, 0); // +$1C/$1E internal
        // +$18 data output buffer: the value of a faulted write, so
        // a handler can complete the write itself (clear DF).
        self.push_32(bus, if write { fault_wdata } else { 0 });
        self.push_32(bus, 0); // +$14/$16 internal
        self.push_32(bus, if data_fault { address } else { 0 }); // +$10 data fault address
        self.push_16(bus, 0); // +$0E pipe stage B
        self.push_16(bus, 0); // +$0C pipe stage C
        self.push_16(bus, ssw); // +$0A special status word
        self.push_16(bus, 0); // +$08 internal
        self.push_16(bus, fmt_vec); // +$06 format/vector
        self.push_32(bus, self.ppc); // +$02 restart PC
        self.push_16(bus, old_sr); // +$00 SR
    }

    /// Process TRAP #n instruction.
    ///
    /// TRAP #n pushes the four-word format $0 frame on every 68010+ model
    /// (M68020UM table 6-5 lists TRAP #N under format $0; the earlier
    /// Musashi-derived format $2 frame on the 020/030 was wrong) with the
    /// next instruction's PC stacked.
    pub fn trap<B: AddressBus>(&mut self, bus: &mut B, trap_num: u8) -> i32 {
        let vector = vector::TRAP_BASE + (trap_num & 0xF) as u32;
        self.take_exception(bus, vector)
    }

    /// Group-2 instruction exceptions: CHK/CHK2, TRAPcc/TRAPV (and FTRAPcc),
    /// zero divide, and trace. The stacked PC is the next instruction on
    /// every model; the 68020+ push the six-word format $2 frame whose
    /// extra long is the address of the instruction that caused the
    /// exception (M68020UM table 6-5), so a handler can decode or skip it.
    pub(crate) fn take_group2_exception<B: AddressBus>(&mut self, bus: &mut B, vector: u32) -> i32 {
        let old_sr = self.get_sr();
        self.settle_pending_trace(vector);

        // Enter supervisor, clear trace.
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;

        let next_pc = self.pc;
        match self.cpu_type {
            super::types::CpuType::M68000 => {
                self.push_exception_frame_68000(bus, next_pc, old_sr);
            }
            super::types::CpuType::M68010 | super::types::CpuType::SCC68070 => {
                self.push_16(bus, (vector as u16) << 2);
                self.push_32(bus, next_pc);
                self.push_16(bus, old_sr);
            }
            _ => {
                let vec_word = (vector as u16) << 2;
                self.push_32(bus, self.ppc);
                self.push_16(bus, 0x2000 | (vec_word & 0x0FFF));
                self.push_32(bus, next_pc);
                self.push_16(bus, old_sr);
            }
        }

        self.jump_vector(bus, vector);
        self.exception_cycles(vector)
    }

    /// Process CHK exception.
    ///
    /// The caller (exec_chk) reports the comparison's internal clocks before
    /// calling this (8 for trap-on-too-big, 10 for trap-on-negative).
    pub fn exception_chk<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.take_group2_exception(bus, vector::CHK)
    }

    /// Process zero divide exception.
    pub fn exception_zero_divide<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.take_group2_exception(bus, vector::ZERO_DIVIDE)
    }

    /// Process privilege violation exception.
    pub fn exception_privilege<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.take_exception(bus, vector::PRIVILEGE_VIOLATION)
    }

    /// Process trace exception.
    pub fn exception_trace<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        // A pending trace recovers the CPU from the STOP state: STOP executed
        // with the trace bit set in the incoming SR takes the trace exception
        // instead of remaining stopped (the trace has priority over both the
        // stopped state and its supervisor check).
        self.stopped &= !super::execute::STOP_LEVEL_STOP;
        // 4 internal clocks precede the trace frame's first stack write
        // (Moira execException TRACE: SYNC(4)).
        self.internal_cycles(4);
        self.take_group2_exception(bus, vector::TRACE)
    }

    /// Process address error exception.
    ///
    /// The 68000 stacks its seven-word group 0 frame, the 68010 format $8,
    /// the 68020/68030 the format $B bus fault frame and the 68040/68060 the
    /// six-word format $2 frame.
    pub fn exception_address_error<B: AddressBus>(
        &mut self,
        bus: &mut B,
        address: u32,
        write: bool,
        instruction: bool,
    ) -> i32 {
        let old_sr = self.get_sr();
        let was_supervisor = (old_sr & 0x2000) != 0;

        // Enter supervisor mode, clear trace
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;

        // Build function code / status word
        // Bits: R/W (4), I/N (3), Function Code (2:0)
        let fc = if was_supervisor {
            if instruction {
                fc::SUPERVISOR_PROGRAM
            } else {
                fc::SUPERVISOR_DATA
            }
        } else if instruction {
            fc::USER_PROGRAM
        } else {
            fc::USER_DATA
        };
        match self.cpu_type {
            CpuType::M68000 => {
                // I/N is set when the fault interrupted exception processing
                // (an odd vector, for example) rather than an instruction.
                let in_exception = self.exception_processing;
                let stacked_pc = self.group0_stacked_pc_68000(write, instruction);
                // A PC-relative operand is read from program space.
                let ir = self.ir as u16;
                let pc_relative = (ir >> 3) & 7 == 7 && matches!(ir & 7, 2 | 3);
                let fc = if !write && !instruction && pc_relative {
                    fc + 1
                } else {
                    fc
                };
                self.push_group0_frame_68000(
                    bus,
                    fc,
                    write,
                    in_exception,
                    address,
                    old_sr,
                    stacked_pc,
                );
            }
            CpuType::M68010 | CpuType::SCC68070 => {
                let ssw = ssw_010(fc, write, instruction, 2);
                self.push_fault_frame_010(bus, vector::ADDRESS_ERROR, old_sr, ssw, address);
            }
            _ if self.is_040() || self.is_060() => {
                // An odd instruction prefetch uses the six-word format-$2
                // address-error frame (MC68040UM 8.2.2, 8.4.3; MC68060UM
                // 8.2.2), not the access-error frame used for bus and ATC
                // faults. The extra longword contains the referenced address
                // with A0 cleared, while the stacked PC identifies the
                // instruction that caused the address error.
                self.push_32_raw(bus, address & !1);
                self.push_16_raw(bus, 0x2000 | ((vector::ADDRESS_ERROR as u16) << 2));
                self.push_32_raw(bus, self.ppc);
                self.push_16_raw(bus, old_sr);
            }
            _ => {
                // 68020/68030: an address error stacks the same bus fault
                // frame as a bus error (M68020UM 6.1.2, M68030UM 8.1.2).
                // Only an instruction fetch can be misaligned on these
                // parts, so this is always the stage-B rerun shape.
                self.push_fault_frame_030(
                    bus,
                    vector::ADDRESS_ERROR,
                    old_sr,
                    fc,
                    address,
                    false,
                    true,
                    2,
                    0,
                );
            }
        }

        // Jump to vector
        self.jump_vector(bus, vector::ADDRESS_ERROR);

        50 // Cycles for address error
    }

    /// Process bus error exception.
    pub fn exception_bus_error<B: AddressBus>(
        &mut self,
        bus: &mut B,
        address: u32,
        write: bool,
        instruction: bool,
        size: u32,
        cause: Option<crate::mmu::MmuFaultCause>,
    ) -> i32 {
        // Read before the frame pushes below translate their own addresses.
        let transparent = std::mem::take(&mut self.pending_fault_transparent);
        let old_sr = self.get_sr();
        let was_supervisor = (old_sr & 0x2000) != 0;
        // The faulted write's data, captured before any frame push: the
        // pushes below are ordinary translated writes and each one replaces
        // pending_fault_wdata, so reading the field mid-frame would hand the
        // handler's writeback (WB3D / the 030 data output buffer) a zero
        // instead of the value the guest was storing.
        let fault_wdata = self.pending_fault_wdata;

        // Enter supervisor mode, clear trace
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;

        // Build function code / status word. A MOVES data fault carries the
        // SFC/DFC space it faulted in, not the CPU-state code: the handler
        // reads the fc back out of the frame's SSW to PTEST the faulted
        // space (mmu.library does exactly this). The override is consumed
        // here: SFC/DFC only governs the MOVES operand cycle, never the
        // exception dispatch itself -- the frame pushes and vector fetch
        // below run in supervisor space (a stuck override would walk the
        // user root pointer for the vector fetch, which on a kernel with
        // split user/supervisor trees reads garbage).
        let fc = match self.mmu_fc_override.take() {
            Some(ofc) => ofc as u16,
            None => {
                if was_supervisor {
                    if instruction {
                        fc::SUPERVISOR_PROGRAM
                    } else {
                        fc::SUPERVISOR_DATA
                    }
                } else if instruction {
                    fc::USER_PROGRAM
                } else {
                    fc::USER_DATA
                }
            }
        };
        match self.cpu_type {
            CpuType::M68000 => {
                // The 68000 bus error frame is the address error's. A bus
                // error during exception processing is a double fault and
                // never reaches this point, so I/N is always clear.
                let stacked_pc = self.group0_stacked_pc_68000(write, instruction);
                self.push_group0_frame_68000(bus, fc, write, false, address, old_sr, stacked_pc);
            }
            CpuType::M68010 | CpuType::SCC68070 => {
                let ssw = ssw_010(fc, write, instruction, size);
                self.push_fault_frame_010(bus, vector::BUS_ERROR, old_sr, ssw, address);
            }
            _ if self.is_060() => {
                // 68060 access-error stack frame (format $4, 8 words): the
                // fault address long at +$08 and the fault status long word
                // (FSLW) at +$0C. The instruction has been rolled back, so
                // RTE restarts it (same demand-paging model as the 040).
                let mut fslw = fslw_060(write, instruction, size, fc, cause, transparent);
                if self.mmu_misaligned_tail {
                    fslw |= fslw::MA;
                }
                let fmt_vec = 0x4000 | ((vector::BUS_ERROR as u16) << 2);
                self.push_32(bus, fslw);
                self.push_32(bus, address);
                self.push_16(bus, fmt_vec);
                self.push_32(bus, self.ppc); // restart PC
                self.push_16(bus, old_sr);
            }
            _ if self.is_040() => {
                // 68040 access-error stack frame (format 7, 30 words). The
                // caller (trigger_bus_error) has already rolled the instruction
                // back, so we stack PPC and leave the writeback/continuation
                // fields clear: RTE then restarts the faulting instruction
                // (demand-paging / Enforcer fix-and-retry model).
                // Layout (Musashi m68ki_stack_frame_0111), pushed high->low.
                //
                // SSW: RW (bit 8), SZ (bits 6:5, 040 encoding: 00 long,
                // 01 byte, 10 word), TM (bits 2:0) = function code, and ATC
                // (bit 10) when the fault came out of the MMU table walk
                // rather than the physical bus. ATC is what an OS-level
                // page-fault handler (mmu.library, VMM, Enforcer) tests to
                // tell a translation fault it must service from a real bus
                // error it must pass on, so a translation fault without it
                // gurus instead of demand-faulting.
                let rw = if write { 0 } else { 0x0100 }; // SSW bit 8: 1 = read
                let atc = if cause.is_some() { 0x0400u16 } else { 0 }; // SSW bit 10
                // The 68040 bus has no three-byte encoding (SZ 11 is a line
                // transfer), so a three-byte operand reports as a long here.
                let sz = match size {
                    1 => 0x0020, // byte
                    2 => 0x0040, // word
                    _ => 0x0000, // long
                };
                // MA (bit 11): the fault is on a later part of an operand
                // split into several bus cycles.
                let ma = if self.mmu_misaligned_tail { 0x0800 } else { 0 };
                let ssw = rw | ma | atc | sz | (fc & 0x7); // TM = function code
                let fmt_vec = 0x7000 | ((vector::BUS_ERROR as u16) << 2);
                // A normal faulted write is reported in writeback slot 3 (V
                // bit, size and TM in WB3S; address and data in WB3A/WB3D),
                // matching real 68040 silicon and the WinUAE/Amiberry MMU
                // reference (cpummu.cpp sets regs.wb3_status/wb3_data on a
                // write fault and clears wb2 -- WB2 is reserved for MOVE16
                // cacheline writes). MuGuardianAngel completes an allowed
                // write by storing WB3D to WB3A, and Enforcer/MuForce discard
                // a protected store by clearing WB3S.V; RTE (below) honours
                // the cleared V. Putting the store in WB2 instead makes MuGA
                // read a zero WB3D and clobber the target with 0.
                let wb3s = if write {
                    0x0080 | sz | (fc & 0x7) // V | SZ | TM
                } else {
                    0
                };
                for _ in 0..3 {
                    self.push_32(bus, 0); // PD3, PD2, PD1
                }
                self.push_32(bus, 0); // WB1D / PD0
                self.push_32(bus, 0); // WB1A
                self.push_32(bus, 0); // WB2D
                self.push_32(bus, 0); // WB2A
                self.push_32(bus, if write { fault_wdata } else { 0 }); // WB3D
                self.push_32(bus, if write { address } else { 0 }); // WB3A
                self.push_32(bus, address); // fault address
                self.push_16(bus, 0); // WB1S
                self.push_16(bus, 0); // WB2S
                self.push_16(bus, wb3s); // WB3S
                self.push_16(bus, ssw); // special status word
                self.push_32(bus, address); // effective address
                self.push_16(bus, fmt_vec); // format 7 / vector offset
                self.push_32(bus, self.ppc); // restart PC
                self.push_16(bus, old_sr);
            }
            _ => {
                self.push_fault_frame_030(
                    bus,
                    vector::BUS_ERROR,
                    old_sr,
                    fc,
                    address,
                    write,
                    instruction,
                    size,
                    fault_wdata,
                );
            }
        }

        // Jump to vector
        self.jump_vector(bus, vector::BUS_ERROR);

        50 // Cycles for bus error
    }

    /// Common exception processing (simple frame: SR, PC).
    ///
    /// Implements double-fault detection: if an exception occurs while already
    /// processing an exception, the CPU halts (similar to x86 triple fault).
    pub fn take_exception<B: AddressBus>(&mut self, bus: &mut B, vector: u32) -> i32 {
        // Double-fault detection: if we're already processing an exception and
        // another exception occurs, halt the CPU. This prevents infinite recursion.
        if self.exception_processing {
            // Double fault - halt the CPU
            self.stopped = 1;
            self.run_mode = RUN_MODE_BERR_AERR_RESET;
            return 0;
        }

        // Mark that we're processing an exception. This flag is checked by translate()
        // to bypass MMU translation during exception frame writes.
        self.exception_processing = true;

        // Exception entry spends 4 internal clocks (vector number / state
        // capture) before the first stack write.
        self.internal_cycles(4);

        let old_sr = self.get_sr();
        self.settle_pending_trace(vector);

        // Match Musashi `m68ki_init_exception`: enter supervisor mode but do not modify M.
        self.set_s_flag(SFLAG_SET);

        // Clear trace flags
        self.t1_flag = 0;
        self.t0_flag = 0;

        // Select stacked PC (Musashi-style: traps/interrupts stack the next PC; faults stack PPC).
        let stacked_pc = if vector == vector::TRAPV
            || vector == vector::TRACE
            || vector == vector::ZERO_DIVIDE
            || (vector::TRAP_BASE..vector::TRAP_BASE + 16).contains(&vector)
            || (24..=31).contains(&vector)
        {
            self.pc
        } else {
            self.ppc
        };

        // Match Musashi `m68ki_stack_frame_0000`:
        // - 68000: push PC, then SR (3-word frame)
        // - 68010+: push vector offset word (vector<<2), then PC, then SR (format 0)
        if self.cpu_type == super::types::CpuType::M68000 {
            self.push_exception_frame_68000(bus, stacked_pc, old_sr);
        } else {
            self.push_16(bus, (vector as u16) << 2);
            self.push_32(bus, stacked_pc);
            self.push_16(bus, old_sr);
        }

        // Read vector and jump
        self.jump_vector(bus, vector);

        // Done processing exception
        self.exception_processing = false;

        self.exception_cycles(vector)
    }

    /// 68040/68060 "FP unimplemented instruction" exception: Line-F vector with
    /// the six-word format $2 frame the software package dispatches on (fmt/vector
    /// word $202C). The frame's PC is the NEXT instruction (every extension
    /// word must be consumed before calling this), the EA field holds the
    /// calculated operand address (0 when the operand is not in memory),
    /// and FPIAR points at the faulting instruction - the 060SP fetches
    /// the opcode through FPIAR, not the frame.
    pub(crate) fn take_fp_unimp_060<B: AddressBus>(&mut self, bus: &mut B, ea: u32) -> i32 {
        self.fpiar = self.ppc;
        let old_sr = self.get_sr();
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;
        let vec_word = (vector::LINE_1111 as u16) << 2;
        self.push_32(bus, ea);
        self.push_16(bus, 0x2000 | (vec_word & 0x0FFF));
        self.push_32(bus, self.pc);
        self.push_16(bus, old_sr);
        self.jump_vector(bus, vector::LINE_1111);
        self.exception_cycles(vector::LINE_1111)
    }

    /// An enabled exception (vectors 49-54) of a 6888x FMOVE out, which
    /// the coprocessor signals during the instruction, after the
    /// destination has been written. The 68020/68030 stacks the ten-word
    /// format $9 coprocessor mid-instruction frame: the stacked PC is the
    /// next instruction, the FMOVE's address is at +$08, and the evaluated
    /// destination address `ea` is at +$10, where the handler finds the
    /// operand it has to repair (MC68881/MC68882UM 6.1.3 and 7.4.2.6,
    /// MC68020UM 10.5.3.3). `ea` is 0 for a data-register destination; the
    /// other two internal words are not modelled and read as zero.
    pub(crate) fn take_fp_mid_instruction_exception<B: AddressBus>(
        &mut self,
        bus: &mut B,
        vector: u32,
        ea: u32,
    ) -> i32 {
        if self.exception_processing {
            self.stopped = 1;
            self.run_mode = RUN_MODE_BERR_AERR_RESET;
            return 0;
        }
        self.exception_processing = true;
        self.internal_cycles(4);
        let old_sr = self.get_sr();
        self.settle_pending_trace(vector);
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;
        self.push_32(bus, ea); // internal words at +$10: destination address
        self.push_32(bus, 0); // internal words at +$0C
        self.push_32(bus, self.ppc); // instruction address
        self.push_16(bus, 0x9000 | ((vector as u16) << 2));
        self.push_32(bus, self.pc);
        self.push_16(bus, old_sr);
        self.jump_vector(bus, vector);
        self.exception_processing = false;
        self.exception_cycles(vector)
    }

    /// A 68040 floating-point exception whose handler finishes the
    /// instruction itself: the four-word format $0 frame with the address
    /// of the next instruction as the stacked PC (M68040UM 9.6.2).
    pub(crate) fn take_fp_exception_past_instruction<B: AddressBus>(
        &mut self,
        bus: &mut B,
        vector: u32,
    ) -> i32 {
        if self.exception_processing {
            self.stopped = 1;
            self.run_mode = RUN_MODE_BERR_AERR_RESET;
            return 0;
        }
        self.exception_processing = true;
        self.internal_cycles(4);
        let old_sr = self.get_sr();
        self.settle_pending_trace(vector);
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;
        self.push_16(bus, (vector as u16) << 2);
        self.push_32(bus, self.pc);
        self.push_16(bus, old_sr);
        self.jump_vector(bus, vector);
        self.exception_processing = false;
        self.exception_cycles(vector)
    }

    /// 68040 unsupported data type on FMOVE to memory: a post-instruction
    /// exception through vector 55 with the six-word format $3 frame, whose
    /// PC is the next instruction and whose EA field is the destination
    /// (M68040UM 9.6.2).
    pub(crate) fn take_fp_unsupp_post_040<B: AddressBus>(&mut self, bus: &mut B, ea: u32) -> i32 {
        self.fpiar = self.ppc;
        let old_sr = self.get_sr();
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;
        let vec_word = (vector::FP_UNSUPP_DATA_TYPE as u16) << 2;
        self.push_32(bus, ea);
        self.push_16(bus, 0x3000 | (vec_word & 0x0FFF));
        self.push_32(bus, self.pc);
        self.push_16(bus, old_sr);
        self.jump_vector(bus, vector::FP_UNSUPP_DATA_TYPE);
        self.exception_cycles(vector::FP_UNSUPP_DATA_TYPE)
    }

    /// 68060 "FPU disabled" exception (PCR.DFP set, or an LC/EC060): the
    /// eight-word format $4 frame ($402C) whose +$0C long holds the PC of
    /// the faulted instruction so the OS can enable the FPU and restart.
    /// The stacked PC also restarts the instruction; FPIAR is untouched
    /// (the FPU never saw the instruction).
    pub(crate) fn take_fp_disabled_060<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        let old_sr = self.get_sr();
        self.set_s_flag(SFLAG_SET);
        self.t1_flag = 0;
        self.t0_flag = 0;
        let vec_word = (vector::LINE_1111 as u16) << 2;
        self.push_32(bus, self.ppc); // PC of the faulted instruction
        self.push_32(bus, 0); // effective address (unused for disabled)
        self.push_16(bus, 0x4000 | (vec_word & 0x0FFF));
        self.push_32(bus, self.ppc); // restart the instruction
        self.push_16(bus, old_sr);
        self.jump_vector(bus, vector::LINE_1111);
        self.exception_cycles(vector::LINE_1111)
    }

    /// Get cycles for exception processing.
    fn exception_cycles(&self, vector: u32) -> i32 {
        match vector {
            vector::RESET_SSP | vector::RESET_PC => 40,
            vector::BUS_ERROR | vector::ADDRESS_ERROR => 50,
            vector::ILLEGAL_INSTRUCTION => 34,
            vector::ZERO_DIVIDE => 38,
            vector::CHK => 40,
            vector::TRAPV => 34,
            vector::PRIVILEGE_VIOLATION => 34,
            vector::TRACE => 34,
            vector::LINE_1010 | vector::LINE_1111 => 34,
            24..=31 => 44, // Autovector interrupts
            _ => 34,       // TRAPs and user vectors
        }
    }

    /// Decide whether the trace pending for the current instruction survives
    /// the exception it is taking through `vector`.
    ///
    /// An instruction that completes by trapping (zero divide, CHK/CHK2,
    /// TRAPV/TRAPcc, TRAP #n) is still traced on the 68000 through 68030:
    /// the trace exception is processed right after the trap's, so its
    /// handler runs first with the trap handler's address stacked
    /// (M68000UM 6.3.8, M68020UM 6.1.8). An instruction that did not
    /// execute (illegal, line A/F, privilege violation, format error) is
    /// not traced on any model, and the 68040 and 68060 do not report a
    /// trace pending together with an instruction trap at all (M68040UM
    /// 8.3, M68060UM 8.2.6).
    ///
    /// The trace decision reads the trace bits of [`CpuCore::sr_save`], so
    /// a cancelled trace clears them there.
    fn settle_pending_trace(&mut self, vector: u32) {
        let instruction_trap = matches!(vector, vector::ZERO_DIVIDE | vector::CHK | vector::TRAPV)
            || (vector::TRAP_BASE..vector::TRAP_BASE + 16).contains(&vector);
        if !instruction_trap || self.is_040() || self.is_060() {
            self.sr_save &= !0xC000;
        }
    }

    /// Check for trace exception after instruction execution.
    /// SR-writing instructions (MOVE to SR, ORI/ANDI/EORI to SR) run the
    /// pending-trace check against the OLD T0 on the 68020+: writing SR is
    /// a pipeline-synchronizing event, so a set T0 traces the instruction
    /// even when the write clears it. Marking the boundary as a change of
    /// flow lets check_trace() (which reads the pre-instruction SR) see it.
    pub(crate) fn trace_t0_sr_write(&mut self) {
        if !self.is_pre_68020 {
            self.change_of_flow = true;
        }
    }

    /// The 68040 additionally runs the T0 check after a small set of
    /// pipeline-synchronizing instructions that are not flow changes:
    /// NOP, MOVEC, MOVE to USP, CAS, CAS2, MOVES, TAS, PFLUSH and PTEST
    /// (WinUAE's trace_t0_68040_only sites).
    pub(crate) fn trace_t0_68040_sync(&mut self) {
        if matches!(
            self.cpu_type,
            super::types::CpuType::M68EC040
                | super::types::CpuType::M68LC040
                | super::types::CpuType::M68040
        ) {
            self.change_of_flow = true;
        }
    }

    /// Determine whether the instruction that just retired requests a trace
    /// exception and clear the one-instruction flow-change latch.
    ///
    /// T1 traces every instruction. On the 68020 and later, T0 traces only
    /// flow-changing or pipeline-synchronizing instructions. The decision
    /// uses [`CpuCore::sr_save`], the status register captured before the
    /// instruction, so an RTE that restores a trace bit does not trace itself.
    pub fn check_trace(&mut self) -> bool {
        // T1 trace: trace after every instruction
        // T0 trace: trace only on change-of-flow (68020+)
        // We check the T1/T0 bits from sr_save (SR BEFORE instruction), not current SR.
        // This is important for RTE: if RTE restores T1=1, we don't take trace immediately.
        let t1_before = (self.sr_save & 0x8000) != 0;
        let t0_before = (self.sr_save & 0x4000) != 0;
        let should_trace = t1_before || (t0_before && self.change_of_flow);
        // Reset change_of_flow flag after checking
        self.change_of_flow = false;
        should_trace
    }

    // ========== Fallback Exception Methods for Unhandled Traps ==========

    /// Take A-line exception for an unhandled trap.
    ///
    /// Call this to manually take an A-line exception (vector 10).
    /// Note: With `step()`, this is called automatically. This method is
    /// primarily used internally by `step_with_hle_handler()` when the
    /// handler returns `false`.
    ///
    /// This rewinds the PC to the trap instruction before taking the exception.
    pub fn take_aline_exception<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.pc = self.ppc; // Rewind PC to the trap instruction
        self.take_exception(bus, vector::LINE_1010)
    }

    /// Take F-line exception for an unhandled trap.
    ///
    /// Call this after receiving `StepResult::FlineTrap` if you cannot handle the trap.
    /// This rewinds the PC and takes the real hardware exception (vector 11).
    pub fn take_fline_exception<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.pc = self.ppc; // Rewind PC to the trap instruction
        self.take_exception(bus, vector::LINE_1111)
    }

    /// Take TRAP #n exception for an unhandled trap.
    ///
    /// Call this after receiving `StepResult::TrapInstruction` if you cannot handle the trap.
    /// This takes the real hardware exception (vector 32+n).
    ///
    /// Note: Unlike A-line/F-line exceptions, TRAP exceptions stack the PC of the
    /// instruction AFTER the TRAP, so we do NOT rewind PC here.
    pub fn take_trap_exception<B: AddressBus>(&mut self, bus: &mut B, trap_num: u8) -> i32 {
        // Don't rewind PC - TRAP stacks the NEXT instruction address
        self.trap(bus, trap_num)
    }

    /// Take BKPT exception for an unhandled breakpoint.
    ///
    /// Call this after receiving `StepResult::Breakpoint` if you cannot handle it.
    /// This rewinds the PC and takes the illegal instruction exception (vector 4).
    pub fn take_bkpt_exception<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.pc = self.ppc; // Rewind PC to the breakpoint instruction
        self.take_exception(bus, vector::ILLEGAL_INSTRUCTION)
    }

    /// Take illegal instruction exception for an unhandled illegal opcode.
    ///
    /// Call this after receiving `StepResult::IllegalInstruction` if you cannot handle it.
    /// This rewinds the PC and takes the real hardware exception (vector 4).
    pub fn take_illegal_exception<B: AddressBus>(&mut self, bus: &mut B) -> i32 {
        self.pc = self.ppc; // Rewind PC to the illegal instruction
        self.take_exception(bus, vector::ILLEGAL_INSTRUCTION)
    }
}

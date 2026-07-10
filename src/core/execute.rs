//! Main execution loop.
//!
//! Implements the fetch-decode-execute cycle.

use super::cpu::{CpuCore, SFLAG_SET};
use super::decode::{dispatch_instruction, needs_rollback_snapshot};
use super::memory::AddressBus;
use super::op_cache::{BatchInnerExit, CachedRunResult};
#[cfg(not(target_family = "wasm"))]
use super::op_cache::DecodedSimpleOp;
use super::trace_jit;
use super::types::{BatchExit, BatchResult, StepResult};

/// Stop level constants.
pub const STOP_LEVEL_STOP: u32 = 1;
pub const STOP_LEVEL_HALT: u32 = 2;

/// Run mode constants.
pub const RUN_MODE_NORMAL: u32 = 0;
pub const RUN_MODE_BERR_AERR_RESET: u32 = 1;

impl CpuCore {
    #[inline]
    fn prepare_rollback_snapshot(&mut self, opcode: u16) {
        if needs_rollback_snapshot(opcode) {
            self.dar_save = self.dar;
            self.sr_save = self.get_sr();
        } else if (self.t1_flag | self.t0_flag) != 0 {
            // Trace checks still need the pre-instruction SR. Simple no-fault instructions do
            // not need the D/A rollback snapshot.
            self.sr_save = self.get_sr();
        } else {
            self.sr_save = 0;
        }
    }

    /// Unconditional snapshot for paths where the opcode is already known
    /// not to be a simple op (a decoded-op-cache miss). Skips
    /// `needs_rollback_snapshot`, which would re-run the full simple-op
    /// decode just to conclude the same thing.
    #[inline]
    fn prepare_rollback_snapshot_full(&mut self) {
        self.dar_save = self.dar;
        self.sr_save = self.get_sr();
    }

    /// Caller must have checked [`CpuCore::can_run_decoded_simple_ops`].
    #[inline]
    fn try_execute_decoded_simple_step(&mut self, opcode: u16) -> Option<StepResult> {
        let op = self.decoded_simple_op(opcode, self.cpu_type)?;
        #[cfg(not(target_family = "wasm"))]
        let branch_pc = if matches!(op, DecodedSimpleOp::BranchShort { .. }) {
            Some(self.ppc)
        } else {
            None
        };
        let cycles = op.execute(self);
        #[cfg(not(target_family = "wasm"))]
        if let Some(branch_pc) = branch_pc
            && self.pc <= branch_pc
        {
            let _ = trace_jit::note_backward_branch(self, self.cpu_type);
        }

        Some(StepResult::Ok { cycles })
    }

    /// Execute instructions for the given number of cycles.
    ///
    /// Returns the number of cycles actually consumed.
    ///
    /// **Note**: This function is intended for batch execution without HLE support.
    /// A-line and F-line traps are silently ignored (treated as 0 cycles).
    /// For HLE support, use `step()` and handle `StepResult::AlineTrap`/`FlineTrap`.
    pub fn execute<B: AddressBus>(&mut self, bus: &mut B, num_cycles: i32) -> i32 {
        // Handle reset cycles
        if self.reset_cycles > 0 {
            let rc = self.reset_cycles as i32;
            self.reset_cycles = 0;
            let remaining = num_cycles - rc;
            if remaining <= 0 {
                return rc;
            }
            self.cycles_remaining = remaining;
        } else {
            self.cycles_remaining = num_cycles;
        }
        self.initial_cycles = num_cycles;

        // Check for pending interrupts
        self.check_and_service_interrupts(bus);

        // If stopped, consume no cycles
        if self.stopped != 0 {
            self.cycles_remaining = 0;
            return self.initial_cycles;
        }

        // Main execution loop
        let mut probe_on_entry = true;
        while self.cycles_remaining > 0 {
            let mut known_complex = false;
            let opcode = if self.can_run_decoded_simple_ops() {
                match self.execute_decoded_simple_run(bus, probe_on_entry) {
                    CachedRunResult::Ran => continue,
                    CachedRunResult::Fault => {
                        self.run_mode = RUN_MODE_NORMAL;
                        probe_on_entry = true;
                        continue;
                    }
                    CachedRunResult::Miss(opcode) => {
                        known_complex = true;
                        opcode
                    }
                }
            } else {
                // Save previous PC
                self.ppc = self.pc;

                // Fetch opcode
                let opcode = self.read_opcode_16(bus);

                // If a bus/address error occurred during fetch, the exception is already taken.
                if self.run_mode == RUN_MODE_BERR_AERR_RESET {
                    self.run_mode = RUN_MODE_NORMAL;
                    probe_on_entry = true;
                    continue;
                }
                self.ir = opcode as u32;
                opcode
            };

            if self.ir != opcode as u32 {
                self.ir = opcode as u32;
            }

            if known_complex {
                self.prepare_rollback_snapshot_full();
            } else {
                self.prepare_rollback_snapshot(opcode);
            }

            // Dispatch instruction
            let result = dispatch_instruction(self, bus, opcode);

            // Auto-take all trap exceptions, extract cycles
            use crate::core::types::InternalStepResult;
            let cycles = match result {
                InternalStepResult::Ok { cycles } => cycles,
                InternalStepResult::AlineTrap { .. } => self.take_aline_exception(bus),
                InternalStepResult::FlineTrap { .. } => self.take_fline_exception(bus),
                InternalStepResult::TrapInstruction { trap_num } => {
                    self.take_trap_exception(bus, trap_num)
                }
                InternalStepResult::Breakpoint { .. } => self.take_bkpt_exception(bus),
                InternalStepResult::IllegalInstruction { .. } => self.take_illegal_exception(bus),
            };
            self.cycles_remaining -= cycles;

            // If a bus/address error occurred mid-instruction, we already built the exception frame
            // and jumped to the handler. Skip trace/interrupt checks for the faulting instruction.
            if self.run_mode == RUN_MODE_BERR_AERR_RESET {
                self.run_mode = RUN_MODE_NORMAL;
                probe_on_entry = true;
                continue;
            }

            // Only a backward branch can land on a (potential) trace head,
            // so straight-line dispatches re-enter the fast loop without a
            // trace-cache probe.
            probe_on_entry =
                self.pc <= self.ppc && trace_jit::note_backward_branch(self, self.cpu_type);

            // Check for trace exception (T1 flag set before instruction)
            if self.check_trace() {
                let trace_cycles = self.exception_trace(bus);
                self.cycles_remaining -= trace_cycles;
            }

            // Check for interrupts after each instruction
            if self.int_level > 0 {
                self.check_and_service_interrupts(bus);
            }

            // Check if stopped/halted
            if self.stopped != 0 {
                break;
            }
        }

        // Return cycles consumed
        self.initial_cycles - self.cycles_remaining
    }

    /// Execute up to `max_instructions` instructions, returning on the first
    /// interesting event.
    ///
    /// This is the fast path for High-Level Emulation embedders that would
    /// otherwise call [`step`](Self::step) in a loop: the whole batch runs
    /// inside the decoded-op cache / trace-JIT inner loop, and control only
    /// returns to the caller when it has something to do:
    ///
    /// - a trap the embedder wants to intercept (A-line/F-line/TRAP/BKPT/
    ///   illegal — surfaced exactly like [`step`](Self::step), never taken
    ///   as a hardware exception),
    /// - the CPU stopping (STOP instruction),
    /// - execution reaching a PC listed in `watch_pcs`, or
    /// - the instruction budget running out.
    ///
    /// Watch semantics: `watch_pcs` is checked after every retired
    /// instruction, *before* the instruction at the new PC executes. The
    /// entry PC is intentionally **not** checked, so a caller that resumes
    /// from a watched PC does not loop forever; keep the list short (it is
    /// scanned linearly).
    ///
    /// Unlike [`execute`](Self::execute), this entry point is
    /// instruction-budgeted and does not maintain cycle accounting
    /// (`cycles_remaining` is clobbered). Trace exceptions are taken
    /// internally and pending interrupts are serviced between instructions,
    /// matching [`step`](Self::step) semantics.
    pub fn run_batch<B: AddressBus>(
        &mut self,
        bus: &mut B,
        max_instructions: u32,
        watch_pcs: &[u32],
    ) -> BatchResult {
        // Capture the bus's fastmem window for the duration of this batch.
        // Never with an active MMU: fastmem addresses are physical.
        if !(self.has_pmmu && self.pmmu_enabled)
            && let Some(fm) = bus.fast_mem()
            && fm.len >= 4
            && !fm.ptr.is_null()
        {
            self.fm_ptr = fm.ptr as usize;
            self.fm_base = fm.base;
            self.fm_len = fm.len;
        }
        let result = self.run_batch_inner(bus, max_instructions, watch_pcs);
        self.fm_ptr = 0;
        self.fm_base = 0;
        self.fm_len = 0;
        result
    }

    fn run_batch_inner<B: AddressBus>(
        &mut self,
        bus: &mut B,
        max_instructions: u32,
        watch_pcs: &[u32],
    ) -> BatchResult {
        use crate::core::types::InternalStepResult;

        if self.stopped != 0 {
            return BatchResult {
                instructions: 0,
                exit: BatchExit::Stopped,
            };
        }

        let mut retired: u32 = 0;
        let mut probe_on_entry = true;

        loop {
            // The trace JIT's headroom guard compares against
            // `cycles_remaining`; keep it topped up so it can never gate a
            // trace in this instruction-budgeted mode (traces decrement it
            // as they run).
            self.cycles_remaining = i32::MAX / 2;

            if retired >= max_instructions {
                return BatchResult {
                    instructions: retired,
                    exit: BatchExit::BudgetExhausted,
                };
            }

            let mut known_complex = false;
            let opcode = if self.can_run_decoded_simple_ops() {
                match self.run_decoded_simple_batch(
                    bus,
                    max_instructions - retired,
                    watch_pcs,
                    &mut retired,
                    probe_on_entry,
                ) {
                    BatchInnerExit::Budget => {
                        return BatchResult {
                            instructions: retired,
                            exit: BatchExit::BudgetExhausted,
                        };
                    }
                    BatchInnerExit::Watched(pc) => {
                        return BatchResult {
                            instructions: retired,
                            exit: BatchExit::WatchedPc { pc },
                        };
                    }
                    BatchInnerExit::Fault => {
                        self.run_mode = RUN_MODE_NORMAL;
                        probe_on_entry = true;
                        continue;
                    }
                    BatchInnerExit::Miss(opcode) => {
                        known_complex = true;
                        opcode
                    }
                }
            } else {
                self.ppc = self.pc;
                let opcode = self.read_opcode_16(bus);
                if self.run_mode == RUN_MODE_BERR_AERR_RESET {
                    self.run_mode = RUN_MODE_NORMAL;
                    probe_on_entry = true;
                    continue;
                }
                self.ir = opcode as u32;
                opcode
            };

            if self.ir != opcode as u32 {
                self.ir = opcode as u32;
            }

            if known_complex {
                self.prepare_rollback_snapshot_full();
            } else {
                self.prepare_rollback_snapshot(opcode);
            }

            let result = dispatch_instruction(self, bus, opcode);

            // A dispatched instruction may have enabled the MMU
            // (PMOVE/MOVEC); fastmem addresses are physical, so drop the
            // window as soon as translation turns on.
            if self.fm_len != 0 && self.has_pmmu && self.pmmu_enabled {
                self.fm_ptr = 0;
                self.fm_base = 0;
                self.fm_len = 0;
            }

            let exit = match result {
                InternalStepResult::Ok { .. } => None,
                InternalStepResult::AlineTrap { opcode } => Some(BatchExit::AlineTrap { opcode }),
                InternalStepResult::FlineTrap { opcode } => Some(BatchExit::FlineTrap { opcode }),
                InternalStepResult::TrapInstruction { trap_num } => {
                    Some(BatchExit::TrapInstruction { trap_num })
                }
                InternalStepResult::Breakpoint { bp_num } => {
                    Some(BatchExit::Breakpoint { bp_num })
                }
                InternalStepResult::IllegalInstruction { opcode } => {
                    Some(BatchExit::IllegalInstruction { opcode })
                }
            };
            if let Some(exit) = exit {
                return BatchResult {
                    instructions: retired,
                    exit,
                };
            }
            retired += 1;

            // A bus/address error mid-instruction already built the exception
            // frame and jumped to the handler; skip trace/interrupt checks
            // for the faulting instruction (mirrors `execute`).
            if self.run_mode == RUN_MODE_BERR_AERR_RESET {
                self.run_mode = RUN_MODE_NORMAL;
                probe_on_entry = true;
            } else {
                // Mirrors `execute`: only backward branches can reach a
                // trace head, so straight-line dispatches re-enter the
                // fast loop without a trace-cache probe.
                probe_on_entry =
                    self.pc <= self.ppc && trace_jit::note_backward_branch(self, self.cpu_type);

                if !self.sst_m68000_compat && self.check_trace() {
                    let _ = self.exception_trace(bus);
                }

                if self.int_level > 0 {
                    self.check_and_service_interrupts(bus);
                }
            }

            if self.stopped != 0 {
                return BatchResult {
                    instructions: retired,
                    exit: BatchExit::Stopped,
                };
            }

            if !watch_pcs.is_empty() && watch_pcs.contains(&self.pc) {
                return BatchResult {
                    instructions: retired,
                    exit: BatchExit::WatchedPc { pc: self.pc },
                };
            }
        }
    }

    /// Execute a single instruction.
    ///
    /// Returns a `StepResult` indicating:
    /// - `Ok { cycles }` - Normal instruction execution
    /// - `Stopped` - CPU is stopped
    ///
    /// Traps are surfaced as `StepResult` variants; exceptions are not taken
    /// automatically in this mode. For HLE interception with automatic fallback
    /// to exceptions, use `step_with_hle_handler()`.
    pub fn step<B: AddressBus>(&mut self, bus: &mut B) -> StepResult {
        use crate::core::types::{InternalStepResult, StepResult};

        if self.stopped != 0 {
            return StepResult::Stopped;
        }

        self.ppc = self.pc;
        self.ir = self.read_opcode_16(bus) as u32;

        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            self.run_mode = RUN_MODE_NORMAL;
            return StepResult::Ok { cycles: 0 };
        }

        if self.can_run_decoded_simple_ops() {
            if let Some(result) = self.try_execute_decoded_simple_step(self.ir as u16) {
                return result;
            }
            // The decoded-op cache just said this is not a simple op;
            // snapshot without re-running the simple-op decode.
            self.prepare_rollback_snapshot_full();
        } else {
            self.prepare_rollback_snapshot(self.ir as u16);
        }

        let result = dispatch_instruction(self, bus, self.ir as u16);

        let res = match result {
            InternalStepResult::Ok { cycles } => StepResult::Ok { cycles },
            InternalStepResult::AlineTrap { opcode } => StepResult::AlineTrap { opcode },
            InternalStepResult::FlineTrap { opcode } => StepResult::FlineTrap { opcode },
            InternalStepResult::TrapInstruction { trap_num } => {
                StepResult::TrapInstruction { trap_num }
            }
            InternalStepResult::Breakpoint { bp_num } => StepResult::Breakpoint { bp_num },
            InternalStepResult::IllegalInstruction { opcode } => {
                StepResult::IllegalInstruction { opcode }
            }
        };

        if matches!(res, StepResult::Ok { .. }) {
            if self.run_mode == RUN_MODE_BERR_AERR_RESET {
                self.run_mode = RUN_MODE_NORMAL;
                return res;
            }

            // Check for trace exception
            if !self.sst_m68000_compat && self.check_trace() {
                let trace_cycles = self.exception_trace(bus);
                if let StepResult::Ok { cycles } = res {
                    return StepResult::Ok {
                        cycles: cycles + trace_cycles,
                    };
                }
            }

            // Check for interrupts after instruction
            if self.int_level > 0 {
                self.check_and_service_interrupts(bus);
            }
        }

        res
    }

    /// Execute a single instruction with HLE trap handling (CPU + bus access).
    ///
    /// This method is the preferred way to run the CPU with High-Level Emulation.
    /// When a trap instruction is encountered, the appropriate `HleHandler` method
    /// is called. If the handler returns `true`, the trap is considered handled
    /// and execution continues. If it returns `false` (or is not implemented),
    /// the real hardware exception is taken automatically.
    ///
    /// # Example
    /// ```
    /// use m68k::{AddressBus, CpuCore, HleHandler};
    ///
    /// struct MyHandler { handled: bool }
    /// impl HleHandler for MyHandler {
    ///     fn handle_aline(
    ///         &mut self,
    ///         _cpu: &mut CpuCore,
    ///         _bus: &mut dyn AddressBus,
    ///         _opcode: u16,
    ///     ) -> bool {
    ///         self.handled = true;
    ///         true // HLE handled it
    ///     }
    /// }
    /// ```
    pub fn step_with_hle_handler<B: AddressBus, T: super::types::HleHandler>(
        &mut self,
        bus: &mut B,
        handler: &mut T,
    ) -> StepResult {
        use crate::core::types::{InternalStepResult, StepResult};

        if self.stopped != 0 {
            return StepResult::Stopped;
        }

        self.ppc = self.pc;
        self.ir = self.read_opcode_16(bus) as u32;

        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            self.run_mode = RUN_MODE_NORMAL;
            return StepResult::Ok { cycles: 0 };
        }

        if self.can_run_decoded_simple_ops() {
            if let Some(result) = self.try_execute_decoded_simple_step(self.ir as u16) {
                return result;
            }
            // The decoded-op cache just said this is not a simple op;
            // snapshot without re-running the simple-op decode.
            self.prepare_rollback_snapshot_full();
        } else {
            self.prepare_rollback_snapshot(self.ir as u16);
        }

        let result = dispatch_instruction(self, bus, self.ir as u16);

        // Handle trap results via callbacks, fallback to exception if not handled
        let cycles = match result {
            InternalStepResult::Ok { cycles } => cycles,
            InternalStepResult::AlineTrap { opcode } => {
                if !handler.handle_aline(self, bus, opcode) {
                    self.take_aline_exception(bus)
                } else {
                    0 // HLE handled, 0 cycles for now
                }
            }
            InternalStepResult::FlineTrap { opcode } => {
                if !handler.handle_fline(self, bus, opcode) {
                    self.take_fline_exception(bus)
                } else {
                    0
                }
            }
            InternalStepResult::TrapInstruction { trap_num } => {
                if !handler.handle_trap(self, bus, trap_num) {
                    self.take_trap_exception(bus, trap_num)
                } else {
                    0
                }
            }
            InternalStepResult::Breakpoint { bp_num } => {
                if !handler.handle_breakpoint(self, bus, bp_num) {
                    self.take_bkpt_exception(bus)
                } else {
                    0
                }
            }
            InternalStepResult::IllegalInstruction { opcode } => {
                if !handler.handle_illegal(self, bus, opcode) {
                    self.take_illegal_exception(bus)
                } else {
                    0
                }
            }
        };

        if self.run_mode == RUN_MODE_BERR_AERR_RESET {
            self.run_mode = RUN_MODE_NORMAL;
            return StepResult::Ok { cycles };
        }

        // Check for trace exception
        if !self.sst_m68000_compat && self.check_trace() {
            let trace_cycles = self.exception_trace(bus);
            return StepResult::Ok {
                cycles: cycles + trace_cycles,
            };
        }

        // Check for interrupts after instruction
        if self.int_level > 0 {
            self.check_and_service_interrupts(bus);
        }

        StepResult::Ok { cycles }
    }

    // step_with_trap_handler removed in favor of step_with_hle_handler.

    // ========== Stack Operations ==========

    /// Push a word onto the stack.
    #[inline]
    pub fn push_16<B: AddressBus>(&mut self, bus: &mut B, value: u16) {
        self.dar[15] = self.dar[15].wrapping_sub(2);
        self.write_16(bus, self.dar[15], value);
    }

    /// Push a long onto the stack.
    #[inline]
    pub fn push_32<B: AddressBus>(&mut self, bus: &mut B, value: u32) {
        self.dar[15] = self.dar[15].wrapping_sub(4);
        self.write_32(bus, self.dar[15], value);
    }

    /// Pull a word from the stack.
    #[inline]
    pub fn pull_16<B: AddressBus>(&mut self, bus: &mut B) -> u16 {
        let value = self.read_16(bus, self.dar[15]);
        self.dar[15] = self.dar[15].wrapping_add(2);
        value
    }

    /// Pull a long from the stack.
    #[inline]
    pub fn pull_32<B: AddressBus>(&mut self, bus: &mut B) -> u32 {
        let value = self.read_32(bus, self.dar[15]);
        self.dar[15] = self.dar[15].wrapping_add(4);
        value
    }

    // ========== Program Flow ==========

    /// Jump to a new PC.
    #[inline]
    pub fn jump(&mut self, new_pc: u32) {
        self.pc = self.address(new_pc);
    }

    /// Jump to an exception vector.
    pub fn jump_vector<B: AddressBus>(&mut self, bus: &mut B, vector: u32) {
        let addr = (vector << 2).wrapping_add(self.vbr);
        self.pc = self.read_32(bus, addr);
    }

    /// Branch with 8-bit displacement.
    #[inline]
    pub fn branch_8(&mut self, offset: u8) {
        self.pc = self.pc.wrapping_add(offset as i8 as i32 as u32);
    }

    /// Branch with 16-bit displacement.
    #[inline]
    pub fn branch_16(&mut self, offset: u16) {
        self.pc = self.pc.wrapping_add(offset as i16 as i32 as u32);
    }

    /// Branch with 32-bit displacement.
    #[inline]
    pub fn branch_32(&mut self, offset: u32) {
        self.pc = self.pc.wrapping_add(offset);
    }

    // ========== Interrupt Handling ==========

    /// Check and service pending interrupts.
    fn check_and_service_interrupts<B: AddressBus>(&mut self, bus: &mut B) {
        // NMI (level 7) always triggers, others compare to mask
        let mask_level = (self.int_mask >> 8) & 7;
        let int_level = self.int_level & 7;

        if int_level == 7 || int_level > mask_level {
            self.service_interrupt(bus, int_level as u8);
            // Clear pending interrupt level - bus.interrupt_acknowledge was called in
            // service_interrupt, so the device has had a chance to update its state.
            // We clear cpu.int_level here; the test harness will re-poll and set it
            // again in the next step if another interrupt is pending.
            self.int_level = 0;
        }
    }

    /// Service an interrupt.
    fn service_interrupt<B: AddressBus>(&mut self, bus: &mut B, level: u8) {
        // Get vector from interrupt acknowledge
        let vector = bus.interrupt_acknowledge(level);
        let vector = if vector == 0xFFFFFFFF {
            // Autovector
            24 + level as u32
        } else {
            vector & 0xFF
        };

        // Match Musashi `m68ki_exception_interrupt`:
        // - save old SR
        // - clear trace, enter supervisor (but do not modify M)
        // - set interrupt mask
        // - stack format-0 frame; if M=1 and 68020+ also stack a format-1 throwaway frame on ISP
        let old_sr = self.get_sr();
        self.t1_flag = 0;
        self.t0_flag = 0;
        self.set_s_flag(SFLAG_SET);
        self.int_mask = ((level as u32) & 7) << 8;

        let stacked_pc = self.pc;
        let vec_word = (vector as u16) << 2;

        if self.cpu_type == super::types::CpuType::M68000 {
            // 68000: 3-word frame (PC, SR)
            self.push_32(bus, stacked_pc);
            self.push_16(bus, old_sr);
        } else {
            // 68010+: format 0 frame: (vector<<2), PC, SR (vector word ends up at +6)
            self.push_16(bus, vec_word);
            self.push_32(bus, stacked_pc);
            self.push_16(bus, old_sr);
        }

        // If we were in supervisor master state, generate a throwaway frame on ISP.
        // (Musashi: clear M, force S in the stacked SR, then stack format-1 frame.)
        let is_ec020_plus = matches!(
            self.cpu_type,
            super::types::CpuType::M68EC020
                | super::types::CpuType::M68020
                | super::types::CpuType::M68EC030
                | super::types::CpuType::M68030
                | super::types::CpuType::M68EC040
                | super::types::CpuType::M68LC040
                | super::types::CpuType::M68040
        );
        if is_ec020_plus && self.m_flag != 0 {
            self.set_sm_flag(SFLAG_SET); // clear M => ISP active
            let sr2 = old_sr | 0x2000;
            self.push_16(bus, 0x1000 | (vec_word & 0x0FFF));
            self.push_32(bus, stacked_pc);
            self.push_16(bus, sr2);
        }

        // Jump to vector
        self.jump_vector(bus, vector);

        // Clear stopped state
        self.stopped = 0;

        // Use exception cycles
        self.cycles_remaining -= 44; // Approximate interrupt cycles
    }

    /// Halt the CPU.
    pub fn halt(&mut self) {
        self.stopped |= STOP_LEVEL_HALT;
    }

    /// Stop the CPU (STOP instruction).
    pub fn stop(&mut self, new_sr: u16) {
        self.set_sr(new_sr);
        self.stopped |= STOP_LEVEL_STOP;
    }
}

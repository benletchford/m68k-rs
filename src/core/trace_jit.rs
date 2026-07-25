//! Trace execution for hot simple-op loops.
//!
//! Native targets lower hot traces to Cranelift machine code. WebAssembly targets keep the same
//! trace detection and validation path, but execute the trace through a compact Rust micro-op loop.

#[cfg(not(target_family = "wasm"))]
use super::cpu::{CFLAG_SET, VFLAG_SET};
use super::cpu::{CpuCore, NFLAG_SET};
use super::execute::RUN_MODE_BERR_AERR_RESET;
use super::mem_ops::{BitSource, DecodedMemOp, FastEa};
use super::memory::AddressBus;
use super::op_cache::{BinaryOp, BitOp, CachedRunResult, DecodedSimpleOp, is_pre_68020};
use super::types::{CpuType, Size};
#[cfg(not(target_family = "wasm"))]
use cranelift_codegen::Context;
#[cfg(not(target_family = "wasm"))]
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, Function, InstBuilder, MemFlags, Type, UserFuncName, Value,
    condcodes::IntCC, types,
};
#[cfg(not(target_family = "wasm"))]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(not(target_family = "wasm"))]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(not(target_family = "wasm"))]
use cranelift_module::{Linkage, Module, default_libcall_names};
use std::cell::{Cell, RefCell};
use std::fmt;
#[cfg(not(target_family = "wasm"))]
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicBool, Ordering};

const TRACE_CACHE_SIZE: usize = 4096;
pub(crate) const TRACE_MAX_OPS: usize = 128;
pub(crate) const TRACE_MIN_OPS: usize = 3;
const TRACE_MIN_SELF_LOOP_OPS: usize = 2;
/// Indirect calls pay trace validation plus a native/Rust boundary on every
/// visit. In same-binary paired 100-million-instruction runs, six-op register
/// traces were only 0.6% faster at the median and regressed in one of five
/// trials. Every seven-op trial won: at least 7.2% across register,
/// memory-ALU, and memory-heavy mixes.
const TRACE_MIN_INDIRECT_JSR_OPS: usize = 7;
const TRACE_HOT_THRESHOLD: u8 = 2;
const TRACE_ADAPT_WINDOW: u8 = 64;
const TRACE_ADAPT_MISMATCHES: u8 = 48;
const TRACE_MAX_ADAPTIVE_RERECORDS: u8 = 1;

/// Sentinel for `CpuCore::trace_record_skip` / `trace_probe_skip`: no PC.
pub(crate) const TRACE_PC_NONE: u32 = u32::MAX;

#[cfg(not(target_family = "wasm"))]
/// Original one-pass compiled trace entry point.
type TraceOnceFn = unsafe extern "C" fn(*mut CpuCore) -> u64;

#[cfg(not(target_family = "wasm"))]
/// Counted self-loop entry point. Keeping repeated guest iterations inside
/// generated code avoids an ABI round trip for every tiny loop.
type TraceLoopFn = unsafe extern "C" fn(*mut CpuCore, u32) -> u64;

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
enum NativeTraceFn {
    Once(TraceOnceFn),
    Loop(TraceLoopFn),
}

static TRACE_JIT_HAS_CANDIDATES: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TRACE_JIT: RefCell<TraceJit> = RefCell::new(TraceJit::new());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitDirectReg {
    Data(u8),
    Addr(u8),
}

/// Effective-address forms allowed in memory trace ops. Extension words are
/// captured in the trace so indexed/displacement operands remain cheap to
/// validate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitEa {
    Data(u8),
    Addr(u8),
    /// (An)
    Ind(u8),
    /// (An)+
    PostInc(u8),
    /// -(An)
    PreDec(u8),
    /// (d16,An), with the extension word captured in the trace.
    Disp(u8, i16),
    /// Brief (d8,An,Xn), decoded once when the trace is recorded.
    Index {
        base: u8,
        index: JitDirectReg,
        index_long: bool,
        scale: u8,
        displacement: i8,
    },
}

impl JitEa {
    fn is_mem(self) -> bool {
        matches!(
            self,
            Self::Ind(_)
                | Self::PostInc(_)
                | Self::PreDec(_)
                | Self::Disp(_, _)
                | Self::Index { .. }
        )
    }
}

/// Post-inc/pre-dec step: byte accesses through A7 keep the stack pointer
/// even (matches `mem_ops::ea_step`).
fn jit_ea_step(size: Size, reg: u8) -> u32 {
    if size == Size::Byte && reg == 7 {
        2
    } else {
        size.bytes()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitUnaryOp {
    Clr,
    Neg,
    Negx,
    Not,
    Tst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitBinaryOp {
    Add,
    Sub,
    And,
    Or,
    Eor,
    Cmp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitAddrOp {
    Adda,
    Suba,
    Cmpa,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitBitOp {
    Test,
    Change,
    Clear,
    Set,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitBitSource {
    Reg(u8),
    Imm(u8),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitTraceOp {
    Nop,
    MoveReg {
        src: JitDirectReg,
        dst: JitDirectReg,
        size: Size,
    },
    Moveq {
        reg: u8,
        data: u32,
    },
    UnaryDataReg {
        op: JitUnaryOp,
        reg: u8,
        size: Size,
    },
    AddqSubqReg {
        reg: u8,
        data: u32,
        size: Size,
        is_sub: bool,
    },
    AddqSubqAddr {
        reg: u8,
        data: u32,
        is_sub: bool,
    },
    BinaryDataReg {
        op: JitBinaryOp,
        src: JitDirectReg,
        dst: u8,
        size: Size,
        cycles: i32,
    },
    AddrDataReg {
        op: JitAddrOp,
        src: JitDirectReg,
        dst: u8,
        size: Size,
    },
    AddSubxReg {
        src: u8,
        dst: u8,
        size: Size,
        is_sub: bool,
    },
    BitReg {
        op: JitBitOp,
        bit_reg: u8,
        dst: u8,
    },
    Exg {
        opcode: u16,
    },
    Ext {
        reg: u8,
        size: Size,
    },
    Extb {
        reg: u8,
    },
    SccDataReg {
        condition: u8,
        reg: u8,
    },
    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    ShiftReg {
        reg: u8,
        size: Size,
        count_or_reg: u8,
        count_is_register: bool,
        direction: u8,
        op: u8,
    },
    Swap {
        reg: u8,
    },
    Branch {
        condition: u8,
        displacement: i32,
        length: u8,
        /// Recorded direction for an interior conditional branch. `None`
        /// means this branch ends the trace; `Some` emits a guarded side
        /// exit and continues along the recorded path on a match.
        expected_taken: Option<bool>,
    },
    Dbcc {
        condition: u8,
        reg: u8,
        displacement: i16,
    },
    /// Terminal `JSR (An)`. The target is dynamic, and the return address
    /// store is checked against the active fastmem window before any CPU
    /// state is committed.
    IndirectJsr {
        reg: u8,
    },
    /// MOVE/MOVEA with at least one register-indirect operand, executed
    /// against the fastmem window (`dst == Addr` is MOVEA). Traces
    /// containing this op only run while a window is active; every access
    /// is bounds/alignment/self-modification checked and bails to the
    /// interpreter mid-trace with nothing from this op committed.
    MoveMem {
        size: Size,
        src: JitEa,
        dst: JitEa,
    },
    /// MOVEM.W (An)+,<data-register mask>. Keeping this deliberately narrow
    /// avoids the architectural corner cases of address registers in a
    /// postincrement MOVEM list.
    MovemWordPostInc {
        base: u8,
        data_mask: u8,
        cycles: i32,
    },
    /// Read-only ALU operation from fast memory to a data register. The
    /// decoder admits measured CMP/ADD/SUB `(An)`/`d16(An)` sources.
    AluMemToReg {
        op: JitBinaryOp,
        size: Size,
        src: JitEa,
        dst: u8,
    },
    /// ADD.W/L Dn,(An)+ store/accumulate operations.
    AddRegToPostInc {
        size: Size,
        src: u8,
        dst: u8,
    },
    /// Displacement-memory forms that require extension words, represented
    /// explicitly rather than through the register-only trace operations.
    AnDispUnary {
        op: JitUnaryOp,
        size: Size,
        reg: u8,
        displacement: i16,
    },
    AnDispAddqSubq {
        data: u32,
        size: Size,
        reg: u8,
        displacement: i16,
        is_sub: bool,
    },
    AnDispBit {
        op: JitBitOp,
        bit: JitBitSource,
        reg: u8,
        displacement: i16,
    },
}

#[derive(Debug, Clone, Copy)]
struct TraceBuildOp {
    opcode: u16,
    extension: Option<u16>,
    extension2: Option<u16>,
    pc: u32,
    op: JitTraceOp,
}

impl TraceBuildOp {
    fn length(self) -> u8 {
        2 + 2 * u8::from(self.extension.is_some()) + 2 * u8::from(self.extension2.is_some())
    }
}

struct CompiledTrace {
    pc: u32,
    cpu_type: CpuType,
    ops: Vec<TraceBuildOp>,
    /// The exact instruction bytes the trace was compiled from (ops are
    /// in execution order. Contiguous traces can validate this with one
    /// compare; recorded multi-block paths validate each instruction.
    code: Vec<u8>,
    contiguous_code: bool,
    max_cycles: i32,
    /// The final branch's taken-target is the trace head, so the trace is
    /// a whole loop iteration and can be re-run (budget permitting)
    /// without re-validating: trace stores that would touch code bail out
    /// before committing, and nothing observable happens between
    /// iterations.
    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    self_loop: bool,
    /// The native body was generated as a counted loop. Short read/write
    /// MoveMem loops deliberately retain the original one-pass body: the
    /// extra loop-carried state costs more than the saved call boundary.
    #[cfg(not(target_family = "wasm"))]
    native_loop: bool,
    /// Contains memory ops: only executable while a fastmem window is active
    /// (i.e. inside `run_batch`).
    needs_window: bool,
    /// Address-masked range of the trace's code bytes; trace stores into
    /// this range bail so self-modification is observed like the
    /// interpreter would. Baked into the compiled function on native
    /// targets; read at execution time by the portable path.
    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    code_start: u32,
    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    code_end: u32,
    /// A recorded interior branch is a path prediction eligible for adaptive
    /// rerecording. Cleared after the one allowed rerecord so completed traces
    /// stay off the accounting path.
    adaptive_branch: bool,
    adaptive_calls: Cell<u32>,
    adaptive_guard_exits: Cell<u32>,
    adaptive_rerecords: u8,
    #[cfg(not(target_family = "wasm"))]
    func: NativeTraceFn,
}

impl CompiledTrace {
    #[cfg(all(not(target_family = "wasm"), test))]
    unsafe fn call_native(&self, cpu: *mut CpuCore, max_iters: u32) -> u64 {
        match self.func {
            NativeTraceFn::Once(func) => unsafe { func(cpu) },
            NativeTraceFn::Loop(func) => unsafe { func(cpu, max_iters) },
        }
    }

    fn is_guarded_branch_exit(&self, cpu: &CpuCore, ops_done: u32) -> bool {
        let Some(index) = ops_done.checked_sub(1).map(|index| index as usize) else {
            return false;
        };
        self.ops.get(index).is_some_and(|op| {
            matches!(
                op.op,
                JitTraceOp::Branch {
                    expected_taken: Some(_),
                    ..
                }
            ) && cpu.ppc == op.pc
        })
    }
}

struct TraceRecording {
    start_pc: u32,
    cpu_type: CpuType,
    ops: Vec<TraceBuildOp>,
    adaptive_rerecords: u8,
}

enum TraceSlot {
    Empty,
    Counting {
        pc: u32,
        cpu_type: CpuType,
        hits: u8,
        adaptive_rerecords: u8,
    },
    Rejected {
        pc: u32,
        cpu_type: CpuType,
    },
    Compiled(CompiledTrace),
}

pub(crate) struct TraceJit {
    #[cfg(not(target_family = "wasm"))]
    module: Option<JITModule>,
    #[cfg(not(target_family = "wasm"))]
    func_ctx: FunctionBuilderContext,
    #[cfg(not(target_family = "wasm"))]
    next_func: u32,
    slots: Vec<TraceSlot>,
    recording: Option<TraceRecording>,
}

impl fmt::Debug for TraceJit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("TraceJit");
        #[cfg(not(target_family = "wasm"))]
        {
            debug.field("native_enabled", &self.module.is_some());
            debug.field("next_func", &self.next_func);
        }
        #[cfg(target_family = "wasm")]
        {
            debug.field("native_enabled", &false);
        }
        debug.finish_non_exhaustive()
    }
}

impl TraceJit {
    fn new() -> Self {
        #[cfg(not(target_family = "wasm"))]
        let module = JITBuilder::new(default_libcall_names())
            .ok()
            .map(JITModule::new);
        Self {
            #[cfg(not(target_family = "wasm"))]
            module,
            #[cfg(not(target_family = "wasm"))]
            func_ctx: FunctionBuilderContext::new(),
            #[cfg(not(target_family = "wasm"))]
            next_func: 0,
            slots: (0..TRACE_CACHE_SIZE).map(|_| TraceSlot::Empty).collect(),
            recording: None,
        }
    }

    /// Attempt to execute a compiled trace at the current PC.
    ///
    /// On `CachedRunResult::Ran`, the returned count is the number of
    /// guest instructions the trace retired. The count is 0 for
    /// `Fault`/`Miss`.
    ///
    /// A self-looping trace (one whose closing branch targets its own
    /// head) may run many iterations per call: up to `instr_budget`
    /// retired instructions, always within the CPU's remaining cycle
    /// budget, and only one iteration when `single_iter` is set (callers
    /// that must observe the PC between iterations, e.g. watchpoints).
    fn try_execute<B: AddressBus>(
        &mut self,
        cpu: &mut CpuCore,
        bus: &mut B,
        cpu_type: CpuType,
        instr_budget: u32,
        single_iter: bool,
        watch_pcs: &[u32],
    ) -> Option<(CachedRunResult, u32)> {
        #[cfg(not(target_family = "wasm"))]
        self.module.as_ref()?;

        if cpu.has_pmmu && cpu.pmmu_enabled || cpu.cycles_remaining <= 0 {
            return None;
        }
        if cpu.trace_recording || self.recording.is_some() {
            // A recorder is already following an executed path on this
            // thread. Nested backward edges and interleaved CPU instances
            // stay in the interpreter until that path closes.
            return None;
        }

        let pc = cpu.pc;
        let idx = trace_cache_index(pc);

        if let TraceSlot::Compiled(trace) = &self.slots[idx]
            && trace.pc == pc
            && trace.cpu_type == cpu_type
        {
            // run_batch observes watched PCs between guest instructions.
            // If a recorded region reaches one internally, leave it to the
            // interpreter so the watch fires before that instruction. The
            // entry PC is intentionally excluded: run_batch does not check
            // watches on entry, and self-loop entry watches are handled by
            // `single_iter` after one complete iteration.
            if watch_pcs.iter().any(|&watched| {
                let masked = cpu.address(watched);
                masked >= trace.code_start
                    && masked < trace.code_end
                    && trace.ops.iter().skip(1).any(|op| op.pc == watched)
            }) {
                return None;
            }
            if trace.needs_window && cpu.fm_len == 0 {
                // Memory traces only run against a fastmem window (i.e.
                // inside run_batch). Stop this cycle-budgeted caller from
                // probing the target again; run_batch clears the filter on
                // entry so the trace still runs there.
                push_probe_skip(cpu, pc);
                return None;
            }
            if cpu.cycles_remaining < trace.max_cycles {
                return None;
            }

            // Fast validation: when a fastmem window covers the whole
            // trace, one slice compare against the live instruction bytes
            // replaces per-op bus reads. (SMC through the window is still
            // caught: we compare the actual RAM.)
            let mut validated = false;
            if trace.contiguous_code && cpu.fm_len != 0 {
                let n = trace.code.len() as u32;
                let off = cpu.address(pc).wrapping_sub(cpu.fm_base);
                if n <= cpu.fm_len && off <= cpu.fm_len - n {
                    let live = unsafe {
                        std::slice::from_raw_parts(
                            (cpu.fm_ptr as *const u8).add(off as usize),
                            n as usize,
                        )
                    };
                    if live == trace.code.as_slice() {
                        validated = true;
                    }
                }
            }

            let mut miss = None;
            if !validated {
                for (index, op) in trace.ops.iter().enumerate() {
                    let addr = cpu.address(op.pc);
                    match bus.try_read_word(addr) {
                        Ok(opcode) if opcode == op.opcode => {}
                        Ok(opcode) => {
                            miss = Some((index, op.pc, opcode));
                            break;
                        }
                        Err(_) => return None,
                    }

                    if let Some(expected) = op.extension {
                        let addr = cpu.address(op.pc.wrapping_add(2));
                        match bus.try_read_word(addr) {
                            Ok(extension) if extension == expected => {}
                            Ok(_) => {
                                miss = Some((index, op.pc, op.opcode));
                                break;
                            }
                            Err(_) => return None,
                        }
                    }
                    if let Some(expected) = op.extension2 {
                        let addr = cpu.address(op.pc.wrapping_add(4));
                        match bus.try_read_word(addr) {
                            Ok(extension) if extension == expected => {}
                            Ok(_) => {
                                miss = Some((index, op.pc, op.opcode));
                                break;
                            }
                            Err(_) => return None,
                        }
                    }
                }
            }

            if let Some((index, ppc, opcode)) = miss {
                self.slots[idx] = TraceSlot::Empty;
                // The trace at this target is gone; re-arm the per-CPU
                // filters so the loop can be re-recorded and re-probed.
                cpu.trace_record_skip = [TRACE_PC_NONE; 4];
                cpu.trace_probe_skip = [TRACE_PC_NONE; 4];
                if index > 0 {
                    // Instruction memory changed mid-trace. Nothing has
                    // executed yet (validation precedes the trace call),
                    // so consuming the changed opcode here would silently
                    // skip the still-valid ops before it. Leave PC at the
                    // trace head and let the caller re-decode from there.
                    return None;
                }
                cpu.ppc = ppc;
                cpu.ir = opcode as u32;
                cpu.pc = cpu.ppc.wrapping_add(2);
                return Some((CachedRunResult::Miss(opcode), 0));
            }

            let ops_len = trace.ops.len() as u32;
            if instr_budget < ops_len {
                return None;
            }
            // A generated loop clearly amortizes the ABI boundary for
            // profiled mixed 3+-op and read-only loops. A
            // two-op read/write MoveMem loop is already dominated by its two
            // checked guest accesses; carrying native loop state made that
            // case 3.5% slower at the median, so retain the old one-pass
            // function and repeat it in this already-validated Rust entry.
            #[cfg(not(target_family = "wasm"))]
            let batch_self_loop = trace.native_loop;
            // How many whole iterations fit in both budgets. The guards
            // above ensure at least one; the instruction budget is the
            // caller's (u32::MAX on the cycle-budgeted paths).
            let max_iters = if single_iter || !trace.self_loop {
                1
            } else {
                let by_instrs = (instr_budget / ops_len).max(1);
                let by_cycles = (cpu.cycles_remaining / trace.max_cycles).max(1) as u32;
                by_instrs.min(by_cycles)
            };
            let mut cycles_total = 0i64;
            let mut retired = 0u32;
            let mut full_iters = 0u32;
            #[cfg(not(target_family = "wasm"))]
            let (guarded_branch_exit, partial_call_this_entry) = if batch_self_loop {
                let NativeTraceFn::Loop(func) = trace.func else {
                    unreachable!("a batched trace must have a counted entry point")
                };
                loop {
                    let call_max_iters = max_iters - full_iters;
                    let packed = unsafe { func(cpu as *mut CpuCore, call_max_iters) };
                    cycles_total += (packed as u32) as i64;
                    let ops_done = (packed >> 32) as u32;
                    let completed = ops_done / ops_len;
                    let remainder = ops_done % ops_len;
                    let partial_call = completed < call_max_iters;
                    // A side exit at the last op has a zero remainder after
                    // one or more complete numeric trace lengths. PC/ppc
                    // distinguish it from an op-zero memory bail.
                    let exit_ops = if partial_call && remainder == 0 && ops_done != 0 {
                        ops_len
                    } else {
                        remainder
                    };
                    let guarded_branch_exit =
                        partial_call && trace.is_guarded_branch_exit(cpu, exit_ops);
                    #[cfg(feature = "trace-profile")]
                    super::trace_profile::note_native_call(pc, cpu_type, ops_done);
                    #[cfg(feature = "trace-profile")]
                    if guarded_branch_exit {
                        super::trace_profile::note_guarded_branch_exit(pc, cpu_type);
                    }
                    retired += ops_done;
                    full_iters += completed;
                    if partial_call {
                        break (guarded_branch_exit, true);
                    }
                    if full_iters >= max_iters || cpu.pc != pc {
                        break (false, false);
                    }
                }
            } else {
                // This is intentionally the original direct-call driver.
                // Tiny memory loops can execute this path once per two guest
                // instructions, so even generalized result accounting is
                // measurable here.
                let NativeTraceFn::Once(func) = trace.func else {
                    unreachable!("a one-pass trace must have a linear entry point")
                };
                loop {
                    let packed = unsafe { func(cpu as *mut CpuCore) };
                    cycles_total += (packed as u32) as i64;
                    let ops_done = (packed >> 32) as u32;
                    let partial_call = ops_done < ops_len;
                    let guarded_branch_exit =
                        partial_call && trace.is_guarded_branch_exit(cpu, ops_done);
                    #[cfg(feature = "trace-profile")]
                    super::trace_profile::note_native_call(pc, cpu_type, ops_done);
                    #[cfg(feature = "trace-profile")]
                    if guarded_branch_exit {
                        super::trace_profile::note_guarded_branch_exit(pc, cpu_type);
                    }
                    retired += ops_done;
                    if partial_call {
                        break (guarded_branch_exit, true);
                    }
                    full_iters += 1;
                    if full_iters >= max_iters || cpu.pc != pc {
                        break (false, false);
                    }
                }
            };
            #[cfg(target_family = "wasm")]
            let (guarded_branch_exit, partial_call_this_entry) = loop {
                let packed =
                    execute_portable_trace(cpu, &trace.ops, trace.code_start, trace.code_end);
                cycles_total += (packed as u32) as i64;
                let ops_done = (packed >> 32) as u32;
                let partial_call = ops_done < ops_len;
                let guarded_branch_exit =
                    partial_call && trace.is_guarded_branch_exit(cpu, ops_done);
                #[cfg(feature = "trace-profile")]
                super::trace_profile::note_native_call(pc, cpu_type, ops_done);
                #[cfg(feature = "trace-profile")]
                if guarded_branch_exit {
                    super::trace_profile::note_guarded_branch_exit(pc, cpu_type);
                }
                retired += ops_done;
                if partial_call {
                    break (guarded_branch_exit, true);
                }
                full_iters += 1;
                if full_iters >= max_iters || cpu.pc != pc {
                    break (false, false);
                }
            };
            let mut rerecord_dominant_path = false;
            if trace.adaptive_branch {
                // Account once per Rust entry, not once per guest operation.
                // Non-self-loop traces normally make one native call per
                // entry, while self-loops may make many; both successful
                // predictions and guarded exits belong in the denominator.
                let calls = trace
                    .adaptive_calls
                    .get()
                    .saturating_add(full_iters.saturating_add(u32::from(partial_call_this_entry)));
                let exits = trace
                    .adaptive_guard_exits
                    .get()
                    .saturating_add(u32::from(guarded_branch_exit));
                trace.adaptive_calls.set(calls);
                trace.adaptive_guard_exits.set(exits);
                if calls >= u32::from(TRACE_ADAPT_WINDOW) {
                    rerecord_dominant_path = exits >= u32::from(TRACE_ADAPT_MISMATCHES)
                        && u64::from(exits) * u64::from(TRACE_ADAPT_WINDOW)
                            >= u64::from(calls) * u64::from(TRACE_ADAPT_MISMATCHES);
                    trace.adaptive_calls.set(0);
                    trace.adaptive_guard_exits.set(0);
                }
            }
            cpu.cycles_remaining -= i32::try_from(cycles_total).unwrap_or(i32::MAX);
            if retired == 0 {
                // The very first op bailed: nothing executed. Fall back to
                // the interpreter so the offending instruction makes
                // progress through full dispatch.
                return None;
            }
            if rerecord_dominant_path {
                let adaptive_rerecords = trace.adaptive_rerecords.saturating_add(1);
                self.slots[idx] = TraceSlot::Counting {
                    pc,
                    cpu_type,
                    hits: 0,
                    adaptive_rerecords,
                };
                cpu.trace_record_skip = [TRACE_PC_NONE; 4];
                cpu.trace_probe_skip = [TRACE_PC_NONE; 4];
                #[cfg(feature = "trace-profile")]
                super::trace_profile::note_adaptive_rerecord(pc, cpu_type);
            }
            return Some((CachedRunResult::Ran, retired));
        }

        match &mut self.slots[idx] {
            TraceSlot::Counting {
                pc: counted_pc,
                cpu_type: counted_type,
                hits,
                adaptive_rerecords,
            } if *counted_pc == pc && *counted_type == cpu_type => {
                *hits = hits.saturating_add(1);
                if *hits < TRACE_HOT_THRESHOLD {
                    return None;
                }
                self.recording = Some(TraceRecording {
                    start_pc: pc,
                    cpu_type,
                    ops: Vec::with_capacity(TRACE_MAX_OPS),
                    adaptive_rerecords: *adaptive_rerecords,
                });
                #[cfg(feature = "trace-profile")]
                super::trace_profile::note_recording(pc, cpu_type);
                cpu.trace_recording = true;
                None
            }
            TraceSlot::Rejected {
                pc: rejected_pc,
                cpu_type: rejected_type,
            } if *rejected_pc == pc && *rejected_type == cpu_type => {
                // Known-uncompilable target: tell the loop to stop probing
                // it (note_backward_branch consults this filter).
                push_probe_skip(cpu, pc);
                None
            }
            _ => None,
        }
    }

    fn record_trace_target(&mut self, pc: u32, cpu_type: CpuType) {
        #[cfg(not(target_family = "wasm"))]
        if self.module.is_none() {
            return;
        }

        let idx = trace_cache_index(pc);
        match &self.slots[idx] {
            TraceSlot::Compiled(CompiledTrace {
                pc: compiled_pc,
                cpu_type: compiled_type,
                ..
            }) if *compiled_pc == pc && *compiled_type == cpu_type => {}
            TraceSlot::Counting {
                pc: counted_pc,
                cpu_type: counted_type,
                ..
            } if *counted_pc == pc && *counted_type == cpu_type => {}
            TraceSlot::Rejected {
                pc: rejected_pc,
                cpu_type: rejected_type,
            } if *rejected_pc == pc && *rejected_type == cpu_type => {}
            _ => {
                self.slots[idx] = TraceSlot::Counting {
                    pc,
                    cpu_type,
                    hits: 1,
                    adaptive_rerecords: 0,
                };
                TRACE_JIT_HAS_CANDIDATES.store(true, Ordering::Relaxed);
            }
        }
    }

    #[cfg(feature = "trace-profile")]
    fn is_rejected(&self, pc: u32, cpu_type: CpuType) -> bool {
        matches!(
            &self.slots[trace_cache_index(pc)],
            TraceSlot::Rejected {
                pc: rejected_pc,
                cpu_type: rejected_type,
            } if *rejected_pc == pc && *rejected_type == cpu_type
        )
    }

    fn reject_recording(&mut self, cpu: &mut CpuCore) {
        if let Some(recording) = self.recording.take() {
            let idx = trace_cache_index(recording.start_pc);
            self.slots[idx] = TraceSlot::Rejected {
                pc: recording.start_pc,
                cpu_type: recording.cpu_type,
            };
            push_probe_skip(cpu, recording.start_pc);
        }
        cpu.trace_recording = false;
    }

    fn finish_recording(&mut self, cpu: &mut CpuCore, exit_pc: u32) {
        let Some(mut recording) = self.recording.take() else {
            cpu.trace_recording = false;
            return;
        };
        cpu.trace_recording = false;

        // An interior recorded branch becomes the region's ordinary final
        // branch when recording stops at its destination.
        if let Some(last) = recording.ops.last_mut()
            && let JitTraceOp::Branch { expected_taken, .. } = &mut last.op
        {
            *expected_taken = None;
        }

        let idx = trace_cache_index(recording.start_pc);
        let start_pc = recording.start_pc;
        let cpu_type = recording.cpu_type;
        let adaptive_rerecords = recording.adaptive_rerecords;
        #[cfg(feature = "trace-profile")]
        let recorded_ops = recording.ops.len();
        self.slots[idx] =
            match self.compile_decoded_ops(cpu, start_pc, cpu_type, recording.ops, Some(exit_pc)) {
                Some(mut trace) => {
                    trace.adaptive_rerecords = adaptive_rerecords;
                    if adaptive_rerecords >= TRACE_MAX_ADAPTIVE_RERECORDS {
                        trace.adaptive_branch = false;
                    }
                    TraceSlot::Compiled(trace)
                }
                None => {
                    push_probe_skip(cpu, start_pc);
                    TraceSlot::Rejected {
                        pc: start_pc,
                        cpu_type,
                    }
                }
            };
        #[cfg(feature = "trace-profile")]
        if matches!(self.slots[idx], TraceSlot::Compiled(_)) {
            super::trace_profile::note_compiled(start_pc, cpu_type, recorded_ops);
        }
    }

    fn record_executed<B: AddressBus>(
        &mut self,
        cpu: &mut CpuCore,
        bus: &mut B,
        executed_pc: u32,
        next_pc: u32,
    ) {
        let Some(recording) = self.recording.as_ref() else {
            cpu.trace_recording = false;
            return;
        };
        let start_pc = recording.start_pc;
        let cpu_type = recording.cpu_type;
        if cpu_type != cpu.cpu_type {
            self.reject_recording(cpu);
            return;
        }

        let Some(mut op) = decode_trace_op(cpu, bus, executed_pc, cpu_type) else {
            #[cfg(feature = "trace-profile")]
            super::trace_profile::note_blocker(
                start_pc,
                cpu_type,
                recording.ops.len(),
                executed_pc,
                cpu.ir as u16,
            );
            self.finish_recording(cpu, executed_pc);
            return;
        };
        let op_len = op.length();
        let taken_target = op.op.taken_target(op.pc);

        match &mut op.op {
            JitTraceOp::Branch { expected_taken, .. } => {
                if next_pc != start_pc {
                    let taken = taken_target == Some(next_pc);
                    *expected_taken = Some(taken);
                }
            }
            // DBcc remains a natural region boundary for now. Its data-
            // dependent counter update is already compiled efficiently.
            JitTraceOp::Dbcc { .. } => {
                self.recording.as_mut().unwrap().ops.push(op);
                self.finish_recording(cpu, next_pc);
                return;
            }
            JitTraceOp::IndirectJsr { .. } => {
                self.recording.as_mut().unwrap().ops.push(op);
                self.finish_recording(cpu, next_pc);
                return;
            }
            _ if next_pc != executed_pc.wrapping_add(op_len as u32) => {
                self.finish_recording(cpu, executed_pc);
                return;
            }
            _ => {}
        }

        let recording = self.recording.as_mut().unwrap();
        recording.ops.push(op);
        if next_pc == start_pc {
            self.finish_recording(cpu, next_pc);
            return;
        }

        let repeated = recording.ops.iter().any(|op| op.pc == next_pc);
        if recording.ops.len() >= TRACE_MAX_OPS || repeated {
            self.finish_recording(cpu, next_pc);
        }
    }

    fn compile_decoded_ops(
        &mut self,
        cpu: &CpuCore,
        start_pc: u32,
        cpu_type: CpuType,
        ops: Vec<TraceBuildOp>,
        recorded_exit_pc: Option<u32>,
    ) -> Option<CompiledTrace> {
        if !ops.last().is_some_and(|op| op.op.ends_trace()) {
            return None;
        }

        let self_loop = recorded_exit_pc == Some(start_pc)
            || ops
                .last()
                .is_some_and(|op| op.op.taken_target(op.pc) == Some(start_pc));
        let min_ops = if self_loop {
            TRACE_MIN_SELF_LOOP_OPS
        } else {
            TRACE_MIN_OPS
        };
        if ops.len() < min_ops {
            return None;
        }

        let max_cycles = ops.iter().map(|op| op.op.max_cycles()).sum();
        let contiguous_code = ops.first().is_some_and(|op| op.pc == start_pc)
            && ops
                .windows(2)
                .all(|pair| pair[0].pc.wrapping_add(pair[0].length() as u32) == pair[1].pc);

        let mut code = Vec::with_capacity(ops.len() * 4);
        for op in &ops {
            code.extend_from_slice(&op.opcode.to_be_bytes());
            if let Some(extension) = op.extension {
                code.extend_from_slice(&extension.to_be_bytes());
            }
            if let Some(extension) = op.extension2 {
                code.extend_from_slice(&extension.to_be_bytes());
            }
        }
        if contiguous_code {
            debug_assert_eq!(
                code.len() as u32,
                ops.last()
                    .map(|op| op.pc.wrapping_add(op.length() as u32))
                    .unwrap_or(start_pc)
                    .wrapping_sub(start_pc)
            );
        }

        let ends_in_indirect_jsr = ops
            .last()
            .is_some_and(|op| matches!(op.op, JitTraceOp::IndirectJsr { .. }));
        if ends_in_indirect_jsr && ops.len() < TRACE_MIN_INDIRECT_JSR_OPS {
            return None;
        }

        // Short checked memory ALU regions do not amortize trace validation
        // and the native/Rust boundary. Keep those on the decoded-memory path
        // unless the measured indirect-call length threshold above provides
        // enough independent work to cover the fixed cost.
        if !self_loop
            && !ends_in_indirect_jsr
            && ops
                .iter()
                .any(|op| matches!(op.op, JitTraceOp::AluMemToReg { .. }))
        {
            return None;
        }

        let needs_window = ops.iter().any(|op| {
            matches!(
                op.op,
                JitTraceOp::MoveMem { .. }
                    | JitTraceOp::MovemWordPostInc { .. }
                    | JitTraceOp::AluMemToReg { .. }
                    | JitTraceOp::AddRegToPostInc { .. }
                    | JitTraceOp::AnDispUnary { .. }
                    | JitTraceOp::AnDispAddqSubq { .. }
                    | JitTraceOp::AnDispBit { .. }
                    | JitTraceOp::IndirectJsr { .. }
            )
        });

        // Address-masked code range, used by the store-overlap (SMC) bail
        // checks. Reject the exotic case of a trace wrapping the address
        // space so the range stays a simple interval.
        let mut code_start = u32::MAX;
        let mut code_end = 0u32;
        for op in &ops {
            let start = cpu.address(op.pc);
            let end = start as u64 + op.length() as u64;
            if end > cpu.address_mask as u64 + 1 || end > u32::MAX as u64 {
                return None;
            }
            code_start = code_start.min(start);
            code_end = code_end.max(end as u32);
        }

        self.compile_ops(CompileParams {
            start_pc,
            cpu_type,
            ops: &ops,
            code,
            contiguous_code,
            max_cycles,
            self_loop,
            needs_window,
            code_start,
            code_end,
            aligned_only: cpu.is_pre_68020,
            address_mask: cpu.address_mask,
        })
    }

    #[cfg(not(target_family = "wasm"))]
    fn compile_ops(&mut self, params: CompileParams<'_>) -> Option<CompiledTrace> {
        let CompileParams {
            start_pc,
            cpu_type,
            ops,
            code,
            contiguous_code,
            max_cycles,
            self_loop,
            needs_window,
            code_start,
            code_end,
            aligned_only,
            address_mask,
        } = params;
        // Matched application and microbenchmark profiles show a clear win
        // for mixed 3+-op and read-only self-loops. A two-op read/write MoveMem
        // loop regresses when it carries counters around the generated loop,
        // so compile that shape with the original linear body instead of
        // trying to disable batching only at call time.
        let native_loop = self_loop
            && (ops.len() >= 3
                || !ops
                    .iter()
                    .any(|op| matches!(op.op, JitTraceOp::MoveMem { .. })));
        let module = self.module.as_mut()?;
        let ptr_ty = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));
        if native_loop {
            sig.params.push(AbiParam::new(types::I32));
        }
        sig.returns.push(AbiParam::new(types::I64));

        let name = format!("m68k_trace_{}", self.next_func);
        self.next_func = self.next_func.wrapping_add(1);
        let func_id = module.declare_function(&name, Linkage::Local, &sig).ok()?;

        let mut ctx = Context::new();
        ctx.func = Function::with_name_signature(UserFuncName::user(0, func_id.as_u32()), sig);

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut self.func_ctx);
            let block = builder.create_block();
            builder.switch_to_block(block);
            builder.append_block_params_for_function_params(block);
            let cpu_ptr = builder.block_params(block)[0];

            // Window state is constant for the whole `run_batch` call that
            // executes this trace; load it once.
            let mem_env = if needs_window {
                let fm_ptr = builder.ins().load(
                    ptr_ty,
                    MemFlags::trusted(),
                    cpu_ptr,
                    offset_of!(CpuCore, fm_ptr) as i32,
                );
                let fm_base = load_u32(&mut builder, cpu_ptr, offset_of!(CpuCore, fm_base));
                let fm_len = load_u32(&mut builder, cpu_ptr, offset_of!(CpuCore, fm_len));
                Some(MemEnv {
                    fm_ptr,
                    fm_ptr_ty: ptr_ty,
                    fm_base,
                    fm_len,
                    address_mask,
                    aligned_only,
                    code_start,
                    code_end,
                })
            } else {
                None
            };

            let zero = builder.ins().iconst(types::I32, 0);
            let max_iters = if native_loop {
                builder.block_params(block)[1]
            } else {
                builder.ins().iconst(types::I32, 1)
            };
            let trace_body = if native_loop {
                let trace_body = builder.create_block();
                builder.append_block_param(trace_body, types::I32); // accumulated cycles
                builder.append_block_param(trace_body, types::I32); // retired instructions
                builder.append_block_param(trace_body, types::I32); // iterations remaining
                let initial_args: [BlockArg; 3] = [zero.into(), zero.into(), max_iters.into()];
                builder.ins().jump(trace_body, &initial_args);
                builder.switch_to_block(trace_body);
                Some(trace_body)
            } else {
                None
            };
            let (cycles_before_iter, retired_before_iter, iterations_left) =
                if let Some(trace_body) = trace_body {
                    let params = builder.block_params(trace_body);
                    (params[0], params[1], params[2])
                } else {
                    (zero, zero, max_iters)
                };

            let mut bails: Vec<BailReq> = Vec::new();
            let mut cycles_value = cycles_before_iter;
            for (index, op) in ops.iter().enumerate() {
                let bail_at = BailAt {
                    ops_before: if native_loop {
                        RetiredBefore::Dynamic(
                            builder.ins().iadd_imm(retired_before_iter, index as i64),
                        )
                    } else {
                        RetiredBefore::Constant(index as u32)
                    },
                    cycles_before: cycles_value,
                };
                let op_cycles = match op.op {
                    JitTraceOp::Branch {
                        condition,
                        displacement,
                        length,
                        expected_taken: Some(expected_taken),
                    } => emit_guarded_branch(
                        &mut builder,
                        cpu_ptr,
                        op.pc,
                        condition,
                        displacement,
                        length,
                        expected_taken,
                        cycles_value,
                        if native_loop {
                            RetiredBefore::Dynamic(retired_before_iter)
                        } else {
                            RetiredBefore::Constant(0)
                        },
                        (index + 1) as u32,
                    ),
                    JitTraceOp::MoveMem { size, src, dst } => {
                        let env = mem_env.as_ref().expect("MoveMem implies a window env");
                        emit_move_mem(
                            &mut builder,
                            cpu_ptr,
                            MoveMemOp {
                                pc: op.pc,
                                size,
                                src,
                                dst,
                            },
                            env,
                            &mut bails,
                            bail_at,
                        )
                    }
                    JitTraceOp::MovemWordPostInc { .. } => {
                        let env = mem_env
                            .as_ref()
                            .expect("MovemWordPostInc implies a window env");
                        emit_movem_word_postinc(
                            &mut builder,
                            cpu_ptr,
                            *op,
                            env,
                            &mut bails,
                            bail_at,
                        )
                    }
                    JitTraceOp::AluMemToReg { .. } => {
                        let env = mem_env.as_ref().expect("AluMemToReg implies a window env");
                        emit_alu_mem_to_reg(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::AddRegToPostInc { .. } => {
                        let env = mem_env
                            .as_ref()
                            .expect("AddRegToPostInc implies a window env");
                        emit_add_reg_to_postinc(
                            &mut builder,
                            cpu_ptr,
                            *op,
                            env,
                            &mut bails,
                            bail_at,
                        )
                    }
                    JitTraceOp::IndirectJsr { reg } => {
                        let env = mem_env.as_ref().expect("IndirectJsr implies a window env");
                        emit_indirect_jsr(&mut builder, cpu_ptr, *op, reg, env, &mut bails, bail_at)
                    }
                    JitTraceOp::AnDispUnary { .. }
                    | JitTraceOp::AnDispAddqSubq { .. }
                    | JitTraceOp::AnDispBit { .. } => {
                        let env = mem_env.as_ref().expect("AnDisp implies a window env");
                        emit_an_disp_mem(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    _ => emit_jit_op(&mut builder, cpu_ptr, *op, aligned_only),
                };
                cycles_value = builder.ins().iadd(cycles_value, op_cycles);
            }

            if let Some(last) = ops.last() {
                store_u32(&mut builder, cpu_ptr, offset_of!(CpuCore, ppc), last.pc);
                store_u32(
                    &mut builder,
                    cpu_ptr,
                    offset_of!(CpuCore, ir),
                    last.opcode as u32,
                );
            }

            let retired_value = builder
                .ins()
                .iadd_imm(retired_before_iter, ops.len() as i64);
            if let Some(trace_body) = trace_body {
                let iterations_left = builder.ins().iadd_imm(iterations_left, -1);
                let more_iterations = builder.ins().icmp_imm(IntCC::NotEqual, iterations_left, 0);
                let live_pc = load_u32(&mut builder, cpu_ptr, offset_of!(CpuCore, pc));
                let at_head = builder
                    .ins()
                    .icmp_imm(IntCC::Equal, live_pc, i64::from(start_pc));
                let repeat = builder.ins().band(more_iterations, at_head);
                let done = builder.create_block();
                let repeat_args: [BlockArg; 3] = [
                    cycles_value.into(),
                    retired_value.into(),
                    iterations_left.into(),
                ];
                builder
                    .ins()
                    .brif(repeat, trace_body, &repeat_args, done, &[]);
                builder.switch_to_block(done);
            }

            let cycles64 = builder.ins().uextend(types::I64, cycles_value);
            let retired64 = if native_loop {
                let retired64 = builder.ins().uextend(types::I64, retired_value);
                builder.ins().ishl_imm(retired64, 32)
            } else {
                builder.ins().iconst(types::I64, (ops.len() as i64) << 32)
            };
            let packed = builder.ins().bor(cycles64, retired64);
            builder.ins().return_(&[packed]);

            // Bail exits: set PC to the un-executed op, return the ops and
            // accumulated cycles/instructions retired before it.
            for bail in bails {
                builder.switch_to_block(bail.block);
                store_u32(&mut builder, cpu_ptr, offset_of!(CpuCore, pc), bail.pc);
                let cycles64 = builder.ins().uextend(types::I64, bail.at.cycles_before);
                let retired = match bail.at.ops_before {
                    RetiredBefore::Constant(ops) => {
                        builder.ins().iconst(types::I64, i64::from(ops) << 32)
                    }
                    RetiredBefore::Dynamic(ops) => {
                        let retired = builder.ins().uextend(types::I64, ops);
                        builder.ins().ishl_imm(retired, 32)
                    }
                };
                let packed = builder.ins().bor(cycles64, retired);
                builder.ins().return_(&[packed]);
            }

            builder.seal_all_blocks();
            builder.finalize();
        }

        module.define_function(func_id, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(func_id);
        let func = if native_loop {
            NativeTraceFn::Loop(unsafe { transmute::<*const u8, TraceLoopFn>(ptr) })
        } else {
            NativeTraceFn::Once(unsafe { transmute::<*const u8, TraceOnceFn>(ptr) })
        };

        Some(CompiledTrace {
            pc: start_pc,
            cpu_type,
            ops: ops.to_vec(),
            code,
            contiguous_code,
            max_cycles,
            self_loop,
            native_loop,
            needs_window,
            code_start,
            code_end,
            adaptive_branch: ops.iter().any(|op| {
                matches!(
                    op.op,
                    JitTraceOp::Branch {
                        expected_taken: Some(_),
                        ..
                    }
                )
            }),
            adaptive_calls: Cell::new(0),
            adaptive_guard_exits: Cell::new(0),
            adaptive_rerecords: 0,
            func,
        })
    }

    #[cfg(target_family = "wasm")]
    fn compile_ops(&mut self, params: CompileParams<'_>) -> Option<CompiledTrace> {
        Some(CompiledTrace {
            pc: params.start_pc,
            cpu_type: params.cpu_type,
            ops: params.ops.to_vec(),
            code: params.code,
            contiguous_code: params.contiguous_code,
            max_cycles: params.max_cycles,
            self_loop: params.self_loop,
            needs_window: params.needs_window,
            code_start: params.code_start,
            code_end: params.code_end,
            adaptive_branch: params.ops.iter().any(|op| {
                matches!(
                    op.op,
                    JitTraceOp::Branch {
                        expected_taken: Some(_),
                        ..
                    }
                )
            }),
            adaptive_calls: Cell::new(0),
            adaptive_guard_exits: Cell::new(0),
            adaptive_rerecords: 0,
        })
    }
}

/// Everything `compile_ops` needs, gathered by `compile_trace`.
struct CompileParams<'a> {
    start_pc: u32,
    cpu_type: CpuType,
    ops: &'a [TraceBuildOp],
    code: Vec<u8>,
    contiguous_code: bool,
    max_cycles: i32,
    self_loop: bool,
    needs_window: bool,
    code_start: u32,
    code_end: u32,
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    aligned_only: bool,
    #[cfg_attr(target_family = "wasm", allow(dead_code))]
    address_mask: u32,
}

/// Attempt to execute a compiled trace at the current PC. See
/// [`TraceJit::try_execute`] for the meaning of the returned count and of
/// `instr_budget`/`single_iter`.
pub(crate) fn try_execute_trace<B: AddressBus>(
    cpu: &mut CpuCore,
    bus: &mut B,
    cpu_type: CpuType,
    instr_budget: u32,
    single_iter: bool,
    watch_pcs: &[u32],
) -> Option<(CachedRunResult, u32)> {
    if cpu.run_mode == RUN_MODE_BERR_AERR_RESET {
        return None;
    }

    TRACE_JIT.with_borrow_mut(|jit| {
        jit.try_execute(cpu, bus, cpu_type, instr_budget, single_iter, watch_pcs)
    })
}

pub(crate) fn record_trace_target(pc: u32, cpu_type: CpuType) {
    TRACE_JIT.with_borrow_mut(|jit| jit.record_trace_target(pc, cpu_type));
}

/// Append one instruction that the interpreter just executed while a hot
/// multi-block path is being recorded. The normal path checks the CPU flag
/// first, so no TLS access occurs when recording is inactive.
pub(crate) fn record_executed<B: AddressBus>(
    cpu: &mut CpuCore,
    bus: &mut B,
    executed_pc: u32,
    next_pc: u32,
) {
    if cpu.trace_recording {
        TRACE_JIT.with_borrow_mut(|jit| jit.record_executed(cpu, bus, executed_pc, next_pc));
    }
}

/// End an in-progress recording before control leaves the fast decoded-op
/// path. A usable prefix ending in a branch is compiled; otherwise the
/// target is marked rejected.
pub(crate) fn stop_recording(cpu: &mut CpuCore) {
    if cpu.trace_recording {
        TRACE_JIT.with_borrow_mut(|jit| jit.finish_recording(cpu, cpu.pc));
    }
}

/// End a recording because the decoded fast loop reached an opcode that it
/// cannot execute. Profiling builds retain the stranded prefix and blocker;
/// ordinary builds are identical to `stop_recording`.
#[cfg(feature = "trace-profile")]
pub(crate) fn stop_recording_at_blocker(cpu: &mut CpuCore, pc: u32, opcode: u16) {
    if cpu.trace_recording {
        TRACE_JIT.with_borrow_mut(|jit| {
            if let Some(recording) = jit.recording.as_ref() {
                super::trace_profile::note_blocker(
                    recording.start_pc,
                    recording.cpu_type,
                    recording.ops.len(),
                    pc,
                    opcode,
                );
            }
            jit.finish_recording(cpu, pc);
        });
    }
}

/// Note that execution just took a backward branch to `cpu.pc` (a potential
/// trace head) and return whether the caller should probe the trace cache.
///
/// This is the cheap front door to the thread-local trace state: tight
/// loops hit their branch target every iteration, so re-recording it (a
/// no-op) and re-probing known-rejected targets are filtered out with two
/// per-CPU compares before any TLS access. `TraceJit::try_execute` re-arms
/// the filters whenever it invalidates or rejects a trace.
#[inline]
pub(crate) fn note_backward_branch(cpu: &mut CpuCore, cpu_type: CpuType) -> bool {
    let pc = cpu.pc;
    #[cfg(feature = "trace-profile")]
    {
        // Consult the actual direct-mapped slot instead of relying only on
        // the CPU's four-entry skip cache: a busy workload can evict a PC
        // from that tiny filter even though its trace remains rejected.
        let rejected = TRACE_JIT.with_borrow(|jit| jit.is_rejected(pc, cpu_type));
        super::trace_profile::note_backward_edge(pc, cpu_type, rejected);
    }
    if cpu.trace_probe_skip.contains(&pc) {
        // Known-uncompilable target: recording is a no-op and probing
        // cannot succeed.
        return false;
    }
    if !cpu.trace_record_skip.contains(&pc) {
        let at = (cpu.trace_record_skip_at & 3) as usize;
        cpu.trace_record_skip[at] = pc;
        cpu.trace_record_skip_at = cpu.trace_record_skip_at.wrapping_add(1);
        record_trace_target(pc, cpu_type);
    }
    true
}

pub(crate) fn has_trace_candidates() -> bool {
    TRACE_JIT_HAS_CANDIDATES.load(Ordering::Relaxed)
}

#[inline]
fn push_probe_skip(cpu: &mut CpuCore, pc: u32) {
    if !cpu.trace_probe_skip.contains(&pc) {
        let at = (cpu.trace_probe_skip_at & 3) as usize;
        cpu.trace_probe_skip[at] = pc;
        cpu.trace_probe_skip_at = cpu.trace_probe_skip_at.wrapping_add(1);
    }
}

impl JitTraceOp {
    fn max_cycles(self) -> i32 {
        match self {
            Self::Nop => 4,
            Self::MoveReg { .. } => 4,
            Self::Moveq { .. } => 4,
            Self::UnaryDataReg { .. } => 6,
            Self::Swap { .. } => 4,
            Self::Ext { .. } => 4,
            Self::Extb { .. } => 4,
            Self::AddqSubqReg { .. } => 8,
            Self::AddqSubqAddr { .. } => 8,
            Self::BinaryDataReg { cycles, .. } => cycles,
            Self::AddrDataReg {
                op: JitAddrOp::Cmpa,
                ..
            } => 6,
            Self::AddrDataReg { .. } => 8,
            Self::AddSubxReg { .. } => 8,
            Self::BitReg {
                op: JitBitOp::Test, ..
            } => 6,
            Self::BitReg {
                op: JitBitOp::Clear,
                ..
            } => 10,
            Self::BitReg { .. } => 8,
            Self::SccDataReg { .. } => 6,
            Self::Exg { .. } => 6,
            Self::ShiftReg {
                count_or_reg,
                count_is_register,
                ..
            } => {
                if count_is_register {
                    132
                } else {
                    let count = if count_or_reg == 0 { 8 } else { count_or_reg };
                    6 + 2 * count as i32
                }
            }
            Self::Branch { length, .. } => {
                // Taken branches cost 10 cycles; a not-taken word branch
                // costs 12. This is a headroom bound, so use the slower arm.
                if length == 4 { 12 } else { 10 }
            }
            Self::Dbcc { .. } => 14,
            Self::MoveMem { size, src, dst } => {
                // 4 + source-EA fetch + destination-EA store (M68000UM).
                let long = size == Size::Long;
                let src_c = match src {
                    JitEa::Data(_) | JitEa::Addr(_) => 0,
                    JitEa::Ind(_) | JitEa::PostInc(_) => {
                        if long {
                            8
                        } else {
                            4
                        }
                    }
                    JitEa::PreDec(_) => {
                        if long {
                            10
                        } else {
                            6
                        }
                    }
                    JitEa::Disp(_, _) => {
                        if long {
                            12
                        } else {
                            8
                        }
                    }
                    JitEa::Index { .. } => {
                        if long {
                            14
                        } else {
                            10
                        }
                    }
                };
                let dst_c = match dst {
                    JitEa::Disp(_, _) => {
                        if long {
                            12
                        } else {
                            8
                        }
                    }
                    JitEa::Index { .. } => {
                        if long {
                            14
                        } else {
                            10
                        }
                    }
                    _ if dst.is_mem() => {
                        if long {
                            8
                        } else {
                            4
                        }
                    }
                    _ => 0,
                };
                4 + src_c + dst_c
            }
            Self::MovemWordPostInc { cycles, .. } => cycles,
            Self::AluMemToReg { .. } => 24,
            Self::AddRegToPostInc { size, .. } => {
                if size == Size::Long {
                    20
                } else {
                    12
                }
            }
            Self::IndirectJsr { .. } => 16,
            // These ops only execute in instruction-budgeted fastmem mode;
            // conservative cycle maxima preserve the trace headroom guard.
            Self::AnDispUnary { .. } | Self::AnDispAddqSubq { .. } | Self::AnDispBit { .. } => 24,
        }
    }

    fn ends_trace(self) -> bool {
        matches!(
            self,
            Self::Branch { .. } | Self::Dbcc { .. } | Self::IndirectJsr { .. }
        )
    }

    /// The PC a taken closing branch at `pc` jumps to, if this op is one.
    fn taken_target(self, pc: u32) -> Option<u32> {
        match self {
            Self::Branch { displacement, .. } => {
                Some((pc.wrapping_add(2) as i32).wrapping_add(displacement) as u32)
            }
            Self::Dbcc { displacement, .. } => {
                Some((pc.wrapping_add(2) as i32).wrapping_add(displacement as i32) as u32)
            }
            _ => None,
        }
    }
}

fn decode_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let opcode = bus.try_read_word(cpu.address(pc)).ok()?;
    if let Some(op) = decode_dbcc_trace_op(cpu, bus, pc, opcode) {
        return Some(op);
    }
    if let Some(op) = decode_branch_word_trace_op(cpu, bus, pc, opcode) {
        return Some(op);
    }
    if let Some(op) = decode_indirect_jsr_trace_op(pc, opcode) {
        return Some(op);
    }
    if let Some(op) = decode_an_disp_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_alu_mem_to_reg_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_add_reg_to_postinc_trace_op(pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_movem_word_postinc_trace_op(cpu, bus, pc, opcode) {
        return Some(op);
    }
    if let Some(op) = decode_move_mem_trace_op(cpu, bus, pc, opcode) {
        return Some(op);
    }

    let decoded = DecodedSimpleOp::decode(cpu_type, opcode)?;
    let op = decoded.to_jit_trace_op()?;
    Some(TraceBuildOp {
        opcode,
        extension: None,
        extension2: None,
        pc,
        op,
    })
}

/// Decode data-register-only MOVEM.W postincrement. Restricting the mask to
/// D0-D7 makes the loads and final address update independent: no loaded
/// register can alias the base An, so the operation has a simple all-or-bail
/// implementation.
fn decode_movem_word_postinc_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
) -> Option<TraceBuildOp> {
    // 0100 1100 10 011 rrr = MOVEM.W (Ar)+,<register list>.
    if (opcode & 0xFFF8) != 0x4C98 {
        return None;
    }
    let mask = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    if mask == 0 || (mask & 0xFF00) != 0 {
        return None;
    }
    let data_mask = mask as u8;
    let cycles = 12 + 4 * data_mask.count_ones() as i32;
    // The per-mode timing overhead is zero for (An)+ on every CPU path.
    Some(TraceBuildOp {
        opcode,
        extension: Some(mask),
        extension2: None,
        pc,
        op: JitTraceOp::MovemWordPostInc {
            base: (opcode & 7) as u8,
            data_mask,
            cycles,
        },
    })
}

fn decode_indirect_jsr_trace_op(pc: u32, opcode: u16) -> Option<TraceBuildOp> {
    if (opcode & 0xFFF8) != 0x4E90 {
        return None;
    }
    Some(TraceBuildOp {
        opcode,
        extension: None,
        extension2: None,
        pc,
        op: JitTraceOp::IndirectJsr {
            reg: (opcode & 7) as u8,
        },
    })
}

fn decode_dbcc_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
) -> Option<TraceBuildOp> {
    if (opcode >> 12) != 0x5 || ((opcode >> 6) & 3) != 3 || ((opcode >> 3) & 7) != 1 {
        return None;
    }

    let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    Some(TraceBuildOp {
        opcode,
        extension: Some(extension),
        extension2: None,
        pc,
        op: JitTraceOp::Dbcc {
            condition: ((opcode >> 8) & 0xF) as u8,
            reg: (opcode & 7) as u8,
            displacement: extension as i16,
        },
    })
}

fn decode_branch_word_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
) -> Option<TraceBuildOp> {
    if (opcode >> 12) != 0x6 || (opcode & 0xFF) != 0 {
        return None;
    }

    let condition = ((opcode >> 8) & 0xF) as u8;
    if condition == 1 {
        return None;
    }

    let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    Some(TraceBuildOp {
        opcode,
        extension: Some(extension),
        extension2: None,
        pc,
        op: JitTraceOp::Branch {
            condition,
            displacement: extension as i16 as i32,
            length: 4,
            expected_taken: None,
        },
    })
}

fn decode_an_disp_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let decoded = DecodedMemOp::decode(cpu_type, opcode)?;
    let read_ext =
        |offset: u32, bus: &mut B| bus.try_read_word(cpu.address(pc.wrapping_add(offset))).ok();
    let (extension, extension2, op) = match decoded {
        DecodedMemOp::Tst {
            size,
            ea: FastEa::AnDisp(reg),
        } => {
            let displacement = read_ext(2, bus)?;
            (
                displacement,
                None,
                JitTraceOp::AnDispUnary {
                    op: JitUnaryOp::Tst,
                    size,
                    reg,
                    displacement: displacement as i16,
                },
            )
        }
        DecodedMemOp::Clr {
            size,
            ea: FastEa::AnDisp(reg),
        } => {
            let displacement = read_ext(2, bus)?;
            (
                displacement,
                None,
                JitTraceOp::AnDispUnary {
                    op: JitUnaryOp::Clr,
                    size,
                    reg,
                    displacement: displacement as i16,
                },
            )
        }
        DecodedMemOp::AddqSubq {
            data,
            size,
            ea: FastEa::AnDisp(reg),
            is_sub,
        } => {
            let displacement = read_ext(2, bus)?;
            (
                displacement,
                None,
                JitTraceOp::AnDispAddqSubq {
                    data,
                    size,
                    reg,
                    displacement: displacement as i16,
                    is_sub,
                },
            )
        }
        DecodedMemOp::BitMem {
            op,
            bit,
            ea: FastEa::AnDisp(reg),
        } => {
            let op = match op {
                BitOp::Test => JitBitOp::Test,
                BitOp::Change => JitBitOp::Change,
                BitOp::Clear => JitBitOp::Clear,
                BitOp::Set => JitBitOp::Set,
            };
            match bit {
                BitSource::Reg(bit_reg) => {
                    let displacement = read_ext(2, bus)?;
                    (
                        displacement,
                        None,
                        JitTraceOp::AnDispBit {
                            op,
                            bit: JitBitSource::Reg(bit_reg),
                            reg,
                            displacement: displacement as i16,
                        },
                    )
                }
                BitSource::Imm => {
                    let bit_word = read_ext(2, bus)?;
                    let displacement = read_ext(4, bus)?;
                    (
                        bit_word,
                        Some(displacement),
                        JitTraceOp::AnDispBit {
                            op,
                            bit: JitBitSource::Imm((bit_word & 7) as u8),
                            reg,
                            displacement: displacement as i16,
                        },
                    )
                }
            }
        }
        _ => return None,
    };
    Some(TraceBuildOp {
        opcode,
        extension: Some(extension),
        extension2,
        pc,
        op,
    })
}

/// MOVE/MOVEA (groups 1-3) using register, register-indirect, or d16(An)
/// EAs. At least one side must be memory; displacement words are captured
/// in execution order for validation and self-modification checks.
fn decode_move_mem_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
) -> Option<TraceBuildOp> {
    let size = match opcode >> 12 {
        1 => Size::Byte,
        2 => Size::Long,
        3 => Size::Word,
        _ => return None,
    };
    let src_mode = (opcode >> 3) & 7;
    let dst_mode = (opcode >> 6) & 7;
    let mut next_ext = pc.wrapping_add(2);
    let mut extensions = [None, None];
    let mut extension_count = 0usize;
    let mut read_ea_ext = |mode: u16| -> Option<u16> {
        if mode != 5 && mode != 6 {
            return Some(0);
        }
        let value = bus.try_read_word(cpu.address(next_ext)).ok()?;
        next_ext = next_ext.wrapping_add(2);
        extensions[extension_count] = Some(value);
        extension_count += 1;
        Some(value)
    };
    let src_ext = read_ea_ext(src_mode)?;
    let dst_ext = read_ea_ext(dst_mode)?;
    let src = decode_jit_ea(src_mode, opcode & 7, src_ext, cpu.cpu_type)?;
    let dst = decode_jit_ea(dst_mode, (opcode >> 9) & 7, dst_ext, cpu.cpu_type)?;
    // The measured loop only needs indexed reads. Indexed stores can be
    // added when a profile demonstrates that their extra emitter paths pay.
    if matches!(dst, JitEa::Index { .. }) {
        return None;
    }
    if !src.is_mem() && !dst.is_mem() {
        return None;
    }
    // MOVEA.B does not exist, and An is not a legal byte source.
    if size == Size::Byte && (matches!(src, JitEa::Addr(_)) || matches!(dst, JitEa::Addr(_))) {
        return None;
    }
    Some(TraceBuildOp {
        opcode,
        extension: extensions[0],
        extension2: extensions[1],
        pc,
        op: JitTraceOp::MoveMem { size, src, dst },
    })
}

fn decode_add_reg_to_postinc_trace_op(
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let DecodedMemOp::AluToMem {
        op: BinaryOp::Add,
        size,
        src,
        dst: FastEa::AnPostInc(dst),
    } = DecodedMemOp::decode(cpu_type, opcode)?
    else {
        return None;
    };
    if !matches!(size, Size::Word | Size::Long) {
        return None;
    }
    Some(TraceBuildOp {
        opcode,
        extension: None,
        extension2: None,
        pc,
        op: JitTraceOp::AddRegToPostInc { size, src, dst },
    })
}

/// CMP/ADD/SUB `<ea>,Dn` for indirect and displacement source forms. The
/// access itself is emitted against the fastmem window; the extension word
/// is captured for validation and displacement baking.
fn decode_alu_mem_to_reg_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let DecodedMemOp::AluToReg { op, size, src, dst } = DecodedMemOp::decode(cpu_type, opcode)?
    else {
        return None;
    };
    let op = match op {
        BinaryOp::Cmp => JitBinaryOp::Cmp,
        BinaryOp::Add => JitBinaryOp::Add,
        BinaryOp::Sub => JitBinaryOp::Sub,
        _ => return None,
    };
    let (src, extension) = match src {
        FastEa::AnInd(reg) => (JitEa::Ind(reg), None),
        FastEa::AnDisp(reg) => {
            let displacement = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
            (JitEa::Disp(reg, displacement as i16), Some(displacement))
        }
        _ => return None,
    };
    Some(TraceBuildOp {
        opcode,
        extension,
        extension2: None,
        pc,
        op: JitTraceOp::AluMemToReg { op, size, src, dst },
    })
}

fn decode_jit_ea(mode: u16, reg: u16, extension: u16, cpu_type: CpuType) -> Option<JitEa> {
    Some(match mode & 7 {
        0 => JitEa::Data(reg as u8),
        1 => JitEa::Addr(reg as u8),
        2 => JitEa::Ind(reg as u8),
        3 => JitEa::PostInc(reg as u8),
        4 => JitEa::PreDec(reg as u8),
        5 => JitEa::Disp(reg as u8, extension as i16),
        6 => {
            if !is_pre_68020(cpu_type) && (extension & 0x0100) != 0 {
                return None;
            }
            let index_num = ((extension >> 12) & 7) as u8;
            let index = if (extension & 0x8000) != 0 {
                JitDirectReg::Addr(index_num)
            } else {
                JitDirectReg::Data(index_num)
            };
            JitEa::Index {
                base: reg as u8,
                index,
                index_long: (extension & 0x0800) != 0,
                scale: if is_pre_68020(cpu_type) {
                    0
                } else {
                    ((extension >> 9) & 3) as u8
                },
                displacement: extension as u8 as i8,
            }
        }
        _ => return None,
    })
}

/// Interpreted trace execution (wasm and unit tests). Same contract as a
/// compiled native trace: returns `(ops_retired << 32) | cycles`, and a
/// mem-op bail sets `pc` to the un-executed op.
#[cfg(any(target_family = "wasm", test))]
fn execute_portable_trace(
    cpu: &mut CpuCore,
    ops: &[TraceBuildOp],
    code_start: u32,
    code_end: u32,
) -> u64 {
    let mut cycles: i32 = 0;
    for (index, op) in ops.iter().enumerate() {
        match execute_portable_op(cpu, *op, code_start, code_end) {
            Some(c) => {
                cycles += c;
                if let JitTraceOp::Branch {
                    expected_taken: Some(expected),
                    ..
                } = op.op
                {
                    let taken = op.op.taken_target(op.pc) == Some(cpu.pc);
                    if taken != expected {
                        cpu.ppc = op.pc;
                        cpu.ir = op.opcode as u32;
                        return (((index + 1) as u64) << 32) | cycles as u32 as u64;
                    }
                }
            }
            None => {
                cpu.pc = op.pc;
                return ((index as u64) << 32) | cycles as u32 as u64;
            }
        }
    }
    if let Some(last) = ops.last() {
        cpu.ppc = last.pc;
        cpu.ir = last.opcode as u32;
    }
    ((ops.len() as u64) << 32) | cycles as u32 as u64
}

/// Execute one trace op; `None` means a mem-op check failed and nothing
/// from this op was committed.
#[cfg(any(target_family = "wasm", test))]
fn execute_portable_op(
    cpu: &mut CpuCore,
    op: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    if let JitTraceOp::MoveMem { size, src, dst } = op.op {
        return execute_portable_move_mem(cpu, size, src, dst, code_start, code_end);
    }
    if matches!(op.op, JitTraceOp::MovemWordPostInc { .. }) {
        return execute_portable_movem_word_postinc(cpu, op);
    }
    if matches!(op.op, JitTraceOp::AluMemToReg { .. }) {
        return execute_portable_alu_mem_to_reg(cpu, op);
    }
    if matches!(op.op, JitTraceOp::AddRegToPostInc { .. }) {
        return execute_portable_add_reg_to_postinc(cpu, op, code_start, code_end);
    }
    if matches!(
        op.op,
        JitTraceOp::AnDispUnary { .. }
            | JitTraceOp::AnDispAddqSubq { .. }
            | JitTraceOp::AnDispBit { .. }
    ) {
        return execute_portable_an_disp(cpu, op, code_start, code_end);
    }
    if let JitTraceOp::IndirectJsr { reg } = op.op {
        return execute_portable_indirect_jsr(cpu, op, reg);
    }
    Some(execute_portable_reg_op(cpu, op))
}

#[cfg(any(target_family = "wasm", test))]
fn execute_portable_movem_word_postinc(cpu: &mut CpuCore, trace: TraceBuildOp) -> Option<i32> {
    let JitTraceOp::MovemWordPostInc {
        base,
        data_mask,
        cycles,
    } = trace.op
    else {
        return None;
    };
    let bytes = data_mask.count_ones() * 2;
    let raw = cpu.dar[8 + base as usize];
    if cpu.is_pre_68020 && (raw & 1) != 0 {
        return None;
    }
    let masked = raw & cpu.address_mask;
    if bytes == 0
        || bytes > cpu.fm_len
        || masked as u64 + bytes as u64 > cpu.address_mask as u64 + 1
    {
        return None;
    }
    let off = masked.wrapping_sub(cpu.fm_base);
    if off > cpu.fm_len - bytes {
        return None;
    }

    let mut next_off = off as usize;
    for reg in 0..8 {
        if (data_mask & (1 << reg)) == 0 {
            continue;
        }
        let value = unsafe {
            let p = (cpu.fm_ptr as *const u8).add(next_off);
            u16::from_be_bytes([*p, *p.add(1)]) as i16 as i32 as u32
        };
        cpu.dar[reg] = value;
        next_off += 2;
    }
    cpu.dar[8 + base as usize] = raw.wrapping_add(bytes);
    cpu.pc = trace.pc.wrapping_add(4);
    Some(cycles)
}

#[cfg(any(target_family = "wasm", test))]
fn execute_portable_indirect_jsr(cpu: &mut CpuCore, trace: TraceBuildOp, reg: u8) -> Option<i32> {
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(
        cpu,
        DecodedMemOp::Jsr {
            ea: FastEa::AnInd(reg),
        },
    ) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(target_family = "wasm", test))]
fn execute_portable_an_disp(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let store = match trace.op {
        JitTraceOp::AnDispUnary {
            op: JitUnaryOp::Clr,
            size,
            reg,
            displacement,
        }
        | JitTraceOp::AnDispAddqSubq {
            size,
            reg,
            displacement,
            ..
        } => Some((size, reg, displacement)),
        JitTraceOp::AnDispBit {
            op: JitBitOp::Change | JitBitOp::Clear | JitBitOp::Set,
            reg,
            displacement,
            ..
        } => Some((Size::Byte, reg, displacement)),
        _ => None,
    };
    if let Some((size, reg, displacement)) = store {
        let raw = cpu.dar[8 + reg as usize].wrapping_add(displacement as i32 as u32);
        let masked = raw & cpu.address_mask;
        if masked < code_end && masked.wrapping_add(size.bytes()) > code_start {
            return None;
        }
    }
    let op = match trace.op {
        JitTraceOp::AnDispUnary {
            op: JitUnaryOp::Tst,
            size,
            reg,
            ..
        } => DecodedMemOp::Tst {
            size,
            ea: FastEa::AnDisp(reg),
        },
        JitTraceOp::AnDispUnary {
            op: JitUnaryOp::Clr,
            size,
            reg,
            ..
        } => DecodedMemOp::Clr {
            size,
            ea: FastEa::AnDisp(reg),
        },
        JitTraceOp::AnDispAddqSubq {
            data,
            size,
            reg,
            is_sub,
            ..
        } => DecodedMemOp::AddqSubq {
            data,
            size,
            ea: FastEa::AnDisp(reg),
            is_sub,
        },
        JitTraceOp::AnDispBit { op, bit, reg, .. } => DecodedMemOp::BitMem {
            op: match op {
                JitBitOp::Test => BitOp::Test,
                JitBitOp::Change => BitOp::Change,
                JitBitOp::Clear => BitOp::Clear,
                JitBitOp::Set => BitOp::Set,
            },
            bit: match bit {
                JitBitSource::Reg(reg) => BitSource::Reg(reg),
                JitBitSource::Imm(_) => BitSource::Imm,
            },
            ea: FastEa::AnDisp(reg),
        },
        _ => return None,
    };
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(cpu, op) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(target_family = "wasm", test))]
fn execute_portable_alu_mem_to_reg(cpu: &mut CpuCore, trace: TraceBuildOp) -> Option<i32> {
    let JitTraceOp::AluMemToReg { op, size, src, dst } = trace.op else {
        return None;
    };
    let op = match op {
        JitBinaryOp::Cmp => BinaryOp::Cmp,
        JitBinaryOp::Add => BinaryOp::Add,
        JitBinaryOp::Sub => BinaryOp::Sub,
        _ => return None,
    };
    let src = match src {
        JitEa::Ind(reg) => FastEa::AnInd(reg),
        JitEa::Disp(reg, _) => FastEa::AnDisp(reg),
        _ => return None,
    };
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(cpu, DecodedMemOp::AluToReg { op, size, src, dst }) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(target_family = "wasm", test))]
fn execute_portable_add_reg_to_postinc(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let JitTraceOp::AddRegToPostInc { size, src, dst } = trace.op else {
        return None;
    };
    let raw = cpu.dar[8 + dst as usize];
    let masked = raw & cpu.address_mask;
    if masked < code_end && masked.wrapping_add(size.bytes()) > code_start {
        return None;
    }
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(
        cpu,
        DecodedMemOp::AluToMem {
            op: BinaryOp::Add,
            size,
            src,
            dst: FastEa::AnPostInc(dst),
        },
    ) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

/// Portable MoveMem, mirroring `emit_move_mem` exactly: all checks before
/// any commit; window reads/writes via the fastmem scratch fields.
#[cfg(any(target_family = "wasm", test))]
fn execute_portable_move_mem(
    cpu: &mut CpuCore,
    size: Size,
    src: JitEa,
    dst: JitEa,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let bytes = size.bytes();
    let aligned_only = cpu.is_pre_68020;
    let locate = |cpu: &CpuCore, raw: u32| -> Option<u32> {
        if aligned_only && size != Size::Byte && (raw & 1) != 0 {
            return None;
        }
        if cpu.fm_len == 0 {
            return None;
        }
        let off = (raw & cpu.address_mask).wrapping_sub(cpu.fm_base);
        if off <= cpu.fm_len - bytes {
            Some(off)
        } else {
            None
        }
    };
    let read = |cpu: &CpuCore, off: u32| -> u32 {
        unsafe {
            let p = (cpu.fm_ptr as *const u8).add(off as usize);
            match size {
                Size::Byte => *p as u32,
                Size::Word => u16::from_be_bytes([*p, *p.add(1)]) as u32,
                Size::Long => u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)]),
            }
        }
    };

    let mut staged: Option<(usize, u32)> = None;
    let value = match src {
        JitEa::Data(r) => cpu.dar[r as usize] & size.mask(),
        JitEa::Addr(r) => cpu.dar[8 + r as usize] & size.mask(),
        JitEa::Ind(r) => read(cpu, locate(cpu, cpu.dar[8 + r as usize])?),
        JitEa::PostInc(r) => {
            let a = cpu.dar[8 + r as usize];
            let off = locate(cpu, a)?;
            staged = Some((8 + r as usize, a.wrapping_add(jit_ea_step(size, r))));
            read(cpu, off)
        }
        JitEa::PreDec(r) => {
            let a = cpu.dar[8 + r as usize].wrapping_sub(jit_ea_step(size, r));
            let off = locate(cpu, a)?;
            staged = Some((8 + r as usize, a));
            read(cpu, off)
        }
        JitEa::Disp(r, displacement) => {
            let a = cpu.dar[8 + r as usize].wrapping_add(displacement as i32 as u32);
            read(cpu, locate(cpu, a)?)
        }
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base = cpu.dar[8 + base as usize];
            let raw_index = match index {
                JitDirectReg::Data(r) => cpu.dar[r as usize],
                JitDirectReg::Addr(r) => cpu.dar[8 + r as usize],
            };
            let index = if index_long {
                raw_index
            } else {
                raw_index as u16 as i16 as i32 as u32
            };
            let a = base
                .wrapping_add(index.wrapping_shl(scale as u32))
                .wrapping_add(displacement as i32 as u32);
            read(cpu, locate(cpu, a)?)
        }
    };

    let dst_base = |cpu: &CpuCore, r: u8| match staged {
        Some((idx, v)) if idx == 8 + r as usize => v,
        _ => cpu.dar[8 + r as usize],
    };

    match dst {
        JitEa::Data(r) => {
            if let Some((idx, v)) = staged {
                cpu.dar[idx] = v;
            }
            let mask = size.mask();
            cpu.dar[r as usize] = (cpu.dar[r as usize] & !mask) | value;
            cpu.set_logic_flags(value, size);
        }
        JitEa::Addr(r) => {
            if let Some((idx, v)) = staged {
                cpu.dar[idx] = v;
            }
            cpu.dar[8 + r as usize] = if size == Size::Word {
                value as u16 as i16 as i32 as u32
            } else {
                value
            };
        }
        JitEa::Ind(r) | JitEa::PostInc(r) | JitEa::PreDec(r) | JitEa::Disp(r, _) => {
            let base = dst_base(cpu, r);
            let (addr, new_reg) = match dst {
                JitEa::Ind(_) => (base, None),
                JitEa::PostInc(_) => (base, Some(base.wrapping_add(jit_ea_step(size, r)))),
                JitEa::PreDec(_) => {
                    let a = base.wrapping_sub(jit_ea_step(size, r));
                    (a, Some(a))
                }
                JitEa::Disp(_, displacement) => {
                    (base.wrapping_add(displacement as i32 as u32), None)
                }
                _ => unreachable!(),
            };
            let off = locate(cpu, addr)?;
            let masked = addr & cpu.address_mask;
            // Self-modification guard, as in the compiled version.
            if masked < code_end && masked.wrapping_add(bytes) > code_start {
                return None;
            }
            if let Some((idx, v)) = staged {
                cpu.dar[idx] = v;
            }
            if let Some(v) = new_reg {
                cpu.dar[8 + r as usize] = v;
            }
            unsafe {
                let p = (cpu.fm_ptr as *mut u8).add(off as usize);
                match size {
                    Size::Byte => *p = value as u8,
                    Size::Word => {
                        let b = (value as u16).to_be_bytes();
                        *p = b[0];
                        *p.add(1) = b[1];
                    }
                    Size::Long => {
                        let b = value.to_be_bytes();
                        *p = b[0];
                        *p.add(1) = b[1];
                        *p.add(2) = b[2];
                        *p.add(3) = b[3];
                    }
                }
            }
            cpu.set_logic_flags(value, size);
        }
        JitEa::Index { .. } => return None,
    }

    Some(JitTraceOp::MoveMem { size, src, dst }.max_cycles())
}

#[cfg(any(target_family = "wasm", test))]
fn execute_portable_reg_op(cpu: &mut CpuCore, op: TraceBuildOp) -> i32 {
    match op.op {
        JitTraceOp::Nop => 4,
        JitTraceOp::Moveq { reg, data } => {
            cpu.dar[reg as usize] = data;
            cpu.n_flag = if (data as i32) < 0 { NFLAG_SET } else { 0 };
            cpu.not_z_flag = data;
            cpu.v_flag = 0;
            cpu.c_flag = 0;
            4
        }
        JitTraceOp::MoveReg { src, dst, size } => {
            let value = portable_read_reg(cpu, src, size);
            match dst {
                JitDirectReg::Data(reg) => {
                    portable_write_data_reg(cpu, reg, size, value);
                    cpu.set_logic_flags(value, size);
                }
                JitDirectReg::Addr(reg) => {
                    let value = if size == Size::Word {
                        value as i16 as i32 as u32
                    } else {
                        value
                    };
                    cpu.dar[8 + reg as usize] = value;
                }
            }
            4
        }
        JitTraceOp::UnaryDataReg {
            op: unary_op,
            reg,
            size,
        } => {
            let reg = reg as usize;
            let mask = size.mask();
            let src = cpu.dar[reg] & mask;
            match unary_op {
                JitUnaryOp::Clr => {
                    portable_write_data_reg(cpu, reg as u8, size, 0);
                    cpu.n_flag = 0;
                    cpu.not_z_flag = 0;
                    cpu.v_flag = 0;
                    cpu.c_flag = 0;
                }
                JitUnaryOp::Neg => {
                    let result = 0u32.wrapping_sub(src);
                    portable_write_data_reg(cpu, reg as u8, size, result);
                    cpu.set_sub_flags(src, 0, result, size);
                }
                JitUnaryOp::Negx => {
                    let result = cpu.exec_subx(size, src, 0);
                    portable_write_data_reg(cpu, reg as u8, size, result);
                }
                JitUnaryOp::Not => {
                    let result = !src & mask;
                    portable_write_data_reg(cpu, reg as u8, size, result);
                    cpu.set_logic_flags(result, size);
                }
                JitUnaryOp::Tst => {
                    cpu.set_logic_flags(src, size);
                }
            }
            if cpu.is_pre_68020 && size == Size::Long && unary_op != JitUnaryOp::Tst {
                6
            } else {
                4
            }
        }
        JitTraceOp::Swap { reg } => cpu.exec_swap(reg as usize),
        JitTraceOp::Ext { reg, size } => cpu.exec_ext(size, reg as usize),
        JitTraceOp::Extb { reg } => cpu.exec_extb(reg as usize),
        JitTraceOp::AddqSubqReg {
            reg,
            data,
            size,
            is_sub,
        } => {
            let reg = reg as usize;
            let mask = size.mask();
            let dst = cpu.dar[reg] & mask;
            let result = if is_sub {
                let result = dst.wrapping_sub(data);
                cpu.set_sub_flags(data, dst, result, size);
                result & mask
            } else {
                let result = dst.wrapping_add(data);
                cpu.set_add_flags(data, dst, result, size);
                result & mask
            };
            cpu.dar[reg] = (cpu.dar[reg] & !mask) | result;
            if cpu.is_pre_68020 && size == Size::Long {
                8
            } else {
                4
            }
        }
        JitTraceOp::AddqSubqAddr { reg, data, is_sub } => {
            let reg = 8 + reg as usize;
            cpu.dar[reg] = if is_sub {
                cpu.dar[reg].wrapping_sub(data)
            } else {
                cpu.dar[reg].wrapping_add(data)
            };
            if cpu.is_pre_68020 { 8 } else { 4 }
        }
        JitTraceOp::BinaryDataReg {
            op: binary_op,
            src,
            dst,
            size,
            cycles,
        } => {
            let src = portable_read_reg(cpu, src, size);
            let dst = dst as usize;
            let mask = size.mask();
            let dst_value = cpu.dar[dst] & mask;
            match binary_op {
                JitBinaryOp::Add => {
                    let result = dst_value.wrapping_add(src);
                    cpu.set_add_flags(src, dst_value, result, size);
                    portable_write_data_reg(cpu, dst as u8, size, result);
                }
                JitBinaryOp::Sub => {
                    let result = dst_value.wrapping_sub(src);
                    cpu.set_sub_flags(src, dst_value, result, size);
                    portable_write_data_reg(cpu, dst as u8, size, result);
                }
                JitBinaryOp::And => {
                    let result = (src & dst_value) & mask;
                    cpu.set_logic_flags(result, size);
                    portable_write_data_reg(cpu, dst as u8, size, result);
                }
                JitBinaryOp::Or => {
                    let result = (src | dst_value) & mask;
                    cpu.set_logic_flags(result, size);
                    portable_write_data_reg(cpu, dst as u8, size, result);
                }
                JitBinaryOp::Eor => {
                    let result = (src ^ dst_value) & mask;
                    cpu.set_logic_flags(result, size);
                    portable_write_data_reg(cpu, dst as u8, size, result);
                }
                JitBinaryOp::Cmp => {
                    let result = dst_value.wrapping_sub(src);
                    cpu.set_cmp_flags(src, dst_value, result, size);
                }
            }
            cycles
        }
        JitTraceOp::AddrDataReg { op, src, dst, size } => {
            let mut src = portable_read_reg(cpu, src, size);
            if size == Size::Word {
                src = src as i16 as i32 as u32;
            }
            let dst = dst as usize;
            let dst_value = cpu.dar[8 + dst];
            match op {
                JitAddrOp::Adda => {
                    cpu.dar[8 + dst] = dst_value.wrapping_add(src);
                    8
                }
                JitAddrOp::Suba => {
                    cpu.dar[8 + dst] = dst_value.wrapping_sub(src);
                    8
                }
                JitAddrOp::Cmpa => {
                    let result = dst_value.wrapping_sub(src);
                    cpu.set_cmp_flags(src, dst_value, result, Size::Long);
                    6
                }
            }
        }
        JitTraceOp::AddSubxReg {
            src,
            dst,
            size,
            is_sub,
        } => {
            let src = src as usize;
            let dst = dst as usize;
            let mask = size.mask();
            let src_value = cpu.dar[src] & mask;
            let dst_value = cpu.dar[dst] & mask;
            let result = if is_sub {
                cpu.exec_subx(size, src_value, dst_value)
            } else {
                cpu.exec_addx(size, src_value, dst_value)
            };
            portable_write_data_reg(cpu, dst as u8, size, result);
            if cpu.is_pre_68020 && size == Size::Long {
                8
            } else {
                4
            }
        }
        JitTraceOp::BitReg { op, bit_reg, dst } => {
            let bit = cpu.dar[bit_reg as usize] & 31;
            let mask = 1u32 << bit;
            let dst = dst as usize;
            let value = cpu.dar[dst];
            cpu.not_z_flag = if value & mask != 0 { 1 } else { 0 };
            let hi_bit_extra = if cpu.is_pre_68020 && bit >= 16 { 2 } else { 0 };
            match op {
                JitBitOp::Test => 6,
                JitBitOp::Change => {
                    cpu.dar[dst] = value ^ mask;
                    if cpu.is_pre_68020 {
                        6 + hi_bit_extra
                    } else {
                        8
                    }
                }
                JitBitOp::Clear => {
                    cpu.dar[dst] = value & !mask;
                    if cpu.is_pre_68020 {
                        8 + hi_bit_extra
                    } else {
                        10
                    }
                }
                JitBitOp::Set => {
                    cpu.dar[dst] = value | mask;
                    if cpu.is_pre_68020 {
                        6 + hi_bit_extra
                    } else {
                        8
                    }
                }
            }
        }
        JitTraceOp::Exg { opcode } => cpu.exec_exg(opcode),
        JitTraceOp::SccDataReg { condition, reg } => {
            let value = if cpu.test_condition(condition) {
                0xFF
            } else {
                0
            };
            portable_write_data_reg(cpu, reg, Size::Byte, value);
            if cpu.is_pre_68020 && value != 0 { 6 } else { 4 }
        }
        JitTraceOp::ShiftReg {
            reg,
            size,
            count_or_reg,
            count_is_register,
            direction,
            op: shift_op,
        } => {
            let shift = if count_is_register {
                cpu.dar[count_or_reg as usize] & 63
            } else {
                let count = count_or_reg as u32;
                if count == 0 { 8 } else { count }
            };
            let reg = reg as usize;
            let value = cpu.dar[reg] & size.mask();
            let (result, cycles) = match (shift_op, direction) {
                (0, 0) => cpu.exec_asr(size, shift, value),
                (0, 1) => cpu.exec_asl(size, shift, value),
                (1, 0) => cpu.exec_lsr(size, shift, value),
                (1, 1) => cpu.exec_lsl(size, shift, value),
                (2, 0) => cpu.exec_roxr(size, shift, value),
                (2, 1) => cpu.exec_roxl(size, shift, value),
                (3, 0) => cpu.exec_ror(size, shift, value),
                (3, 1) => cpu.exec_rol(size, shift, value),
                _ => unreachable!(),
            };
            let mask = size.mask();
            cpu.dar[reg] = (cpu.dar[reg] & !mask) | result;
            cycles
        }
        JitTraceOp::Branch {
            condition,
            displacement,
            length,
            ..
        } => {
            if condition == 0 || cpu.test_condition(condition) {
                cpu.change_of_flow = true;
                cpu.pc = (op.pc.wrapping_add(2) as i32).wrapping_add(displacement) as u32;
                10
            } else {
                cpu.pc = op.pc.wrapping_add(length as u32);
                if length == 4 { 12 } else { 8 }
            }
        }
        JitTraceOp::Dbcc {
            condition,
            reg,
            displacement,
        } => {
            if !cpu.test_condition(condition) {
                let reg = reg as usize;
                let counter = cpu.dar[reg] as u16;
                let new_counter = counter.wrapping_sub(1);
                cpu.dar[reg] = (cpu.dar[reg] & 0xFFFF_0000) | new_counter as u32;
                if new_counter != 0xFFFF {
                    cpu.pc =
                        (op.pc.wrapping_add(2) as i32).wrapping_add(displacement as i32) as u32;
                    10
                } else {
                    cpu.pc = op.pc.wrapping_add(4);
                    14
                }
            } else {
                cpu.pc = op.pc.wrapping_add(4);
                12
            }
        }
        JitTraceOp::IndirectJsr { .. } => {
            unreachable!("IndirectJsr is handled by execute_portable_indirect_jsr")
        }
        JitTraceOp::MoveMem { .. } => {
            unreachable!("MoveMem is handled by execute_portable_move_mem")
        }
        JitTraceOp::MovemWordPostInc { .. } => {
            unreachable!("MovemWordPostInc is handled by execute_portable_movem_word_postinc")
        }
        JitTraceOp::AluMemToReg { .. } => {
            unreachable!("AluMemToReg is handled by execute_portable_alu_mem_to_reg")
        }
        JitTraceOp::AddRegToPostInc { .. } => {
            unreachable!("AddRegToPostInc is handled by execute_portable_add_reg_to_postinc")
        }
        JitTraceOp::AnDispUnary { .. }
        | JitTraceOp::AnDispAddqSubq { .. }
        | JitTraceOp::AnDispBit { .. } => {
            unreachable!("AnDisp ops are handled by execute_portable_an_disp")
        }
    }
}

#[cfg(any(target_family = "wasm", test))]
fn portable_read_reg(cpu: &CpuCore, reg: JitDirectReg, size: Size) -> u32 {
    match reg {
        JitDirectReg::Data(reg) => cpu.dar[reg as usize] & size.mask(),
        JitDirectReg::Addr(reg) => cpu.dar[8 + reg as usize] & size.mask(),
    }
}

#[cfg(any(target_family = "wasm", test))]
fn portable_write_data_reg(cpu: &mut CpuCore, reg: u8, size: Size, value: u32) {
    let reg = reg as usize;
    let mask = size.mask();
    cpu.dar[reg] = (cpu.dar[reg] & !mask) | (value & mask);
}

#[cfg(not(target_family = "wasm"))]
fn emit_jit_op(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    op: TraceBuildOp,
    pre020: bool,
) -> Value {
    let trace_pc = op.pc;
    match op.op {
        JitTraceOp::Nop => cycles_const(builder, 4),
        JitTraceOp::Moveq { reg, data } => {
            let data = iconst_u32(builder, data);
            store_reg(builder, cpu, JitDirectReg::Data(reg), data);
            set_logic_flags(builder, cpu, data, Size::Long);
            cycles_const(builder, 4)
        }
        JitTraceOp::MoveReg { src, dst, size } => {
            let value = load_reg_sized(builder, cpu, src, size);
            match dst {
                JitDirectReg::Data(reg) => {
                    write_data_reg_sized(builder, cpu, reg, size, value);
                    set_logic_flags(builder, cpu, value, size);
                }
                JitDirectReg::Addr(reg) => {
                    let value = if size == Size::Word {
                        sign_extend_word(builder, value)
                    } else {
                        value
                    };
                    store_reg(builder, cpu, JitDirectReg::Addr(reg), value);
                }
            }
            cycles_const(builder, 4)
        }
        JitTraceOp::UnaryDataReg {
            op: unary_op,
            reg,
            size,
        } => {
            let value = load_reg_sized(builder, cpu, JitDirectReg::Data(reg), size);
            match unary_op {
                JitUnaryOp::Clr => {
                    let zero = iconst_u32(builder, 0);
                    write_data_reg_sized(builder, cpu, reg, size, zero);
                    store_u32(builder, cpu, offset_of!(CpuCore, n_flag), 0);
                    store_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), 0);
                    store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
                    store_u32(builder, cpu, offset_of!(CpuCore, c_flag), 0);
                }
                JitUnaryOp::Neg => {
                    let zero = iconst_u32(builder, 0);
                    let result = builder.ins().isub(zero, value);
                    write_data_reg_sized(builder, cpu, reg, size, result);
                    set_sub_flags(builder, cpu, value, zero, result, size);
                }
                JitUnaryOp::Negx => {
                    let zero = iconst_u32(builder, 0);
                    let result = emit_subx(builder, cpu, value, zero, size);
                    write_data_reg_sized(builder, cpu, reg, size, result);
                }
                JitUnaryOp::Not => {
                    let result = builder.ins().bxor_imm(value, -1);
                    let result = mask_value(builder, result, size);
                    write_data_reg_sized(builder, cpu, reg, size, result);
                    set_logic_flags(builder, cpu, result, size);
                }
                JitUnaryOp::Tst => {
                    set_logic_flags(builder, cpu, value, size);
                }
            }
            let cycles = if pre020 && size == Size::Long && unary_op != JitUnaryOp::Tst {
                6
            } else {
                4
            };
            cycles_const(builder, cycles)
        }
        JitTraceOp::Swap { reg } => {
            let value = load_reg(builder, cpu, JitDirectReg::Data(reg));
            let lo = builder.ins().ishl_imm(value, 16);
            let hi = builder.ins().ushr_imm(value, 16);
            let result = builder.ins().bor(lo, hi);
            store_reg(builder, cpu, JitDirectReg::Data(reg), result);
            set_logic_flags(builder, cpu, result, Size::Long);
            cycles_const(builder, 4)
        }
        JitTraceOp::Ext { reg, size } => {
            let value = load_reg(builder, cpu, JitDirectReg::Data(reg));
            let result = match size {
                Size::Word => {
                    let extended = sign_extend_byte(builder, value);
                    let upper_mask = iconst_u32(builder, 0xFFFF_0000);
                    let old_upper = builder.ins().band(value, upper_mask);
                    let low_word = mask_value(builder, extended, Size::Word);
                    builder.ins().bor(old_upper, low_word)
                }
                Size::Long => sign_extend_word(builder, value),
                Size::Byte => value,
            };
            store_reg(builder, cpu, JitDirectReg::Data(reg), result);
            set_logic_flags(builder, cpu, result, size);
            cycles_const(builder, 4)
        }
        JitTraceOp::Extb { reg } => {
            let value = load_reg(builder, cpu, JitDirectReg::Data(reg));
            let result = sign_extend_byte(builder, value);
            store_reg(builder, cpu, JitDirectReg::Data(reg), result);
            set_logic_flags(builder, cpu, result, Size::Long);
            cycles_const(builder, 4)
        }
        JitTraceOp::AddqSubqReg {
            reg,
            data,
            size,
            is_sub,
        } => {
            let dst = load_reg_sized(builder, cpu, JitDirectReg::Data(reg), size);
            let src = iconst_u32(builder, data);
            let result = if is_sub {
                builder.ins().isub(dst, src)
            } else {
                builder.ins().iadd(dst, src)
            };
            write_data_reg_sized(builder, cpu, reg, size, result);
            if is_sub {
                set_sub_flags(builder, cpu, src, dst, result, size);
            } else {
                set_add_flags(builder, cpu, src, dst, result, size);
            }
            cycles_const(builder, if pre020 && size == Size::Long { 8 } else { 4 })
        }
        JitTraceOp::AddqSubqAddr { reg, data, is_sub } => {
            let dst_reg = JitDirectReg::Addr(reg);
            let dst = load_reg(builder, cpu, dst_reg);
            let src = iconst_u32(builder, data);
            let result = if is_sub {
                builder.ins().isub(dst, src)
            } else {
                builder.ins().iadd(dst, src)
            };
            store_reg(builder, cpu, dst_reg, result);
            cycles_const(builder, if pre020 { 8 } else { 4 })
        }
        JitTraceOp::BinaryDataReg {
            op: binary_op,
            src,
            dst,
            size,
            ..
        } => {
            let src_value = load_reg_sized(builder, cpu, src, size);
            let dst_reg = JitDirectReg::Data(dst);
            let dst_value = load_reg_sized(builder, cpu, dst_reg, size);
            match binary_op {
                JitBinaryOp::Add => {
                    let result = builder.ins().iadd(dst_value, src_value);
                    write_data_reg_sized(builder, cpu, dst, size, result);
                    set_add_flags(builder, cpu, src_value, dst_value, result, size);
                }
                JitBinaryOp::Sub => {
                    let result = builder.ins().isub(dst_value, src_value);
                    write_data_reg_sized(builder, cpu, dst, size, result);
                    set_sub_flags(builder, cpu, src_value, dst_value, result, size);
                }
                JitBinaryOp::And => {
                    let result = builder.ins().band(dst_value, src_value);
                    write_data_reg_sized(builder, cpu, dst, size, result);
                    set_logic_flags(builder, cpu, result, size);
                }
                JitBinaryOp::Or => {
                    let result = builder.ins().bor(dst_value, src_value);
                    write_data_reg_sized(builder, cpu, dst, size, result);
                    set_logic_flags(builder, cpu, result, size);
                }
                JitBinaryOp::Eor => {
                    let result = builder.ins().bxor(dst_value, src_value);
                    write_data_reg_sized(builder, cpu, dst, size, result);
                    set_logic_flags(builder, cpu, result, size);
                }
                JitBinaryOp::Cmp => {
                    let result = builder.ins().isub(dst_value, src_value);
                    set_cmp_flags(builder, cpu, src_value, dst_value, result, size);
                }
            }
            cycles_const(builder, op.op.max_cycles())
        }
        JitTraceOp::AddrDataReg {
            op: addr_op,
            src,
            dst,
            size,
        } => {
            let src_value = load_reg_sized(builder, cpu, src, size);
            let src_value = if size == Size::Word {
                sign_extend_word(builder, src_value)
            } else {
                src_value
            };
            let dst_reg = JitDirectReg::Addr(dst);
            let dst_value = load_reg(builder, cpu, dst_reg);
            match addr_op {
                JitAddrOp::Adda => {
                    let result = builder.ins().iadd(dst_value, src_value);
                    store_reg(builder, cpu, dst_reg, result);
                    cycles_const(builder, 8)
                }
                JitAddrOp::Suba => {
                    let result = builder.ins().isub(dst_value, src_value);
                    store_reg(builder, cpu, dst_reg, result);
                    cycles_const(builder, 8)
                }
                JitAddrOp::Cmpa => {
                    let result = builder.ins().isub(dst_value, src_value);
                    set_cmp_flags(builder, cpu, src_value, dst_value, result, Size::Long);
                    cycles_const(builder, 6)
                }
            }
        }
        JitTraceOp::AddSubxReg {
            src,
            dst,
            size,
            is_sub,
        } => {
            let src_value = load_reg_sized(builder, cpu, JitDirectReg::Data(src), size);
            let dst_value = load_reg_sized(builder, cpu, JitDirectReg::Data(dst), size);
            let result = if is_sub {
                emit_subx(builder, cpu, src_value, dst_value, size)
            } else {
                emit_addx(builder, cpu, src_value, dst_value, size)
            };
            write_data_reg_sized(builder, cpu, dst, size, result);
            cycles_const(builder, if pre020 && size == Size::Long { 8 } else { 4 })
        }
        JitTraceOp::BitReg { op, bit_reg, dst } => {
            let bit = load_reg(builder, cpu, JitDirectReg::Data(bit_reg));
            let bit = builder.ins().band_imm(bit, 31);
            let one = iconst_u32(builder, 1);
            let mask = builder.ins().ishl(one, bit);
            let value = load_reg(builder, cpu, JitDirectReg::Data(dst));
            let tested = builder.ins().band(value, mask);
            let not_z = flag_from_nonzero(builder, tested, 1);
            store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), not_z);
            // Pre-020: base cycles + 2 when the (dynamic) bit number is >= 16.
            let dyn_cycles = |builder: &mut FunctionBuilder<'_>, base: i32, legacy: i32| {
                if pre020 {
                    let hi = builder
                        .ins()
                        .icmp_imm(IntCC::UnsignedGreaterThanOrEqual, bit, 16);
                    let with_extra = cycles_const(builder, base + 2);
                    let base = cycles_const(builder, base);
                    builder.ins().select(hi, with_extra, base)
                } else {
                    cycles_const(builder, legacy)
                }
            };
            match op {
                JitBitOp::Test => cycles_const(builder, 6),
                JitBitOp::Change => {
                    let result = builder.ins().bxor(value, mask);
                    store_reg(builder, cpu, JitDirectReg::Data(dst), result);
                    dyn_cycles(builder, 6, 8)
                }
                JitBitOp::Clear => {
                    let inverted = builder.ins().bxor_imm(mask, -1);
                    let result = builder.ins().band(value, inverted);
                    store_reg(builder, cpu, JitDirectReg::Data(dst), result);
                    dyn_cycles(builder, 8, 10)
                }
                JitBitOp::Set => {
                    let result = builder.ins().bor(value, mask);
                    store_reg(builder, cpu, JitDirectReg::Data(dst), result);
                    dyn_cycles(builder, 6, 8)
                }
            }
        }
        JitTraceOp::Exg { opcode } => {
            let rx = ((opcode >> 9) & 7) as u8;
            let ry = (opcode & 7) as u8;
            match (opcode >> 3) & 0x1F {
                0x08 => swap_regs(builder, cpu, JitDirectReg::Data(rx), JitDirectReg::Data(ry)),
                0x09 => swap_regs(builder, cpu, JitDirectReg::Addr(rx), JitDirectReg::Addr(ry)),
                0x11 => swap_regs(builder, cpu, JitDirectReg::Data(rx), JitDirectReg::Addr(ry)),
                _ => {}
            }
            cycles_const(builder, 6)
        }
        JitTraceOp::SccDataReg { condition, reg } => {
            let condition = emit_condition(builder, cpu, condition);
            let true_value = iconst_u32(builder, 0xFF);
            let false_value = iconst_u32(builder, 0);
            let value = builder.ins().select(condition, true_value, false_value);
            write_data_reg_sized(builder, cpu, reg, Size::Byte, value);
            if pre020 {
                let taken = cycles_const(builder, 6);
                let not_taken = cycles_const(builder, 4);
                builder.ins().select(condition, taken, not_taken)
            } else {
                cycles_const(builder, 4)
            }
        }
        JitTraceOp::ShiftReg {
            reg,
            size,
            count_or_reg,
            count_is_register,
            direction,
            op,
        } => {
            debug_assert!(!count_is_register && matches!((op, direction), (0, 0) | (1, 1)));
            let shift = if count_or_reg == 0 {
                8
            } else {
                u32::from(count_or_reg)
            };
            let value = load_reg_sized(builder, cpu, JitDirectReg::Data(reg), size);
            let bits = size.bits() as u32;
            let (result, shifted_out) = match (op, direction) {
                (0, 0) => {
                    let signed = match size {
                        Size::Byte => sign_extend_byte(builder, value),
                        Size::Word => sign_extend_word(builder, value),
                        Size::Long => value,
                    };
                    let result = builder.ins().sshr_imm(signed, i64::from(shift));
                    let shifted_out = if shift >= bits {
                        let msb = iconst_u32(builder, size_msb(size));
                        builder.ins().band(value, msb)
                    } else {
                        let bit = iconst_u32(builder, 1u32 << (shift - 1));
                        builder.ins().band(value, bit)
                    };
                    (result, shifted_out)
                }
                (1, 1) => {
                    let result = builder.ins().ishl_imm(value, i64::from(shift));
                    let shifted_out = if shift > bits {
                        iconst_u32(builder, 0)
                    } else {
                        let bit = iconst_u32(builder, 1u32 << (bits - shift));
                        builder.ins().band(value, bit)
                    };
                    (result, shifted_out)
                }
                _ => unreachable!("unsupported native register shift"),
            };
            let result = mask_value(builder, result, size);
            write_data_reg_sized(builder, cpu, reg, size, result);
            let carry = flag_from_nonzero(builder, shifted_out, CFLAG_SET);
            store_value_u32(builder, cpu, offset_of!(CpuCore, c_flag), carry);
            store_value_u32(builder, cpu, offset_of!(CpuCore, x_flag), carry);
            store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
            set_logic_flags_nv(builder, cpu, result, size);

            let base = if pre020 && size == Size::Long { 8 } else { 6 };
            cycles_const(builder, base + 2 * shift as i32)
        }
        JitTraceOp::MoveMem { .. } => unreachable!("MoveMem is emitted by emit_move_mem"),
        JitTraceOp::MovemWordPostInc { .. } => {
            unreachable!("MovemWordPostInc is emitted by emit_movem_word_postinc")
        }
        JitTraceOp::AluMemToReg { .. } => {
            unreachable!("AluMemToReg is emitted by emit_alu_mem_to_reg")
        }
        JitTraceOp::AddRegToPostInc { .. } => {
            unreachable!("AddRegToPostInc is emitted by emit_add_reg_to_postinc")
        }
        JitTraceOp::AnDispUnary { .. }
        | JitTraceOp::AnDispAddqSubq { .. }
        | JitTraceOp::AnDispBit { .. } => {
            unreachable!("AnDisp ops are emitted by emit_an_disp_mem")
        }
        JitTraceOp::IndirectJsr { .. } => {
            unreachable!("IndirectJsr is emitted by emit_indirect_jsr")
        }
        JitTraceOp::Branch {
            condition,
            displacement,
            length,
            ..
        } => emit_branch(builder, cpu, trace_pc, condition, displacement, length),
        JitTraceOp::Dbcc {
            condition,
            reg,
            displacement,
        } => emit_dbcc(builder, cpu, trace_pc, condition, reg, displacement),
    }
}

/// Window/bounds context shared by all mem ops in one trace function.
#[cfg(not(target_family = "wasm"))]
struct MemEnv {
    fm_ptr: Value,
    fm_ptr_ty: Type,
    fm_base: Value,
    fm_len: Value,
    address_mask: u32,
    aligned_only: bool,
    code_start: u32,
    code_end: u32,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
struct BailAt {
    ops_before: RetiredBefore,
    cycles_before: Value,
}

#[cfg(not(target_family = "wasm"))]
#[derive(Clone, Copy)]
enum RetiredBefore {
    Constant(u32),
    Dynamic(Value),
}

#[cfg(not(target_family = "wasm"))]
struct BailReq {
    block: Block,
    pc: u32,
    at: BailAt,
}

#[cfg(not(target_family = "wasm"))]
struct MoveMemOp {
    pc: u32,
    size: Size,
    src: JitEa,
    dst: JitEa,
}

/// Branch to `bail` when `bad` holds; continue emitting in a fresh block.
#[cfg(not(target_family = "wasm"))]
fn branch_guard(builder: &mut FunctionBuilder<'_>, bail: Block, bad: Value) {
    let cont = builder.create_block();
    builder.ins().brif(bad, bail, &[], cont, &[]);
    builder.switch_to_block(cont);
}

/// Alignment + window-range checks for an access of `size` at raw address
/// `addr`. Returns `(window_offset, masked_address)`; branches to `bail`
/// on any miss.
#[cfg(not(target_family = "wasm"))]
fn checked_window_off(
    builder: &mut FunctionBuilder<'_>,
    env: &MemEnv,
    bail: Block,
    addr: Value,
    size: Size,
) -> (Value, Value) {
    if env.aligned_only && size != Size::Byte {
        let low = builder.ins().band_imm(addr, 1);
        let bad = builder.ins().icmp_imm(IntCC::NotEqual, low, 0);
        branch_guard(builder, bail, bad);
    }
    let masked = builder.ins().band_imm(addr, env.address_mask as i64);
    let off = builder.ins().isub(masked, env.fm_base);
    let limit = builder.ins().iadd_imm(env.fm_len, -(size.bytes() as i64));
    let bad = builder.ins().icmp(IntCC::UnsignedGreaterThan, off, limit);
    branch_guard(builder, bail, bad);
    (off, masked)
}

#[cfg(not(target_family = "wasm"))]
fn window_host_addr(builder: &mut FunctionBuilder<'_>, env: &MemEnv, off: Value) -> Value {
    let off_ptr = if env.fm_ptr_ty == types::I32 {
        off
    } else {
        builder.ins().uextend(env.fm_ptr_ty, off)
    };
    builder.ins().iadd(env.fm_ptr, off_ptr)
}

/// Big-endian sized load from the window; result is a zero-extended I32.
#[cfg(not(target_family = "wasm"))]
fn window_load(builder: &mut FunctionBuilder<'_>, env: &MemEnv, off: Value, size: Size) -> Value {
    let addr = window_host_addr(builder, env, off);
    let mut flags = MemFlags::new();
    flags.set_notrap();
    match size {
        Size::Byte => {
            let v = builder.ins().load(types::I8, flags, addr, 0);
            builder.ins().uextend(types::I32, v)
        }
        Size::Word => {
            let v = builder.ins().load(types::I16, flags, addr, 0);
            let v = builder.ins().bswap(v);
            builder.ins().uextend(types::I32, v)
        }
        Size::Long => {
            let v = builder.ins().load(types::I32, flags, addr, 0);
            builder.ins().bswap(v)
        }
    }
}

/// Emit a data-register-only MOVEM.W (An)+. A single check covers the
/// contiguous register list before any register or address state changes;
/// the individual big-endian loads are then safe to emit without guards.
#[cfg(not(target_family = "wasm"))]
fn emit_movem_word_postinc(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::MovemWordPostInc {
        base,
        data_mask,
        cycles,
    } = trace.op
    else {
        unreachable!("expected MOVEM.W postincrement trace")
    };
    let bytes = data_mask.count_ones() * 2;
    debug_assert!(bytes != 0 && bytes <= 16);
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let raw = load_reg(builder, cpu, JitDirectReg::Addr(base));
    if env.aligned_only {
        let low = builder.ins().band_imm(raw, 1);
        let bad = builder.ins().icmp_imm(IntCC::NotEqual, low, 0);
        branch_guard(builder, bail, bad);
    }
    let masked = builder.ins().band_imm(raw, env.address_mask as i64);
    let last_valid_start = env.address_mask.saturating_sub(bytes - 1);
    let wraps = builder.ins().icmp_imm(
        IntCC::UnsignedGreaterThan,
        masked,
        i64::from(last_valid_start),
    );
    branch_guard(builder, bail, wraps);
    let too_short = builder
        .ins()
        .icmp_imm(IntCC::UnsignedLessThan, env.fm_len, i64::from(bytes));
    branch_guard(builder, bail, too_short);
    let off = builder.ins().isub(masked, env.fm_base);
    let limit = builder.ins().iadd_imm(env.fm_len, -i64::from(bytes));
    let outside = builder.ins().icmp(IntCC::UnsignedGreaterThan, off, limit);
    branch_guard(builder, bail, outside);

    let mut ordinal = 0i64;
    for reg in 0..8 {
        if (data_mask & (1 << reg)) == 0 {
            continue;
        }
        let word_off = if ordinal == 0 {
            off
        } else {
            builder.ins().iadd_imm(off, ordinal * 2)
        };
        let word = window_load(builder, env, word_off, Size::Word);
        let value = sign_extend_word(builder, word);
        store_reg(builder, cpu, JitDirectReg::Data(reg), value);
        ordinal += 1;
    }
    let next = builder.ins().iadd_imm(raw, i64::from(bytes));
    store_reg(builder, cpu, JitDirectReg::Addr(base), next);
    cycles_const(builder, cycles)
}

/// Big-endian sized store of (sized) `value` into the window.
#[cfg(not(target_family = "wasm"))]
fn window_store(
    builder: &mut FunctionBuilder<'_>,
    env: &MemEnv,
    off: Value,
    size: Size,
    value: Value,
) {
    let addr = window_host_addr(builder, env, off);
    let mut flags = MemFlags::new();
    flags.set_notrap();
    match size {
        Size::Byte => {
            let v = builder.ins().ireduce(types::I8, value);
            builder.ins().store(flags, v, addr, 0);
        }
        Size::Word => {
            let v = builder.ins().ireduce(types::I16, value);
            let v = builder.ins().bswap(v);
            builder.ins().store(flags, v, addr, 0);
        }
        Size::Long => {
            let v = builder.ins().bswap(value);
            builder.ins().store(flags, v, addr, 0);
        }
    }
}

#[cfg(not(target_family = "wasm"))]
fn guard_store_not_code(
    builder: &mut FunctionBuilder<'_>,
    env: &MemEnv,
    bail: Block,
    masked: Value,
    size: Size,
) {
    let lt_end = builder
        .ins()
        .icmp_imm(IntCC::UnsignedLessThan, masked, env.code_end as i64);
    let past = builder.ins().iadd_imm(masked, size.bytes() as i64);
    let gt_start = builder
        .ins()
        .icmp_imm(IntCC::UnsignedGreaterThan, past, env.code_start as i64);
    let bad = builder.ins().band(lt_end, gt_start);
    branch_guard(builder, bail, bad);
}

/// Emit a read-only memory-to-register ALU operation. All address checks run
/// before flags are committed, so a miss can re-execute the instruction via
/// full dispatch without rolling back architectural state.
#[cfg(not(target_family = "wasm"))]
fn emit_alu_mem_to_reg(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::AluMemToReg { op, size, src, dst } = trace.op else {
        unreachable!("expected memory-to-register ALU trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let base_reg = match src {
        JitEa::Ind(reg) | JitEa::Disp(reg, _) => reg,
        _ => unreachable!("ALU trace decoder only admits (An) and d16(An)"),
    };
    let base = load_reg(builder, cpu, JitDirectReg::Addr(base_reg));
    let addr = match src {
        JitEa::Ind(_) => base,
        JitEa::Disp(_, displacement) => builder.ins().iadd_imm(base, displacement as i64),
        _ => unreachable!(),
    };
    let (off, _) = checked_window_off(builder, env, bail, addr, size);
    let src_value = window_load(builder, env, off, size);
    let dst_value = load_reg(builder, cpu, JitDirectReg::Data(dst));
    let dst_value = mask_value(builder, dst_value, size);
    match op {
        JitBinaryOp::Cmp => {
            let result = builder.ins().isub(dst_value, src_value);
            set_cmp_flags(builder, cpu, src_value, dst_value, result, size);
        }
        JitBinaryOp::Add => {
            let result = builder.ins().iadd(dst_value, src_value);
            write_data_reg_sized(builder, cpu, dst, size, result);
            set_add_flags(builder, cpu, src_value, dst_value, result, size);
        }
        JitBinaryOp::Sub => {
            let result = builder.ins().isub(dst_value, src_value);
            write_data_reg_sized(builder, cpu, dst, size, result);
            set_sub_flags(builder, cpu, src_value, dst_value, result, size);
        }
        _ => unreachable!("unsupported memory-to-register ALU operation"),
    }
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit ADD.W/L Dn,(An)+. The window, alignment, and self-modification
/// guards all run before memory, address-register, or flag state is changed.
#[cfg(not(target_family = "wasm"))]
fn emit_add_reg_to_postinc(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::AddRegToPostInc { size, src, dst } = trace.op else {
        unreachable!("expected register-to-postincrement ADD trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let addr = load_reg(builder, cpu, JitDirectReg::Addr(dst));
    let (off, masked) = checked_window_off(builder, env, bail, addr, size);
    guard_store_not_code(builder, env, bail, masked, size);
    let dst_value = window_load(builder, env, off, size);
    let src_value = load_reg(builder, cpu, JitDirectReg::Data(src));
    let src_value = mask_value(builder, src_value, size);
    let result = builder.ins().iadd(dst_value, src_value);

    window_store(builder, env, off, size, result);
    let next = builder
        .ins()
        .iadd_imm(addr, i64::from(jit_ea_step(size, dst)));
    store_reg(builder, cpu, JitDirectReg::Addr(dst), next);
    set_add_flags(builder, cpu, src_value, dst_value, result, size);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit terminal `JSR (An)`. The stack write is checked before the stack
/// pointer, flow state, or PC changes, so a miss can re-execute the call via
/// full dispatch. A successful call ends this non-self-loop trace; writing
/// into its code is therefore safe because any later entry revalidates it.
#[cfg(not(target_family = "wasm"))]
fn emit_indirect_jsr(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    reg: u8,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let target = load_reg(builder, cpu, JitDirectReg::Addr(reg));
    let old_sp = load_reg(builder, cpu, JitDirectReg::Addr(7));
    let new_sp = builder.ins().iadd_imm(old_sp, -4);
    let (off, _) = checked_window_off(builder, env, bail, new_sp, Size::Long);
    let return_pc = iconst_u32(builder, trace.pc.wrapping_add(2));
    window_store(builder, env, off, Size::Long, return_pc);

    store_reg(builder, cpu, JitDirectReg::Addr(7), new_sp);
    store_bool(builder, cpu, offset_of!(CpuCore, change_of_flow), true);
    store_value_u32(builder, cpu, offset_of!(CpuCore, pc), target);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit displacement-memory operations with the displacement baked into the
/// trace, leaving only the live An value and fastmem bounds to check at
/// runtime.
#[cfg(not(target_family = "wasm"))]
fn emit_an_disp_mem(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });
    let (reg, displacement, size) = match trace.op {
        JitTraceOp::AnDispUnary {
            reg,
            displacement,
            size,
            ..
        }
        | JitTraceOp::AnDispAddqSubq {
            reg,
            displacement,
            size,
            ..
        } => (reg, displacement, size),
        JitTraceOp::AnDispBit {
            reg, displacement, ..
        } => (reg, displacement, Size::Byte),
        _ => unreachable!(),
    };
    let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
    let addr = builder.ins().iadd_imm(base, displacement as i64);
    let (off, masked) = checked_window_off(builder, env, bail, addr, size);
    let value = window_load(builder, env, off, size);

    match trace.op {
        JitTraceOp::AnDispUnary {
            op: JitUnaryOp::Tst,
            ..
        } => set_logic_flags(builder, cpu, value, size),
        JitTraceOp::AnDispUnary {
            op: JitUnaryOp::Clr,
            ..
        } => {
            guard_store_not_code(builder, env, bail, masked, size);
            let zero = iconst_u32(builder, 0);
            window_store(builder, env, off, size, zero);
            store_u32(builder, cpu, offset_of!(CpuCore, n_flag), 0);
            store_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), 0);
            store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
            store_u32(builder, cpu, offset_of!(CpuCore, c_flag), 0);
        }
        JitTraceOp::AnDispAddqSubq { data, is_sub, .. } => {
            guard_store_not_code(builder, env, bail, masked, size);
            let src = iconst_u32(builder, data);
            let result = if is_sub {
                builder.ins().isub(value, src)
            } else {
                builder.ins().iadd(value, src)
            };
            window_store(builder, env, off, size, result);
            if is_sub {
                set_sub_flags(builder, cpu, src, value, result, size);
            } else {
                set_add_flags(builder, cpu, src, value, result, size);
            }
        }
        JitTraceOp::AnDispBit { op, bit, .. } => {
            let bit = match bit {
                JitBitSource::Reg(reg) => {
                    let value = load_reg(builder, cpu, JitDirectReg::Data(reg));
                    builder.ins().band_imm(value, 7)
                }
                JitBitSource::Imm(bit) => iconst_u32(builder, bit as u32),
            };
            let one = iconst_u32(builder, 1);
            let mask = builder.ins().ishl(one, bit);
            let tested = builder.ins().band(value, mask);
            let not_z = flag_from_nonzero(builder, tested, 1);
            match op {
                JitBitOp::Test => {}
                JitBitOp::Change | JitBitOp::Clear | JitBitOp::Set => {
                    guard_store_not_code(builder, env, bail, masked, Size::Byte);
                    let result = match op {
                        JitBitOp::Change => builder.ins().bxor(value, mask),
                        JitBitOp::Clear => {
                            let inverted = builder.ins().bxor_imm(mask, -1);
                            builder.ins().band(value, inverted)
                        }
                        JitBitOp::Set => builder.ins().bor(value, mask),
                        JitBitOp::Test => unreachable!(),
                    };
                    window_store(builder, env, off, Size::Byte, result);
                }
            }
            store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), not_z);
        }
        _ => unreachable!(),
    }
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit a MOVE/MOVEA with memory operands. All alignment/window/code-overlap
/// checks run before anything commits; each check branches to a bail block
/// that sets `pc = op.pc` and returns the ops retired before this one, so a
/// bailing instruction re-executes through full dispatch.
#[cfg(not(target_family = "wasm"))]
fn emit_move_mem(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    op: MoveMemOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: op.pc,
        at,
    });
    let size = op.size;

    let load_an =
        |builder: &mut FunctionBuilder<'_>, r: u8| load_reg(builder, cpu, JitDirectReg::Addr(r));

    // Resolve the source: its value plus any staged post-inc/pre-dec
    // register update (not committed until every check has passed).
    let mut staged: Option<(u8, Value)> = None; // (An index, new value)
    let value = match op.src {
        JitEa::Data(r) => {
            let v = load_reg(builder, cpu, JitDirectReg::Data(r));
            mask_value(builder, v, size)
        }
        JitEa::Addr(r) => {
            let v = load_an(builder, r);
            mask_value(builder, v, size)
        }
        JitEa::Ind(r) => {
            let a = load_an(builder, r);
            let (off, _) = checked_window_off(builder, env, bail, a, size);
            window_load(builder, env, off, size)
        }
        JitEa::PostInc(r) => {
            let a = load_an(builder, r);
            let (off, _) = checked_window_off(builder, env, bail, a, size);
            let next = builder.ins().iadd_imm(a, jit_ea_step(size, r) as i64);
            staged = Some((r, next));
            window_load(builder, env, off, size)
        }
        JitEa::PreDec(r) => {
            let a0 = load_an(builder, r);
            let a = builder.ins().iadd_imm(a0, -(jit_ea_step(size, r) as i64));
            let (off, _) = checked_window_off(builder, env, bail, a, size);
            staged = Some((r, a));
            window_load(builder, env, off, size)
        }
        JitEa::Disp(r, displacement) => {
            let base = load_an(builder, r);
            let a = builder.ins().iadd_imm(base, displacement as i64);
            let (off, _) = checked_window_off(builder, env, bail, a, size);
            window_load(builder, env, off, size)
        }
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base = load_an(builder, base);
            let raw_index = load_reg(builder, cpu, index);
            let index = if index_long {
                raw_index
            } else {
                let word = builder.ins().ireduce(types::I16, raw_index);
                builder.ins().sextend(types::I32, word)
            };
            let index = if scale == 0 {
                index
            } else {
                builder.ins().ishl_imm(index, i64::from(scale))
            };
            let a = builder.ins().iadd(base, index);
            let a = builder.ins().iadd_imm(a, displacement as i64);
            let (off, _) = checked_window_off(builder, env, bail, a, size);
            window_load(builder, env, off, size)
        }
    };

    // A destination base register must observe a same-register source
    // adjustment (e.g. `MOVE.L (A0)+,(A0)+`).
    let dst_base = |builder: &mut FunctionBuilder<'_>, r: u8| match staged {
        Some((sr, v)) if sr == r => v,
        _ => load_an(builder, r),
    };
    let commit_staged = |builder: &mut FunctionBuilder<'_>| {
        if let Some((r, v)) = staged {
            store_reg(builder, cpu, JitDirectReg::Addr(r), v);
        }
    };

    match op.dst {
        JitEa::Data(r) => {
            commit_staged(builder);
            write_data_reg_sized(builder, cpu, r, size, value);
            set_logic_flags(builder, cpu, value, size);
        }
        JitEa::Addr(r) => {
            // MOVEA: sign-extend word, no flags.
            commit_staged(builder);
            let v = if size == Size::Word {
                sign_extend_word(builder, value)
            } else {
                value
            };
            store_reg(builder, cpu, JitDirectReg::Addr(r), v);
        }
        JitEa::Ind(r) | JitEa::PostInc(r) | JitEa::PreDec(r) | JitEa::Disp(r, _) => {
            let base = dst_base(builder, r);
            let (addr, new_reg) = match op.dst {
                JitEa::Ind(_) => (base, None),
                JitEa::PostInc(_) => {
                    let next = builder.ins().iadd_imm(base, jit_ea_step(size, r) as i64);
                    (base, Some(next))
                }
                JitEa::PreDec(_) => {
                    let a = builder.ins().iadd_imm(base, -(jit_ea_step(size, r) as i64));
                    (a, Some(a))
                }
                JitEa::Disp(_, displacement) => {
                    (builder.ins().iadd_imm(base, displacement as i64), None)
                }
                _ => unreachable!(),
            };
            let (off, masked) = checked_window_off(builder, env, bail, addr, size);

            // Self-modification guard: a store overlapping this trace's
            // own code bails (before committing) so the interpreter
            // re-runs it and the next fetch sees the new bytes.
            let lt_end =
                builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, masked, env.code_end as i64);
            let past = builder.ins().iadd_imm(masked, size.bytes() as i64);
            let gt_start =
                builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedGreaterThan, past, env.code_start as i64);
            let bad = builder.ins().band(lt_end, gt_start);
            branch_guard(builder, bail, bad);

            commit_staged(builder);
            if let Some(v) = new_reg {
                store_reg(builder, cpu, JitDirectReg::Addr(r), v);
            }
            window_store(builder, env, off, size, value);
            set_logic_flags(builder, cpu, value, size);
        }
        JitEa::Index { .. } => unreachable!("indexed MOVE destination is not traceable"),
    }

    cycles_const(
        builder,
        JitTraceOp::MoveMem {
            size,
            src: op.src,
            dst: op.dst,
        }
        .max_cycles(),
    )
}

#[cfg(not(target_family = "wasm"))]
fn load_reg(builder: &mut FunctionBuilder<'_>, cpu: Value, reg: JitDirectReg) -> Value {
    let index = match reg {
        JitDirectReg::Data(reg) => reg as usize,
        JitDirectReg::Addr(reg) => 8 + reg as usize,
    };
    load_u32(
        builder,
        cpu,
        offset_of!(CpuCore, dar) + index * size_of::<u32>(),
    )
}

#[cfg(not(target_family = "wasm"))]
fn store_reg(builder: &mut FunctionBuilder<'_>, cpu: Value, reg: JitDirectReg, value: Value) {
    let index = match reg {
        JitDirectReg::Data(reg) => reg as usize,
        JitDirectReg::Addr(reg) => 8 + reg as usize,
    };
    store_value_u32(
        builder,
        cpu,
        offset_of!(CpuCore, dar) + index * size_of::<u32>(),
        value,
    );
}

#[cfg(not(target_family = "wasm"))]
fn cycles_const(builder: &mut FunctionBuilder<'_>, cycles: i32) -> Value {
    builder.ins().iconst(types::I32, cycles as i64)
}

#[cfg(not(target_family = "wasm"))]
fn swap_regs(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    left: JitDirectReg,
    right: JitDirectReg,
) {
    let left_value = load_reg(builder, cpu, left);
    let right_value = load_reg(builder, cpu, right);
    store_reg(builder, cpu, left, right_value);
    store_reg(builder, cpu, right, left_value);
}

#[cfg(not(target_family = "wasm"))]
fn load_reg_sized(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    reg: JitDirectReg,
    size: Size,
) -> Value {
    let value = load_reg(builder, cpu, reg);
    mask_value(builder, value, size)
}

#[cfg(not(target_family = "wasm"))]
fn write_data_reg_sized(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    reg: u8,
    size: Size,
    value: Value,
) {
    let value = mask_value(builder, value, size);
    if size == Size::Long {
        store_reg(builder, cpu, JitDirectReg::Data(reg), value);
        return;
    }

    let old = load_reg(builder, cpu, JitDirectReg::Data(reg));
    let upper_mask = iconst_u32(builder, !size_mask(size));
    let upper = builder.ins().band(old, upper_mask);
    let result = builder.ins().bor(upper, value);
    store_reg(builder, cpu, JitDirectReg::Data(reg), result);
}

#[cfg(not(target_family = "wasm"))]
fn mask_value(builder: &mut FunctionBuilder<'_>, value: Value, size: Size) -> Value {
    if size == Size::Long {
        value
    } else {
        let mask = iconst_u32(builder, size_mask(size));
        builder.ins().band(value, mask)
    }
}

#[cfg(not(target_family = "wasm"))]
fn sign_extend_byte(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let shifted = builder.ins().ishl_imm(value, 24);
    builder.ins().sshr_imm(shifted, 24)
}

#[cfg(not(target_family = "wasm"))]
fn sign_extend_word(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let shifted = builder.ins().ishl_imm(value, 16);
    builder.ins().sshr_imm(shifted, 16)
}

#[cfg(not(target_family = "wasm"))]
fn size_mask(size: Size) -> u32 {
    match size {
        Size::Byte => 0xFF,
        Size::Word => 0xFFFF,
        Size::Long => 0xFFFF_FFFF,
    }
}

#[cfg(not(target_family = "wasm"))]
fn size_msb(size: Size) -> u32 {
    match size {
        Size::Byte => 0x80,
        Size::Word => 0x8000,
        Size::Long => 0x8000_0000,
    }
}

#[cfg(not(target_family = "wasm"))]
fn set_logic_flags(builder: &mut FunctionBuilder<'_>, cpu: Value, value: Value, size: Size) {
    set_logic_flags_nv(builder, cpu, value, size);
    store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
    store_u32(builder, cpu, offset_of!(CpuCore, c_flag), 0);
}

#[cfg(not(target_family = "wasm"))]
fn set_logic_flags_nv(builder: &mut FunctionBuilder<'_>, cpu: Value, value: Value, size: Size) {
    let value = mask_value(builder, value, size);
    let msb = iconst_u32(builder, size_msb(size));
    let sign_bits = builder.ins().band(value, msb);
    let n = flag_from_nonzero(builder, sign_bits, NFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, n_flag), n);
    store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), value);
}

#[cfg(not(target_family = "wasm"))]
fn set_add_flags(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    src: Value,
    dst: Value,
    result: Value,
    size: Size,
) {
    let src = mask_value(builder, src, size);
    let dst = mask_value(builder, dst, size);
    let masked_result = mask_value(builder, result, size);
    let msb = iconst_u32(builder, size_msb(size));
    let sign_bits = builder.ins().band(masked_result, msb);
    let n = flag_from_nonzero(builder, sign_bits, NFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, n_flag), n);
    store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), masked_result);

    let src_xor_result = builder.ins().bxor(src, masked_result);
    let dst_xor_result = builder.ins().bxor(dst, masked_result);
    let overflow_bits = builder.ins().band(src_xor_result, dst_xor_result);
    let overflow_sign_bits = builder.ins().band(overflow_bits, msb);
    let v = flag_from_nonzero(builder, overflow_sign_bits, VFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, v_flag), v);

    let c = if size == Size::Long {
        let src_and_dst = builder.ins().band(src, dst);
        let src_or_dst = builder.ins().bor(src, dst);
        let not_result = builder.ins().bxor_imm(masked_result, -1);
        let not_result_and_src_or_dst = builder.ins().band(not_result, src_or_dst);
        let carry_bits = builder.ins().bor(src_and_dst, not_result_and_src_or_dst);
        let carry_sign_bits = builder.ins().band(carry_bits, msb);
        flag_from_nonzero(builder, carry_sign_bits, CFLAG_SET)
    } else {
        let carry_mask = iconst_u32(builder, size_mask(size) + 1);
        let carry_bits = builder.ins().band(result, carry_mask);
        flag_from_nonzero(builder, carry_bits, CFLAG_SET)
    };
    store_value_u32(builder, cpu, offset_of!(CpuCore, c_flag), c);
    store_value_u32(builder, cpu, offset_of!(CpuCore, x_flag), c);
}

#[cfg(not(target_family = "wasm"))]
fn set_sub_flags(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    src: Value,
    dst: Value,
    result: Value,
    size: Size,
) {
    let src = mask_value(builder, src, size);
    let dst = mask_value(builder, dst, size);
    let masked_result = mask_value(builder, result, size);
    let msb = iconst_u32(builder, size_msb(size));
    let sign_bits = builder.ins().band(masked_result, msb);
    let n = flag_from_nonzero(builder, sign_bits, NFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, n_flag), n);
    store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), masked_result);

    let src_xor_dst = builder.ins().bxor(src, dst);
    let result_xor_dst = builder.ins().bxor(masked_result, dst);
    let overflow_bits = builder.ins().band(src_xor_dst, result_xor_dst);
    let overflow_sign_bits = builder.ins().band(overflow_bits, msb);
    let v = flag_from_nonzero(builder, overflow_sign_bits, VFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, v_flag), v);

    let c = if size == Size::Long {
        let src_and_result = builder.ins().band(src, masked_result);
        let src_or_result = builder.ins().bor(src, masked_result);
        let not_dst = builder.ins().bxor_imm(dst, -1);
        let not_dst_and_src_or_result = builder.ins().band(not_dst, src_or_result);
        let carry_bits = builder.ins().bor(src_and_result, not_dst_and_src_or_result);
        let carry_sign_bits = builder.ins().band(carry_bits, msb);
        flag_from_nonzero(builder, carry_sign_bits, CFLAG_SET)
    } else {
        let carry = builder.ins().icmp(IntCC::UnsignedGreaterThan, src, dst);
        select_flag(builder, carry, CFLAG_SET)
    };
    store_value_u32(builder, cpu, offset_of!(CpuCore, c_flag), c);
    store_value_u32(builder, cpu, offset_of!(CpuCore, x_flag), c);
}

#[cfg(not(target_family = "wasm"))]
fn set_cmp_flags(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    src: Value,
    dst: Value,
    result: Value,
    size: Size,
) {
    let src = mask_value(builder, src, size);
    let dst = mask_value(builder, dst, size);
    let masked_result = mask_value(builder, result, size);
    let msb = iconst_u32(builder, size_msb(size));
    let sign_bits = builder.ins().band(masked_result, msb);
    let n = flag_from_nonzero(builder, sign_bits, NFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, n_flag), n);
    store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), masked_result);

    let src_xor_dst = builder.ins().bxor(src, dst);
    let result_xor_dst = builder.ins().bxor(masked_result, dst);
    let overflow_bits = builder.ins().band(src_xor_dst, result_xor_dst);
    let overflow_sign_bits = builder.ins().band(overflow_bits, msb);
    let v = flag_from_nonzero(builder, overflow_sign_bits, VFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, v_flag), v);

    let carry = builder.ins().icmp(IntCC::UnsignedGreaterThan, src, dst);
    let c = select_flag(builder, carry, CFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, c_flag), c);
}

#[cfg(not(target_family = "wasm"))]
fn emit_addx(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    src: Value,
    dst: Value,
    size: Size,
) -> Value {
    let src = mask_value(builder, src, size);
    let dst = mask_value(builder, dst, size);
    let x = extend_flag_value(builder, cpu);
    let src64 = builder.ins().uextend(types::I64, src);
    let dst64 = builder.ins().uextend(types::I64, dst);
    let x64 = builder.ins().uextend(types::I64, x);
    let sum64 = builder.ins().iadd(dst64, src64);
    let sum64 = builder.ins().iadd(sum64, x64);
    let result32 = builder.ins().ireduce(types::I32, sum64);
    let result = mask_value(builder, result32, size);

    set_addx_subx_common_flags(builder, cpu, src, dst, result, size, false);
    let carry = builder
        .ins()
        .icmp_imm(IntCC::UnsignedGreaterThan, sum64, size_mask(size) as i64);
    let c = select_flag(builder, carry, CFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, c_flag), c);
    store_value_u32(builder, cpu, offset_of!(CpuCore, x_flag), c);
    result
}

#[cfg(not(target_family = "wasm"))]
fn emit_subx(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    src: Value,
    dst: Value,
    size: Size,
) -> Value {
    let src = mask_value(builder, src, size);
    let dst = mask_value(builder, dst, size);
    let x = extend_flag_value(builder, cpu);
    let src64 = builder.ins().uextend(types::I64, src);
    let dst64 = builder.ins().uextend(types::I64, dst);
    let x64 = builder.ins().uextend(types::I64, x);
    let sub64 = builder.ins().iadd(src64, x64);
    let result64 = builder.ins().isub(dst64, sub64);
    let result32 = builder.ins().ireduce(types::I32, result64);
    let result = mask_value(builder, result32, size);

    set_addx_subx_common_flags(builder, cpu, src, dst, result, size, true);
    let borrow = builder.ins().icmp(IntCC::UnsignedGreaterThan, sub64, dst64);
    let c = select_flag(builder, borrow, CFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, c_flag), c);
    store_value_u32(builder, cpu, offset_of!(CpuCore, x_flag), c);
    result
}

#[cfg(not(target_family = "wasm"))]
fn set_addx_subx_common_flags(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    src: Value,
    dst: Value,
    result: Value,
    size: Size,
    is_sub: bool,
) {
    let msb = iconst_u32(builder, size_msb(size));
    let sign_bits = builder.ins().band(result, msb);
    let n = flag_from_nonzero(builder, sign_bits, NFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, n_flag), n);

    let result_nonzero = builder.ins().icmp_imm(IntCC::NotEqual, result, 0);
    let old_not_z = load_u32(builder, cpu, offset_of!(CpuCore, not_z_flag));
    let not_z = builder.ins().select(result_nonzero, result, old_not_z);
    store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), not_z);

    let v = if is_sub {
        let src_xor_dst = builder.ins().bxor(src, dst);
        let result_xor_dst = builder.ins().bxor(result, dst);
        let overflow_bits = builder.ins().band(src_xor_dst, result_xor_dst);
        let overflow_sign_bits = builder.ins().band(overflow_bits, msb);
        flag_from_nonzero(builder, overflow_sign_bits, VFLAG_SET)
    } else {
        let src_xor_result = builder.ins().bxor(src, result);
        let dst_xor_result = builder.ins().bxor(dst, result);
        let overflow_bits = builder.ins().band(src_xor_result, dst_xor_result);
        let overflow_sign_bits = builder.ins().band(overflow_bits, msb);
        flag_from_nonzero(builder, overflow_sign_bits, VFLAG_SET)
    };
    store_value_u32(builder, cpu, offset_of!(CpuCore, v_flag), v);
}

#[cfg(not(target_family = "wasm"))]
fn extend_flag_value(builder: &mut FunctionBuilder<'_>, cpu: Value) -> Value {
    let x_flag = load_u32(builder, cpu, offset_of!(CpuCore, x_flag));
    let has_x = builder.ins().icmp_imm(IntCC::NotEqual, x_flag, 0);
    let one = iconst_u32(builder, 1);
    let zero = iconst_u32(builder, 0);
    builder.ins().select(has_x, one, zero)
}

#[cfg(not(target_family = "wasm"))]
/// Logical NOT for the 0/1 booleans produced by `icmp`.
///
/// `bnot` is bitwise and must not be used here: `bnot(0x01) == 0xFE`,
/// which is still non-zero and therefore still "true" to `select`/`brif`.
/// Flipping the low bit keeps the value a canonical 0/1 boolean.
#[cfg(not(target_family = "wasm"))]
fn not_bool(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    builder.ins().bxor_imm(value, 1)
}

#[cfg(not(target_family = "wasm"))]
fn emit_condition(builder: &mut FunctionBuilder<'_>, cpu: Value, cond: u8) -> Value {
    let c = flag_is_set(builder, cpu, offset_of!(CpuCore, c_flag));
    let z = flag_is_zero_set(builder, cpu);
    let v = flag_is_set(builder, cpu, offset_of!(CpuCore, v_flag));
    let n = flag_is_set(builder, cpu, offset_of!(CpuCore, n_flag));

    match cond & 0x0F {
        0x0 => bool_const(builder, true),
        0x1 => bool_const(builder, false),
        0x2 => {
            let not_c = not_bool(builder, c);
            let not_z = not_bool(builder, z);
            builder.ins().band(not_c, not_z)
        }
        0x3 => builder.ins().bor(c, z),
        0x4 => not_bool(builder, c),
        0x5 => c,
        0x6 => not_bool(builder, z),
        0x7 => z,
        0x8 => not_bool(builder, v),
        0x9 => v,
        0xA => not_bool(builder, n),
        0xB => n,
        0xC => {
            let different = builder.ins().bxor(n, v);
            not_bool(builder, different)
        }
        0xD => builder.ins().bxor(n, v),
        0xE => {
            let not_z = not_bool(builder, z);
            let different = builder.ins().bxor(n, v);
            let same = not_bool(builder, different);
            builder.ins().band(not_z, same)
        }
        0xF => {
            let different = builder.ins().bxor(n, v);
            builder.ins().bor(z, different)
        }
        _ => bool_const(builder, true),
    }
}

#[cfg(not(target_family = "wasm"))]
fn emit_branch(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace_pc: u32,
    condition: u8,
    displacement: i32,
    length: u8,
) -> Value {
    let target_pc = (trace_pc.wrapping_add(2) as i32).wrapping_add(displacement) as u32;
    if condition == 0 {
        store_bool(builder, cpu, offset_of!(CpuCore, change_of_flow), true);
        store_pc(builder, cpu, target_pc);
        return cycles_const(builder, 10);
    }

    let taken = emit_condition(builder, cpu, condition);
    let target = iconst_u32(builder, target_pc);
    let next = iconst_u32(builder, trace_pc.wrapping_add(length as u32));
    let pc = builder.ins().select(taken, target, next);
    store_pc_value(builder, cpu, pc);

    let old_change = load_u8(builder, cpu, offset_of!(CpuCore, change_of_flow));
    let true_change = builder.ins().iconst(types::I8, 1);
    let change = builder.ins().select(taken, true_change, old_change);
    store_value(builder, cpu, offset_of!(CpuCore, change_of_flow), change);

    let taken_cycles = cycles_const(builder, 10);
    let not_taken_cycles = cycles_const(builder, if length == 4 { 12 } else { 8 });
    builder.ins().select(taken, taken_cycles, not_taken_cycles)
}

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::too_many_arguments)]
fn emit_guarded_branch(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace_pc: u32,
    condition: u8,
    displacement: i32,
    length: u8,
    expected_taken: bool,
    cycles_before: Value,
    retired_before_iter: RetiredBefore,
    ops_done: u32,
) -> Value {
    let target_pc = (trace_pc.wrapping_add(2) as i32).wrapping_add(displacement) as u32;
    let taken = if condition == 0 {
        bool_const(builder, true)
    } else {
        emit_condition(builder, cpu, condition)
    };
    let target = iconst_u32(builder, target_pc);
    let next = iconst_u32(builder, trace_pc.wrapping_add(length as u32));
    let pc = builder.ins().select(taken, target, next);
    store_pc_value(builder, cpu, pc);

    let old_change = load_u8(builder, cpu, offset_of!(CpuCore, change_of_flow));
    let true_change = builder.ins().iconst(types::I8, 1);
    let change = builder.ins().select(taken, true_change, old_change);
    store_value(builder, cpu, offset_of!(CpuCore, change_of_flow), change);

    let taken_cycles = cycles_const(builder, 10);
    let not_taken_cycles = cycles_const(builder, if length == 4 { 12 } else { 8 });
    let op_cycles = builder.ins().select(taken, taken_cycles, not_taken_cycles);

    let expected = bool_const(builder, expected_taken);
    let matches = builder.ins().icmp(IntCC::Equal, taken, expected);
    let continue_block = builder.create_block();
    let side_exit = builder.create_block();
    builder
        .ins()
        .brif(matches, continue_block, &[], side_exit, &[]);

    builder.switch_to_block(side_exit);
    store_u32(builder, cpu, offset_of!(CpuCore, ppc), trace_pc);
    let total_cycles = builder.ins().iadd(cycles_before, op_cycles);
    let cycles64 = builder.ins().uextend(types::I64, total_cycles);
    let retired = match retired_before_iter {
        RetiredBefore::Constant(retired) => builder
            .ins()
            .iconst(types::I64, i64::from(retired + ops_done) << 32),
        RetiredBefore::Dynamic(retired) => {
            let retired = builder.ins().iadd_imm(retired, i64::from(ops_done));
            let retired = builder.ins().uextend(types::I64, retired);
            builder.ins().ishl_imm(retired, 32)
        }
    };
    let packed = builder.ins().bor(cycles64, retired);
    builder.ins().return_(&[packed]);

    builder.switch_to_block(continue_block);
    op_cycles
}

#[cfg(not(target_family = "wasm"))]
fn emit_dbcc(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace_pc: u32,
    condition: u8,
    reg: u8,
    displacement: i16,
) -> Value {
    let condition_true = emit_condition(builder, cpu, condition);
    let dreg = load_reg(builder, cpu, JitDirectReg::Data(reg));
    let counter = mask_value(builder, dreg, Size::Word);
    let one = iconst_u32(builder, 1);
    let new_counter = builder.ins().isub(counter, one);
    let new_counter = mask_value(builder, new_counter, Size::Word);
    let upper_mask = iconst_u32(builder, 0xFFFF_0000);
    let upper = builder.ins().band(dreg, upper_mask);
    let updated_dreg = builder.ins().bor(upper, new_counter);
    let stored_dreg = builder.ins().select(condition_true, dreg, updated_dreg);
    store_reg(builder, cpu, JitDirectReg::Data(reg), stored_dreg);

    let false_condition = not_bool(builder, condition_true);
    let not_expired = builder.ins().icmp_imm(IntCC::NotEqual, new_counter, 0xFFFF);
    let false_value = bool_const(builder, false);
    let branch_taken = builder
        .ins()
        .select(false_condition, not_expired, false_value);

    let target_pc = (trace_pc.wrapping_add(2) as i32).wrapping_add(displacement as i32) as u32;
    let target = iconst_u32(builder, target_pc);
    let next = iconst_u32(builder, trace_pc.wrapping_add(4));
    let pc = builder.ins().select(branch_taken, target, next);
    store_pc_value(builder, cpu, pc);

    let taken_cycles = cycles_const(builder, 10);
    let expired_cycles = cycles_const(builder, 14);
    let false_cycles = builder
        .ins()
        .select(branch_taken, taken_cycles, expired_cycles);
    let true_cycles = cycles_const(builder, 12);
    builder
        .ins()
        .select(condition_true, true_cycles, false_cycles)
}

#[cfg(not(target_family = "wasm"))]
fn flag_is_set(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize) -> Value {
    let flag = load_u32(builder, cpu, offset);
    builder.ins().icmp_imm(IntCC::NotEqual, flag, 0)
}

#[cfg(not(target_family = "wasm"))]
fn flag_is_zero_set(builder: &mut FunctionBuilder<'_>, cpu: Value) -> Value {
    let not_z = load_u32(builder, cpu, offset_of!(CpuCore, not_z_flag));
    builder.ins().icmp_imm(IntCC::Equal, not_z, 0)
}

#[cfg(not(target_family = "wasm"))]
fn bool_const(builder: &mut FunctionBuilder<'_>, value: bool) -> Value {
    let zero = iconst_u32(builder, 0);
    if value {
        builder.ins().icmp_imm(IntCC::Equal, zero, 0)
    } else {
        builder.ins().icmp_imm(IntCC::NotEqual, zero, 0)
    }
}

#[cfg(not(target_family = "wasm"))]
fn flag_from_nonzero(builder: &mut FunctionBuilder<'_>, value: Value, flag: u32) -> Value {
    let condition = builder.ins().icmp_imm(IntCC::NotEqual, value, 0);
    select_flag(builder, condition, flag)
}

#[cfg(not(target_family = "wasm"))]
fn select_flag(builder: &mut FunctionBuilder<'_>, condition: Value, flag: u32) -> Value {
    let flag_value = iconst_u32(builder, flag);
    let zero = iconst_u32(builder, 0);
    builder.ins().select(condition, flag_value, zero)
}

#[cfg(not(target_family = "wasm"))]
fn load_u32(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize) -> Value {
    builder
        .ins()
        .load(types::I32, MemFlags::trusted(), cpu, offset as i32)
}

#[cfg(not(target_family = "wasm"))]
fn load_u8(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize) -> Value {
    builder
        .ins()
        .load(types::I8, MemFlags::trusted(), cpu, offset as i32)
}

#[cfg(not(target_family = "wasm"))]
fn store_pc(builder: &mut FunctionBuilder<'_>, cpu: Value, pc: u32) {
    store_u32(builder, cpu, offset_of!(CpuCore, pc), pc);
}

#[cfg(not(target_family = "wasm"))]
fn store_pc_value(builder: &mut FunctionBuilder<'_>, cpu: Value, pc: Value) {
    store_value_u32(builder, cpu, offset_of!(CpuCore, pc), pc);
}

#[cfg(not(target_family = "wasm"))]
fn store_bool(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: bool) {
    let value = builder.ins().iconst(types::I8, i64::from(value as u8));
    builder
        .ins()
        .store(MemFlags::trusted(), value, cpu, offset as i32);
}

#[cfg(not(target_family = "wasm"))]
fn store_u32(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: u32) {
    let value = iconst_u32(builder, value);
    store_value_u32(builder, cpu, offset, value);
}

#[cfg(not(target_family = "wasm"))]
fn store_value_u32(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: Value) {
    store_value(builder, cpu, offset, value);
}

#[cfg(not(target_family = "wasm"))]
fn store_value(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: Value) {
    builder
        .ins()
        .store(MemFlags::trusted(), value, cpu, offset as i32);
}

#[cfg(not(target_family = "wasm"))]
fn iconst_u32(builder: &mut FunctionBuilder<'_>, value: u32) -> Value {
    builder.ins().iconst(types::I32, value as i32 as i64)
}

fn trace_cache_index(pc: u32) -> usize {
    ((pc >> 1) as usize) & (TRACE_CACHE_SIZE - 1)
}

#[cfg(test)]
mod portable_tests {
    use super::*;

    fn cpu() -> CpuCore {
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68000);
        cpu.set_sr(0x2700);
        cpu.pc = 0x0100;
        cpu
    }

    /// Wire a byte buffer up as the CPU's fastmem window at guest base 0.
    fn attach_window(cpu: &mut CpuCore, mem: &mut [u8]) {
        cpu.fm_ptr = mem.as_mut_ptr() as usize;
        cpu.fm_base = 0;
        cpu.fm_len = mem.len() as u32;
    }

    /// `MOVE.L (A0)+,(A1)+ ; DBRA D0` at $0100 — the memcpy inner loop.
    fn move_mem_loop_ops() -> [TraceBuildOp; 2] {
        [
            TraceBuildOp {
                opcode: 0x22D8,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::MoveMem {
                    size: Size::Long,
                    src: JitEa::PostInc(0),
                    dst: JitEa::PostInc(1),
                },
            },
            TraceBuildOp {
                opcode: 0x51C8,
                extension: Some(0xFFFC),
                extension2: None,
                pc: 0x0102,
                op: JitTraceOp::Dbcc {
                    condition: 1,
                    reg: 0,
                    displacement: -4,
                },
            },
        ]
    }

    #[test]
    fn portable_move_mem_copies_through_window() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x200..0x204].copy_from_slice(&0xDEADBEEFu32.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x200);
        cpu.set_a(1, 0x300);
        cpu.set_d(0, 5);

        let ops = move_mem_loop_ops();
        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0106);

        assert_eq!((packed >> 32) as u32, 2, "both ops retired");
        assert_eq!(&mem[0x300..0x304], &0xDEADBEEFu32.to_be_bytes());
        assert_eq!(cpu.a(0), 0x204);
        assert_eq!(cpu.a(1), 0x304);
        assert_eq!(cpu.d(0), 4, "DBRA decremented");
        assert_eq!(cpu.pc, 0x0100, "DBRA branched back to the head");
    }

    #[test]
    fn portable_move_mem_bails_outside_window_with_nothing_committed() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x00FF_F000); // masked address beyond the window
        cpu.set_a(1, 0x300);
        cpu.set_d(0, 5);

        let ops = move_mem_loop_ops();
        cpu.pc = 0x0104;
        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0106);

        assert_eq!((packed >> 32) as u32, 0, "nothing retired");
        assert_eq!(packed as u32, 0, "no cycles charged");
        assert_eq!(cpu.pc, 0x0100, "pc points at the bailing op");
        assert_eq!(cpu.a(0), 0x00FF_F000, "no post-increment committed");
        assert_eq!(cpu.d(0), 5);
    }

    #[test]
    fn portable_move_mem_bails_on_store_into_own_code() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x200..0x204].copy_from_slice(&0x4E714E71u32.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x200);
        cpu.set_a(1, 0x0102); // store would overwrite the trace's DBRA
        cpu.set_d(0, 5);

        let ops = move_mem_loop_ops();
        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0106);

        assert_eq!((packed >> 32) as u32, 0, "store into code bails");
        assert_eq!(cpu.pc, 0x0100);
        assert_eq!(cpu.a(0), 0x200, "source post-increment not committed");
        assert_eq!(&mem[0x102..0x106], &[0u8; 4], "no store happened");
    }

    #[test]
    fn portable_move_mem_same_register_postinc_pair() {
        // MOVE.W (A0)+,(A0)+ — destination must see the incremented A0.
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x200..0x202].copy_from_slice(&0xBEEFu16.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x200);

        let op = TraceBuildOp {
            opcode: 0x30D8,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::MoveMem {
                size: Size::Word,
                src: JitEa::PostInc(0),
                dst: JitEa::PostInc(0),
            },
        };
        // Single-op traces never compile, but the executor semantics are
        // shared; drive the op directly.
        let cycles = execute_portable_op(&mut cpu, op, 0x0100, 0x0102);

        assert!(cycles.is_some());
        assert_eq!(&mem[0x202..0x204], &0xBEEFu16.to_be_bytes());
        assert_eq!(cpu.a(0), 0x204);
    }

    fn movem_word_postinc_op() -> TraceBuildOp {
        TraceBuildOp {
            opcode: 0x4C98,
            extension: Some(0x00FE),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::MovemWordPostInc {
                base: 0,
                data_mask: 0xFE,
                cycles: 40,
            },
        }
    }

    #[test]
    fn decodes_movem_word_postincrement() {
        let cpu = cpu();
        let mut mem = super::super::memory::LinearMemoryBus::new(0x1000);
        mem.write_word(0x0100, 0x4C98);
        mem.write_word(0x0102, 0x00FE);
        let op = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68000).unwrap();
        assert_eq!(op.length(), 4);
        assert!(matches!(
            op.op,
            JitTraceOp::MovemWordPostInc {
                base: 0,
                data_mask: 0xFE,
                cycles: 40,
            }
        ));

        mem.write_word(0x0102, 0x0101); // address-register masks stay interpreted
        assert!(decode_movem_word_postinc_trace_op(&cpu, &mut mem, 0x0100, 0x4C98).is_none());
    }

    #[test]
    fn portable_movem_word_postincrement_sign_extends_and_preserves_flags() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        let words = [0x8000u16, 1, 0x7FFF, 0xFFFF, 0, 0x1234, 0xABCD];
        for (index, word) in words.into_iter().enumerate() {
            mem[0x0200 + index * 2..0x0202 + index * 2].copy_from_slice(&word.to_be_bytes());
        }
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x0200);
        cpu.set_ccr(0x1F);

        assert_eq!(
            execute_portable_op(&mut cpu, movem_word_postinc_op(), 0x0100, 0x0104),
            Some(40)
        );
        assert_eq!(cpu.d(0), 0);
        assert_eq!(cpu.d(1), 0xFFFF_8000);
        assert_eq!(cpu.d(2), 1);
        assert_eq!(cpu.d(3), 0x0000_7FFF);
        assert_eq!(cpu.d(4), 0xFFFF_FFFF);
        assert_eq!(cpu.d(5), 0);
        assert_eq!(cpu.d(6), 0x0000_1234);
        assert_eq!(cpu.d(7), 0xFFFF_ABCD);
        assert_eq!(cpu.a(0), 0x020E);
        assert_eq!(cpu.pc, 0x0104);
        assert_eq!(cpu.get_ccr(), 0x1F, "MOVEM does not affect flags");
    }

    #[test]
    fn portable_movem_word_postincrement_bails_without_partial_state() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x0208]; // seven words do not fit at $0200
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x0200);
        for reg in 1..8 {
            cpu.set_d(reg, 0xA000_0000 | reg as u32);
        }
        cpu.pc = 0x0444;

        assert_eq!(
            execute_portable_op(&mut cpu, movem_word_postinc_op(), 0x0100, 0x0104),
            None
        );
        assert_eq!(cpu.a(0), 0x0200);
        assert_eq!(cpu.pc, 0x0444);
        for reg in 1..8 {
            assert_eq!(cpu.d(reg), 0xA000_0000 | reg as u32);
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_movem_word_postincrement_matches_portable_and_bails_atomically() {
        let ops = vec![
            movem_word_postinc_op(),
            TraceBuildOp {
                opcode: 0x51C8,
                extension: Some(0xFFFA),
                extension2: None,
                pc: 0x0104,
                op: JitTraceOp::Dbcc {
                    condition: 1,
                    reg: 0,
                    displacement: -6,
                },
            },
        ];
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        let words = [0x8000u16, 1, 0x7FFF, 0xFFFF, 0, 0x1234, 0xABCD];
        for (index, word) in words.into_iter().enumerate() {
            mem[0x0200 + index * 2..0x0202 + index * 2].copy_from_slice(&word.to_be_bytes());
        }
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x0200);
        cpu.set_d(0, 2);
        cpu.set_ccr(0x1F);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&cpu, 0x0100, CpuType::M68000, ops, Some(0x0100))
            .expect("MOVEM/DBRA loop should compile");

        let packed = unsafe { compiled.call_native(&mut cpu, 1) };
        assert_eq!((packed >> 32) as u32, 2);
        assert_eq!(packed as u32, 50);
        assert_eq!(cpu.d(0), 1);
        assert_eq!(cpu.d(1), 0xFFFF_8000);
        assert_eq!(cpu.d(7), 0xFFFF_ABCD);
        assert_eq!(cpu.a(0), 0x020E);
        assert_eq!(cpu.pc, 0x0100);
        assert_eq!(cpu.get_ccr(), 0x1F);

        cpu.set_a(0, 0x00FF_FFF8); // the register list crosses the address mask
        for reg in 1..8 {
            cpu.set_d(reg, 0xB000_0000 | reg as u32);
        }
        let packed = unsafe { compiled.call_native(&mut cpu, 1) };
        assert_eq!(packed, 0, "bail retires no instructions or cycles");
        assert_eq!(cpu.pc, 0x0100);
        assert_eq!(cpu.a(0), 0x00FF_FFF8);
        for reg in 1..8 {
            assert_eq!(cpu.d(reg), 0xB000_0000 | reg as u32);
        }
    }

    #[test]
    fn decodes_hot_alu_memory_sources() {
        let cpu = cpu();
        let mut mem = super::super::memory::LinearMemoryBus::new(0x1000);
        mem.write_word(0x0100, 0xB210); // CMP.B (A0),D1
        let indirect = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68000).unwrap();
        assert_eq!(indirect.extension, None);
        assert!(matches!(
            indirect.op,
            JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Byte,
                src: JitEa::Ind(0),
                dst: 1,
            }
        ));

        mem.write_word(0x0100, 0xBC6E); // CMP.W $0010(A6),D6
        mem.write_word(0x0102, 0x0010);
        let displacement = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68000).unwrap();
        assert_eq!(displacement.extension, Some(0x0010));
        assert!(matches!(
            displacement.op,
            JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Word,
                src: JitEa::Disp(6, 0x0010),
                dst: 6,
            }
        ));

        mem.write_word(0x0100, 0xDE6D); // ADD.W $0010(A5),D7
        mem.write_word(0x0102, 0x0010);
        let add = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68000).unwrap();
        assert_eq!(add.extension, Some(0x0010));
        assert!(matches!(
            add.op,
            JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Add,
                size: Size::Word,
                src: JitEa::Disp(5, 0x0010),
                dst: 7,
            }
        ));

        mem.write_word(0x0100, 0x986D); // SUB.W $0010(A5),D4
        mem.write_word(0x0102, 0x0010);
        let sub = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68000).unwrap();
        assert_eq!(sub.extension, Some(0x0010));
        assert!(matches!(
            sub.op,
            JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Sub,
                size: Size::Word,
                src: JitEa::Disp(5, 0x0010),
                dst: 4,
            }
        ));
    }

    #[test]
    fn decodes_indirect_jsr_trace_boundary() {
        let cpu = cpu();
        let mut mem = super::super::memory::LinearMemoryBus::new(0x1000);
        mem.write_word(0x0100, 0x4E90); // JSR (A0)
        let jsr = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68000).unwrap();
        assert_eq!(jsr.extension, None);
        assert_eq!(jsr.length(), 2);
        assert!(matches!(jsr.op, JitTraceOp::IndirectJsr { reg: 0 }));
        assert!(jsr.op.ends_trace());
    }

    #[test]
    fn portable_indirect_jsr_pushes_return_and_changes_flow() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x0340);
        cpu.set_a(7, 0x0800);
        cpu.change_of_flow = false;

        let op = TraceBuildOp {
            opcode: 0x4E90,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::IndirectJsr { reg: 0 },
        };
        assert_eq!(execute_portable_op(&mut cpu, op, 0x0100, 0x0102), Some(16));
        assert_eq!(cpu.a(7), 0x07FC);
        assert_eq!(&mem[0x07FC..0x0800], &0x0102u32.to_be_bytes());
        assert_eq!(cpu.pc, 0x0340);
        assert!(cpu.change_of_flow);
    }

    #[test]
    fn portable_indirect_jsr_bails_without_partial_state() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x0340);
        cpu.set_a(7, 2); // decremented SP wraps outside the window
        cpu.pc = 0x0444;
        cpu.change_of_flow = false;

        let op = TraceBuildOp {
            opcode: 0x4E90,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::IndirectJsr { reg: 0 },
        };
        assert_eq!(execute_portable_op(&mut cpu, op, 0x0100, 0x0102), None);
        assert_eq!(cpu.a(7), 2);
        assert_eq!(cpu.pc, 0x0444);
        assert!(!cpu.change_of_flow);
        assert!(mem.iter().all(|&byte| byte == 0));
    }

    #[test]
    fn portable_cmp_memory_sets_nzvc_and_preserves_x() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(0, 0x0200);
        cpu.set_d(1, 0x1234_567F);
        cpu.set_ccr(0x10); // X set; CMP must preserve it.
        mem[0x0200] = 0x80;

        let op = TraceBuildOp {
            opcode: 0xB210,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Byte,
                src: JitEa::Ind(0),
                dst: 1,
            },
        };
        assert!(execute_portable_op(&mut cpu, op, 0x0100, 0x0102).is_some());
        assert_eq!(cpu.d(1), 0x1234_567F, "CMP does not write its destination");
        assert_eq!(cpu.get_ccr(), 0x1B, "X/N/V/C set and Z clear");
    }

    #[test]
    fn portable_cmp_displacement_bails_without_changing_state() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(6, 0x00FF_F000); // displacement remains outside the window
        cpu.set_d(6, 0xCAFE_BEEF);
        cpu.set_ccr(0x15);
        mem[0x0102..0x0104].copy_from_slice(&0x0010u16.to_be_bytes());

        let op = TraceBuildOp {
            opcode: 0xBC6E,
            extension: Some(0x0010),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Word,
                src: JitEa::Disp(6, 0x0010),
                dst: 6,
            },
        };
        cpu.pc = 0x0444;
        assert_eq!(execute_portable_op(&mut cpu, op, 0x0100, 0x0104), None);
        assert_eq!(cpu.pc, 0x0444);
        assert_eq!(cpu.d(6), 0xCAFE_BEEF);
        assert_eq!(cpu.get_ccr(), 0x15);
    }

    #[test]
    fn portable_add_displacement_updates_register_and_flags() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0102..0x0104].copy_from_slice(&0x0010u16.to_be_bytes());
        mem[0x0210..0x0212].copy_from_slice(&1u16.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(5, 0x0200);
        cpu.set_d(7, 0xA5A5_7FFF);
        cpu.set_ccr(0x1F);

        let op = TraceBuildOp {
            opcode: 0xDE6D,
            extension: Some(0x0010),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Add,
                size: Size::Word,
                src: JitEa::Disp(5, 0x0010),
                dst: 7,
            },
        };
        assert!(execute_portable_op(&mut cpu, op, 0x0100, 0x0104).is_some());
        assert_eq!(cpu.d(7), 0xA5A5_8000);
        assert_eq!(cpu.get_ccr(), 0x0A, "N/V set; X/Z/C clear");
    }

    #[test]
    fn portable_sub_displacement_updates_register_and_flags() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0102..0x0104].copy_from_slice(&0x0010u16.to_be_bytes());
        mem[0x0210..0x0212].copy_from_slice(&1u16.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(5, 0x0200);
        cpu.set_d(4, 0xA5A5_8000);
        cpu.set_ccr(0x1F);

        let op = TraceBuildOp {
            opcode: 0x986D,
            extension: Some(0x0010),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Sub,
                size: Size::Word,
                src: JitEa::Disp(5, 0x0010),
                dst: 4,
            },
        };
        assert!(execute_portable_op(&mut cpu, op, 0x0100, 0x0104).is_some());
        assert_eq!(cpu.d(4), 0xA5A5_7FFF);
        assert_eq!(cpu.get_ccr(), 0x02, "V set; X/N/Z/C clear");
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_add_displacement_matches_interpreter_result() {
        let cases = [
            (Size::Byte, None, 0x81, 0xA5A5_557F),
            (Size::Word, Some(0x0010), 1, 0xA5A5_7FFF),
            (Size::Long, None, 0x8000_0000, 0x8000_0000),
        ];

        for cpu_type in [CpuType::M68000, CpuType::M68040] {
            for (size, displacement, src_value, initial) in cases {
                let dst = 7usize;
                let addr_reg = 5usize;
                let op_mode = match size {
                    Size::Byte => 0,
                    Size::Word => 1,
                    Size::Long => 2,
                };
                let ea_mode = if displacement.is_some() { 5 } else { 2 };
                let opcode = 0xD000
                    | ((dst as u16) << 9)
                    | (op_mode << 6)
                    | (ea_mode << 3)
                    | addr_reg as u16;
                let src = displacement
                    .map(|disp| JitEa::Disp(addr_reg as u8, disp))
                    .unwrap_or(JitEa::Ind(addr_reg as u8));
                let branch_pc = if displacement.is_some() {
                    0x0104
                } else {
                    0x0102
                };
                let branch_displacement = if displacement.is_some() { -6 } else { -4 };
                let ops = vec![
                    TraceBuildOp {
                        opcode,
                        extension: displacement.map(|disp| disp as u16),
                        extension2: None,
                        pc: 0x0100,
                        op: JitTraceOp::AluMemToReg {
                            op: JitBinaryOp::Add,
                            size,
                            src,
                            dst: dst as u8,
                        },
                    },
                    TraceBuildOp {
                        opcode: 0x6000 | branch_displacement as u8 as u16,
                        extension: None,
                        extension2: None,
                        pc: branch_pc,
                        op: JitTraceOp::Branch {
                            condition: 0,
                            displacement: branch_displacement,
                            length: 2,
                            expected_taken: None,
                        },
                    },
                ];

                let mut expected = cpu();
                expected.set_cpu_type(cpu_type);
                expected.set_d(dst, initial);
                expected.set_ccr(0x1F);
                let mut unused_bus = super::super::memory::LinearMemoryBus::new(2);
                let (result, _) =
                    expected.exec_add(&mut unused_bus, size, src_value, initial & size.mask());
                expected.set_d(dst, (initial & !size.mask()) | result);

                let mut actual = cpu();
                let mut mem = vec![0u8; 0x1000];
                let address = 0x0200usize + displacement.unwrap_or(0) as usize;
                match size {
                    Size::Byte => mem[address] = src_value as u8,
                    Size::Word => {
                        mem[address..address + 2]
                            .copy_from_slice(&(src_value as u16).to_be_bytes());
                    }
                    Size::Long => {
                        mem[address..address + 4].copy_from_slice(&src_value.to_be_bytes());
                    }
                }
                attach_window(&mut actual, &mut mem);
                actual.set_cpu_type(cpu_type);
                actual.set_a(addr_reg, 0x0200);
                actual.set_d(dst, initial);
                actual.set_ccr(0x1F);
                let mut jit = TraceJit::new();
                let compiled = jit
                    .compile_decoded_ops(&actual, 0x0100, cpu_type, ops, Some(0x0100))
                    .expect("native ADD loop should compile");
                let packed = unsafe { compiled.call_native(&mut actual, 1) };

                assert_eq!((packed >> 32) as u32, 2, "{cpu_type:?} {size:?}");
                assert_eq!(actual.d(dst), expected.d(dst), "{cpu_type:?} {size:?}");
                assert_eq!(
                    actual.get_ccr(),
                    expected.get_ccr(),
                    "{cpu_type:?} {size:?} flags"
                );
                assert_eq!(actual.pc, 0x0100);
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_sub_displacement_matches_interpreter_result() {
        for cpu_type in [CpuType::M68000, CpuType::M68040] {
            let mut expected = cpu();
            expected.set_cpu_type(cpu_type);
            expected.set_d(4, 0xA5A5_8000);
            expected.set_ccr(0x1F);
            let mut unused_bus = super::super::memory::LinearMemoryBus::new(2);
            let (result, _) = expected.exec_sub(&mut unused_bus, Size::Word, 1, 0x8000);
            expected.set_d(4, 0xA5A5_0000 | result);

            let mut actual = cpu();
            let mut mem = vec![0u8; 0x1000];
            mem[0x0210..0x0212].copy_from_slice(&1u16.to_be_bytes());
            attach_window(&mut actual, &mut mem);
            actual.set_cpu_type(cpu_type);
            actual.set_a(5, 0x0200);
            actual.set_d(4, 0xA5A5_8000);
            actual.set_ccr(0x1F);
            let ops = vec![
                TraceBuildOp {
                    opcode: 0x986D,
                    extension: Some(0x0010),
                    extension2: None,
                    pc: 0x0100,
                    op: JitTraceOp::AluMemToReg {
                        op: JitBinaryOp::Sub,
                        size: Size::Word,
                        src: JitEa::Disp(5, 0x0010),
                        dst: 4,
                    },
                },
                TraceBuildOp {
                    opcode: 0x60FA,
                    extension: None,
                    extension2: None,
                    pc: 0x0104,
                    op: JitTraceOp::Branch {
                        condition: 0,
                        displacement: -6,
                        length: 2,
                        expected_taken: None,
                    },
                },
            ];
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(&actual, 0x0100, cpu_type, ops, Some(0x0100))
                .expect("native SUB loop should compile");
            let packed = unsafe { compiled.call_native(&mut actual, 1) };

            assert_eq!((packed >> 32) as u32, 2, "{cpu_type:?}");
            assert_eq!(actual.d(4), expected.d(4), "{cpu_type:?}");
            assert_eq!(actual.get_ccr(), expected.get_ccr(), "{cpu_type:?} flags");
            assert_eq!(actual.pc, 0x0100);
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn indirect_jsr_profitability_threshold_is_enforced() {
        let ops_for = |count: usize| {
            let mut ops = Vec::new();
            for index in 0..count - 1 {
                let reg = (index & 7) as u8;
                ops.push(TraceBuildOp {
                    opcode: 0x7001 | (u16::from(reg) << 9),
                    extension: None,
                    extension2: None,
                    pc: 0x0100 + index as u32 * 2,
                    op: JitTraceOp::Moveq { reg, data: 1 },
                });
            }
            ops.push(TraceBuildOp {
                opcode: 0x4E90,
                extension: None,
                extension2: None,
                pc: 0x0100 + (count - 1) as u32 * 2,
                op: JitTraceOp::IndirectJsr { reg: 0 },
            });
            ops
        };

        let compile_cpu = cpu();
        let mut jit = TraceJit::new();
        assert!(
            jit.compile_decoded_ops(
                &compile_cpu,
                0x0100,
                CpuType::M68000,
                ops_for(TRACE_MIN_INDIRECT_JSR_OPS - 1),
                Some(0x0340),
            )
            .is_none(),
            "six-op indirect-call region should remain decoded"
        );
        assert!(
            jit.compile_decoded_ops(
                &compile_cpu,
                0x0100,
                CpuType::M68000,
                ops_for(TRACE_MIN_INDIRECT_JSR_OPS),
                Some(0x0340),
            )
            .is_some(),
            "seven-op indirect-call region should compile"
        );
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_indirect_jsr_commits_only_after_stack_check() {
        let ops = vec![
            TraceBuildOp {
                opcode: 0x7201,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::Moveq { reg: 1, data: 1 },
            },
            TraceBuildOp {
                opcode: 0x7402,
                extension: None,
                extension2: None,
                pc: 0x0102,
                op: JitTraceOp::Moveq { reg: 2, data: 2 },
            },
            TraceBuildOp {
                opcode: 0x7603,
                extension: None,
                extension2: None,
                pc: 0x0104,
                op: JitTraceOp::Moveq { reg: 3, data: 3 },
            },
            TraceBuildOp {
                opcode: 0x7804,
                extension: None,
                extension2: None,
                pc: 0x0106,
                op: JitTraceOp::Moveq { reg: 4, data: 4 },
            },
            TraceBuildOp {
                opcode: 0x7A05,
                extension: None,
                extension2: None,
                pc: 0x0108,
                op: JitTraceOp::Moveq { reg: 5, data: 5 },
            },
            TraceBuildOp {
                opcode: 0xDE6D,
                extension: Some(0x0010),
                extension2: None,
                pc: 0x010A,
                op: JitTraceOp::AluMemToReg {
                    op: JitBinaryOp::Add,
                    size: Size::Word,
                    src: JitEa::Disp(5, 0x0010),
                    dst: 7,
                },
            },
            TraceBuildOp {
                opcode: 0x4E90,
                extension: None,
                extension2: None,
                pc: 0x010E,
                op: JitTraceOp::IndirectJsr { reg: 0 },
            },
        ];
        let mut compile_cpu = cpu();
        compile_cpu.set_a(0, 0x0340);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&compile_cpu, 0x0100, CpuType::M68000, ops, Some(0x0340))
            .expect("indirect JSR region should compile");

        let prepare = |stack: u32| {
            let mut cpu = cpu();
            let mut mem = vec![0u8; 0x1000];
            mem[0x0210..0x0212].copy_from_slice(&1u16.to_be_bytes());
            attach_window(&mut cpu, &mut mem);
            cpu.set_a(0, 0x0340);
            cpu.set_a(5, 0x0200);
            cpu.set_a(7, stack);
            cpu.set_d(7, 0xA5A5_7FFF);
            cpu.set_ccr(0x1F);
            cpu.change_of_flow = false;
            (cpu, mem)
        };

        let (mut success, success_mem) = prepare(0x0800);
        let packed = unsafe { compiled.call_native(&mut success, 1) };
        assert_eq!((packed >> 32) as u32, 7);
        assert_eq!(packed as u32 as i32, 60);
        assert_eq!(success.d(1), 1);
        assert_eq!(success.d(7), 0xA5A5_8000);
        assert_eq!(success.a(7), 0x07FC);
        assert_eq!(&success_mem[0x07FC..0x0800], &0x0110u32.to_be_bytes());
        assert_eq!(success.pc, 0x0340);
        assert_eq!(success.ppc, 0x010E);
        assert_eq!(success.ir, 0x4E90);
        assert!(success.change_of_flow);

        let (mut bail, bail_mem) = prepare(2);
        let packed = unsafe { compiled.call_native(&mut bail, 1) };
        assert_eq!((packed >> 32) as u32, 6);
        assert_eq!(packed as u32 as i32, 44);
        assert_eq!(bail.d(1), 1, "prefix remains committed");
        assert_eq!(bail.d(7), 0xA5A5_8000, "prefix remains committed");
        assert_eq!(bail.a(7), 2, "call itself did not commit");
        assert_eq!(bail.pc, 0x010E, "retry the unexecuted call");
        assert!(!bail.change_of_flow);
        assert!(bail_mem[0x07FC..0x0800].iter().all(|&byte| byte == 0));
    }

    #[test]
    fn portable_trace_executes_displacement_memory_mix() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x10000];
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(5, 0x1000);
        cpu.set_a(7, 0x8000);
        cpu.set_d(0, 0x34);
        mem[0x1100] = 0x08;

        let ops = [
            TraceBuildOp {
                opcode: 0x4A2D,
                extension: Some(0x0100),
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::AnDispUnary {
                    op: JitUnaryOp::Tst,
                    size: Size::Byte,
                    reg: 5,
                    displacement: 0x0100,
                },
            },
            TraceBuildOp {
                opcode: 0x082D,
                extension: Some(3),
                extension2: Some(0x0100),
                pc: 0x0104,
                op: JitTraceOp::AnDispBit {
                    op: JitBitOp::Test,
                    bit: JitBitSource::Imm(3),
                    reg: 5,
                    displacement: 0x0100,
                },
            },
            TraceBuildOp {
                opcode: 0x1B40,
                extension: Some(0x0100),
                extension2: None,
                pc: 0x010A,
                op: JitTraceOp::MoveMem {
                    size: Size::Byte,
                    src: JitEa::Data(0),
                    dst: JitEa::Disp(5, 0x0100),
                },
            },
            TraceBuildOp {
                opcode: 0x422D,
                extension: Some(0x0100),
                extension2: None,
                pc: 0x010E,
                op: JitTraceOp::AnDispUnary {
                    op: JitUnaryOp::Clr,
                    size: Size::Byte,
                    reg: 5,
                    displacement: 0x0100,
                },
            },
            TraceBuildOp {
                opcode: 0x322D,
                extension: Some(0x0100),
                extension2: None,
                pc: 0x0112,
                op: JitTraceOp::MoveMem {
                    size: Size::Word,
                    src: JitEa::Disp(5, 0x0100),
                    dst: JitEa::Data(1),
                },
            },
            TraceBuildOp {
                opcode: 0x526D,
                extension: Some(0x0100),
                extension2: None,
                pc: 0x0116,
                op: JitTraceOp::AnDispAddqSubq {
                    data: 1,
                    size: Size::Word,
                    reg: 5,
                    displacement: 0x0100,
                    is_sub: false,
                },
            },
            TraceBuildOp {
                opcode: 0x2F2D,
                extension: Some(0x0100),
                extension2: None,
                pc: 0x011A,
                op: JitTraceOp::MoveMem {
                    size: Size::Long,
                    src: JitEa::Disp(5, 0x0100),
                    dst: JitEa::PreDec(7),
                },
            },
            TraceBuildOp {
                opcode: 0x588F,
                extension: None,
                extension2: None,
                pc: 0x011E,
                op: JitTraceOp::AddqSubqAddr {
                    reg: 7,
                    data: 4,
                    is_sub: false,
                },
            },
            TraceBuildOp {
                opcode: 0x60DE,
                extension: None,
                extension2: None,
                pc: 0x0120,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -34,
                    length: 2,
                    expected_taken: None,
                },
            },
        ];
        // The portable memory helpers read the live extension words just as
        // the native trace validates them before executing.
        for op in ops {
            let at = op.pc as usize;
            mem[at..at + 2].copy_from_slice(&op.opcode.to_be_bytes());
            if let Some(extension) = op.extension {
                mem[at + 2..at + 4].copy_from_slice(&extension.to_be_bytes());
            }
            if let Some(extension) = op.extension2 {
                mem[at + 4..at + 6].copy_from_slice(&extension.to_be_bytes());
            }
        }

        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0122);

        assert_eq!((packed >> 32) as usize, ops.len());
        assert_eq!(cpu.pc, 0x0100);
        assert_eq!(cpu.d(1) & 0xFFFF, 0);
        assert_eq!(&mem[0x1100..0x1102], &1u16.to_be_bytes());
        assert_eq!(&mem[0x7FFC..0x8000], &0x0001_0000u32.to_be_bytes());
        assert_eq!(cpu.a(7), 0x8000);
    }

    #[test]
    fn portable_trace_executes_unconditional_loop_iteration() {
        let mut cpu = cpu();
        let ops = [
            TraceBuildOp {
                opcode: 0x5280,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::AddqSubqReg {
                    reg: 0,
                    data: 1,
                    size: Size::Long,
                    is_sub: false,
                },
            },
            TraceBuildOp {
                opcode: 0x60FC,
                extension: None,
                extension2: None,
                pc: 0x0102,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -4,
                    length: 2,
                    expected_taken: None,
                },
            },
        ];

        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0100 + ops.len() as u32 * 2);
        let cycles = packed as u32 as i32;
        assert_eq!((packed >> 32) as u32, ops.len() as u32);

        assert_eq!(cycles, 18);
        assert_eq!(cpu.d(0), 1);
        assert_eq!(cpu.pc, 0x0100);
        assert_eq!(cpu.ppc, 0x0102);
        assert_eq!(cpu.ir, 0x60FC);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_self_loop_batches_iterations_and_accumulates_progress() {
        let ops = vec![
            TraceBuildOp {
                opcode: 0x5280,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::AddqSubqReg {
                    reg: 0,
                    data: 1,
                    size: Size::Long,
                    is_sub: false,
                },
            },
            TraceBuildOp {
                opcode: 0x60FC,
                extension: None,
                extension2: None,
                pc: 0x0102,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -4,
                    length: 2,
                    expected_taken: None,
                },
            },
        ];
        let mut actual = cpu();
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, 0x0100, CpuType::M68000, ops, Some(0x0100))
            .expect("native self-loop should compile");

        let packed = unsafe { compiled.call_native(&mut actual, 5) };

        assert_eq!((packed >> 32) as u32, 10);
        assert_eq!(packed as u32, 90);
        assert_eq!(actual.d(0), 5);
        assert_eq!(actual.pc, 0x0100);
    }

    #[test]
    fn portable_trace_uses_flags_for_conditional_branch() {
        let mut cpu = cpu();
        cpu.set_d(0, 1);
        let ops = [
            TraceBuildOp {
                opcode: 0x5340,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::AddqSubqReg {
                    reg: 0,
                    data: 1,
                    size: Size::Word,
                    is_sub: true,
                },
            },
            TraceBuildOp {
                opcode: 0x66FC,
                extension: None,
                extension2: None,
                pc: 0x0102,
                op: JitTraceOp::Branch {
                    condition: 6,
                    displacement: -4,
                    length: 2,
                    expected_taken: None,
                },
            },
        ];

        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0100 + ops.len() as u32 * 2);
        let cycles = packed as u32 as i32;
        assert_eq!((packed >> 32) as u32, ops.len() as u32);

        assert_eq!(cycles, 12);
        assert_eq!(cpu.d(0), 0);
        assert!(cpu.flag_z());
        assert_eq!(cpu.pc, 0x0104);
        assert_eq!(cpu.ppc, 0x0102);
        assert_eq!(cpu.ir, 0x66FC);
    }

    #[test]
    fn portable_trace_guard_mismatch_exits_before_later_ops() {
        let mut cpu = cpu();
        cpu.set_d(0, 1);
        let ops = [
            TraceBuildOp {
                opcode: 0x5340,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::AddqSubqReg {
                    reg: 0,
                    data: 1,
                    size: Size::Word,
                    is_sub: true,
                },
            },
            TraceBuildOp {
                opcode: 0x66FC,
                extension: None,
                extension2: None,
                pc: 0x0102,
                op: JitTraceOp::Branch {
                    condition: 6,
                    displacement: -4,
                    length: 2,
                    expected_taken: Some(true),
                },
            },
            TraceBuildOp {
                opcode: 0x5281,
                extension: None,
                extension2: None,
                pc: 0x0104,
                op: JitTraceOp::AddqSubqReg {
                    reg: 1,
                    data: 1,
                    size: Size::Long,
                    is_sub: false,
                },
            },
        ];

        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0106);

        assert_eq!((packed >> 32) as u32, 2);
        assert_eq!(packed as u32 as i32, 12);
        assert_eq!(cpu.d(0), 0);
        assert_eq!(cpu.d(1), 0);
        assert_eq!(cpu.pc, 0x0104);
        assert_eq!(cpu.ppc, 0x0102);
        assert_eq!(cpu.ir, 0x66FC);
    }

    #[test]
    fn portable_trace_executes_register_shift() {
        let mut cpu = cpu();
        cpu.set_d(0, 0x8000_0001);
        let ops = [TraceBuildOp {
            opcode: 0xE188,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::ShiftReg {
                reg: 0,
                size: Size::Long,
                count_or_reg: 0,
                count_is_register: false,
                direction: 1,
                op: 1,
            },
        }];

        let packed = execute_portable_trace(&mut cpu, &ops, 0x0100, 0x0100 + ops.len() as u32 * 2);
        let cycles = packed as u32 as i32;
        assert_eq!((packed >> 32) as u32, ops.len() as u32);

        assert_eq!(cycles, 24);
        assert_eq!(cpu.d(0), 0x0000_0100);
        assert_eq!(cpu.ppc, 0x0100);
        assert_eq!(cpu.ir, 0xE188);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_trace_accepts_only_supported_immediate_shift_forms() {
        let asr = DecodedSimpleOp::decode(CpuType::M68040, 0xE247)
            .unwrap()
            .to_jit_trace_op();
        assert!(matches!(
            asr,
            Some(JitTraceOp::ShiftReg {
                reg: 7,
                size: Size::Word,
                count_or_reg: 1,
                count_is_register: false,
                direction: 0,
                op: 0,
            })
        ));

        let immediate_asl = DecodedSimpleOp::decode(CpuType::M68040, 0xE347)
            .unwrap()
            .to_jit_trace_op();
        assert!(immediate_asl.is_none());

        let immediate_lsl = DecodedSimpleOp::decode(CpuType::M68040, 0xE788)
            .unwrap()
            .to_jit_trace_op();
        assert!(matches!(
            immediate_lsl,
            Some(JitTraceOp::ShiftReg {
                reg: 0,
                size: Size::Long,
                count_or_reg: 3,
                count_is_register: false,
                direction: 1,
                op: 1,
            })
        ));

        let register_asr = DecodedSimpleOp::decode(CpuType::M68040, 0xE267)
            .unwrap()
            .to_jit_trace_op();
        assert!(register_asr.is_none());
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_immediate_asr_matches_interpreter() {
        let cases = [
            (Size::Byte, 1u8, 0xA5A5_5581u32),
            (Size::Byte, 7, 0xA5A5_557Fu32),
            (Size::Byte, 0, 0xA5A5_5580u32), // encoded zero means eight
            (Size::Word, 1, 0xA5A5_8001u32),
            (Size::Word, 4, 0xA5A5_7FF0u32),
            (Size::Word, 0, 0xA5A5_8100u32),
            (Size::Long, 1, 0x8000_0001u32),
            (Size::Long, 5, 0x7FFF_FFE0u32),
            (Size::Long, 0, 0x8100_0080u32),
        ];

        for cpu_type in [CpuType::M68000, CpuType::M68040] {
            for (size, encoded_count, initial) in cases {
                let shift = if encoded_count == 0 {
                    8
                } else {
                    u32::from(encoded_count)
                };
                let size_code = match size {
                    Size::Byte => 0,
                    Size::Word => 1,
                    Size::Long => 2,
                };
                let opcode = 0xE000 | (u16::from(encoded_count) << 9) | (size_code << 6) | 7;
                let ops = vec![
                    TraceBuildOp {
                        opcode,
                        extension: None,
                        extension2: None,
                        pc: 0x0100,
                        op: JitTraceOp::ShiftReg {
                            reg: 7,
                            size,
                            count_or_reg: encoded_count,
                            count_is_register: false,
                            direction: 0,
                            op: 0,
                        },
                    },
                    TraceBuildOp {
                        opcode: 0x60FC,
                        extension: None,
                        extension2: None,
                        pc: 0x0102,
                        op: JitTraceOp::Branch {
                            condition: 0,
                            displacement: -4,
                            length: 2,
                            expected_taken: None,
                        },
                    },
                ];

                let mut expected = cpu();
                expected.set_cpu_type(cpu_type);
                expected.set_d(7, initial);
                expected.set_ccr(0x1F);
                let (result, shift_cycles) = expected.exec_asr(size, shift, initial & size.mask());
                expected.set_d(7, (initial & !size.mask()) | result);

                let mut actual = cpu();
                actual.set_cpu_type(cpu_type);
                actual.set_d(7, initial);
                actual.set_ccr(0x1F);
                let mut jit = TraceJit::new();
                let compiled = jit
                    .compile_decoded_ops(&actual, 0x0100, cpu_type, ops, Some(0x0100))
                    .expect("native ASR loop should compile");
                let packed = unsafe { compiled.call_native(&mut actual, 1) };

                assert_eq!((packed >> 32) as u32, 2, "{cpu_type:?} {size:?} #{shift}");
                assert_eq!(
                    packed as u32 as i32,
                    shift_cycles + 10,
                    "{cpu_type:?} {size:?} #{shift} cycles"
                );
                assert_eq!(actual.d(7), expected.d(7), "{cpu_type:?} {size:?} #{shift}");
                assert_eq!(
                    actual.get_ccr(),
                    expected.get_ccr(),
                    "{cpu_type:?} {size:?} #{shift} flags"
                );
                assert_eq!(actual.pc, 0x0100);
                assert_eq!(actual.ppc, 0x0102);
                assert_eq!(actual.ir, 0x60FC);
            }
        }
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn native_immediate_lsl_matches_interpreter() {
        let cases = [
            (Size::Byte, 1u8, 0xA5A5_5581u32),
            (Size::Byte, 0, 0xA5A5_5580u32), // encoded zero means eight
            (Size::Word, 4, 0xA5A5_1801u32),
            (Size::Word, 0, 0xA5A5_8180u32),
            (Size::Long, 3, 0x9000_0001u32),
            (Size::Long, 0, 0x0180_0081u32),
        ];

        for cpu_type in [CpuType::M68000, CpuType::M68040] {
            for (size, encoded_count, initial) in cases {
                let shift = if encoded_count == 0 {
                    8
                } else {
                    u32::from(encoded_count)
                };
                let size_code = match size {
                    Size::Byte => 0,
                    Size::Word => 1,
                    Size::Long => 2,
                };
                let opcode = 0xE108 | (u16::from(encoded_count) << 9) | (size_code << 6);
                let ops = vec![
                    TraceBuildOp {
                        opcode,
                        extension: None,
                        extension2: None,
                        pc: 0x0100,
                        op: JitTraceOp::ShiftReg {
                            reg: 0,
                            size,
                            count_or_reg: encoded_count,
                            count_is_register: false,
                            direction: 1,
                            op: 1,
                        },
                    },
                    TraceBuildOp {
                        opcode: 0x60FC,
                        extension: None,
                        extension2: None,
                        pc: 0x0102,
                        op: JitTraceOp::Branch {
                            condition: 0,
                            displacement: -4,
                            length: 2,
                            expected_taken: None,
                        },
                    },
                ];

                let mut expected = cpu();
                expected.set_cpu_type(cpu_type);
                expected.set_d(0, initial);
                expected.set_ccr(0x1F);
                let (result, shift_cycles) = expected.exec_lsl(size, shift, initial & size.mask());
                expected.set_d(0, (initial & !size.mask()) | result);

                let mut actual = cpu();
                actual.set_cpu_type(cpu_type);
                actual.set_d(0, initial);
                actual.set_ccr(0x1F);
                let mut jit = TraceJit::new();
                let compiled = jit
                    .compile_decoded_ops(&actual, 0x0100, cpu_type, ops, Some(0x0100))
                    .expect("native LSL loop should compile");
                let packed = unsafe { compiled.call_native(&mut actual, 1) };

                assert_eq!((packed >> 32) as u32, 2, "{cpu_type:?} {size:?} #{shift}");
                assert_eq!(
                    packed as u32 as i32,
                    shift_cycles + 10,
                    "{cpu_type:?} {size:?} #{shift} cycles"
                );
                assert_eq!(actual.d(0), expected.d(0), "{cpu_type:?} {size:?} #{shift}");
                assert_eq!(
                    actual.get_ccr(),
                    expected.get_ccr(),
                    "{cpu_type:?} {size:?} #{shift} flags"
                );
                assert_eq!(actual.pc, 0x0100);
                assert_eq!(actual.ppc, 0x0102);
                assert_eq!(actual.ir, 0x60FC);
            }
        }
    }
}

//! Trace execution for hot simple-op loops.
//!
//! Native targets with the `jit` feature lower hot traces to Cranelift machine code. Other builds
//! keep the same trace detection and validation path, but execute the trace through a compact Rust
//! micro-op loop.

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use super::cpu::CFLAG_SET;
use super::cpu::{CpuCore, NFLAG_SET, VFLAG_SET};
use super::execute::RUN_MODE_BERR_AERR_RESET;
use super::mem_ops::{BitSource, DecodedMemOp, FastEa};
use super::memory::AddressBus;
use super::op_cache::{AddrOp, BinaryOp, BitOp, CachedRunResult, DecodedSimpleOp, is_pre_68020};
use super::types::{CpuType, Size};
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use cranelift_codegen::Context;
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use cranelift_codegen::ir::{
    AbiParam, Block, BlockArg, Function, InstBuilder, MemFlags, Type, UserFuncName, Value,
    condcodes::IntCC, types,
};
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use cranelift_module::{Linkage, Module, default_libcall_names};
use std::cell::{Cell, RefCell};
use std::fmt;
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicBool, Ordering};

const TRACE_CACHE_SIZE: usize = 4096;
pub(crate) const TRACE_MAX_OPS: usize = 128;
pub(crate) const TRACE_MIN_OPS: usize = 3;

/// Minimum recorded ops for a blocked recording to be worth salvaging as
/// a trimmed region. Higher than `TRACE_MIN_OPS`: a salvaged trace exits
/// to the interpreter at its trimmed terminal every pass, so short
/// fragments would pay entry/exit costs without covering enough work.
const SALVAGE_MIN_OPS: usize = 8;
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
/// Original one-pass compiled trace entry point.
type TraceOnceFn = unsafe extern "C" fn(*mut CpuCore) -> u64;

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
/// Counted self-loop entry point. Keeping repeated guest iterations inside
/// generated code avoids an ABI round trip for every tiny loop.
type TraceLoopFn = unsafe extern "C" fn(*mut CpuCore, u32) -> u64;

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
    /// Absolute-short memory address, sign-extended when the trace is
    /// recorded so execution does not read the instruction stream.
    AbsWord(u32),
    /// Absolute-long memory address captured from the two extension words.
    AbsLong(u32),
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
                | Self::AbsWord(_)
                | Self::AbsLong(_)
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
    /// ORI/ANDI/SUBI/ADDI/EORI/CMPI with a data-register destination. The
    /// immediate extension words are captured while recording.
    BinaryImmediateDataReg {
        op: JitBinaryOp,
        immediate: u32,
        dst: u8,
        size: Size,
        cycles: i32,
    },
    /// MULU.W/MULS.W with a data-register source. Memory-source forms retain
    /// the checked interpreter path.
    MulWordDataReg {
        src: u8,
        dst: u8,
        signed: bool,
        m68000_timing: bool,
    },
    /// `MULU.W`/`MULS.W #imm,Dn`. The multiplicand is an extension word
    /// rather than a register, so it is captured once while recording.
    MulWordImmediate {
        immediate: u16,
        dst: u8,
        signed: bool,
        m68000_timing: bool,
    },
    /// 68020+ MULU.L/MULS.L with a data-register source and the 32-bit
    /// result form. The 64-bit Dh:Dl form retains the interpreter path.
    MulLongDataReg {
        src: u8,
        dst: u8,
        signed: bool,
    },
    AddrDataReg {
        op: JitAddrOp,
        src: JitDirectReg,
        dst: u8,
        size: Size,
    },
    /// CMPA.W/L with an immediate source captured while recording. CMPA.W
    /// sign-extends its word before comparing with the full address register.
    AddrCmpImmediate {
        immediate: u32,
        dst: u8,
        size: Size,
        cycles: i32,
    },
    /// LEA `(An)`/`d16(An),Am`. The displacement is captured while recording,
    /// so execution is pure address-register arithmetic.
    LeaAn {
        base: u8,
        dst: u8,
        displacement: i16,
        cycles: i32,
    },
    /// `LEA (d8,An,Xn),An` with the brief extension decoded once while
    /// recording. LEA is register-only: it accesses no memory and changes
    /// no condition codes.
    LeaIndex {
        src: JitEa,
        dst: u8,
        cycles: i32,
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
    #[cfg_attr(all(feature = "jit", not(target_family = "wasm")), allow(dead_code))]
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
    /// `MOVEM.W (An)+,<data-register-mask>`. Keeping this deliberately narrow
    /// avoids the architectural corner cases of address registers in a
    /// postincrement MOVEM list.
    MovemWordPostInc {
        base: u8,
        data_mask: u8,
        cycles: i32,
    },
    /// Read-only ALU operation from fast memory to a data register. The
    /// decoder admits measured CMP/ADD/SUB `(An)`/`d16(An)`/`d8(An,Xn)`
    /// sources.
    AluMemToReg {
        op: JitBinaryOp,
        size: Size,
        src: JitEa,
        dst: u8,
    },
    /// Read-only `CMPI.W #imm,d8(An,Xn)` through checked fast memory. Both
    /// extension words are captured while recording, so the trace performs
    /// only the live indexed-address calculation, load, and comparison.
    CmpiWordMem {
        immediate: u16,
        src: JitEa,
    },
    /// Read-only TST through a checked fast-memory effective address.
    TstMem {
        size: Size,
        src: JitEa,
    },
    /// CLR through a checked brief-indexed effective address. The store
    /// address passes the window and code-overlap checks before anything
    /// commits, so a bail leaves memory and flags untouched.
    ClrMem {
        size: Size,
        dst: JitEa,
    },
    /// MOVE of an immediate value to a memory destination through the
    /// checked window. No source registers, so no staging; a destination
    /// register update (postinc/predec) commits only after every guard.
    MoveImmMem {
        size: Size,
        value: u32,
        dst: JitEa,
    },
    /// CMPA.W/L through `(An)` or `d16(An)`. Unlike ordinary CMP, the
    /// destination is always the full address register and a word source is
    /// sign-extended before the 32-bit comparison.
    AddrCmpMemToReg {
        size: Size,
        src: JitEa,
        dst: u8,
    },
    /// ADD.W/L Dn,`<ea>` store/accumulate operations through measured writable
    /// address-register-relative forms.
    AddRegToMem {
        is_sub: bool,
        size: Size,
        src: u8,
        dst: JitEa,
    },
    /// Displacement-memory forms that require extension words, represented
    /// explicitly rather than through the register-only trace operations.
    AnDispUnary {
        op: JitUnaryOp,
        size: Size,
        reg: u8,
        displacement: i16,
    },
    /// `PEA (d16,An)`: push the effective address through the checked
    /// window. The address is computed from the pre-decrement stack pointer
    /// state, no condition code changes, and every check precedes the store
    /// and the A7 update so a bail commits nothing.
    PeaDisp {
        reg: u8,
        displacement: i16,
    },
    /// `LINK An,#d16`: push An through the checked window, point An at the
    /// pushed frame, then move SP by the displacement. No condition code
    /// changes; every check precedes the store and both register updates so
    /// a bail commits nothing. A7 forms are never admitted (the pushed
    /// value is generation-dependent for LINK A7).
    Link {
        reg: u8,
        displacement: i16,
    },
    /// `UNLK An`: reload SP from An and pop the saved frame pointer back
    /// into An through the checked window. The load is bounds/alignment
    /// checked before either register updates.
    Unlk {
        reg: u8,
    },
    /// ADDQ/SUBQ through a checked address-register-relative memory EA.
    MemAddqSubq {
        data: u32,
        size: Size,
        dst: JitEa,
        is_sub: bool,
    },
    AnDispBit {
        op: JitBitOp,
        bit: JitBitSource,
        reg: u8,
        displacement: i16,
    },
}

/// Why a compile-stage gate declined a recorded region.
///
/// Kept separate from the public profiling enum so the always-compiled
/// gate logic does not depend on the optional profiler; the mapping in
/// `profile_reject_reason` is exhaustive, so a new gate must declare how
/// it is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RegionRejectReason {
    /// The recorded self-loop is a pure poll: a wait, deliberately left
    /// uncompiled. Reported as a wait rather than as a silent rejection.
    WaitLoop,
    NoTraceTerminal,
    TooShort,
    IndirectJsrTooShort,
    LinearMemoryAlu,
    AddressWrap,
    Backend,
}

/// Why an in-progress recording was stopped from outside the recorder.
///
/// The distinction is load-bearing for reporting: a trap or exception is a
/// structural bound on how far a head can ever record, while a host
/// boundary is an artifact of how the embedder sliced execution and says
/// nothing about the head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecordingStop {
    /// A trap or exception surfaced. The embedder handles it and may resume
    /// at an unrelated guest PC, so no amount of instruction coverage
    /// extends a recording past this point.
    TrapOrException,
    /// The batch ended at a host boundary: instruction budget exhausted, a
    /// watched PC, a stopped CPU, or a decoded fast-path miss. The head may
    /// well record further in a later batch.
    HostBoundary,
}

/// How a recording ended, which decides whether the profiler has already
/// attributed it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RecordingEnd {
    /// Decoding refused an executed instruction. `note_blocker` has already
    /// recorded the head, its prefix, and the offending opcode.
    Blocker,
    /// Something outside the recorder stopped it mid-region.
    Stopped(RecordingStop),
    /// The region closed on its own terms: a back edge, a recorded branch,
    /// or an operation limit.
    Region,
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
    #[cfg_attr(all(feature = "jit", not(target_family = "wasm")), allow(dead_code))]
    self_loop: bool,
    /// The native body was generated as a counted loop. Short read/write
    /// MoveMem loops deliberately retain the original one-pass body: the
    /// extra loop-carried state costs more than the saved call boundary.
    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    native_loop: bool,
    /// Contains memory ops: only executable while a fastmem window is active
    /// (i.e. inside `run_batch`).
    needs_window: bool,
    /// Address-masked range of the trace's code bytes; trace stores into
    /// this range bail so self-modification is observed like the
    /// interpreter would. Baked into the compiled function on native
    /// targets; read at execution time by the portable path.
    #[cfg_attr(all(feature = "jit", not(target_family = "wasm")), allow(dead_code))]
    code_start: u32,
    #[cfg_attr(all(feature = "jit", not(target_family = "wasm")), allow(dead_code))]
    code_end: u32,
    /// A recorded interior branch is a path prediction eligible for adaptive
    /// rerecording. Cleared after the one allowed rerecord so completed traces
    /// stay off the accounting path.
    adaptive_branch: bool,
    adaptive_calls: Cell<u32>,
    adaptive_guard_exits: Cell<u32>,
    adaptive_rerecords: u8,
    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    func: NativeTraceFn,
}

impl CompiledTrace {
    #[cfg(all(feature = "jit", not(target_family = "wasm"), test))]
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
    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    module: Option<JITModule>,
    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    func_ctx: FunctionBuilderContext,
    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    next_func: u32,
    slots: Vec<TraceSlot>,
    recording: Option<TraceRecording>,
}

impl fmt::Debug for TraceJit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug = f.debug_struct("TraceJit");
        #[cfg(all(feature = "jit", not(target_family = "wasm")))]
        {
            debug.field("native_enabled", &self.module.is_some());
            debug.field("next_func", &self.next_func);
        }
        #[cfg(any(not(feature = "jit"), target_family = "wasm"))]
        {
            debug.field("native_enabled", &false);
        }
        debug.finish_non_exhaustive()
    }
}

impl TraceJit {
    fn new() -> Self {
        #[cfg(all(feature = "jit", not(target_family = "wasm")))]
        let module = JITBuilder::new(default_libcall_names())
            .ok()
            .map(JITModule::new);
        Self {
            #[cfg(all(feature = "jit", not(target_family = "wasm")))]
            module,
            #[cfg(all(feature = "jit", not(target_family = "wasm")))]
            func_ctx: FunctionBuilderContext::new(),
            #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
        #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
            #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
            #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
            #[cfg(any(not(feature = "jit"), target_family = "wasm"))]
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
        #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    #[cfg(feature = "trace-profile")]
    fn profile_reject_reason(
        reason: RegionRejectReason,
    ) -> super::trace_profile::TraceRejectReason {
        use super::trace_profile::TraceRejectReason as Public;
        // Exhaustive so a new compile gate must declare how it is reported.
        match reason {
            // A classified wait is routed to `note_wait_loop` by
            // `finish_recording` before this mapping is consulted; the
            // profiler represents it through the shape-keyed wait table
            // flag and the wait-loops report section, never as a silent
            // rejection.
            RegionRejectReason::WaitLoop => {
                unreachable!("classified waits are reported via note_wait_loop")
            }
            RegionRejectReason::NoTraceTerminal => Public::NoTraceTerminal,
            RegionRejectReason::TooShort => Public::TooShort,
            RegionRejectReason::IndirectJsrTooShort => Public::IndirectJsrTooShort,
            RegionRejectReason::LinearMemoryAlu => Public::LinearMemoryAlu,
            RegionRejectReason::AddressWrap => Public::AddressWrap,
            RegionRejectReason::Backend => Public::Backend,
        }
    }

    /// Exhaustive so a new stop cause must declare how it is reported.
    #[cfg(feature = "trace-profile")]
    fn profile_stop_reason(cause: RecordingStop) -> super::trace_profile::TraceRejectReason {
        use super::trace_profile::TraceRejectReason as Public;
        match cause {
            RecordingStop::TrapOrException => Public::TrapOrException,
            RecordingStop::HostBoundary => Public::HostBoundary,
        }
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

    #[cfg_attr(not(feature = "trace-profile"), allow(unused_variables))]
    fn finish_recording(&mut self, cpu: &mut CpuCore, mut exit_pc: u32, end: RecordingEnd) {
        let Some(mut recording) = self.recording.take() else {
            cpu.trace_recording = false;
            return;
        };
        cpu.trace_recording = false;

        // A recording stopped by an instruction the decoder refuses ends
        // without a trace terminal, so compilation would reject the whole
        // region. If the prefix already crossed a recorded branch, trim
        // back to the last one and compile that instead: partial coverage
        // of a long region beats none, and the guest re-enters full
        // dispatch exactly where the trimmed tail began (the recorded op
        // after the terminal names the true continuation, whichever way
        // the branch went). Ops are execution-ordered, so the trimmed
        // trace revalidates and runs exactly what the guest executed.
        if end == RecordingEnd::Blocker
            && !recording.ops.last().is_some_and(|op| op.op.ends_trace())
            && let Some(last_terminal) = recording.ops.iter().rposition(|op| op.op.ends_trace())
            && last_terminal + 1 >= SALVAGE_MIN_OPS
        {
            exit_pc = recording.ops[last_terminal + 1].pc;
            recording.ops.truncate(last_terminal + 1);
        }

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
        let recorded_shape = recording
            .ops
            .iter()
            .map(|op| super::trace_profile::TraceShapeOp {
                pc: op.pc,
                opcode: op.opcode,
                extension: op.extension,
                extension2: op.extension2,
            })
            .collect();
        let outcome =
            self.compile_decoded_ops_reason(cpu, start_pc, cpu_type, recording.ops, Some(exit_pc));
        #[cfg(feature = "trace-profile")]
        let reject_reason = outcome.as_ref().err().copied();
        self.slots[idx] = match outcome {
            Ok(mut trace) => {
                trace.adaptive_rerecords = adaptive_rerecords;
                if adaptive_rerecords >= TRACE_MAX_ADAPTIVE_RERECORDS {
                    trace.adaptive_branch = false;
                }
                TraceSlot::Compiled(trace)
            }
            Err(_) => {
                push_probe_skip(cpu, start_pc);
                TraceSlot::Rejected {
                    pc: start_pc,
                    cpu_type,
                }
            }
        };
        #[cfg(feature = "trace-profile")]
        if matches!(self.slots[idx], TraceSlot::Compiled(_)) {
            super::trace_profile::note_compiled(start_pc, cpu_type, recorded_shape);
        } else if let Some(reason) = reject_reason {
            // A recording that ends at an unsupported opcode is already
            // attributed by `note_blocker`; anything else would leave no
            // report entry at all, so attribute it here. Leaving the
            // decoded path outranks the compile-stage reason: it bounds how
            // far this head can ever record, which no opcode coverage can
            // change.
            if reason == RegionRejectReason::WaitLoop {
                // A classified wait is accounted as a wait, keyed by the
                // recorded shape so the accounting is independent of any
                // blocker shapes at the same head, in either recording
                // order. It keeps its own report section and its exclusion
                // from the opportunity ranking, rather than being filed as
                // a generic rejection.
                super::trace_profile::note_wait_loop(start_pc, cpu_type, recorded_shape);
            } else if end != RecordingEnd::Blocker {
                let reason = match end {
                    RecordingEnd::Stopped(cause) => Self::profile_stop_reason(cause),
                    _ => Self::profile_reject_reason(reason),
                };
                // Report the instruction that stopped the recording. For a
                // trap `pc` has already advanced past its opcode word, so
                // `ppc` is the actionable address. This is reporting only:
                // `exit_pc` still drives self-loop detection above.
                let reported_pc = match end {
                    RecordingEnd::Stopped(RecordingStop::TrapOrException) => cpu.ppc,
                    _ => exit_pc,
                };
                // Name the instruction that stopped the recording. For a trap
                // this is the A-line word identifying the Toolbox or OS call,
                // which decides whether a trace could ever be compiled through
                // it. Window-only read: the profiler must not add bus traffic.
                let reported_opcode = super::mem_ops::peek_window_word(cpu, reported_pc);
                super::trace_profile::note_silent_rejection(
                    start_pc,
                    cpu_type,
                    reported_pc,
                    reported_opcode,
                    recorded_shape,
                    reason,
                );
            }
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
            {
                // Diagnostics read through the fastmem window only. The
                // profiler must not add bus transactions: on buses where a
                // read is host-visible (MMIO, watchpoints, fault counters),
                // extra reads would change guest-observable behavior.
                let memory_opcode = super::mem_ops::peek_window_word(cpu, executed_pc);
                let next_word = super::mem_ops::peek_window_word(cpu, executed_pc.wrapping_add(2));
                let next_word2 = super::mem_ops::peek_window_word(cpu, executed_pc.wrapping_add(4));
                let prefix = recording
                    .ops
                    .iter()
                    .map(|op| super::trace_profile::TraceShapeOp {
                        pc: op.pc,
                        opcode: op.opcode,
                        extension: op.extension,
                        extension2: op.extension2,
                    })
                    .collect();
                super::trace_profile::note_blocker(
                    start_pc,
                    cpu_type,
                    prefix,
                    super::trace_profile::TraceBlocker {
                        pc: executed_pc,
                        executed_opcode: cpu.ir as u16,
                        memory_opcode,
                        next_word,
                        next_word2,
                    },
                );
            }
            self.finish_recording(cpu, executed_pc, RecordingEnd::Blocker);
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
                self.finish_recording(cpu, next_pc, RecordingEnd::Region);
                return;
            }
            JitTraceOp::IndirectJsr { .. } => {
                self.recording.as_mut().unwrap().ops.push(op);
                self.finish_recording(cpu, next_pc, RecordingEnd::Region);
                return;
            }
            _ if next_pc != executed_pc.wrapping_add(op_len as u32) => {
                self.finish_recording(cpu, executed_pc, RecordingEnd::Region);
                return;
            }
            _ => {}
        }

        let recording = self.recording.as_mut().unwrap();
        recording.ops.push(op);
        if next_pc == start_pc {
            self.finish_recording(cpu, next_pc, RecordingEnd::Region);
            return;
        }

        let repeated = recording.ops.iter().any(|op| op.pc == next_pc);
        if recording.ops.len() >= TRACE_MAX_OPS || repeated {
            self.finish_recording(cpu, next_pc, RecordingEnd::Region);
        }
    }

    /// Outcome-only wrapper over [`Self::compile_decoded_ops_reason`], kept
    /// for the many tests that assert only whether a region compiled.
    #[cfg(test)]
    fn compile_decoded_ops(
        &mut self,
        cpu: &CpuCore,
        start_pc: u32,
        cpu_type: CpuType,
        ops: Vec<TraceBuildOp>,
        recorded_exit_pc: Option<u32>,
    ) -> Option<CompiledTrace> {
        self.compile_decoded_ops_reason(cpu, start_pc, cpu_type, ops, recorded_exit_pc)
            .ok()
    }

    /// Compile a recorded region, reporting *why* a region was declined.
    ///
    /// The reason exists so the profiler can attribute recordings that end
    /// without any unsupported opcode; a head declined here has no blocker
    /// and would otherwise leave no trace in the report at all.
    fn compile_decoded_ops_reason(
        &mut self,
        cpu: &CpuCore,
        start_pc: u32,
        cpu_type: CpuType,
        ops: Vec<TraceBuildOp>,
        recorded_exit_pc: Option<u32>,
    ) -> Result<CompiledTrace, RegionRejectReason> {
        if !ops.last().is_some_and(|op| op.op.ends_trace()) {
            return Err(RegionRejectReason::NoTraceTerminal);
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
            return Err(RegionRejectReason::TooShort);
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
            return Err(RegionRejectReason::IndirectJsrTooShort);
        }

        // Short checked memory ALU regions do not amortize trace validation
        // and the native/Rust boundary. Keep those on the decoded-memory path
        // unless the measured indirect-call length threshold above provides
        // enough independent work to cover the fixed cost.
        if !self_loop
            && !ends_in_indirect_jsr
            && ops.iter().any(|op| {
                matches!(
                    op.op,
                    JitTraceOp::AluMemToReg { .. }
                        | JitTraceOp::CmpiWordMem { .. }
                        | JitTraceOp::AddrCmpMemToReg { .. }
                )
            })
        {
            return Err(RegionRejectReason::LinearMemoryAlu);
        }

        // A pure poll loop burns wall time by design: its exit depends only
        // on memory it never writes, so executing it faster cannot make it
        // exit sooner. It is also exactly the class no trap-anchored wait
        // detector can see. Compiling it as a native spin multiplies the
        // instructions burned per wall second while the guest waits, so
        // refuse the compilation and leave the loop on the decoded path.
        //
        // This requires the recording to have actually gone round, not
        // merely to end in a branch whose static target is the head.
        // `self_loop` is true for a fall-through path as well, and a
        // recording that fell through to an unsupported operation is a
        // different shape entirely: reporting it as a wait would remove a
        // real blocker from the opportunity ranking.
        if recorded_exit_pc == Some(start_pc) && is_pure_poll_loop(&ops) {
            return Err(RegionRejectReason::WaitLoop);
        }

        let needs_window = ops.iter().any(|op| {
            matches!(
                op.op,
                JitTraceOp::MoveMem { .. }
                    | JitTraceOp::MovemWordPostInc { .. }
                    | JitTraceOp::AluMemToReg { .. }
                    | JitTraceOp::CmpiWordMem { .. }
                    | JitTraceOp::TstMem { .. }
                    | JitTraceOp::ClrMem { .. }
                    | JitTraceOp::MoveImmMem { .. }
                    | JitTraceOp::AddrCmpMemToReg { .. }
                    | JitTraceOp::AddRegToMem { .. }
                    | JitTraceOp::AnDispUnary { .. }
                    | JitTraceOp::PeaDisp { .. }
                    | JitTraceOp::Link { .. }
                    | JitTraceOp::Unlk { .. }
                    | JitTraceOp::MemAddqSubq { .. }
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
                return Err(RegionRejectReason::AddressWrap);
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
        .ok_or(RegionRejectReason::Backend)
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
                    JitTraceOp::CmpiWordMem { .. } => {
                        let env = mem_env.as_ref().expect("CmpiWordMem implies a window env");
                        emit_cmpi_word_mem(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::TstMem { .. } => {
                        let env = mem_env.as_ref().expect("TstMem implies a window env");
                        emit_tst_mem(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::ClrMem { .. } => {
                        let env = mem_env.as_ref().expect("ClrMem implies a window env");
                        emit_clr_mem(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::MoveImmMem { .. } => {
                        let env = mem_env.as_ref().expect("MoveImmMem implies a window env");
                        emit_move_imm_mem(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::AddrCmpMemToReg { .. } => {
                        let env = mem_env
                            .as_ref()
                            .expect("AddrCmpMemToReg implies a window env");
                        emit_addr_cmp_mem_to_reg(
                            &mut builder,
                            cpu_ptr,
                            *op,
                            env,
                            &mut bails,
                            bail_at,
                        )
                    }
                    JitTraceOp::AddRegToMem { .. } => {
                        let env = mem_env.as_ref().expect("AddRegToMem implies a window env");
                        emit_add_reg_to_mem(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::IndirectJsr { reg } => {
                        let env = mem_env.as_ref().expect("IndirectJsr implies a window env");
                        emit_indirect_jsr(&mut builder, cpu_ptr, *op, reg, env, &mut bails, bail_at)
                    }
                    JitTraceOp::MemAddqSubq { .. } => {
                        let env = mem_env.as_ref().expect("MemAddqSubq implies a window env");
                        emit_mem_addq_subq(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::AnDispUnary { .. } | JitTraceOp::AnDispBit { .. } => {
                        let env = mem_env.as_ref().expect("AnDisp implies a window env");
                        emit_an_disp_mem(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::Link { .. } => {
                        let env = mem_env.as_ref().expect("Link implies a window env");
                        emit_link(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::Unlk { .. } => {
                        let env = mem_env.as_ref().expect("Unlk implies a window env");
                        emit_unlk(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
                    }
                    JitTraceOp::PeaDisp { .. } => {
                        let env = mem_env.as_ref().expect("PeaDisp implies a window env");
                        emit_pea_disp(&mut builder, cpu_ptr, *op, env, &mut bails, bail_at)
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

    #[cfg(any(not(feature = "jit"), target_family = "wasm"))]
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
    #[cfg_attr(any(not(feature = "jit"), target_family = "wasm"), allow(dead_code))]
    aligned_only: bool,
    #[cfg_attr(any(not(feature = "jit"), target_family = "wasm"), allow(dead_code))]
    address_mask: u32,
}

const CC_N: u8 = 0b1000;
const CC_Z: u8 = 0b0100;
const CC_V: u8 = 0b0010;
const CC_C: u8 = 0b0001;

/// Condition codes a 68k conditional branch reads. `T`/`F` read nothing.
fn branch_flags_read(condition: u8) -> u8 {
    match condition & 0xF {
        0 | 1 => 0,              // T / F
        2 | 3 => CC_C | CC_Z,    // HI / LS
        4 | 5 => CC_C,           // CC / CS
        6 | 7 => CC_Z,           // NE / EQ
        8 | 9 => CC_V,           // VC / VS
        10 | 11 => CC_N,         // PL / MI
        12 | 13 => CC_N | CC_V,  // GE / LT
        _ => CC_N | CC_V | CC_Z, // GT / LE
    }
}

/// A recorded self-loop is a pure poll when it consists of memory reads
/// that mutate nothing except the condition codes, followed by a single
/// terminal conditional branch whose consumed flags were all written by
/// those reads (conservative flag provenance: a BTST-only loop branching
/// on carry does not classify, because BTST writes only Z). Interior
/// branches disqualify: a guarded interior branch means the head can
/// record multiple dynamic shapes, and the profiler's per-head wait flag
/// is only sound when the recorded path is structurally unique. Such a
/// loop's exit can only be driven by memory it never writes, so
/// executing it faster cannot make it exit sooner — it is a wait. The
/// match is exhaustive so every future trace operation must declare its
/// classification here.
fn is_pure_poll_loop(ops: &[TraceBuildOp]) -> bool {
    let Some((terminal, body)) = ops.split_last() else {
        return false;
    };
    let JitTraceOp::Branch { condition, .. } = terminal.op else {
        return false;
    };
    let consumed = branch_flags_read(condition);
    if consumed == 0 {
        // An unconditional terminal branch has no memory-driven exit.
        return false;
    }
    let mut mem_written_flags: u8 = 0;
    for op in body {
        match op.op {
            // No-ops mutate nothing.
            JitTraceOp::Nop => {}
            // Interior branches make the recorded path non-unique for
            // this head; per-head wait accounting would then erase
            // opportunity data belonging to other shapes.
            JitTraceOp::Branch { .. } => return false,
            // Memory-reading compares and tests write NZVC only; the
            // polled value is the only input that can change.
            JitTraceOp::TstMem { .. }
            | JitTraceOp::CmpiWordMem { .. }
            | JitTraceOp::AddrCmpMemToReg { .. }
            | JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                ..
            }
            | JitTraceOp::AnDispUnary {
                op: JitUnaryOp::Tst,
                ..
            } => mem_written_flags |= CC_N | CC_Z | CC_V | CC_C,
            // A bit test writes only Z.
            JitTraceOp::AnDispBit {
                op: JitBitOp::Test, ..
            } => mem_written_flags |= CC_Z,
            // Everything else mutates registers or memory, or transfers
            // control in a way that can end the wait from inside: any of
            // those makes the loop's progress self-driven, so it is not a
            // pure poll.
            JitTraceOp::MoveReg { .. }
            | JitTraceOp::Moveq { .. }
            | JitTraceOp::UnaryDataReg { .. }
            | JitTraceOp::AddqSubqReg { .. }
            | JitTraceOp::AddqSubqAddr { .. }
            | JitTraceOp::BinaryDataReg { .. }
            | JitTraceOp::BinaryImmediateDataReg { .. }
            | JitTraceOp::MulWordDataReg { .. }
            | JitTraceOp::MulWordImmediate { .. }
            | JitTraceOp::MulLongDataReg { .. }
            | JitTraceOp::AddrDataReg { .. }
            | JitTraceOp::AddrCmpImmediate { .. }
            | JitTraceOp::LeaAn { .. }
            | JitTraceOp::LeaIndex { .. }
            | JitTraceOp::AddSubxReg { .. }
            | JitTraceOp::BitReg { .. }
            | JitTraceOp::Exg { .. }
            | JitTraceOp::Ext { .. }
            | JitTraceOp::Extb { .. }
            | JitTraceOp::SccDataReg { .. }
            | JitTraceOp::ShiftReg { .. }
            | JitTraceOp::Swap { .. }
            | JitTraceOp::Dbcc { .. }
            | JitTraceOp::IndirectJsr { .. }
            | JitTraceOp::MoveMem { .. }
            | JitTraceOp::MovemWordPostInc { .. }
            | JitTraceOp::AluMemToReg { .. }
            | JitTraceOp::AnDispUnary { .. }
            | JitTraceOp::AddRegToMem { .. }
            | JitTraceOp::MemAddqSubq { .. }
            | JitTraceOp::AnDispBit { .. }
            | JitTraceOp::PeaDisp { .. }
            | JitTraceOp::Link { .. }
            | JitTraceOp::Unlk { .. }
            | JitTraceOp::ClrMem { .. }
            | JitTraceOp::MoveImmMem { .. } => return false,
        }
    }
    mem_written_flags != 0 && consumed & !mem_written_flags == 0
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
pub(crate) fn stop_recording(cpu: &mut CpuCore, cause: RecordingStop) {
    if cpu.trace_recording {
        TRACE_JIT
            .with_borrow_mut(|jit| jit.finish_recording(cpu, cpu.pc, RecordingEnd::Stopped(cause)));
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
            Self::BinaryImmediateDataReg { cycles, .. } => cycles,
            // MC68000 multiply timing depends on the source bits. Use its
            // slowest possible result for headroom; later supported CPUs use
            // the interpreter's fixed pre-scaled value.
            Self::MulWordDataReg {
                m68000_timing: true,
                ..
            } => 70,
            Self::MulWordDataReg { .. } => 42,
            Self::MulWordImmediate {
                m68000_timing: true,
                ..
            } => 74,
            Self::MulWordImmediate { .. } => 42,
            Self::MulLongDataReg { .. } => 40,
            Self::AddrDataReg {
                op: JitAddrOp::Cmpa,
                ..
            } => 6,
            Self::AddrDataReg { .. } => 8,
            Self::AddrCmpImmediate { cycles, .. } => cycles,
            Self::LeaAn { cycles, .. } => cycles,
            Self::LeaIndex { cycles, .. } => cycles,
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
                    JitEa::AbsWord(_) => {
                        if long {
                            12
                        } else {
                            8
                        }
                    }
                    JitEa::AbsLong(_) => {
                        if long {
                            16
                        } else {
                            12
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
                    JitEa::AbsWord(_) => {
                        if long {
                            12
                        } else {
                            8
                        }
                    }
                    JitEa::AbsLong(_) => {
                        if long {
                            16
                        } else {
                            12
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
            // MC68000 CMPI.W uses an eight-cycle base plus the ten-cycle
            // brief-indexed effective-address read.
            Self::CmpiWordMem { .. } => 18,
            // TST is a four-cycle operation plus the indexed EA read
            // (M68000UM); byte and word reads have the same EA cost.
            Self::TstMem { size, .. } => {
                if size == Size::Long {
                    18
                } else {
                    14
                }
            }
            // CLR is the M68000UM store cost (8, or 12 for long) plus the
            // EA calculation: indexed as in the read above, predecrement
            // per Table 8-2.
            Self::ClrMem { size, dst } => match (size, dst) {
                (Size::Long, JitEa::Index { .. }) => 26,
                (_, JitEa::Index { .. }) => 18,
                (Size::Long, JitEa::AbsWord(_)) => 24,
                (_, JitEa::AbsWord(_)) => 16,
                (Size::Long, JitEa::AbsLong(_)) => 28,
                (_, JitEa::AbsLong(_)) => 20,
                (Size::Long, _) => 22,
                _ => 14,
            },
            // MOVE #imm to memory per the M68000UM move table: 12 for
            // byte/word to the extension-less destinations, 16 with a
            // displacement; long pays the extra immediate fetch.
            Self::MoveImmMem { size, dst, .. } => match (size, dst) {
                (Size::Long, JitEa::Disp(_, _)) => 24,
                (Size::Long, _) => 20,
                (_, JitEa::Index { .. }) => 18,
                (_, JitEa::Disp(_, _)) => 16,
                _ => 12,
            },
            Self::AddrCmpMemToReg { .. } => 24,
            Self::AddRegToMem { size, dst, .. } => match (size, dst) {
                (Size::Long, JitEa::Disp(_, _)) => 24,
                (Size::Long, _) => 20,
                (_, JitEa::Disp(_, _)) => 16,
                _ => 12,
            },
            Self::IndirectJsr { .. } => 16,
            // These ops only execute in instruction-budgeted fastmem mode;
            // conservative cycle maxima preserve the trace headroom guard.
            Self::MemAddqSubq { .. } | Self::AnDispUnary { .. } | Self::AnDispBit { .. } => 24,
            // MC68000 PEA (d16,An): twelve-cycle push plus the four-cycle
            // displacement extension fetch.
            Self::PeaDisp { .. } => 16,
            // Interpreter parity: exec_link charges 16, exec_unlk 12.
            Self::Link { .. } => 16,
            Self::Unlk { .. } => 12,
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
    if let Some(op) = decode_link_unlk_trace_op(cpu, bus, pc, opcode) {
        return Some(op);
    }
    if let Some(op) = decode_binary_immediate_data_reg_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_mul_word_immediate_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_cmpi_word_mem_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_addr_cmp_immediate_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_long_mul_data_reg_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_an_disp_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_alu_mem_to_reg_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_addr_cmp_mem_to_reg_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_add_reg_to_mem_trace_op(cpu, bus, pc, opcode, cpu_type) {
        return Some(op);
    }
    if let Some(op) = decode_movem_word_postinc_trace_op(cpu, bus, pc, opcode) {
        return Some(op);
    }
    if let Some(op) = decode_move_imm_mem_trace_op(cpu, bus, pc, opcode) {
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

/// Decode `LINK An,#d16` / `UNLK An`. A7 forms are excluded: LINK A7's
/// pushed value is generation-dependent (the 68040 decrements first) and
/// UNLK A7 is degenerate; ROM prologues use A6.
fn decode_link_unlk_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
) -> Option<TraceBuildOp> {
    let reg = (opcode & 7) as u8;
    if opcode & 0xFFF8 == 0x4E50 && reg != 7 {
        let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
        return Some(TraceBuildOp {
            opcode,
            extension: Some(extension),
            extension2: None,
            pc,
            op: JitTraceOp::Link {
                reg,
                displacement: extension as i16,
            },
        });
    }
    if opcode & 0xFFF8 == 0x4E58 && reg != 7 {
        return Some(TraceBuildOp {
            opcode,
            extension: None,
            extension2: None,
            pc,
            op: JitTraceOp::Unlk { reg },
        });
    }
    None
}

/// Decode the register-source, 32-bit-result form of the 68020 long
/// multiply family. Capturing the extension word fixes signedness and the
/// destination register for the lifetime of the validated trace.
fn decode_long_mul_data_reg_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    if is_pre_68020(cpu_type) || (opcode & 0xFFF8) != 0x4C00 {
        return None;
    }
    let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    if (extension & 0x0400) != 0 {
        return None;
    }
    Some(TraceBuildOp {
        opcode,
        extension: Some(extension),
        extension2: None,
        pc,
        op: JitTraceOp::MulLongDataReg {
            src: (opcode & 7) as u8,
            dst: ((extension >> 12) & 7) as u8,
            signed: (extension & 0x0800) != 0,
        },
    })
}

/// Decode the register-direct subset of the immediate ALU family. Once the
/// extension words have been captured, these operations cannot fault and do
/// not need effective-address handling inside the trace.
fn decode_binary_immediate_data_reg_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let DecodedMemOp::AluImm {
        op,
        size,
        dst: FastEa::DataReg(dst),
    } = DecodedMemOp::decode(cpu_type, opcode)?
    else {
        return None;
    };
    let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    let (immediate, extension2) = if size == Size::Long {
        let low = bus.try_read_word(cpu.address(pc.wrapping_add(4))).ok()?;
        (((u32::from(extension)) << 16) | u32::from(low), Some(low))
    } else {
        (u32::from(extension) & size.mask(), None)
    };
    let op = match op {
        BinaryOp::Add => JitBinaryOp::Add,
        BinaryOp::Sub => JitBinaryOp::Sub,
        BinaryOp::And => JitBinaryOp::And,
        BinaryOp::Or => JitBinaryOp::Or,
        BinaryOp::Eor => JitBinaryOp::Eor,
        BinaryOp::Cmp => JitBinaryOp::Cmp,
    };
    let cycles = if size == Size::Long {
        if op == JitBinaryOp::Cmp { 14 } else { 16 }
    } else {
        8
    };
    Some(TraceBuildOp {
        opcode,
        extension: Some(extension),
        extension2,
        pc,
        op: JitTraceOp::BinaryImmediateDataReg {
            op,
            immediate,
            dst,
            size,
            cycles,
        },
    })
}

/// `MULU.W`/`MULS.W #imm,Dn`. The register-source forms already decode
/// through `DecodedSimpleOp`, which cannot see the extension word holding
/// the multiplicand; capture it here instead.
///
/// This form heads nine separate hot loops in a capped gameplay profile
/// (1.69M rejected recordings), each blocking after a single operation,
/// which is what kept the loop bodies behind it out of every trace.
fn decode_mul_word_immediate_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    // 1100 ddd 0mm 111 100: opmode 3 = MULU.W, 7 = MULS.W; ea = immediate.
    if (opcode & 0xF000) != 0xC000 || (opcode & 0x003F) != 0x003C {
        return None;
    }
    let op_mode = (opcode >> 6) & 7;
    if !matches!(op_mode, 3 | 7) {
        return None;
    }
    let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    Some(TraceBuildOp {
        opcode,
        extension: Some(extension),
        extension2: None,
        pc,
        op: JitTraceOp::MulWordImmediate {
            immediate: extension,
            dst: ((opcode >> 9) & 7) as u8,
            signed: op_mode == 7,
            m68000_timing: cpu_type == CpuType::M68000,
        },
    })
}

/// Decode the measured indexed word-memory form of CMPI. Unlike the other
/// immediate ALU operations, CMPI does not write its effective address, so a
/// checked fast-memory read can bail before any architectural state changes.
fn decode_cmpi_word_mem_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let DecodedMemOp::AluImm {
        op: BinaryOp::Cmp,
        size: Size::Word,
        dst: FastEa::AnIndex(base),
    } = DecodedMemOp::decode(cpu_type, opcode)?
    else {
        return None;
    };
    let immediate = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    let ea_extension = bus.try_read_word(cpu.address(pc.wrapping_add(4))).ok()?;
    let src = decode_jit_ea(6, u16::from(base), ea_extension, cpu_type)?;
    Some(TraceBuildOp {
        opcode,
        extension: Some(immediate),
        extension2: Some(ea_extension),
        pc,
        op: JitTraceOp::CmpiWordMem { immediate, src },
    })
}

/// Decode immediate CMPA without routing it through checked guest memory.
/// The extension words are part of the validated trace code, and the
/// operation itself can neither fault nor mutate its destination.
fn decode_addr_cmp_immediate_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let DecodedMemOp::AluAddr {
        op: AddrOp::Cmpa,
        size,
        src: FastEa::Imm,
        dst,
    } = DecodedMemOp::decode(cpu_type, opcode)?
    else {
        return None;
    };
    let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
    let (immediate, extension2, cycles) = if size == Size::Long {
        let low = bus.try_read_word(cpu.address(pc.wrapping_add(4))).ok()?;
        ((u32::from(extension) << 16) | u32::from(low), Some(low), 14)
    } else {
        (u32::from(extension), None, 10)
    };
    Some(TraceBuildOp {
        opcode,
        extension: Some(extension),
        extension2,
        pc,
        op: JitTraceOp::AddrCmpImmediate {
            immediate,
            dst,
            size,
            cycles,
        },
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
                Some(displacement),
                None,
                JitTraceOp::AnDispUnary {
                    op: JitUnaryOp::Tst,
                    size,
                    reg,
                    displacement: displacement as i16,
                },
            )
        }
        DecodedMemOp::Tst {
            size,
            ea: FastEa::AnIndex(reg),
        } => {
            let extension = read_ext(2, bus)?;
            let src = decode_jit_ea(6, u16::from(reg), extension, cpu_type)?;
            (Some(extension), None, JitTraceOp::TstMem { size, src })
        }
        DecodedMemOp::Clr {
            size,
            ea: FastEa::AnDisp(reg),
        } => {
            let displacement = read_ext(2, bus)?;
            (
                Some(displacement),
                None,
                JitTraceOp::AnDispUnary {
                    op: JitUnaryOp::Clr,
                    size,
                    reg,
                    displacement: displacement as i16,
                },
            )
        }
        DecodedMemOp::Clr {
            size,
            ea: FastEa::AnIndex(reg),
        } => {
            let extension = read_ext(2, bus)?;
            let dst = decode_jit_ea(6, u16::from(reg), extension, cpu_type)?;
            (Some(extension), None, JitTraceOp::ClrMem { size, dst })
        }
        DecodedMemOp::Clr {
            size,
            ea: FastEa::AnPreDec(reg),
        } => (
            None,
            None,
            JitTraceOp::ClrMem {
                size,
                dst: JitEa::PreDec(reg),
            },
        ),
        DecodedMemOp::Clr {
            size,
            ea: FastEa::AbsW,
        } => {
            let extension = read_ext(2, bus)?;
            (
                Some(extension),
                None,
                JitTraceOp::ClrMem {
                    size,
                    dst: JitEa::AbsWord(extension as i16 as i32 as u32),
                },
            )
        }
        DecodedMemOp::Clr {
            size,
            ea: FastEa::AbsL,
        } => {
            let hi = read_ext(2, bus)?;
            let lo = read_ext(4, bus)?;
            (
                Some(hi),
                Some(lo),
                JitTraceOp::ClrMem {
                    size,
                    dst: JitEa::AbsLong((u32::from(hi) << 16) | u32::from(lo)),
                },
            )
        }
        DecodedMemOp::Pea {
            ea: FastEa::AnDisp(reg),
        } => {
            let displacement = read_ext(2, bus)?;
            (
                Some(displacement),
                None,
                JitTraceOp::PeaDisp {
                    reg,
                    displacement: displacement as i16,
                },
            )
        }
        DecodedMemOp::AddqSubq {
            data,
            size,
            ea: FastEa::AnInd(reg),
            is_sub,
        } => (
            None,
            None,
            JitTraceOp::MemAddqSubq {
                data,
                size,
                dst: JitEa::Ind(reg),
                is_sub,
            },
        ),
        DecodedMemOp::AddqSubq {
            data,
            size,
            ea: FastEa::AnDisp(reg),
            is_sub,
        } => {
            let displacement = read_ext(2, bus)?;
            (
                Some(displacement),
                None,
                JitTraceOp::MemAddqSubq {
                    data,
                    size,
                    dst: JitEa::Disp(reg, displacement as i16),
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
                        Some(displacement),
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
                        Some(bit_word),
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
        DecodedMemOp::Lea {
            reg: dst,
            ea: FastEa::AnInd(base),
        } => (
            None,
            None,
            JitTraceOp::LeaAn {
                base,
                dst,
                displacement: 0,
                cycles: 4,
            },
        ),
        DecodedMemOp::Lea {
            reg: dst,
            ea: FastEa::AnDisp(base),
        } => {
            let displacement = read_ext(2, bus)?;
            (
                Some(displacement),
                None,
                JitTraceOp::LeaAn {
                    base,
                    dst,
                    displacement: displacement as i16,
                    cycles: if is_pre_68020(cpu_type) { 8 } else { 4 },
                },
            )
        }
        DecodedMemOp::Lea {
            reg: dst,
            ea: FastEa::AnIndex(base),
        } => {
            let extension = read_ext(2, bus)?;
            let src = decode_jit_ea(6, u16::from(base), extension, cpu_type)?;
            (
                Some(extension),
                None,
                JitTraceOp::LeaIndex {
                    src,
                    dst,
                    // MC68000 LEA (d8,An,Xn) is twelve cycles.
                    cycles: if is_pre_68020(cpu_type) { 12 } else { 4 },
                },
            )
        }
        _ => return None,
    };
    Some(TraceBuildOp {
        opcode,
        extension,
        extension2,
        pc,
        op,
    })
}

/// MOVE/MOVEA (groups 1-3) using register, register-indirect, d16(An), brief
/// indexed, or absolute EAs. At least one side must be memory; extension
/// words are captured in execution order for validation and
/// self-modification checks.
/// Decode `MOVE #imm,<memory>` for the destination forms that fit the
/// two extension-word budget: byte/word immediates to `(An)`, `(An)+`,
/// `-(An)`, and `(d16,An)`, and long immediates to the extension-less
/// destinations. `MOVE.L #imm,(d16,An)` needs three extension words and
/// stays on the decoded path.
fn decode_move_imm_mem_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
) -> Option<TraceBuildOp> {
    let size = match opcode >> 12 {
        1 => Size::Byte,
        3 => Size::Word,
        2 => Size::Long,
        _ => return None,
    };
    // Source must be immediate: mode 7, register 4.
    if opcode & 0x003F != 0x003C {
        return None;
    }
    let dst_reg = ((opcode >> 9) & 7) as u8;
    let dst_mode = (opcode >> 6) & 7;
    let read_ext =
        |offset: u32, bus: &mut B| bus.try_read_word(cpu.address(pc.wrapping_add(offset))).ok();
    let (value, extension, extension2, imm_words) = match size {
        Size::Long => {
            let hi = read_ext(2, bus)?;
            let lo = read_ext(4, bus)?;
            (
                (u32::from(hi) << 16) | u32::from(lo),
                Some(hi),
                Some(lo),
                2u32,
            )
        }
        Size::Word => {
            let imm = read_ext(2, bus)?;
            (u32::from(imm), Some(imm), None, 1)
        }
        Size::Byte => {
            let imm = read_ext(2, bus)?;
            (u32::from(imm & 0x00FF), Some(imm), None, 1)
        }
    };
    let (dst, extension2) = match dst_mode {
        2 => (JitEa::Ind(dst_reg), extension2),
        3 => (JitEa::PostInc(dst_reg), extension2),
        4 => (JitEa::PreDec(dst_reg), extension2),
        5 => {
            if size == Size::Long {
                // Three extension words in total; TraceBuildOp carries two.
                return None;
            }
            let displacement = read_ext(2 + 2 * imm_words, bus)?;
            (
                JitEa::Disp(dst_reg, displacement as i16),
                Some(displacement),
            )
        }
        6 => {
            if size == Size::Long {
                // Immediate high/low plus the brief extension word exceed
                // TraceBuildOp's two extension slots.
                return None;
            }
            let brief = read_ext(2 + 2 * imm_words, bus)?;
            (
                decode_jit_ea(6, u16::from(dst_reg), brief, cpu.cpu_type)?,
                Some(brief),
            )
        }
        _ => return None,
    };
    Some(TraceBuildOp {
        opcode,
        extension,
        extension2,
        pc,
        op: JitTraceOp::MoveImmMem { size, value, dst },
    })
}

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
    let mut extensions = [0u16; 2];
    let mut extension_count = 0usize;
    let mut read_ext = || -> Option<u16> {
        if extension_count == extensions.len() {
            return None;
        }
        let address = pc.wrapping_add(2 + 2 * extension_count as u32);
        let value = bus.try_read_word(cpu.address(address)).ok()?;
        extensions[extension_count] = value;
        extension_count += 1;
        Some(value)
    };
    let mut decode_ea = |mode: u16, reg: u16| -> Option<JitEa> {
        match (mode, reg) {
            (5 | 6, _) => decode_jit_ea(mode, reg, read_ext()?, cpu.cpu_type),
            (7, 0) => Some(JitEa::AbsWord(read_ext()? as i16 as i32 as u32)),
            (7, 1) => {
                let high = read_ext()?;
                let low = read_ext()?;
                Some(JitEa::AbsLong((u32::from(high) << 16) | u32::from(low)))
            }
            _ => decode_jit_ea(mode, reg, 0, cpu.cpu_type),
        }
    };
    let src = decode_ea(src_mode, opcode & 7)?;
    let dst = decode_ea(dst_mode, (opcode >> 9) & 7)?;
    // Indexed destinations were gated until "a profile demonstrates that
    // their extra emitter paths pay". Three profiled heads across two
    // workloads now terminate on indexed-destination stores (MOVE.W 3180
    // and the CLR forms 4230/4270), so the store side gets the same brief-
    // indexed support the read side has had.
    if !src.is_mem() && !dst.is_mem() {
        return None;
    }
    // MOVEA.B does not exist, and An is not a legal byte source.
    if size == Size::Byte && (matches!(src, JitEa::Addr(_)) || matches!(dst, JitEa::Addr(_))) {
        return None;
    }
    Some(TraceBuildOp {
        opcode,
        extension: (extension_count >= 1).then_some(extensions[0]),
        extension2: (extension_count >= 2).then_some(extensions[1]),
        pc,
        op: JitTraceOp::MoveMem { size, src, dst },
    })
}

fn decode_add_reg_to_mem_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let DecodedMemOp::AluToMem { op, size, src, dst } = DecodedMemOp::decode(cpu_type, opcode)?
    else {
        return None;
    };
    let is_sub = match op {
        BinaryOp::Add => false,
        BinaryOp::Sub => true,
        _ => return None,
    };
    if !matches!(size, Size::Word | Size::Long) {
        return None;
    }
    let (dst, extension) = match dst {
        FastEa::AnInd(reg) => (JitEa::Ind(reg), None),
        FastEa::AnPostInc(reg) => (JitEa::PostInc(reg), None),
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
        op: JitTraceOp::AddRegToMem {
            is_sub,
            size,
            src,
            dst,
        },
    })
}

/// CMP/ADD/SUB `<ea>,Dn` for indirect, displacement, and brief-indexed source
/// forms. The access itself is emitted against the fastmem window; extension
/// words are captured for validation and decoded once while recording.
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
        FastEa::AnIndex(reg) => {
            let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
            (
                decode_jit_ea(6, u16::from(reg), extension, cpu_type)?,
                Some(extension),
            )
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

/// Decode read-only address-register compares through the two simplest
/// address-register-relative memory forms. CMPA differs from CMP in two
/// important ways: it always compares against all 32 destination bits, and
/// its word source is sign-extended to 32 bits. Keeping it separate from the
/// ordinary memory-to-data-register ALU op makes those rules explicit and
/// avoids admitting the mutating ADDA/SUBA forms.
fn decode_addr_cmp_mem_to_reg_trace_op<B: AddressBus>(
    cpu: &CpuCore,
    bus: &mut B,
    pc: u32,
    opcode: u16,
    cpu_type: CpuType,
) -> Option<TraceBuildOp> {
    let DecodedMemOp::AluAddr {
        op: AddrOp::Cmpa,
        size,
        src,
        dst,
    } = DecodedMemOp::decode(cpu_type, opcode)?
    else {
        return None;
    };
    let (src, extension) = match src {
        FastEa::AnInd(reg) => (JitEa::Ind(reg), None),
        FastEa::AnDisp(reg) => {
            let extension = bus.try_read_word(cpu.address(pc.wrapping_add(2))).ok()?;
            (JitEa::Disp(reg, extension as i16), Some(extension))
        }
        _ => return None,
    };
    Some(TraceBuildOp {
        opcode,
        extension,
        extension2: None,
        pc,
        op: JitTraceOp::AddrCmpMemToReg { size, src, dst },
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
#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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
#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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
    if matches!(op.op, JitTraceOp::CmpiWordMem { .. }) {
        return execute_portable_cmpi_word_mem(cpu, op);
    }
    if matches!(op.op, JitTraceOp::TstMem { .. }) {
        return execute_portable_tst_mem(cpu, op);
    }
    if matches!(op.op, JitTraceOp::ClrMem { .. }) {
        return execute_portable_clr_mem(cpu, op, code_start, code_end);
    }
    if matches!(op.op, JitTraceOp::MoveImmMem { .. }) {
        return execute_portable_move_imm_mem(cpu, op, code_start, code_end);
    }
    if matches!(op.op, JitTraceOp::AddrCmpMemToReg { .. }) {
        return execute_portable_addr_cmp_mem_to_reg(cpu, op);
    }
    if matches!(op.op, JitTraceOp::AddRegToMem { .. }) {
        return execute_portable_add_reg_to_mem(cpu, op, code_start, code_end);
    }
    if matches!(op.op, JitTraceOp::MemAddqSubq { .. }) {
        return execute_portable_mem_addq_subq(cpu, op, code_start, code_end);
    }
    if matches!(op.op, JitTraceOp::Link { .. }) {
        return execute_portable_link(cpu, op, code_start, code_end);
    }
    if matches!(op.op, JitTraceOp::Unlk { .. }) {
        return execute_portable_unlk(cpu, op);
    }
    if matches!(op.op, JitTraceOp::PeaDisp { .. }) {
        return execute_portable_pea_disp(cpu, op, code_start, code_end);
    }
    if matches!(
        op.op,
        JitTraceOp::AnDispUnary { .. } | JitTraceOp::AnDispBit { .. }
    ) {
        return execute_portable_an_disp(cpu, op, code_start, code_end);
    }
    if let JitTraceOp::IndirectJsr { reg } = op.op {
        return execute_portable_indirect_jsr(cpu, op, reg);
    }
    Some(execute_portable_reg_op(cpu, op))
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_mem_addq_subq(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let JitTraceOp::MemAddqSubq {
        data,
        size,
        dst,
        is_sub,
    } = trace.op
    else {
        return None;
    };
    let (reg, displacement, ea) = match dst {
        JitEa::Ind(reg) => (reg, 0, FastEa::AnInd(reg)),
        JitEa::Disp(reg, displacement) => (reg, displacement, FastEa::AnDisp(reg)),
        _ => return None,
    };
    let raw = cpu.dar[8 + reg as usize].wrapping_add(displacement as i32 as u32);
    let masked = raw & cpu.address_mask;
    if masked < code_end && masked.wrapping_add(size.bytes()) > code_start {
        return None;
    }

    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(
        cpu,
        DecodedMemOp::AddqSubq {
            data,
            size,
            ea,
            is_sub,
        },
    ) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_pea_disp(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let JitTraceOp::PeaDisp { reg, .. } = trace.op else {
        return None;
    };
    let sp = cpu.dar[15].wrapping_sub(4);
    let masked = sp & cpu.address_mask;
    if masked < code_end && masked.wrapping_add(4) > code_start {
        return None;
    }
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(
        cpu,
        DecodedMemOp::Pea {
            ea: FastEa::AnDisp(reg),
        },
    ) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_link(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let JitTraceOp::Link { reg, displacement } = trace.op else {
        return None;
    };
    if cpu.fm_len == 0 {
        return None;
    }
    let new_sp = cpu.dar[15].wrapping_sub(4);
    if cpu.is_pre_68020 && (new_sp & 1) != 0 {
        return None;
    }
    let masked = new_sp & cpu.address_mask;
    let off = masked.wrapping_sub(cpu.fm_base);
    if off > cpu.fm_len - 4 {
        return None;
    }
    if masked < code_end && masked.wrapping_add(4) > code_start {
        return None;
    }
    let an = cpu.dar[8 + reg as usize];
    unsafe {
        let p = (cpu.fm_ptr as *mut u8).add(off as usize);
        let b = an.to_be_bytes();
        *p = b[0];
        *p.add(1) = b[1];
        *p.add(2) = b[2];
        *p.add(3) = b[3];
    }
    cpu.dar[8 + reg as usize] = new_sp;
    cpu.dar[15] = new_sp.wrapping_add(displacement as i32 as u32);
    Some(trace.op.max_cycles())
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_unlk(cpu: &mut CpuCore, trace: TraceBuildOp) -> Option<i32> {
    let JitTraceOp::Unlk { reg } = trace.op else {
        return None;
    };
    if cpu.fm_len == 0 {
        return None;
    }
    let addr = cpu.dar[8 + reg as usize];
    if cpu.is_pre_68020 && (addr & 1) != 0 {
        return None;
    }
    let off = (addr & cpu.address_mask).wrapping_sub(cpu.fm_base);
    if off > cpu.fm_len - 4 {
        return None;
    }
    let value = unsafe {
        let p = (cpu.fm_ptr as *const u8).add(off as usize);
        u32::from_be_bytes([*p, *p.add(1), *p.add(2), *p.add(3)])
    };
    cpu.dar[15] = addr.wrapping_add(4);
    cpu.dar[8 + reg as usize] = value;
    Some(trace.op.max_cycles())
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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
        JitEa::Index { base, .. } => FastEa::AnIndex(base),
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

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_cmpi_word_mem(cpu: &mut CpuCore, trace: TraceBuildOp) -> Option<i32> {
    let JitTraceOp::CmpiWordMem {
        src: JitEa::Index { base, .. },
        ..
    } = trace.op
    else {
        return None;
    };
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(
        cpu,
        DecodedMemOp::AluImm {
            op: BinaryOp::Cmp,
            size: Size::Word,
            dst: FastEa::AnIndex(base),
        },
    ) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_tst_mem(cpu: &mut CpuCore, trace: TraceBuildOp) -> Option<i32> {
    let JitTraceOp::TstMem { size, src } = trace.op else {
        return None;
    };
    let ea = match src {
        JitEa::Index { base, .. } => FastEa::AnIndex(base),
        _ => return None,
    };
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(cpu, DecodedMemOp::Tst { size, ea }) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_clr_mem(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let JitTraceOp::ClrMem { size, dst } = trace.op else {
        return None;
    };
    // Pre-check the store target against the trace's own code, as the
    // compiled version does, so a self-modifying store bails instead of
    // executing. For the predecrement form the target is the DECREMENTED
    // address, and nothing (including the register) commits on a bail.
    let (raw, ea) = match dst {
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base_value = cpu.dar[8 + base as usize];
            let raw_index = match index {
                JitDirectReg::Addr(r) => cpu.dar[8 + r as usize],
                JitDirectReg::Data(r) => cpu.dar[r as usize],
            };
            let index_value = if index_long {
                raw_index
            } else {
                raw_index as u16 as i16 as i32 as u32
            };
            (
                base_value
                    .wrapping_add(index_value << scale)
                    .wrapping_add(displacement as i32 as u32),
                FastEa::AnIndex(base),
            )
        }
        JitEa::PreDec(reg) => (
            cpu.dar[8 + reg as usize].wrapping_sub(jit_ea_step(size, reg)),
            FastEa::AnPreDec(reg),
        ),
        JitEa::AbsWord(address) => (address, FastEa::AbsW),
        JitEa::AbsLong(address) => (address, FastEa::AbsL),
        _ => return None,
    };
    let masked = raw & cpu.address_mask;
    if masked < code_end && masked.wrapping_add(size.bytes()) > code_start {
        return None;
    }
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(cpu, DecodedMemOp::Clr { size, ea }) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_move_imm_mem(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let JitTraceOp::MoveImmMem { size, value, dst } = trace.op else {
        return None;
    };
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
    let write = |cpu: &mut CpuCore, off: u32, value: u32| unsafe {
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
    };
    let (r, addr, new_reg) = match dst {
        JitEa::Ind(r) => (r, cpu.dar[8 + r as usize], None),
        JitEa::PostInc(r) => {
            let base = cpu.dar[8 + r as usize];
            (r, base, Some(base.wrapping_add(jit_ea_step(size, r))))
        }
        JitEa::PreDec(r) => {
            let a = cpu.dar[8 + r as usize].wrapping_sub(jit_ea_step(size, r));
            (r, a, Some(a))
        }
        JitEa::Disp(r, displacement) => (
            r,
            cpu.dar[8 + r as usize].wrapping_add(displacement as i32 as u32),
            None,
        ),
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base_value = cpu.dar[8 + base as usize];
            let raw_index = match index {
                JitDirectReg::Addr(r) => cpu.dar[8 + r as usize],
                JitDirectReg::Data(r) => cpu.dar[r as usize],
            };
            let index_value = if index_long {
                raw_index
            } else {
                raw_index as u16 as i16 as i32 as u32
            };
            (
                base,
                base_value
                    .wrapping_add(index_value << scale)
                    .wrapping_add(displacement as i32 as u32),
                None,
            )
        }
        _ => return None,
    };
    let off = locate(cpu, addr)?;
    let masked = addr & cpu.address_mask;
    // Self-modification guard, as in the compiled version: nothing —
    // including the register update — commits on a bail.
    if masked < code_end && masked.wrapping_add(bytes) > code_start {
        return None;
    }
    if let Some(v) = new_reg {
        cpu.dar[8 + r as usize] = v;
    }
    write(cpu, off, value);
    cpu.set_logic_flags(value, size);
    Some(trace.op.max_cycles())
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_addr_cmp_mem_to_reg(cpu: &mut CpuCore, trace: TraceBuildOp) -> Option<i32> {
    let JitTraceOp::AddrCmpMemToReg { size, src, dst } = trace.op else {
        return None;
    };
    let src = match src {
        JitEa::Ind(reg) => FastEa::AnInd(reg),
        JitEa::Disp(reg, _) => FastEa::AnDisp(reg),
        _ => return None,
    };
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(
        cpu,
        DecodedMemOp::AluAddr {
            op: AddrOp::Cmpa,
            size,
            src,
            dst,
        },
    ) {
        Some(trace.op.max_cycles())
    } else {
        cpu.pc = old_pc;
        None
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn execute_portable_add_reg_to_mem(
    cpu: &mut CpuCore,
    trace: TraceBuildOp,
    code_start: u32,
    code_end: u32,
) -> Option<i32> {
    let JitTraceOp::AddRegToMem {
        is_sub,
        size,
        src,
        dst,
    } = trace.op
    else {
        return None;
    };
    let (reg, raw) = match dst {
        JitEa::Ind(reg) => (reg, cpu.dar[8 + reg as usize]),
        JitEa::PostInc(reg) => (reg, cpu.dar[8 + reg as usize]),
        JitEa::Disp(reg, displacement) => (
            reg,
            cpu.dar[8 + reg as usize].wrapping_add(displacement as i32 as u32),
        ),
        _ => return None,
    };
    let masked = raw & cpu.address_mask;
    if masked < code_end && masked.wrapping_add(size.bytes()) > code_start {
        return None;
    }
    let old_pc = cpu.pc;
    cpu.pc = trace.pc.wrapping_add(2);
    if super::mem_ops::execute_mem_op(
        cpu,
        DecodedMemOp::AluToMem {
            op: if is_sub { BinaryOp::Sub } else { BinaryOp::Add },
            size,
            src,
            dst: match dst {
                JitEa::Ind(_) => FastEa::AnInd(reg),
                JitEa::PostInc(_) => FastEa::AnPostInc(reg),
                JitEa::Disp(_, _) => FastEa::AnDisp(reg),
                _ => unreachable!(),
            },
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
#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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
    let write = |cpu: &mut CpuCore, off: u32, value: u32| unsafe {
        let p = (cpu.fm_ptr as *mut u8).add(off as usize);
        match size {
            Size::Byte => *p = value as u8,
            Size::Word => {
                let bytes = (value as u16).to_be_bytes();
                *p = bytes[0];
                *p.add(1) = bytes[1];
            }
            Size::Long => {
                let bytes = value.to_be_bytes();
                *p = bytes[0];
                *p.add(1) = bytes[1];
                *p.add(2) = bytes[2];
                *p.add(3) = bytes[3];
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
        JitEa::AbsWord(address) | JitEa::AbsLong(address) => read(cpu, locate(cpu, address)?),
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
            write(cpu, off, value);
            cpu.set_logic_flags(value, size);
        }
        JitEa::AbsWord(address) | JitEa::AbsLong(address) => {
            let off = locate(cpu, address)?;
            let masked = address & cpu.address_mask;
            if masked < code_end && masked.wrapping_add(bytes) > code_start {
                return None;
            }
            if let Some((idx, value)) = staged {
                cpu.dar[idx] = value;
            }
            write(cpu, off, value);
            cpu.set_logic_flags(value, size);
        }
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base_value = dst_base(cpu, base);
            // The index register must also observe a staged source
            // adjustment: in `MOVE.W (A2)+,(d8,A1,A2.W)` the source
            // post-increment commits before the destination EA evaluates.
            let raw_index = match index {
                JitDirectReg::Addr(r) => match staged {
                    Some((idx, v)) if idx == 8 + r as usize => v,
                    _ => cpu.dar[8 + r as usize],
                },
                JitDirectReg::Data(r) => cpu.dar[r as usize],
            };
            let index_value = if index_long {
                raw_index
            } else {
                raw_index as u16 as i16 as i32 as u32
            };
            let addr = base_value
                .wrapping_add(index_value << scale)
                .wrapping_add(displacement as i32 as u32);
            let off = locate(cpu, addr)?;
            let masked = addr & cpu.address_mask;
            if masked < code_end && masked.wrapping_add(bytes) > code_start {
                return None;
            }
            if let Some((idx, v)) = staged {
                cpu.dar[idx] = v;
            }
            write(cpu, off, value);
            cpu.set_logic_flags(value, size);
        }
    }

    Some(JitTraceOp::MoveMem { size, src, dst }.max_cycles())
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
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
        JitTraceOp::Swap { reg } => {
            let reg = reg as usize;
            let result = cpu.d(reg).rotate_right(16);
            cpu.set_d(reg, result);
            cpu.set_logic_flags(result, Size::Long);
            4
        }
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
        JitTraceOp::BinaryImmediateDataReg {
            op: binary_op,
            immediate,
            dst,
            size,
            cycles,
        } => {
            let dst_value = cpu.dar[dst as usize] & size.mask();
            match binary_op {
                JitBinaryOp::Add => {
                    let result = dst_value.wrapping_add(immediate);
                    cpu.set_add_flags(immediate, dst_value, result, size);
                    portable_write_data_reg(cpu, dst, size, result);
                }
                JitBinaryOp::Sub => {
                    let result = dst_value.wrapping_sub(immediate);
                    cpu.set_sub_flags(immediate, dst_value, result, size);
                    portable_write_data_reg(cpu, dst, size, result);
                }
                JitBinaryOp::And => {
                    let result = immediate & dst_value;
                    cpu.set_logic_flags(result, size);
                    portable_write_data_reg(cpu, dst, size, result);
                }
                JitBinaryOp::Or => {
                    let result = immediate | dst_value;
                    cpu.set_logic_flags(result, size);
                    portable_write_data_reg(cpu, dst, size, result);
                }
                JitBinaryOp::Eor => {
                    let result = immediate ^ dst_value;
                    cpu.set_logic_flags(result, size);
                    portable_write_data_reg(cpu, dst, size, result);
                }
                JitBinaryOp::Cmp => {
                    let result = dst_value.wrapping_sub(immediate);
                    cpu.set_cmp_flags(immediate, dst_value, result, size);
                }
            }
            cycles
        }
        JitTraceOp::MulWordDataReg {
            src,
            dst,
            signed,
            m68000_timing,
        } => {
            let src_word = cpu.dar[src as usize] as u16;
            let dst_word = cpu.dar[dst as usize] as u16;
            let result = if signed {
                (i32::from(src_word as i16) * i32::from(dst_word as i16)) as u32
            } else {
                u32::from(src_word) * u32::from(dst_word)
            };
            cpu.dar[dst as usize] = result;
            cpu.set_logic_flags(result, Size::Long);
            if m68000_timing {
                let variable = if signed {
                    let shifted = u32::from(src_word) << 1;
                    ((shifted ^ (shifted >> 1)) & 0xFFFF).count_ones()
                } else {
                    src_word.count_ones()
                };
                38 + 2 * variable as i32
            } else {
                42
            }
        }
        JitTraceOp::MulWordImmediate {
            immediate,
            dst,
            signed,
            m68000_timing,
        } => {
            let dst_word = cpu.dar[dst as usize] as u16;
            let result = if signed {
                (i32::from(immediate as i16) * i32::from(dst_word as i16)) as u32
            } else {
                u32::from(immediate) * u32::from(dst_word)
            };
            cpu.dar[dst as usize] = result;
            cpu.set_logic_flags(result, Size::Long);
            if m68000_timing {
                // 68000 multiply time varies with the multiplier's bit
                // pattern, exactly as for the register form -- plus the
                // 4-cycle word-immediate operand fetch the interpreter
                // charges via ea_source_cycles(Immediate, Word), which the
                // register form does not have.
                let variable = if signed {
                    let shifted = u32::from(immediate) << 1;
                    ((shifted ^ (shifted >> 1)) & 0xFFFF).count_ones()
                } else {
                    immediate.count_ones()
                };
                42 + 2 * variable as i32
            } else {
                42
            }
        }
        JitTraceOp::MulLongDataReg { src, dst, signed } => {
            let src_value = cpu.dar[src as usize];
            let dst_value = cpu.dar[dst as usize];
            let (result, overflow) = if signed {
                let product = (src_value as i32 as i64).wrapping_mul(dst_value as i32 as i64);
                let result = product as u32;
                (result, product != result as i32 as i64)
            } else {
                let product = u64::from(src_value).wrapping_mul(u64::from(dst_value));
                (product as u32, (product >> 32) != 0)
            };
            cpu.dar[dst as usize] = result;
            cpu.set_logic_flags(result, Size::Long);
            cpu.v_flag = if overflow { VFLAG_SET } else { 0 };
            40
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
        JitTraceOp::AddrCmpImmediate {
            immediate,
            dst,
            size,
            cycles,
        } => {
            let src = if size == Size::Word {
                immediate as u16 as i16 as i32 as u32
            } else {
                immediate
            };
            let dst_value = cpu.dar[8 + dst as usize];
            let result = dst_value.wrapping_sub(src);
            cpu.set_cmp_flags(src, dst_value, result, Size::Long);
            cycles
        }
        JitTraceOp::LeaAn {
            base,
            dst,
            displacement,
            cycles,
        } => {
            cpu.dar[8 + dst as usize] =
                cpu.dar[8 + base as usize].wrapping_add(displacement as i32 as u32);
            cycles
        }
        JitTraceOp::LeaIndex { src, dst, cycles } => {
            let JitEa::Index {
                base,
                index,
                index_long,
                scale,
                displacement,
            } = src
            else {
                unreachable!("indexed LEA decoder admitted an unsupported EA")
            };
            let raw_index = match index {
                JitDirectReg::Data(reg) => cpu.dar[reg as usize],
                JitDirectReg::Addr(reg) => cpu.dar[8 + reg as usize],
            };
            let index_value = if index_long {
                raw_index
            } else {
                raw_index as u16 as i16 as i32 as u32
            };
            cpu.dar[8 + dst as usize] = cpu.dar[8 + base as usize]
                .wrapping_add(index_value.wrapping_shl(u32::from(scale)))
                .wrapping_add(displacement as i32 as u32);
            cycles
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
        JitTraceOp::Exg { opcode } => {
            let rx = ((opcode >> 9) & 7) as usize;
            let ry = (opcode & 7) as usize;
            match (opcode >> 3) & 0x1F {
                0x08 => {
                    let tmp = cpu.d(rx);
                    cpu.set_d(rx, cpu.d(ry));
                    cpu.set_d(ry, tmp);
                }
                0x09 => {
                    let tmp = cpu.a(rx);
                    cpu.set_a(rx, cpu.a(ry));
                    cpu.set_a(ry, tmp);
                }
                0x11 => {
                    let tmp = cpu.d(rx);
                    cpu.set_d(rx, cpu.a(ry));
                    cpu.set_a(ry, tmp);
                }
                _ => {}
            }
            6
        }
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
        JitTraceOp::CmpiWordMem { .. } => {
            unreachable!("CmpiWordMem is handled by execute_portable_cmpi_word_mem")
        }
        JitTraceOp::TstMem { .. } => {
            unreachable!("TstMem is handled by execute_portable_tst_mem")
        }
        JitTraceOp::ClrMem { .. } => {
            unreachable!("ClrMem is handled by execute_portable_clr_mem")
        }
        JitTraceOp::MoveImmMem { .. } => {
            unreachable!("MoveImmMem is handled by execute_portable_move_imm_mem")
        }
        JitTraceOp::AddrCmpMemToReg { .. } => {
            unreachable!("AddrCmpMemToReg is handled by execute_portable_addr_cmp_mem_to_reg")
        }
        JitTraceOp::AddRegToMem { .. } => {
            unreachable!("AddRegToMem is handled by execute_portable_add_reg_to_mem")
        }
        JitTraceOp::MemAddqSubq { .. } => {
            unreachable!("MemAddqSubq is handled by execute_portable_mem_addq_subq")
        }
        JitTraceOp::AnDispUnary { .. } | JitTraceOp::AnDispBit { .. } => {
            unreachable!("AnDisp ops are handled by execute_portable_an_disp")
        }
        JitTraceOp::PeaDisp { .. } => {
            unreachable!("PeaDisp is handled by execute_portable_pea_disp")
        }
        JitTraceOp::Link { .. } | JitTraceOp::Unlk { .. } => {
            unreachable!("LINK/UNLK are handled by execute_portable_link/unlk")
        }
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn portable_read_reg(cpu: &CpuCore, reg: JitDirectReg, size: Size) -> u32 {
    match reg {
        JitDirectReg::Data(reg) => cpu.dar[reg as usize] & size.mask(),
        JitDirectReg::Addr(reg) => cpu.dar[8 + reg as usize] & size.mask(),
    }
}

#[cfg(any(not(feature = "jit"), target_family = "wasm", test))]
fn portable_write_data_reg(cpu: &mut CpuCore, reg: u8, size: Size, value: u32) {
    let reg = reg as usize;
    let mask = size.mask();
    cpu.dar[reg] = (cpu.dar[reg] & !mask) | (value & mask);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
        JitTraceOp::BinaryImmediateDataReg {
            op: binary_op,
            immediate,
            dst,
            size,
            ..
        } => {
            let src_value = iconst_u32(builder, immediate);
            let dst_value = load_reg_sized(builder, cpu, JitDirectReg::Data(dst), size);
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
        JitTraceOp::MulWordImmediate {
            immediate,
            dst,
            signed,
            m68000_timing,
        } => {
            // The multiplicand is fixed, so its sign extension and -- on the
            // 68000 -- its cycle contribution both fold at compile time.
            let dst_word = load_reg_sized(builder, cpu, JitDirectReg::Data(dst), Size::Word);
            let (src_value, dst_value) = if signed {
                (
                    iconst_u32(builder, immediate as i16 as i32 as u32),
                    sign_extend_word(builder, dst_word),
                )
            } else {
                (iconst_u32(builder, u32::from(immediate)), dst_word)
            };
            let result = builder.ins().imul(src_value, dst_value);
            store_reg(builder, cpu, JitDirectReg::Data(dst), result);
            set_logic_flags(builder, cpu, result, Size::Long);

            if m68000_timing {
                let variable = if signed {
                    let shifted = u32::from(immediate) << 1;
                    ((shifted ^ (shifted >> 1)) & 0xFFFF).count_ones()
                } else {
                    immediate.count_ones()
                };
                // 38 + 2*bits + the 4-cycle word-immediate operand fetch,
                // matching the interpreter's exec_mulu/exec_muls exactly.
                cycles_const(builder, 42 + 2 * variable as i32)
            } else {
                cycles_const(builder, 42)
            }
        }
        JitTraceOp::MulWordDataReg {
            src,
            dst,
            signed,
            m68000_timing,
        } => {
            let src_word = load_reg_sized(builder, cpu, JitDirectReg::Data(src), Size::Word);
            let dst_word = load_reg_sized(builder, cpu, JitDirectReg::Data(dst), Size::Word);
            let (src_value, dst_value) = if signed {
                (
                    sign_extend_word(builder, src_word),
                    sign_extend_word(builder, dst_word),
                )
            } else {
                (src_word, dst_word)
            };
            let result = builder.ins().imul(src_value, dst_value);
            store_reg(builder, cpu, JitDirectReg::Data(dst), result);
            set_logic_flags(builder, cpu, result, Size::Long);

            if m68000_timing {
                let bits = if signed {
                    let shifted = builder.ins().ishl_imm(src_word, 1);
                    let previous = builder.ins().ushr_imm(shifted, 1);
                    let transitions = builder.ins().bxor(shifted, previous);
                    builder.ins().band_imm(transitions, 0xFFFF)
                } else {
                    src_word
                };
                let variable = builder.ins().popcnt(bits);
                let doubled = builder.ins().ishl_imm(variable, 1);
                builder.ins().iadd_imm(doubled, 38)
            } else {
                cycles_const(builder, 42)
            }
        }
        JitTraceOp::MulLongDataReg { src, dst, signed } => {
            let src_value = load_reg(builder, cpu, JitDirectReg::Data(src));
            let dst_value = load_reg(builder, cpu, JitDirectReg::Data(dst));
            let (product, overflow) = if signed {
                let src64 = builder.ins().sextend(types::I64, src_value);
                let dst64 = builder.ins().sextend(types::I64, dst_value);
                let product = builder.ins().imul(src64, dst64);
                let result = builder.ins().ireduce(types::I32, product);
                let result64 = builder.ins().sextend(types::I64, result);
                let overflow = builder.ins().icmp(IntCC::NotEqual, product, result64);
                (product, overflow)
            } else {
                let src64 = builder.ins().uextend(types::I64, src_value);
                let dst64 = builder.ins().uextend(types::I64, dst_value);
                let product = builder.ins().imul(src64, dst64);
                let high = builder.ins().ushr_imm(product, 32);
                let overflow = builder.ins().icmp_imm(IntCC::NotEqual, high, 0);
                (product, overflow)
            };
            let result = builder.ins().ireduce(types::I32, product);
            store_reg(builder, cpu, JitDirectReg::Data(dst), result);
            set_logic_flags(builder, cpu, result, Size::Long);
            let overflow = select_flag(builder, overflow, VFLAG_SET);
            store_value_u32(builder, cpu, offset_of!(CpuCore, v_flag), overflow);
            cycles_const(builder, 40)
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
        JitTraceOp::AddrCmpImmediate {
            immediate,
            dst,
            size,
            cycles,
        } => {
            let src_value = iconst_u32(builder, immediate);
            let src_value = if size == Size::Word {
                sign_extend_word(builder, src_value)
            } else {
                src_value
            };
            let dst_value = load_reg(builder, cpu, JitDirectReg::Addr(dst));
            let result = builder.ins().isub(dst_value, src_value);
            set_cmp_flags(builder, cpu, src_value, dst_value, result, Size::Long);
            cycles_const(builder, cycles)
        }
        JitTraceOp::LeaAn {
            base,
            dst,
            displacement,
            ..
        } => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(base));
            let address = builder.ins().iadd_imm(base, displacement as i64);
            store_reg(builder, cpu, JitDirectReg::Addr(dst), address);
            cycles_const(builder, op.op.max_cycles())
        }
        JitTraceOp::LeaIndex { src, dst, .. } => {
            let JitEa::Index {
                base,
                index,
                index_long,
                scale,
                displacement,
            } = src
            else {
                unreachable!("indexed LEA decoder admitted an unsupported EA")
            };
            let base = load_reg(builder, cpu, JitDirectReg::Addr(base));
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
            let address = builder.ins().iadd(base, index);
            let address = builder.ins().iadd_imm(address, displacement as i64);
            store_reg(builder, cpu, JitDirectReg::Addr(dst), address);
            cycles_const(builder, op.op.max_cycles())
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
            debug_assert!(matches!((op, direction), (0, 0) | (1, 0) | (1, 1)));
            if count_is_register {
                let shift = RegisterCountShift {
                    reg,
                    size,
                    count_reg: count_or_reg,
                    direction,
                    op,
                };
                return emit_register_count_shift(builder, cpu, shift, pre020);
            }
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
                (1, 0) => {
                    let result = builder.ins().ushr_imm(value, i64::from(shift));
                    let shifted_out = if shift > bits {
                        iconst_u32(builder, 0)
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

            if pre020 {
                let base = if size == Size::Long { 8 } else { 6 };
                cycles_const(builder, base + 2 * shift as i32)
            } else {
                // 68020+ has a barrel shifter. Its handler returns a fixed
                // pre-scaled cost independent of the encoded count.
                cycles_const(builder, 6)
            }
        }
        JitTraceOp::MoveMem { .. } => unreachable!("MoveMem is emitted by emit_move_mem"),
        JitTraceOp::MovemWordPostInc { .. } => {
            unreachable!("MovemWordPostInc is emitted by emit_movem_word_postinc")
        }
        JitTraceOp::AluMemToReg { .. } => {
            unreachable!("AluMemToReg is emitted by emit_alu_mem_to_reg")
        }
        JitTraceOp::CmpiWordMem { .. } => {
            unreachable!("CmpiWordMem is emitted by emit_cmpi_word_mem")
        }
        JitTraceOp::TstMem { .. } => {
            unreachable!("TstMem is emitted by emit_tst_mem")
        }
        JitTraceOp::ClrMem { .. } => {
            unreachable!("ClrMem is emitted by emit_clr_mem")
        }
        JitTraceOp::MoveImmMem { .. } => {
            unreachable!("MoveImmMem is emitted by emit_move_imm_mem")
        }
        JitTraceOp::AddrCmpMemToReg { .. } => {
            unreachable!("AddrCmpMemToReg is emitted by emit_addr_cmp_mem_to_reg")
        }
        JitTraceOp::AddRegToMem { .. } => {
            unreachable!("AddRegToMem is emitted by emit_add_reg_to_mem")
        }
        JitTraceOp::MemAddqSubq { .. } => {
            unreachable!("MemAddqSubq is emitted by emit_mem_addq_subq")
        }
        JitTraceOp::Link { .. } | JitTraceOp::Unlk { .. } => {
            unreachable!("LINK/UNLK are emitted by emit_link/emit_unlk")
        }
        JitTraceOp::AnDispUnary { .. } | JitTraceOp::AnDispBit { .. } => {
            unreachable!("AnDisp ops are emitted by emit_an_disp_mem")
        }
        JitTraceOp::PeaDisp { .. } => {
            unreachable!("PeaDisp is emitted by emit_pea_disp")
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
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
#[derive(Clone, Copy)]
struct BailAt {
    ops_before: RetiredBefore,
    cycles_before: Value,
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
#[derive(Clone, Copy)]
enum RetiredBefore {
    Constant(u32),
    Dynamic(Value),
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
struct BailReq {
    block: Block,
    pc: u32,
    at: BailAt,
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
struct MoveMemOp {
    pc: u32,
    size: Size,
    src: JitEa,
    dst: JitEa,
}

/// Branch to `bail` when `bad` holds; continue emitting in a fresh block.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn branch_guard(builder: &mut FunctionBuilder<'_>, bail: Block, bad: Value) {
    let cont = builder.create_block();
    builder.ins().brif(bad, bail, &[], cont, &[]);
    builder.switch_to_block(cont);
}

/// Alignment + window-range checks for an access of `size` at raw address
/// `addr`. Returns `(window_offset, masked_address)`; branches to `bail`
/// on any miss.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn window_host_addr(builder: &mut FunctionBuilder<'_>, env: &MemEnv, off: Value) -> Value {
    let off_ptr = if env.fm_ptr_ty == types::I32 {
        off
    } else {
        builder.ins().uextend(env.fm_ptr_ty, off)
    };
    builder.ins().iadd(env.fm_ptr, off_ptr)
}

/// Big-endian sized load from the window; result is a zero-extended I32.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

/// Emit a read-only brief-indexed TST. The extension word is decoded while
/// recording, so the hot path performs only the live address calculation,
/// checked load, and condition-code update.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_tst_mem(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::TstMem {
        size,
        src:
            JitEa::Index {
                base,
                index,
                index_long,
                scale,
                displacement,
            },
    } = trace.op
    else {
        unreachable!("memory TST decoder admitted an unsupported EA")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let base = load_reg(builder, cpu, JitDirectReg::Addr(base));
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
    let address = builder.ins().iadd(base, index);
    let address = builder.ins().iadd_imm(address, displacement as i64);
    let (off, _) = checked_window_off(builder, env, bail, address, size);
    let value = window_load(builder, env, off, size);
    set_logic_flags(builder, cpu, value, size);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit CLR to a brief-indexed destination. The address passes the window
/// and code-overlap checks before the store and the flag writes, so a bail
/// commits nothing. CLR sets Z and clears N, V, and C; X is untouched.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_clr_mem(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::ClrMem { size, dst } = trace.op else {
        unreachable!("expected a memory CLR trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    // (address, register update committed only after every guard passes)
    let (address, updated_reg) = match dst {
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(base));
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
            let address = builder.ins().iadd(base, index);
            (builder.ins().iadd_imm(address, displacement as i64), None)
        }
        JitEa::PreDec(reg) => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            let address = builder
                .ins()
                .iadd_imm(base, -i64::from(jit_ea_step(size, reg)));
            (address, Some((reg, address)))
        }
        JitEa::AbsWord(address) | JitEa::AbsLong(address) => (iconst_u32(builder, address), None),
        _ => unreachable!("memory CLR decoder admitted an unsupported EA"),
    };
    let (off, masked) = checked_window_off(builder, env, bail, address, size);
    guard_store_not_code(builder, env, bail, masked, size);
    if let Some((reg, value)) = updated_reg {
        store_reg(builder, cpu, JitDirectReg::Addr(reg), value);
    }
    let zero = iconst_u32(builder, 0);
    window_store(builder, env, off, size, zero);
    store_u32(builder, cpu, offset_of!(CpuCore, n_flag), 0);
    store_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), 0);
    store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
    store_u32(builder, cpu, offset_of!(CpuCore, c_flag), 0);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit MOVE #imm to a memory destination. The address passes the window
/// and code-overlap checks before the register update, the store, and the
/// flag writes, so a bail commits nothing.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_move_imm_mem(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::MoveImmMem { size, value, dst } = trace.op else {
        unreachable!("expected an immediate memory MOVE trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let (address, updated_reg) = match dst {
        JitEa::Ind(reg) => (load_reg(builder, cpu, JitDirectReg::Addr(reg)), None),
        JitEa::PostInc(reg) => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            let next = builder
                .ins()
                .iadd_imm(base, i64::from(jit_ea_step(size, reg)));
            (base, Some((reg, next)))
        }
        JitEa::PreDec(reg) => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            let address = builder
                .ins()
                .iadd_imm(base, -i64::from(jit_ea_step(size, reg)));
            (address, Some((reg, address)))
        }
        JitEa::Disp(reg, displacement) => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            (builder.ins().iadd_imm(base, displacement as i64), None)
        }
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(base));
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
            let address = builder.ins().iadd(base, index);
            (builder.ins().iadd_imm(address, displacement as i64), None)
        }
        _ => unreachable!("immediate MOVE decoder admitted an unsupported EA"),
    };
    let (off, masked) = checked_window_off(builder, env, bail, address, size);
    guard_store_not_code(builder, env, bail, masked, size);
    if let Some((reg, next)) = updated_reg {
        store_reg(builder, cpu, JitDirectReg::Addr(reg), next);
    }
    let value = iconst_u32(builder, value);
    window_store(builder, env, off, size, value);
    set_logic_flags(builder, cpu, value, size);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit a read-only memory-to-register ALU operation. All address checks run
/// before flags are committed, so a miss can re-execute the instruction via
/// full dispatch without rolling back architectural state.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    let addr = match src {
        JitEa::Ind(reg) => load_reg(builder, cpu, JitDirectReg::Addr(reg)),
        JitEa::Disp(reg, displacement) => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            builder.ins().iadd_imm(base, displacement as i64)
        }
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(base));
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
            let address = builder.ins().iadd(base, index);
            builder.ins().iadd_imm(address, displacement as i64)
        }
        _ => unreachable!("ALU trace decoder admitted an unsupported EA"),
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

/// Emit indexed CMPI.W as a checked read followed by a register-only flag
/// update. A failed address/window/alignment check reaches the side exit
/// before any condition code is changed, so full dispatch can retry it.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_cmpi_word_mem(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::CmpiWordMem {
        immediate,
        src:
            JitEa::Index {
                base,
                index,
                index_long,
                scale,
                displacement,
            },
    } = trace.op
    else {
        unreachable!("indexed CMPI decoder admitted an unsupported EA")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let base = load_reg(builder, cpu, JitDirectReg::Addr(base));
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
    let address = builder.ins().iadd(base, index);
    let address = builder.ins().iadd_imm(address, displacement as i64);
    let (off, _) = checked_window_off(builder, env, bail, address, Size::Word);
    let dst_value = window_load(builder, env, off, Size::Word);
    let immediate = iconst_u32(builder, u32::from(immediate));
    let result = builder.ins().isub(dst_value, immediate);
    set_cmp_flags(builder, cpu, immediate, dst_value, result, Size::Word);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit CMPA.W/L from checked guest memory. Address calculation and the
/// fast-memory bounds/alignment checks precede every flag write, so a failed
/// window access can fall back and re-execute the instruction atomically.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_addr_cmp_mem_to_reg(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::AddrCmpMemToReg { size, src, dst } = trace.op else {
        unreachable!("expected memory-to-address-register compare trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let address = match src {
        JitEa::Ind(reg) => load_reg(builder, cpu, JitDirectReg::Addr(reg)),
        JitEa::Disp(reg, displacement) => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            builder.ins().iadd_imm(base, displacement as i64)
        }
        _ => unreachable!("address compare decoder admitted unsupported EA"),
    };
    let (off, _) = checked_window_off(builder, env, bail, address, size);
    let src_value = window_load(builder, env, off, size);
    let src_value = if size == Size::Word {
        sign_extend_word(builder, src_value)
    } else {
        src_value
    };
    let dst_value = load_reg(builder, cpu, JitDirectReg::Addr(dst));
    let result = builder.ins().isub(dst_value, src_value);
    set_cmp_flags(builder, cpu, src_value, dst_value, result, Size::Long);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit ADD.W/L Dn,`<ea>`. The window, alignment, and self-modification guards
/// all run before memory, address-register, or flag state is changed.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_add_reg_to_mem(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::AddRegToMem {
        is_sub,
        size,
        src,
        dst,
    } = trace.op
    else {
        unreachable!("expected register-to-memory ADD/SUB trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let (reg, addr, next) = match dst {
        JitEa::Ind(reg) => {
            let addr = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            (reg, addr, None)
        }
        JitEa::PostInc(reg) => {
            let addr = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            let next = builder
                .ins()
                .iadd_imm(addr, i64::from(jit_ea_step(size, reg)));
            (reg, addr, Some(next))
        }
        JitEa::Disp(reg, displacement) => {
            let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
            (reg, builder.ins().iadd_imm(base, displacement as i64), None)
        }
        _ => unreachable!("unsupported register-to-memory ADD/SUB destination"),
    };
    let (off, masked) = checked_window_off(builder, env, bail, addr, size);
    guard_store_not_code(builder, env, bail, masked, size);
    let dst_value = window_load(builder, env, off, size);
    let src_value = load_reg(builder, cpu, JitDirectReg::Data(src));
    let src_value = mask_value(builder, src_value, size);
    let result = if is_sub {
        builder.ins().isub(dst_value, src_value)
    } else {
        builder.ins().iadd(dst_value, src_value)
    };

    window_store(builder, env, off, size, result);
    if let Some(next) = next {
        store_reg(builder, cpu, JitDirectReg::Addr(reg), next);
    }
    if is_sub {
        set_sub_flags(builder, cpu, src_value, dst_value, result, size);
    } else {
        set_add_flags(builder, cpu, src_value, dst_value, result, size);
    }
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit terminal `JSR (An)`. The stack write is checked before the stack
/// pointer, flow state, or PC changes, so a miss can re-execute the call via
/// full dispatch. A successful call ends this non-self-loop trace; writing
/// into its code is therefore safe because any later entry revalidates it.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

/// Emit a checked ADDQ/SUBQ read-modify-write through an address-register-
/// relative memory EA. All guards precede the memory and flag writes so a
/// failed operation can be retried through full dispatch without partial
/// architectural state.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_mem_addq_subq(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::MemAddqSubq {
        data,
        size,
        dst,
        is_sub,
    } = trace.op
    else {
        unreachable!()
    };
    let (reg, displacement) = match dst {
        JitEa::Ind(reg) => (reg, 0),
        JitEa::Disp(reg, displacement) => (reg, displacement),
        _ => unreachable!("only address-register-relative ADDQ/SUBQ is traceable"),
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });
    let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
    let addr = builder.ins().iadd_imm(base, displacement as i64);
    let (off, masked) = checked_window_off(builder, env, bail, addr, size);
    let value = window_load(builder, env, off, size);
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
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit displacement-memory operations with the displacement baked into the
/// trace, leaving only the live An value and fastmem bounds to check at
/// runtime.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

/// Emit `PEA (d16,An)`: the pushed address is computed from the
/// pre-decrement register state, then the decremented stack slot passes the
/// alignment/window/code-overlap checks before the store and the A7 update,
/// so a failed check bails with nothing committed. PEA changes no condition
/// codes.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_pea_disp(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::PeaDisp { reg, displacement } = trace.op else {
        unreachable!("expected PEA displacement trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let base = load_reg(builder, cpu, JitDirectReg::Addr(reg));
    let address = builder.ins().iadd_imm(base, displacement as i64);
    let sp = if reg == 7 {
        base
    } else {
        load_reg(builder, cpu, JitDirectReg::Addr(7))
    };
    let new_sp = builder.ins().iadd_imm(sp, -4);
    let (off, masked) = checked_window_off(builder, env, bail, new_sp, Size::Long);
    guard_store_not_code(builder, env, bail, masked, Size::Long);
    window_store(builder, env, off, Size::Long, address);
    store_reg(builder, cpu, JitDirectReg::Addr(7), new_sp);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit `LINK An,#d16`. Interpreter order: push An's original value, then
/// An = the decremented SP, then SP moves by the displacement. All checks
/// precede the store and both register writes.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_link(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::Link { reg, displacement } = trace.op else {
        unreachable!("expected LINK trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let an = load_reg(builder, cpu, JitDirectReg::Addr(reg));
    let sp = load_reg(builder, cpu, JitDirectReg::Addr(7));
    let new_sp = builder.ins().iadd_imm(sp, -4);
    let (off, masked) = checked_window_off(builder, env, bail, new_sp, Size::Long);
    guard_store_not_code(builder, env, bail, masked, Size::Long);
    window_store(builder, env, off, Size::Long, an);
    store_reg(builder, cpu, JitDirectReg::Addr(reg), new_sp);
    let final_sp = builder.ins().iadd_imm(new_sp, displacement as i64);
    store_reg(builder, cpu, JitDirectReg::Addr(7), final_sp);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit `UNLK An`. Interpreter order: SP reloads from An, the saved frame
/// pointer pops into An. The load is checked before either register writes.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn emit_unlk(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    trace: TraceBuildOp,
    env: &MemEnv,
    bails: &mut Vec<BailReq>,
    at: BailAt,
) -> Value {
    let JitTraceOp::Unlk { reg } = trace.op else {
        unreachable!("expected UNLK trace")
    };
    let bail = builder.create_block();
    bails.push(BailReq {
        block: bail,
        pc: trace.pc,
        at,
    });

    let addr = load_reg(builder, cpu, JitDirectReg::Addr(reg));
    let (off, _) = checked_window_off(builder, env, bail, addr, Size::Long);
    let value = window_load(builder, env, off, Size::Long);
    let new_sp = builder.ins().iadd_imm(addr, 4);
    store_reg(builder, cpu, JitDirectReg::Addr(7), new_sp);
    store_reg(builder, cpu, JitDirectReg::Addr(reg), value);
    cycles_const(builder, trace.op.max_cycles())
}

/// Emit a MOVE/MOVEA with memory operands. All alignment/window/code-overlap
/// checks run before anything commits; each check branches to a bail block
/// that sets `pc = op.pc` and returns the ops retired before this one, so a
/// bailing instruction re-executes through full dispatch.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
        JitEa::AbsWord(address) | JitEa::AbsLong(address) => {
            let address = iconst_u32(builder, address);
            let (off, _) = checked_window_off(builder, env, bail, address, size);
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
        JitEa::Index {
            base,
            index,
            index_long,
            scale,
            displacement,
        } => {
            let base_value = dst_base(builder, base);
            // The index register must also observe a staged source
            // adjustment (`MOVE.W (A2)+,(d8,A1,A2.W)`): the source
            // post-increment commits before the destination EA evaluates.
            let raw_index = match (index, staged) {
                (JitDirectReg::Addr(ir), Some((sr, v))) if sr == ir => v,
                _ => load_reg(builder, cpu, index),
            };
            let index_value = if index_long {
                raw_index
            } else {
                let word = builder.ins().ireduce(types::I16, raw_index);
                builder.ins().sextend(types::I32, word)
            };
            let index_value = if scale == 0 {
                index_value
            } else {
                builder.ins().ishl_imm(index_value, i64::from(scale))
            };
            let a = builder.ins().iadd(base_value, index_value);
            let addr = builder.ins().iadd_imm(a, displacement as i64);
            let (off, masked) = checked_window_off(builder, env, bail, addr, size);
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
            window_store(builder, env, off, size, value);
            set_logic_flags(builder, cpu, value, size);
        }
        JitEa::AbsWord(address) | JitEa::AbsLong(address) => {
            let address = iconst_u32(builder, address);
            let (off, masked) = checked_window_off(builder, env, bail, address, size);
            guard_store_not_code(builder, env, bail, masked, size);
            commit_staged(builder);
            window_store(builder, env, off, size, value);
            set_logic_flags(builder, cpu, value, size);
        }
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn cycles_const(builder: &mut FunctionBuilder<'_>, cycles: i32) -> Value {
    builder.ins().iconst(types::I32, cycles as i64)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn load_reg_sized(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    reg: JitDirectReg,
    size: Size,
) -> Value {
    let value = load_reg(builder, cpu, reg);
    mask_value(builder, value, size)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn mask_value(builder: &mut FunctionBuilder<'_>, value: Value, size: Size) -> Value {
    if size == Size::Long {
        value
    } else {
        let mask = iconst_u32(builder, size_mask(size));
        builder.ins().band(value, mask)
    }
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn sign_extend_byte(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let shifted = builder.ins().ishl_imm(value, 24);
    builder.ins().sshr_imm(shifted, 24)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn sign_extend_word(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    let shifted = builder.ins().ishl_imm(value, 16);
    builder.ins().sshr_imm(shifted, 16)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn size_mask(size: Size) -> u32 {
    match size {
        Size::Byte => 0xFF,
        Size::Word => 0xFFFF,
        Size::Long => 0xFFFF_FFFF,
    }
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn size_msb(size: Size) -> u32 {
    match size {
        Size::Byte => 0x80,
        Size::Word => 0x8000,
        Size::Long => 0x8000_0000,
    }
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn set_logic_flags(builder: &mut FunctionBuilder<'_>, cpu: Value, value: Value, size: Size) {
    set_logic_flags_nv(builder, cpu, value, size);
    store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
    store_u32(builder, cpu, offset_of!(CpuCore, c_flag), 0);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn set_logic_flags_nv(builder: &mut FunctionBuilder<'_>, cpu: Value, value: Value, size: Size) {
    let value = mask_value(builder, value, size);
    let msb = iconst_u32(builder, size_msb(size));
    let sign_bits = builder.ins().band(value, msb);
    let n = flag_from_nonzero(builder, sign_bits, NFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, n_flag), n);
    store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), value);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn extend_flag_value(builder: &mut FunctionBuilder<'_>, cpu: Value) -> Value {
    let x_flag = load_u32(builder, cpu, offset_of!(CpuCore, x_flag));
    let has_x = builder.ins().icmp_imm(IntCC::NotEqual, x_flag, 0);
    let one = iconst_u32(builder, 1);
    let zero = iconst_u32(builder, 0);
    builder.ins().select(has_x, one, zero)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
/// Logical NOT for the 0/1 booleans produced by `icmp`.
///
/// `bnot` is bitwise and must not be used here: `bnot(0x01) == 0xFE`,
/// which is still non-zero and therefore still "true" to `select`/`brif`.
/// Flipping the low bit keeps the value a canonical 0/1 boolean.
#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn not_bool(builder: &mut FunctionBuilder<'_>, value: Value) -> Value {
    builder.ins().bxor_imm(value, 1)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn flag_is_set(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize) -> Value {
    let flag = load_u32(builder, cpu, offset);
    builder.ins().icmp_imm(IntCC::NotEqual, flag, 0)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn flag_is_zero_set(builder: &mut FunctionBuilder<'_>, cpu: Value) -> Value {
    let not_z = load_u32(builder, cpu, offset_of!(CpuCore, not_z_flag));
    builder.ins().icmp_imm(IntCC::Equal, not_z, 0)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn bool_const(builder: &mut FunctionBuilder<'_>, value: bool) -> Value {
    let zero = iconst_u32(builder, 0);
    if value {
        builder.ins().icmp_imm(IntCC::Equal, zero, 0)
    } else {
        builder.ins().icmp_imm(IntCC::NotEqual, zero, 0)
    }
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
#[derive(Debug, Clone, Copy)]
struct RegisterCountShift {
    reg: u8,
    size: Size,
    count_reg: u8,
    direction: u8,
    op: u8,
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
/// `ASR`/`LSR`/`LSL Dx,Dy`: the shift distance is only known at run time,
/// so the count clamping, the shifted-out bit, and the cycle cost are all
/// computed in the trace rather than folded at compile time.
///
/// Two architectural details drive the shape of this code. The 68k takes
/// the count modulo 64, and a **zero count is not a no-op for the flags**:
/// it clears C and V and sets N/Z from the unshifted value, but leaves X
/// untouched. X is therefore read back and re-selected rather than simply
/// written.
fn emit_register_count_shift(
    builder: &mut FunctionBuilder<'_>,
    cpu: Value,
    shift: RegisterCountShift,
    pre020: bool,
) -> Value {
    let RegisterCountShift {
        reg,
        size,
        count_reg,
        direction,
        op,
    } = shift;
    let bits = size.bits() as u32;
    let value = load_reg_sized(builder, cpu, JitDirectReg::Data(reg), size);
    let raw_count = load_reg(builder, cpu, JitDirectReg::Data(count_reg));
    let count = builder.ins().band_imm(raw_count, 63);
    let count_is_zero = builder.ins().icmp_imm(IntCC::Equal, count, 0);

    let last_bit = iconst_u32(builder, bits - 1);
    let bits_value = iconst_u32(builder, bits);
    let zero = iconst_u32(builder, 0);
    // Clamped so no shift amount can exceed the value width, and so the
    // count-1 index stays defined when the count is zero (its flag results
    // are discarded below).
    let count_minus_one = builder.ins().iadd_imm(count, -1);
    let over_last = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThan, count_minus_one, last_bit);
    let shifted_out_index = builder.ins().select(over_last, last_bit, count_minus_one);
    let count_past_width = builder
        .ins()
        .icmp(IntCC::UnsignedGreaterThan, count, bits_value);
    let count_reaches_width =
        builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, count, bits_value);

    let (result, carry) = match (op, direction) {
        (0, 0) => {
            // ASR: shifting an all-sign-bits value further changes nothing,
            // so clamping the distance to the width reproduces the
            // architectural result for every larger count.
            let signed = match size {
                Size::Byte => sign_extend_byte(builder, value),
                Size::Word => sign_extend_word(builder, value),
                Size::Long => value,
            };
            // The shift amount clamps at the last bit position: a count of
            // exactly the operand width must still saturate to all sign
            // bits, and shifting an I32 by 32 or more is not defined here.
            let count_over_last = builder
                .ins()
                .icmp(IntCC::UnsignedGreaterThan, count, last_bit);
            let clamped = builder.ins().select(count_over_last, last_bit, count);
            let result = builder.ins().sshr(signed, clamped);
            let bit = builder.ins().ushr(value, shifted_out_index);
            let carry = builder.ins().band_imm(bit, 1);
            (result, carry)
        }
        (1, 0) => {
            // LSR: zero once the count reaches the width; the last bit
            // shifted out is still the carry when the count equals it.
            let plain = builder.ins().ushr(value, count);
            let result = builder.ins().select(count_reaches_width, zero, plain);
            let bit = builder.ins().ushr(value, shifted_out_index);
            let bit = builder.ins().band_imm(bit, 1);
            let carry = builder.ins().select(count_past_width, zero, bit);
            (result, carry)
        }
        (1, 1) => {
            // LSL: mirror of LSR, with the carry taken from the bit that
            // reaches the top on the final iteration.
            let plain = builder.ins().ishl(value, count);
            let result = builder.ins().select(count_reaches_width, zero, plain);
            let index = builder.ins().isub(bits_value, count);
            let index = builder.ins().band_imm(index, 31);
            let bit = builder.ins().ushr(value, index);
            let bit = builder.ins().band_imm(bit, 1);
            let carry = builder.ins().select(count_past_width, zero, bit);
            (result, carry)
        }
        _ => unreachable!("unsupported native register shift"),
    };

    let result = mask_value(builder, result, size);
    // A zero count leaves the register unchanged, which `value` already is.
    let result = builder.ins().select(count_is_zero, value, result);
    let result = mask_value(builder, result, size);
    write_data_reg_sized(builder, cpu, reg, size, result);

    let carry = builder.ins().select(count_is_zero, zero, carry);
    let carry_flag = flag_from_nonzero(builder, carry, CFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, c_flag), carry_flag);
    // X is preserved across a zero-count shift.
    let previous_x = load_u32(builder, cpu, offset_of!(CpuCore, x_flag));
    let x_flag = builder.ins().select(count_is_zero, previous_x, carry_flag);
    store_value_u32(builder, cpu, offset_of!(CpuCore, x_flag), x_flag);
    store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
    set_logic_flags_nv(builder, cpu, result, size);

    if pre020 {
        let base = if size == Size::Long { 8 } else { 6 };
        let scaled = builder.ins().imul_imm(count, 2);
        builder.ins().iadd_imm(scaled, base)
    } else {
        // 68020+ barrel shifter: fixed cost, as in the immediate form.
        cycles_const(builder, 6)
    }
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn flag_from_nonzero(builder: &mut FunctionBuilder<'_>, value: Value, flag: u32) -> Value {
    let condition = builder.ins().icmp_imm(IntCC::NotEqual, value, 0);
    select_flag(builder, condition, flag)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn select_flag(builder: &mut FunctionBuilder<'_>, condition: Value, flag: u32) -> Value {
    let flag_value = iconst_u32(builder, flag);
    let zero = iconst_u32(builder, 0);
    builder.ins().select(condition, flag_value, zero)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn load_u32(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize) -> Value {
    builder
        .ins()
        .load(types::I32, MemFlags::trusted(), cpu, offset as i32)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn load_u8(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize) -> Value {
    builder
        .ins()
        .load(types::I8, MemFlags::trusted(), cpu, offset as i32)
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn store_pc(builder: &mut FunctionBuilder<'_>, cpu: Value, pc: u32) {
    store_u32(builder, cpu, offset_of!(CpuCore, pc), pc);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn store_pc_value(builder: &mut FunctionBuilder<'_>, cpu: Value, pc: Value) {
    store_value_u32(builder, cpu, offset_of!(CpuCore, pc), pc);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn store_bool(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: bool) {
    let value = builder.ins().iconst(types::I8, i64::from(value as u8));
    builder
        .ins()
        .store(MemFlags::trusted(), value, cpu, offset as i32);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn store_u32(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: u32) {
    let value = iconst_u32(builder, value);
    store_value_u32(builder, cpu, offset, value);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn store_value_u32(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: Value) {
    store_value(builder, cpu, offset, value);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn store_value(builder: &mut FunctionBuilder<'_>, cpu: Value, offset: usize, value: Value) {
    builder
        .ins()
        .store(MemFlags::trusted(), value, cpu, offset as i32);
}

#[cfg(all(feature = "jit", not(target_family = "wasm")))]
fn iconst_u32(builder: &mut FunctionBuilder<'_>, value: u32) -> Value {
    builder.ins().iconst(types::I32, value as i32 as i64)
}

fn trace_cache_index(pc: u32) -> usize {
    ((pc >> 1) as usize) & (TRACE_CACHE_SIZE - 1)
}

#[cfg(test)]
mod portable_tests {
    use super::*;

    #[test]
    fn decodes_register_source_word_multiply() {
        let mut cpu = cpu();
        cpu.set_cpu_type(CpuType::M68040);
        let mut mem = super::super::memory::LinearMemoryBus::new(0x1_0000);
        mem.write_word(0x6004, 0xC0C3);
        let op = decode_trace_op(&cpu, &mut mem, 0x6004, CpuType::M68040).expect("MULU.W D3,D0");
        assert!(matches!(
            op.op,
            JitTraceOp::MulWordDataReg {
                src: 3,
                dst: 0,
                signed: false,
                m68000_timing: false,
            }
        ));
    }

    #[test]
    fn register_word_multiply_matches_interpreter_and_decoded_cache() {
        use super::super::ea::AddressingMode;

        for cpu_type in [CpuType::M68000, CpuType::M68010, CpuType::M68040] {
            for signed in [false, true] {
                for source in [0x0000u32, 0x8001, 0xFFFF] {
                    let opcode = 0xC000 | (1 << 9) | ((if signed { 7 } else { 3 }) << 6) | 5;
                    let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);

                    let mut expected = cpu();
                    expected.set_cpu_type(cpu_type);
                    expected.set_d(5, source);
                    expected.set_d(1, 0x8003);
                    expected.set_ccr(0x1F);
                    let expected_cycles = if signed {
                        expected.exec_muls(&mut bus, AddressingMode::DataDirect(5), 1)
                    } else {
                        expected.exec_mulu(&mut bus, AddressingMode::DataDirect(5), 1)
                    };

                    let decoded = DecodedSimpleOp::decode(cpu_type, opcode)
                        .expect("register-source word multiply must be cached");
                    let mut actual = cpu();
                    actual.set_cpu_type(cpu_type);
                    actual.set_d(5, source);
                    actual.set_d(1, 0x8003);
                    actual.set_ccr(0x1F);
                    let actual_cycles = decoded.execute(&mut actual, &mut bus);

                    assert_eq!(
                        actual_cycles, expected_cycles,
                        "{cpu_type:?} signed={signed}"
                    );
                    assert_eq!(actual.d(1), expected.d(1), "{cpu_type:?} signed={signed}");
                    assert_eq!(
                        actual.get_ccr(),
                        expected.get_ccr(),
                        "{cpu_type:?} signed={signed}"
                    );
                }
            }
        }
    }

    #[test]
    fn register_long_multiply_matches_interpreter() {
        let cases = [
            (false, 0x0001_0000u32, 0x0001_0000u32),
            (false, 0xFFFF_FFFF, 2),
            (true, 0xFFFF_FFFEu32, 3),
            (true, 0x4000_0000, 4),
        ];

        for (signed, source, destination) in cases {
            let opcode = 0x4C03; // MULU.L/MULS.L D3,D2
            let extension = (2 << 12) | if signed { 0x0800 } else { 0 };
            let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
            bus.write_word(0x0100, opcode);
            bus.write_word(0x0102, extension);
            let op = decode_trace_op(&cpu(), &mut bus, 0x0100, CpuType::M68040)
                .expect("register long multiply must decode");
            assert!(matches!(
                op.op,
                JitTraceOp::MulLongDataReg {
                    src: 3,
                    dst: 2,
                    signed: decoded_signed,
                } if decoded_signed == signed
            ));

            let mut expected = cpu();
            expected.set_cpu_type(CpuType::M68040);
            expected.set_d(3, source);
            expected.set_d(2, destination);
            expected.set_ccr(0x1F);
            assert!(matches!(
                expected.step(&mut bus),
                super::super::types::StepResult::Ok { .. }
            ));

            let mut actual = cpu();
            actual.set_cpu_type(CpuType::M68040);
            actual.set_d(3, source);
            actual.set_d(2, destination);
            actual.set_ccr(0x1F);
            assert_eq!(
                execute_portable_op(&mut actual, op, 0x0100, 0x0104),
                Some(40)
            );
            assert_eq!(actual.d(2), expected.d(2), "signed={signed}");
            assert_eq!(actual.get_ccr(), expected.get_ccr(), "signed={signed}");
            assert_eq!(actual.get_ccr() & 0x10, 0x10, "X must be preserved");
        }

        let mut pre_020_bus = super::super::memory::LinearMemoryBus::new(0x1000);
        pre_020_bus.write_word(0x0100, 0x4C03);
        pre_020_bus.write_word(0x0102, 0x2800);
        assert!(decode_trace_op(&cpu(), &mut pre_020_bus, 0x0100, CpuType::M68000).is_none());

        let mut wide_bus = super::super::memory::LinearMemoryBus::new(0x1000);
        wide_bus.write_word(0x0100, 0x4C03);
        wide_bus.write_word(0x0102, 0x2C01);
        let mut cpu_040 = cpu();
        cpu_040.set_cpu_type(CpuType::M68040);
        assert!(decode_trace_op(&cpu_040, &mut wide_bus, 0x0100, CpuType::M68040).is_none());
    }

    #[test]
    fn immediate_data_register_alu_matches_interpreter() {
        let operations = [
            (JitBinaryOp::Or, 0x0000),
            (JitBinaryOp::And, 0x0200),
            (JitBinaryOp::Sub, 0x0400),
            (JitBinaryOp::Add, 0x0600),
            (JitBinaryOp::Eor, 0x0A00),
            (JitBinaryOp::Cmp, 0x0C00),
        ];
        let cases = [
            (Size::Byte, 0x81u32, 0xA5A5_557Fu32),
            (Size::Word, 0x8001u32, 0xA5A5_7FFFu32),
            (Size::Long, 0x8000_0001u32, 0x7FFF_FFFFu32),
        ];

        for (binary_op, opcode_base) in operations {
            for (size, immediate, initial) in cases {
                let size_bits = match size {
                    Size::Byte => 0,
                    Size::Word => 1,
                    Size::Long => 2,
                };
                let opcode = opcode_base | (size_bits << 6) | 3;
                let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
                bus.write_word(0x0100, opcode);
                if size == Size::Long {
                    bus.write_word(0x0102, (immediate >> 16) as u16);
                    bus.write_word(0x0104, immediate as u16);
                } else {
                    bus.write_word(0x0102, immediate as u16);
                }
                let op = decode_trace_op(&cpu(), &mut bus, 0x0100, CpuType::M68040)
                    .expect("register immediate must decode");

                let mut expected = cpu();
                expected.set_cpu_type(CpuType::M68040);
                expected.set_d(3, initial);
                expected.set_ccr(0x1F);
                assert!(matches!(
                    expected.step(&mut bus),
                    super::super::types::StepResult::Ok { .. }
                ));

                let mut actual = cpu();
                actual.set_cpu_type(CpuType::M68040);
                actual.set_d(3, initial);
                actual.set_ccr(0x1F);
                assert_eq!(
                    execute_portable_op(&mut actual, op, 0x0100, 0x0100 + op.length() as u32),
                    Some(op.op.max_cycles()),
                    "{binary_op:?} {size:?}"
                );
                assert_eq!(actual.d(3), expected.d(3), "{binary_op:?} {size:?}");
                assert_eq!(
                    actual.get_ccr(),
                    expected.get_ccr(),
                    "{binary_op:?} {size:?}"
                );
            }
        }
    }

    #[test]
    fn absolute_movea_and_displacement_lea_preserve_flags() {
        let mut decode_bus = super::super::memory::LinearMemoryBus::new(0x1000);
        decode_bus.write_word(0x0100, 0x2079); // MOVEA.L $00000200,A0
        decode_bus.write_word(0x0102, 0x0000);
        decode_bus.write_word(0x0104, 0x0200);
        decode_bus.write_word(0x0106, 0x43E8); // LEA -$0100(A0),A1
        decode_bus.write_word(0x0108, 0xFF00);
        let movea = decode_trace_op(&cpu(), &mut decode_bus, 0x0100, CpuType::M68040).unwrap();
        let lea = decode_trace_op(&cpu(), &mut decode_bus, 0x0106, CpuType::M68040).unwrap();

        let mut actual = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0200..0x0204].copy_from_slice(&0x1234_5678u32.to_be_bytes());
        attach_window(&mut actual, &mut mem);
        actual.set_ccr(0x1F);

        assert!(execute_portable_op(&mut actual, movea, 0x0100, 0x0106).is_some());
        assert!(execute_portable_op(&mut actual, lea, 0x0106, 0x010A).is_some());
        assert_eq!(actual.a(0), 0x1234_5678);
        assert_eq!(actual.a(1), 0x1234_5578);
        assert_eq!(actual.get_ccr(), 0x1F, "MOVEA and LEA do not affect CCR");
    }

    #[test]
    fn indexed_tst_bails_without_changing_flags() {
        let op = TraceBuildOp {
            opcode: 0x4A70,
            extension: Some(0x0800),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::TstMem {
                size: Size::Word,
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(0),
                    index_long: true,
                    scale: 0,
                    displacement: 0,
                },
            },
        };
        let mut actual = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0102..0x0104].copy_from_slice(&0x0800u16.to_be_bytes());
        attach_window(&mut actual, &mut mem);
        actual.set_a(0, 0x00FF_F000);
        actual.set_d(0, 4);
        actual.set_ccr(0x1F);

        assert_eq!(execute_portable_op(&mut actual, op, 0x0100, 0x0104), None);
        assert_eq!(actual.pc, 0x0100);
        assert_eq!(actual.get_ccr(), 0x1F);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_guarded_indexed_scan_reaches_its_terminal_bound() {
        const START: u32 = 0x0100;
        let words = [
            0x7038, // MOVEQ #56,D0
            0xC0C3, // MULU.W D3,D0
            0x2079, 0x0000, 0x0300, // MOVEA.L $00000300,A0
            0x41E8, 0x002A, // LEA 42(A0),A0
            0x4A70, 0x0800, // TST.W 0(A0,D0.L)
            0x6F00, 0x00EC, // BLE.W latch (common path)
        ];
        let latch_words = [
            0x5243, // ADDQ.W #1,D3
            0x0C43, 0x0080, // CMPI.W #128,D3
            0x6D00, 0xFEF8, // BLT.W start
        ];
        let mut decode_bus = super::super::memory::LinearMemoryBus::new(0x3000);
        for (index, word) in words.iter().enumerate() {
            decode_bus.write_word(START + index as u32 * 2, *word);
        }
        for (index, word) in latch_words.iter().enumerate() {
            decode_bus.write_word(0x0200 + index as u32 * 2, *word);
        }
        let pcs = [
            0x0100, 0x0102, 0x0104, 0x010A, 0x010E, 0x0112, 0x0200, 0x0202, 0x0206,
        ];
        let mut ops: Vec<_> = pcs
            .iter()
            .map(|&pc| decode_trace_op(&cpu(), &mut decode_bus, pc, CpuType::M68040).unwrap())
            .collect();
        let JitTraceOp::Branch { expected_taken, .. } = &mut ops[5].op else {
            panic!("interior BLE must decode as a branch");
        };
        *expected_taken = Some(true);

        let mut actual = cpu();
        let mut mem = vec![0u8; 0x3000];
        for (index, word) in words.iter().enumerate() {
            let at = START as usize + index * 2;
            mem[at..at + 2].copy_from_slice(&word.to_be_bytes());
        }
        for (index, word) in latch_words.iter().enumerate() {
            let at = 0x0200 + index * 2;
            mem[at..at + 2].copy_from_slice(&word.to_be_bytes());
        }
        mem[0x0300..0x0304].copy_from_slice(&0x0000_0400u32.to_be_bytes());
        attach_window(&mut actual, &mut mem);
        actual.set_cpu_type(CpuType::M68040);
        actual.pc = START;
        actual.set_d(3, 0);

        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, START, CpuType::M68040, ops, Some(START))
            .expect("bounded indexed scan should compile");
        assert!(compiled.native_loop);
        let packed = unsafe { compiled.call_native(&mut actual, 1_000) };

        assert_eq!((packed >> 32) as u32, 128 * 9);
        assert_eq!(actual.d(3) as u16, 128);
        assert_eq!(actual.pc, 0x020A);
    }

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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

        mem.write_word(0x0100, 0xB270); // CMP.W $04(A0,D2.W),D1
        mem.write_word(0x0102, 0x2004);
        let indexed = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68040).unwrap();
        assert_eq!(indexed.extension, Some(0x2004));
        assert!(matches!(
            indexed.op,
            JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Word,
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
                dst: 1,
            }
        ));

        mem.write_word(0x0100, 0xB7E9); // CMPA.L $0010(A1),A3
        mem.write_word(0x0102, 0x0010);
        let address_compare = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68040).unwrap();
        assert_eq!(address_compare.extension, Some(0x0010));
        assert!(matches!(
            address_compare.op,
            JitTraceOp::AddrCmpMemToReg {
                size: Size::Long,
                src: JitEa::Disp(1, 0x0010),
                dst: 3,
            }
        ));

        mem.write_word(0x0100, 0x5497); // ADDQ.L #2,(A7)
        let indirect_addq = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68040).unwrap();
        assert_eq!(indirect_addq.extension, None);
        assert!(matches!(
            indirect_addq.op,
            JitTraceOp::MemAddqSubq {
                data: 2,
                size: Size::Long,
                dst: JitEa::Ind(7),
                is_sub: false,
            }
        ));

        mem.write_word(0x0100, 0xBCFC); // CMPA.W #$FFFF,A6
        mem.write_word(0x0102, 0xFFFF);
        let immediate_cmpa = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68040).unwrap();
        assert_eq!(immediate_cmpa.extension, Some(0xFFFF));
        assert!(matches!(
            immediate_cmpa.op,
            JitTraceOp::AddrCmpImmediate {
                immediate: 0xFFFF,
                dst: 6,
                size: Size::Word,
                cycles: 10,
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
    fn decodes_fixed_point_update_operations() {
        let cpu = cpu();
        let mut mem = super::super::memory::LinearMemoryBus::new(0x1000);

        mem.write_word(0x0100, 0xE088); // LSR.L #8,D0
        let shift = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68040).unwrap();
        assert_eq!(shift.extension, None);
        assert!(matches!(
            shift.op,
            JitTraceOp::ShiftReg {
                reg: 0,
                size: Size::Long,
                count_or_reg: 0,
                count_is_register: false,
                direction: 0,
                op: 1,
            }
        ));

        mem.write_word(0x0100, 0xD1A9); // ADD.L D0,$0018(A1)
        mem.write_word(0x0102, 0x0018);
        let add = decode_trace_op(&cpu, &mut mem, 0x0100, CpuType::M68040).unwrap();
        assert_eq!(add.extension, Some(0x0018));
        assert!(matches!(
            add.op,
            JitTraceOp::AddRegToMem {
                is_sub: false,
                size: Size::Long,
                src: 0,
                dst: JitEa::Disp(1, 0x0018),
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
    fn portable_cmp_indexed_sets_flags_without_changing_registers() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0102..0x0104].copy_from_slice(&0x2004u16.to_be_bytes());
        mem[0x0206..0x0208].copy_from_slice(&0x8000u16.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(0, 0x0200);
        cpu.set_d(1, 0x1234_7FFF);
        cpu.set_d(2, 2);
        cpu.set_ccr(0x10); // X set; CMP must preserve it.

        let op = TraceBuildOp {
            opcode: 0xB270,
            extension: Some(0x2004),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Word,
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
                dst: 1,
            },
        };
        assert_eq!(execute_portable_op(&mut cpu, op, 0x0100, 0x0104), Some(24));
        assert_eq!(cpu.a(0), 0x0200);
        assert_eq!(cpu.d(1), 0x1234_7FFF);
        assert_eq!(cpu.d(2), 2);
        assert_eq!(cpu.pc, 0x0104);
        assert_eq!(cpu.get_ccr(), 0x1B, "X/N/V/C set and Z clear");
    }

    #[test]
    fn indexed_cmpi_word_decode_and_portable_execution_preserve_memory_and_x() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0100..0x0102].copy_from_slice(&0x0C70u16.to_be_bytes());
        mem[0x0102..0x0104].copy_from_slice(&0xFFFFu16.to_be_bytes());
        mem[0x0104..0x0106].copy_from_slice(&0x0000u16.to_be_bytes());
        mem[0x0202..0x0204].copy_from_slice(&0u16.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(0, 0x0200);
        cpu.set_d(0, 2);
        cpu.set_ccr(0x10);

        let mut decode_bus = super::super::memory::LinearMemoryBus::new(0x1000);
        decode_bus.write_word(0x0100, 0x0C70);
        decode_bus.write_word(0x0102, 0xFFFF);
        decode_bus.write_word(0x0104, 0x0000);
        let trace = decode_trace_op(&cpu, &mut decode_bus, 0x0100, CpuType::M68040)
            .expect("indexed CMPI.W should decode");
        assert!(matches!(
            trace.op,
            JitTraceOp::CmpiWordMem {
                immediate: 0xFFFF,
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(0),
                    index_long: false,
                    scale: 0,
                    displacement: 0,
                },
            }
        ));
        assert_eq!(trace.extension, Some(0xFFFF));
        assert_eq!(trace.extension2, Some(0x0000));
        assert_eq!(
            execute_portable_op(&mut cpu, trace, 0x0100, 0x0106),
            Some(18)
        );
        assert_eq!(&mem[0x0202..0x0204], &[0, 0], "CMPI is read-only");
        assert_eq!(cpu.get_ccr(), 0x11, "X is preserved while C is set");
        assert_eq!(cpu.pc, 0x0106);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_indexed_cmpi_word_matches_portable_and_bails_atomically() {
        let cmp = TraceBuildOp {
            opcode: 0x0C70,
            extension: Some(0xFFFF),
            extension2: Some(0x0000),
            pc: 0x0100,
            op: JitTraceOp::CmpiWordMem {
                immediate: 0xFFFF,
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(0),
                    index_long: false,
                    scale: 0,
                    displacement: 0,
                },
            },
        };
        let branch = TraceBuildOp {
            opcode: 0x60F8,
            extension: None,
            extension2: None,
            pc: 0x0106,
            op: JitTraceOp::Branch {
                condition: 0,
                displacement: -8,
                length: 2,
                expected_taken: None,
            },
        };
        let prepare = |mem: &mut [u8], value: u16| {
            mem[0x0102..0x0104].copy_from_slice(&0xFFFFu16.to_be_bytes());
            mem[0x0104..0x0106].copy_from_slice(&0x0000u16.to_be_bytes());
            mem[0x0202..0x0204].copy_from_slice(&value.to_be_bytes());
            let mut cpu = cpu();
            attach_window(&mut cpu, mem);
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_a(0, 0x0200);
            cpu.set_d(0, 2);
            cpu.set_ccr(0x10);
            cpu
        };

        // Memory values covering the NZVC matrix of `dst - 0xFFFF`; X is
        // preserved in every case.
        let cases = [
            (0x0000u16, 0x11), // borrow only: C
            (0xFFFF, 0x14),    // equal operands: Z
            (0x7FFF, 0x1B),    // signed overflow: N, V, C
            (0x8000, 0x19),    // negative without overflow: N, C
        ];
        for (value, expected_ccr) in cases {
            let ops = vec![cmp, branch];
            let mut expected_mem = vec![0u8; 0x1000];
            let mut expected = prepare(&mut expected_mem, value);
            let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0108);
            assert_eq!(
                expected.get_ccr(),
                expected_ccr,
                "portable CCR comparing {value:#06x}"
            );

            let mut actual_mem = vec![0u8; 0x1000];
            let mut actual = prepare(&mut actual_mem, value);
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops, Some(0x0100))
                .expect("indexed CMPI.W loop should compile");
            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
            assert_eq!(actual_packed, expected_packed);
            assert_eq!(actual.pc, expected.pc);
            assert_eq!(actual.dar, expected.dar);
            assert_eq!(actual.get_ccr(), expected.get_ccr());
            assert_eq!(actual_mem, expected_mem);
        }

        // A0 placed so the word read falls outside the window: the native
        // side exit must retire nothing and leave all state untouched.
        let mut bail_mem = vec![0u8; 0x1000];
        let mut bailed = prepare(&mut bail_mem, 0x0000);
        bailed.set_a(0, 0x0FFE);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(
                &bailed,
                0x0100,
                CpuType::M68040,
                vec![cmp, branch],
                Some(0x0100),
            )
            .expect("indexed CMPI.W loop should compile");
        let before = bailed.dar;
        let packed = unsafe { compiled.call_native(&mut bailed, 1) };
        assert_eq!(packed, 0, "bail retires no instructions or cycles");
        assert_eq!(bailed.pc, 0x0100);
        assert_eq!(bailed.dar, before);
        assert_eq!(bailed.get_ccr(), 0x10);
    }

    /// Word-read counting bus for the profiling bus-access regression.
    #[cfg(feature = "trace-profile")]
    struct CountingBus {
        memory: Vec<u8>,
        word_reads: std::collections::BTreeMap<u32, u32>,
    }

    #[cfg(feature = "trace-profile")]
    impl super::super::memory::AddressBus for CountingBus {
        fn read_byte(&mut self, address: u32) -> u8 {
            self.memory[address as usize & 0xFFF]
        }

        fn read_word(&mut self, address: u32) -> u16 {
            *self.word_reads.entry(address).or_default() += 1;
            let addr = address as usize & 0xFFF;
            u16::from_be_bytes([self.memory[addr], self.memory[addr + 1]])
        }

        fn read_long(&mut self, address: u32) -> u32 {
            let high = self.read_word(address);
            let low = self.read_word(address.wrapping_add(2));
            (u32::from(high) << 16) | u32::from(low)
        }

        fn write_byte(&mut self, address: u32, value: u8) {
            self.memory[address as usize & 0xFFF] = value;
        }

        fn write_word(&mut self, address: u32, value: u16) {
            let addr = address as usize & 0xFFF;
            self.memory[addr..addr + 2].copy_from_slice(&value.to_be_bytes());
        }

        fn write_long(&mut self, address: u32, value: u32) {
            let addr = address as usize & 0xFFF;
            self.memory[addr..addr + 4].copy_from_slice(&value.to_be_bytes());
        }
    }

    /// Recording-failure diagnostics must not add bus transactions: the one
    /// bus read is the decoder's own opcode read, and the blocker's memory
    /// opcode and following words come from the fastmem window alone.
    #[cfg(feature = "trace-profile")]
    #[test]
    fn recording_failure_diagnostics_add_no_bus_reads() {
        super::super::trace_profile::reset();

        let swap = TraceBuildOp {
            opcode: 0x4840,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::Swap { reg: 0 },
        };

        let run = |cpu: &mut CpuCore| {
            let mut bus = CountingBus {
                memory: vec![0u8; 0x1000],
                word_reads: std::collections::BTreeMap::new(),
            };
            // An opcode the trace decoder refuses and which carries no
            // extension words, so a failing decode costs exactly the one
            // opcode read. (This was `C0FC` until `MUL.W #imm,Dn` became
            // supported -- pick a still-unsupported form when updating.)
            bus.memory[0x0102..0x0104].copy_from_slice(&0x4AFCu16.to_be_bytes());
            let mut jit = TraceJit::new();
            jit.recording = Some(TraceRecording {
                start_pc: 0x0100,
                cpu_type: CpuType::M68040,
                ops: vec![swap],
                adaptive_rerecords: 0,
            });
            cpu.set_cpu_type(CpuType::M68040);
            cpu.trace_recording = true;
            cpu.ir = 0x4AFC;
            jit.record_executed(cpu, &mut bus, 0x0102, 0x0106);
            bus.word_reads
        };

        // Headless: no window is attached, so diagnostics are unavailable
        // and the failure still costs exactly one decoder opcode read.
        let mut headless = cpu();
        let reads = run(&mut headless);
        assert_eq!(reads.get(&0x0102), Some(&1), "one decoder opcode read");
        assert_eq!(reads.len(), 1, "no other bus reads: {reads:?}");

        // With a window attached, the diagnostics come from the window and
        // the bus still sees exactly the one decoder opcode read.
        super::super::trace_profile::reset();
        let mut windowed = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0102..0x0104].copy_from_slice(&0x4AFCu16.to_be_bytes());
        mem[0x0104..0x0106].copy_from_slice(&0x0005u16.to_be_bytes());
        mem[0x0106..0x0108].copy_from_slice(&0x60F8u16.to_be_bytes());
        attach_window(&mut windowed, &mut mem);
        let reads = run(&mut windowed);
        assert_eq!(reads.get(&0x0102), Some(&1), "one decoder opcode read");
        assert_eq!(reads.len(), 1, "diagnostics must use the window: {reads:?}");

        let snapshot = super::super::trace_profile::snapshot();
        let shape = snapshot
            .failed_shapes
            .iter()
            .find(|row| row.start_pc == 0x0100)
            .expect("the failure was recorded");
        assert_eq!(shape.blocker_pc, 0x0102);
        assert_eq!(shape.executed_opcode, 0x4AFC);
        assert_eq!(shape.memory_opcode, Some(0x4AFC));
        assert_eq!(shape.next_word, Some(0x0005));
        assert_eq!(shape.next_word2, Some(0x60F8));
        assert_eq!(shape.prefix_ops, 1);
        assert_eq!(shape.prefix[0].pc, 0x0100);
        assert_eq!(shape.prefix[0].opcode, 0x4840);
    }

    #[test]
    fn pea_displacement_decode_and_portable_execution_push_without_flags() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0100..0x0102].copy_from_slice(&0x486Du16.to_be_bytes());
        mem[0x0102..0x0104].copy_from_slice(&0x0040u16.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(5, 0x0300);
        cpu.set_a(7, 0x0800);
        cpu.set_ccr(0x1F);

        let mut decode_bus = super::super::memory::LinearMemoryBus::new(0x1000);
        decode_bus.write_word(0x0100, 0x486D);
        decode_bus.write_word(0x0102, 0x0040);
        let trace = decode_trace_op(&cpu, &mut decode_bus, 0x0100, CpuType::M68040)
            .expect("PEA (d16,An) should decode");
        assert!(matches!(
            trace.op,
            JitTraceOp::PeaDisp {
                reg: 5,
                displacement: 0x40,
            }
        ));
        assert_eq!(trace.extension, Some(0x0040));
        assert_eq!(trace.extension2, None);
        assert_eq!(
            execute_portable_op(&mut cpu, trace, 0x0100, 0x0104),
            Some(16)
        );
        assert_eq!(cpu.dar[15], 0x07FC);
        assert_eq!(&mem[0x07FC..0x0800], &0x0340u32.to_be_bytes());
        assert_eq!(cpu.get_ccr(), 0x1F, "PEA changes no condition codes");
        assert_eq!(cpu.pc, 0x0104);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_pea_displacement_matches_portable_and_bails_atomically() {
        let pea = TraceBuildOp {
            opcode: 0x486D,
            extension: Some(0x0040),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::PeaDisp {
                reg: 5,
                displacement: 0x40,
            },
        };
        let branch = TraceBuildOp {
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
        };

        let prepare = |mem: &mut [u8], opcode: u16, ext: u16| {
            mem[0x0100..0x0102].copy_from_slice(&opcode.to_be_bytes());
            mem[0x0102..0x0104].copy_from_slice(&ext.to_be_bytes());
            mem[0x0104..0x0106].copy_from_slice(&0x60FAu16.to_be_bytes());
            let mut cpu = cpu();
            attach_window(&mut cpu, mem);
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_a(5, 0x0300);
            cpu.set_a(7, 0x0800);
            cpu.set_ccr(0x15);
            cpu
        };

        let mut expected_mem = vec![0u8; 0x1000];
        let mut expected = prepare(&mut expected_mem, 0x486D, 0x0040);
        let expected_packed = execute_portable_trace(&mut expected, &[pea, branch], 0x0100, 0x0106);

        let mut actual_mem = vec![0u8; 0x1000];
        let mut actual = prepare(&mut actual_mem, 0x486D, 0x0040);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(
                &actual,
                0x0100,
                CpuType::M68040,
                vec![pea, branch],
                Some(0x0100),
            )
            .expect("PEA displacement loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(actual_packed, expected_packed);
        assert_eq!(actual.pc, expected.pc);
        assert_eq!(actual.dar, expected.dar);
        assert_eq!(actual.get_ccr(), expected.get_ccr());
        assert_eq!(actual_mem, expected_mem);
        assert_eq!(&actual_mem[0x07FC..0x0800], &0x0340u32.to_be_bytes());

        // PEA (d16,A7) pushes an address computed from the pre-decrement
        // stack pointer.
        let pea_sp = TraceBuildOp {
            opcode: 0x486F,
            extension: Some(0x0010),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::PeaDisp {
                reg: 7,
                displacement: 0x10,
            },
        };
        let mut sp_expected_mem = vec![0u8; 0x1000];
        let mut sp_expected = prepare(&mut sp_expected_mem, 0x486F, 0x0010);
        let sp_expected_packed =
            execute_portable_trace(&mut sp_expected, &[pea_sp, branch], 0x0100, 0x0106);
        let mut sp_actual_mem = vec![0u8; 0x1000];
        let mut sp_actual = prepare(&mut sp_actual_mem, 0x486F, 0x0010);
        let mut sp_jit = TraceJit::new();
        let sp_compiled = sp_jit
            .compile_decoded_ops(
                &sp_actual,
                0x0100,
                CpuType::M68040,
                vec![pea_sp, branch],
                Some(0x0100),
            )
            .expect("PEA (d16,A7) loop should compile");
        let sp_actual_packed = unsafe { sp_compiled.call_native(&mut sp_actual, 1) };
        assert_eq!(sp_actual_packed, sp_expected_packed);
        assert_eq!(sp_actual.dar, sp_expected.dar);
        assert_eq!(sp_actual_mem, sp_expected_mem);
        assert_eq!(&sp_actual_mem[0x07FC..0x0800], &0x0810u32.to_be_bytes());

        let mut untouched_mem = vec![0u8; 0x1000];
        prepare(&mut untouched_mem, 0x486D, 0x0040);

        // A stack slot outside the window reaches the side exit with
        // nothing committed.
        let mut bail_mem = vec![0u8; 0x1000];
        let mut bailed = prepare(&mut bail_mem, 0x486D, 0x0040);
        bailed.set_a(7, 0x0002);
        let mut bail_jit = TraceJit::new();
        let bail_compiled = bail_jit
            .compile_decoded_ops(
                &bailed,
                0x0100,
                CpuType::M68040,
                vec![pea, branch],
                Some(0x0100),
            )
            .expect("PEA displacement loop should compile");
        let before = bailed.dar;
        let packed = unsafe { bail_compiled.call_native(&mut bailed, 1) };
        assert_eq!(packed, 0, "bail retires no instructions or cycles");
        assert_eq!(bailed.pc, 0x0100);
        assert_eq!(bailed.dar, before);
        assert_eq!(bailed.get_ccr(), 0x15);
        assert_eq!(bail_mem, untouched_mem, "nothing was stored");

        // A stack slot overlapping the trace's own code hits the
        // store-overlap guard before anything commits.
        let mut smc_mem = vec![0u8; 0x1000];
        let mut smc = prepare(&mut smc_mem, 0x486D, 0x0040);
        smc.set_a(7, 0x0106);
        let mut smc_jit = TraceJit::new();
        let smc_compiled = smc_jit
            .compile_decoded_ops(
                &smc,
                0x0100,
                CpuType::M68040,
                vec![pea, branch],
                Some(0x0100),
            )
            .expect("PEA displacement loop should compile");
        let before = smc.dar;
        let packed = unsafe { smc_compiled.call_native(&mut smc, 1) };
        assert_eq!(packed, 0, "store-overlap guard retires nothing");
        assert_eq!(smc.dar, before);
        assert_eq!(smc_mem, untouched_mem);
    }

    #[test]
    fn indexed_lea_decode_and_portable_execution() {
        let mut cpu = cpu();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(0, 0x1000);
        cpu.set_d(2, 0x0010);
        cpu.set_ccr(0x1F);

        let mut decode_bus = super::super::memory::LinearMemoryBus::new(0x1000);
        decode_bus.write_word(0x0100, 0x43F0);
        decode_bus.write_word(0x0102, 0x2004);
        let trace = decode_trace_op(&cpu, &mut decode_bus, 0x0100, CpuType::M68040)
            .expect("LEA (d8,An,Xn) should decode");
        assert!(matches!(
            trace.op,
            JitTraceOp::LeaIndex {
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
                dst: 1,
                cycles: 4,
            }
        ));
        assert_eq!(trace.extension, Some(0x2004));
        assert_eq!(
            execute_portable_op(&mut cpu, trace, 0x0100, 0x0104),
            Some(4)
        );
        assert_eq!(cpu.dar[9], 0x1014, "A1 = A0 + D2.W + 4");
        assert_eq!(cpu.dar[8], 0x1000, "the base register is untouched");
        assert_eq!(cpu.get_ccr(), 0x1F, "LEA changes no condition codes");

        // Word indexes sign-extend.
        cpu.set_d(2, 0x0001_8000);
        assert_eq!(
            execute_portable_op(&mut cpu, trace, 0x0100, 0x0104),
            Some(4)
        );
        assert_eq!(
            cpu.dar[9],
            0x1000u32.wrapping_sub(0x8000).wrapping_add(4),
            "a word index sign-extends before the add"
        );
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_indexed_lea_matches_portable() {
        let lea = TraceBuildOp {
            opcode: 0x43F0,
            extension: Some(0x2004),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::LeaIndex {
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
                dst: 1,
                cycles: 4,
            },
        };
        let branch = TraceBuildOp {
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
        };

        let prepare = || {
            let mut cpu = cpu();
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_a(0, 0x1000);
            cpu.set_d(2, 0x8000);
            cpu.set_ccr(0x15);
            cpu
        };

        let mut expected = prepare();
        let expected_packed = execute_portable_trace(&mut expected, &[lea, branch], 0x0100, 0x0106);

        let mut actual = prepare();
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(
                &actual,
                0x0100,
                CpuType::M68040,
                vec![lea, branch],
                Some(0x0100),
            )
            .expect("indexed LEA loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(actual_packed, expected_packed);
        assert_eq!(actual.pc, expected.pc);
        assert_eq!(actual.dar, expected.dar);
        assert_eq!(actual.get_ccr(), expected.get_ccr());
        assert_eq!(
            actual.dar[9],
            0x1000u32.wrapping_sub(0x8000).wrapping_add(4),
            "the negative word index case matches natively"
        );
    }

    #[test]
    fn mul_word_immediate_cycle_headroom_covers_the_68000_worst_case() {
        // `try_execute` uses `max_cycles()` both to decide whether a trace
        // fits `cycles_remaining` and to derive the self-loop iteration
        // count, so the metadata must dominate every actual cost. Parity
        // tests cannot catch a shared underestimate -- native and portable
        // agree with each other while both overrun the budget -- so this
        // checks metadata against the portable-returned cycles directly,
        // across immediates including the worst cases (MULU $FFFF and
        // MULS $5555 both cost 38 + 2*16 = 70 on the 68000).
        let mut worst_seen = 0;
        for signed in [false, true] {
            for &immediate in &[0x0000u16, 0x0001, 0x5555, 0x8000, 0xAAAA, 0xFFFF] {
                let op = JitTraceOp::MulWordImmediate {
                    immediate,
                    dst: 0,
                    signed,
                    m68000_timing: true,
                };
                let trace = TraceBuildOp {
                    opcode: if signed { 0xC1FC } else { 0xC0FC },
                    extension: Some(immediate),
                    extension2: None,
                    pc: 0x0100,
                    op,
                };
                let mut cpu = cpu();
                cpu.set_cpu_type(CpuType::M68000);
                cpu.set_d(0, 0x1234);
                let cycles = execute_portable_op(&mut cpu, trace, 0x0100, 0x0104)
                    .expect("portable immediate multiply executes");
                assert!(
                    op.max_cycles() >= cycles,
                    "headroom: max_cycles {} < actual {} for imm {immediate:04X} signed {signed}",
                    op.max_cycles(),
                    cycles
                );
                worst_seen = worst_seen.max(cycles);
            }
        }
        assert_eq!(
            worst_seen, 74,
            "the sweep reaches the true 68000 worst case"
        );
        assert_eq!(
            JitTraceOp::MulWordImmediate {
                immediate: 0xFFFF,
                dst: 0,
                signed: false,
                m68000_timing: true,
            }
            .max_cycles(),
            74,
            "the 68000 bound is tight: 70 as for MulWordDataReg plus the
             4-cycle word-immediate operand fetch"
        );
        assert_eq!(
            JitTraceOp::MulWordImmediate {
                immediate: 0xFFFF,
                dst: 0,
                signed: false,
                m68000_timing: false,
            }
            .max_cycles(),
            42,
            "later CPUs keep the fixed pre-scaled value"
        );
    }

    #[test]
    fn mul_word_immediate_cycles_match_the_interpreter() {
        // The differential Ben asked for: the trace operation's cycle
        // result is compared against the interpreter's own
        // exec_mulu/exec_muls for the same immediate, not against the
        // trace's own arithmetic -- a shared underestimate between the
        // portable and native paths cannot hide from this. The immediate
        // form charges the 4-cycle word-operand fetch on the 68000
        // (ea_source_cycles(Immediate, Word)), which the register form
        // does not.
        use super::super::ea::AddressingMode;
        for signed in [false, true] {
            for &immediate in &[0x0000u16, 0x0001, 0x5555, 0x8000, 0xAAAA, 0xFFFF] {
                // Interpreter side: execute the real instruction with the
                // immediate in the instruction stream at pc.
                let mut icpu = cpu();
                icpu.set_cpu_type(CpuType::M68000);
                icpu.pc = 0x2000;
                icpu.prefetch_queue = [immediate, 0];
                icpu.prefetch_count = 1;
                icpu.set_d(0, 0x9ABC);
                let mut ibus = super::super::memory::LinearMemoryBus::new(0x4000);
                ibus.write_word(0x2000, immediate);
                let interpreter_cycles = if signed {
                    icpu.exec_muls(&mut ibus, AddressingMode::Immediate, 0)
                } else {
                    icpu.exec_mulu(&mut ibus, AddressingMode::Immediate, 0)
                };

                // Trace side: the portable operation for the same form.
                let op = JitTraceOp::MulWordImmediate {
                    immediate,
                    dst: 0,
                    signed,
                    m68000_timing: true,
                };
                let trace = TraceBuildOp {
                    opcode: if signed { 0xC1FC } else { 0xC0FC },
                    extension: Some(immediate),
                    extension2: None,
                    pc: 0x0100,
                    op,
                };
                let mut tcpu = cpu();
                tcpu.set_cpu_type(CpuType::M68000);
                tcpu.set_d(0, 0x9ABC);
                let trace_cycles = execute_portable_op(&mut tcpu, trace, 0x0100, 0x0104)
                    .expect("portable immediate multiply executes");

                assert_eq!(
                    trace_cycles, interpreter_cycles,
                    "imm {immediate:04X} signed {signed}: trace vs interpreter"
                );
                assert_eq!(
                    icpu.d(0),
                    tcpu.d(0),
                    "imm {immediate:04X} signed {signed}: results agree"
                );
                assert!(
                    op.max_cycles() >= interpreter_cycles,
                    "imm {immediate:04X} signed {signed}: headroom {} < interpreter {}",
                    op.max_cycles(),
                    interpreter_cycles
                );
            }
        }
    }

    #[test]
    fn mul_word_immediate_decodes_and_executes_portably() {
        // C0FC = MULU.W #imm,D0; C1FC = MULS.W #imm,D0.
        for (opcode, signed) in [(0xC0FCu16, false), (0xC1FC, true)] {
            for &(immediate, seed) in &[
                (0x0003u16, 0x0000_0005u32),
                (0xFFFF, 0x0000_0002),
                (0x8000, 0x0000_8000),
                (0x0000, 0x1234_5678),
            ] {
                let mut decode_bus = super::super::memory::LinearMemoryBus::new(0x1000);
                decode_bus.write_word(0x0100, opcode);
                decode_bus.write_word(0x0102, immediate);
                let mut cpu = cpu();
                cpu.set_cpu_type(CpuType::M68040);
                cpu.set_d(0, seed);
                let trace = decode_trace_op(&cpu, &mut decode_bus, 0x0100, CpuType::M68040)
                    .expect("MUL.W #imm,Dn should decode");
                assert!(matches!(
                    trace.op,
                    JitTraceOp::MulWordImmediate { immediate: imm, dst: 0, signed: s, .. }
                        if imm == immediate && s == signed
                ));

                execute_portable_op(&mut cpu, trace, 0x0100, 0x0104);
                let expected = if signed {
                    (i32::from(immediate as i16) * i32::from(seed as u16 as i16)) as u32
                } else {
                    u32::from(immediate) * u32::from(seed as u16)
                };
                assert_eq!(
                    cpu.d(0),
                    expected,
                    "opcode {opcode:04X} imm {immediate:04X}"
                );
            }
        }
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_mul_word_immediate_matches_portable() {
        // The 68000 timing path folds the multiplicand's bit pattern at
        // compile time, so check a value with a distinctive popcount as well
        // as the sign-extension boundary.
        for (opcode, signed) in [(0xC0FCu16, false), (0xC1FC, true)] {
            for &immediate in &[0x0003u16, 0xFFFF, 0x8001] {
                let mul = TraceBuildOp {
                    opcode,
                    extension: Some(immediate),
                    extension2: None,
                    pc: 0x0100,
                    op: JitTraceOp::MulWordImmediate {
                        immediate,
                        dst: 0,
                        signed,
                        m68000_timing: true,
                    },
                };
                let branch = TraceBuildOp {
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
                };
                let prepare = || {
                    let mut cpu = cpu();
                    cpu.set_cpu_type(CpuType::M68000);
                    cpu.set_d(0, 0x0000_9ABC);
                    cpu.set_ccr(0x1F);
                    cpu
                };

                let mut expected = prepare();
                let expected_packed =
                    execute_portable_trace(&mut expected, &[mul, branch], 0x0100, 0x0106);

                let mut actual = prepare();
                let mut jit = TraceJit::new();
                let compiled = jit
                    .compile_decoded_ops(
                        &actual,
                        0x0100,
                        CpuType::M68000,
                        vec![mul, branch],
                        Some(0x0100),
                    )
                    .expect("immediate multiply loop should compile");
                let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

                assert_eq!(actual_packed, expected_packed, "cycles/retired");
                assert_eq!(actual.dar, expected.dar, "registers");
                assert_eq!(actual.get_ccr(), expected.get_ccr(), "ccr");
            }
        }
    }

    #[test]
    fn sub_reg_to_mem_decodes_for_all_three_destinations() {
        // 9190 = SUB.L D0,(A0) -- the form blocking a 153k-hit gameplay
        // head -- plus the postinc and displacement forms, and the ADD
        // indirect form newly admitted alongside.
        let mut cpu = cpu();
        cpu.set_cpu_type(CpuType::M68040);
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        for (pc, words, expected_dst, expected_sub) in [
            (0x0100u32, vec![0x9190u16], JitEa::Ind(0), true),
            (0x0110, vec![0x9198], JitEa::PostInc(0), true),
            (0x0120, vec![0x91A8, 0x0018], JitEa::Disp(0, 0x18), true),
            (0x0130, vec![0xD190], JitEa::Ind(0), false),
        ] {
            for (i, w) in words.iter().enumerate() {
                bus.write_word(pc + i as u32 * 2, *w);
            }
            let trace = decode_trace_op(&cpu, &mut bus, pc, CpuType::M68040)
                .expect("ADD/SUB reg-to-mem should decode");
            assert!(
                matches!(
                    trace.op,
                    JitTraceOp::AddRegToMem { is_sub, size: Size::Long, src: 0, dst }
                        if is_sub == expected_sub && dst == expected_dst
                ),
                "words {words:04X?} decoded {:?}",
                trace.op
            );
        }
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_sub_reg_to_mem_matches_portable() {
        // Values chosen to exercise borrow, signed overflow, and zero: the
        // flag cases where an ADD-path mistake would show.
        for &(initial, sub_by) in &[
            (0x0000_0005u32, 0x0000_0003u32), // plain
            (0x0000_0003, 0x0000_0005),       // borrow (C/X set)
            (0x8000_0000, 0x0000_0001),       // signed overflow (V set)
            (0x0000_0007, 0x0000_0007),       // zero (Z set)
        ] {
            let sub = TraceBuildOp {
                opcode: 0x9190,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::AddRegToMem {
                    is_sub: true,
                    size: Size::Long,
                    src: 0,
                    dst: JitEa::Ind(0),
                },
            };
            let branch = TraceBuildOp {
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
            };
            let mut expected_mem = vec![0u8; 0x1000];
            expected_mem[0x0200..0x0204].copy_from_slice(&initial.to_be_bytes());
            let mut actual_mem = expected_mem.clone();

            let prepare = |mem: &mut Vec<u8>| {
                let mut cpu = cpu();
                cpu.set_cpu_type(CpuType::M68040);
                cpu.set_a(0, 0x0200);
                cpu.set_d(0, sub_by);
                cpu.set_ccr(0x00);
                attach_window(&mut cpu, mem);
                cpu
            };

            let mut expected = prepare(&mut expected_mem);
            let expected_packed =
                execute_portable_trace(&mut expected, &[sub, branch], 0x0100, 0x0104);

            let mut actual = prepare(&mut actual_mem);
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(
                    &actual,
                    0x0100,
                    CpuType::M68040,
                    vec![sub, branch],
                    Some(0x0100),
                )
                .expect("SUB.L Dn,(An) loop should compile");
            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

            let context = format!("initial {initial:#010X} sub_by {sub_by:#010X}");
            assert_eq!(actual_packed, expected_packed, "cycles/retired: {context}");
            assert_eq!(actual.dar, expected.dar, "registers: {context}");
            assert_eq!(
                actual.get_ccr(),
                expected.get_ccr(),
                "ccr incl X: {context}"
            );
            assert_eq!(actual_mem, expected_mem, "memory: {context}");
            assert_eq!(
                u32::from_be_bytes(actual_mem[0x0200..0x0204].try_into().unwrap()),
                initial.wrapping_sub(sub_by),
                "stored difference: {context}"
            );
        }
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_register_count_shift_matches_portable_across_counts() {
        // The architecturally interesting counts: zero (flags change but X
        // is preserved and the register does not), one, counts at and past
        // the operand width (result saturates, carry stops), and the
        // modulo-64 wrap (64 behaves as zero, 65 as one).
        for &(op, direction) in &[(0u8, 0u8), (1, 0), (1, 1)] {
            for size in [Size::Byte, Size::Word, Size::Long] {
                for &count in &[0u32, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65] {
                    for &value in &[0x0000_0000u32, 0x8000_0001, 0x1234_5678, 0xFFFF_FFFF] {
                        for &initial_x in &[0u32, 1] {
                            let shift = TraceBuildOp {
                                opcode: 0xE2A0,
                                extension: None,
                                extension2: None,
                                pc: 0x0100,
                                op: JitTraceOp::ShiftReg {
                                    reg: 0,
                                    size,
                                    count_or_reg: 1,
                                    count_is_register: true,
                                    direction,
                                    op,
                                },
                            };
                            let branch = TraceBuildOp {
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
                            };
                            let prepare = || {
                                let mut cpu = cpu();
                                cpu.set_cpu_type(CpuType::M68000);
                                cpu.set_d(0, value);
                                cpu.set_d(1, count);
                                cpu.x_flag = initial_x;
                                cpu
                            };

                            let mut expected = prepare();
                            let expected_packed = execute_portable_trace(
                                &mut expected,
                                &[shift, branch],
                                0x0100,
                                0x0104,
                            );

                            let mut actual = prepare();
                            let mut jit = TraceJit::new();
                            let compiled = jit
                                .compile_decoded_ops(
                                    &actual,
                                    0x0100,
                                    CpuType::M68000,
                                    vec![shift, branch],
                                    Some(0x0100),
                                )
                                .expect("register-count shift loop should compile");
                            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

                            let context = format!(
                                "op={op} dir={direction} size={size:?} count={count} \
                                 value={value:#010X} x={initial_x}"
                            );
                            assert_eq!(actual_packed, expected_packed, "cycles/retired: {context}");
                            assert_eq!(actual.dar, expected.dar, "registers: {context}");
                            assert_eq!(actual.get_ccr(), expected.get_ccr(), "ccr: {context}");
                            assert_eq!(actual.x_flag != 0, expected.x_flag != 0, "X: {context}");
                        }
                    }
                }
            }
        }
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn zero_register_count_shift_preserves_x_and_the_operand() {
        // Called out separately because it is the case an immediate-count
        // encoding can never produce: a zero count clears C and V and sets
        // N/Z from the unshifted value, while leaving X and the register
        // alone.
        let shift = TraceBuildOp {
            opcode: 0xE2A0,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::ShiftReg {
                reg: 0,
                size: Size::Long,
                count_or_reg: 1,
                count_is_register: true,
                direction: 0,
                op: 0,
            },
        };
        let branch = TraceBuildOp {
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
        };
        let mut actual = cpu();
        actual.set_cpu_type(CpuType::M68000);
        actual.set_d(0, 0x8000_0000);
        actual.set_d(1, 0);
        actual.x_flag = 1;
        actual.c_flag = 1;

        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(
                &actual,
                0x0100,
                CpuType::M68000,
                vec![shift, branch],
                Some(0x0100),
            )
            .expect("zero-count shift should compile");
        unsafe { compiled.call_native(&mut actual, 1) };

        assert_eq!(actual.d(0), 0x8000_0000, "the operand is untouched");
        assert_ne!(actual.x_flag, 0, "X survives a zero count");
        assert_eq!(actual.c_flag, 0, "C is cleared by a zero count");
        assert_eq!(actual.v_flag, 0, "V is cleared");
        assert_ne!(actual.n_flag, 0, "N comes from the unshifted value");
    }

    #[test]
    fn move_to_indexed_destination_decodes_and_matches_the_interpreter() {
        // 3180 = MOVE.W D0,(d8,A0,Xn): the indexed-destination store that
        // three profiled heads terminate on. The staged case is the subtle
        // one -- MOVE.W (A2)+,(d8,A1,A2.W) commits the source post-
        // increment before the destination EA evaluates -- and portable
        // and native could share a wrong staging model, so both are
        // checked against the interpreter's own decoded execution rather
        // than only against each other.
        let mut cpu = cpu();
        cpu.set_cpu_type(CpuType::M68040);
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        bus.write_word(0x0100, 0x3180);
        bus.write_word(0x0102, 0x2004); // (4,A0,D2.W)
        let trace = decode_trace_op(&cpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("MOVE.W Dn,(d8,An,Xn) should decode");
        assert!(matches!(
            trace.op,
            JitTraceOp::MoveMem {
                size: Size::Word,
                src: JitEa::Data(0),
                dst: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
            }
        ));

        // Staged-interaction differential: MOVE.W (A2)+,(d8,A1,A2.W).
        // dst reg 1, dst mode 6 (index), src mode 3 (postinc), src reg 2
        // => 0011 001 110 011 010 = 0x339A.
        let mut dbus = super::super::memory::LinearMemoryBus::new(0x1000);
        dbus.write_word(0x0100, 0x339A);
        dbus.write_word(0x0102, 0xA008); // (8,A1,A2.W) index=A2 word
        let staged_trace = decode_trace_op(&cpu, &mut dbus, 0x0100, CpuType::M68040)
            .expect("staged indexed store should decode");
        assert!(matches!(
            staged_trace.op,
            JitTraceOp::MoveMem {
                size: Size::Word,
                src: JitEa::PostInc(2),
                dst: JitEa::Index {
                    base: 1,
                    index: JitDirectReg::Addr(2),
                    index_long: false,
                    scale: 0,
                    displacement: 8,
                },
            }
        ));

        let prepare = |mem: &mut Vec<u8>| {
            let mut c = cpu_fn();
            c.set_cpu_type(CpuType::M68040);
            c.set_a(2, 0x0200); // source pointer; also the dst index register
            c.set_a(1, 0x0400); // dst base
            c.set_a(7, 0x0900);
            mem[0x0200..0x0202].copy_from_slice(&0xBEEFu16.to_be_bytes());
            // The interpreter reads the extension word from guest memory.
            mem[0x0100..0x0102].copy_from_slice(&0x339Au16.to_be_bytes());
            mem[0x0102..0x0104].copy_from_slice(&0xA008u16.to_be_bytes());
            attach_window(&mut c, mem);
            c.pc = 0x0100;
            c
        };
        fn cpu_fn() -> CpuCore {
            let mut c = CpuCore::new();
            c.set_sr(0x2700);
            c
        }

        // Interpreter (decoded fast path) reference.
        let mut imem = vec![0u8; 0x1000];
        let mut icpu = prepare(&mut imem);
        icpu.pc = 0x0102; // execute_mem_op reads extensions from pc
        assert!(super::super::mem_ops::execute_mem_op(
            &mut icpu,
            DecodedMemOp::Move {
                size: Size::Word,
                src: FastEa::AnPostInc(2),
                dst: FastEa::AnIndex(1),
            },
        ));

        // Portable trace op.
        let mut pmem = vec![0u8; 0x1000];
        let mut pcpu = prepare(&mut pmem);
        let cycles = execute_portable_op(&mut pcpu, staged_trace, 0x0100, 0x0104)
            .expect("portable staged indexed store executes");
        assert!(cycles > 0);

        assert_eq!(icpu.a(2), 0x0202, "interpreter commits the post-increment");
        assert_eq!(pcpu.a(2), icpu.a(2), "portable matches interpreter A2");
        // The stored address uses the UPDATED A2 as index: 0x400+0x202+8.
        let addr = 0x0400 + 0x0202 + 8;
        assert_eq!(
            &imem[addr..addr + 2],
            &0xBEEFu16.to_be_bytes(),
            "interpreter stores through the post-incremented index"
        );
        assert_eq!(pmem, imem, "portable memory matches interpreter exactly");
        assert_eq!(pcpu.get_ccr(), icpu.get_ccr(), "flags agree");
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_move_to_indexed_destination_matches_portable() {
        // Plain, negative-word-index, and staged-postinc cases.
        let cases: [(JitEa, u16, u32, u32); 3] = [
            (
                JitEa::Data(0),
                0x3180, // MOVE.W D0,(4,A0,D2.W)
                0x0000_0010,
                0x0000_0004,
            ),
            (
                JitEa::Data(0),
                0x3180,
                0xFFFF_8000, // negative word index, sign-extension boundary
                0x0000_0004,
            ),
            (
                JitEa::PostInc(2),
                0x339A, // MOVE.W (A2)+,(8,A1,A2.W) -- staged interaction
                0,
                0,
            ),
        ];
        for (case, (src, opcode, d2, _)) in cases.iter().enumerate() {
            let dst = if *opcode == 0x3180 {
                JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                }
            } else {
                JitEa::Index {
                    base: 1,
                    index: JitDirectReg::Addr(2),
                    index_long: false,
                    scale: 0,
                    displacement: 8,
                }
            };
            let mv = TraceBuildOp {
                opcode: *opcode,
                extension: Some(if *opcode == 0x3180 { 0x2004 } else { 0xA008 }),
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::MoveMem {
                    size: Size::Word,
                    src: *src,
                    dst,
                },
            };
            let branch = TraceBuildOp {
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
            };
            let mut emem = vec![0u8; 0x1000];
            let mut amem = vec![0u8; 0x1000];
            let prepare = |mem: &mut Vec<u8>| {
                let mut c = cpu();
                c.set_cpu_type(CpuType::M68040);
                c.set_d(0, 0x0000_CAFE);
                c.set_d(2, *d2);
                c.set_a(0, 0x0300);
                c.set_a(1, 0x0400);
                c.set_a(2, 0x0200);
                c.set_a(7, 0x0900);
                mem[0x0200..0x0202].copy_from_slice(&0x1234u16.to_be_bytes());
                attach_window(&mut c, mem);
                c
            };
            let mut expected = prepare(&mut emem);
            let expected_packed =
                execute_portable_trace(&mut expected, &[mv, branch], 0x0100, 0x0106);
            let mut actual = prepare(&mut amem);
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(
                    &actual,
                    0x0100,
                    CpuType::M68040,
                    vec![mv, branch],
                    Some(0x0100),
                )
                .expect("indexed-destination store loop should compile");
            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
            assert_eq!(
                actual_packed, expected_packed,
                "case {case}: cycles/retired"
            );
            assert_eq!(actual.dar, expected.dar, "case {case}: registers");
            assert_eq!(actual.get_ccr(), expected.get_ccr(), "case {case}: ccr");
            assert_eq!(amem, emem, "case {case}: memory");
        }
    }

    #[test]
    fn clr_to_indexed_destination_decodes_and_clears_with_correct_flags() {
        // 4230/4270 = CLR.B/W (d8,An,Xn): the indexed-destination stores
        // two profiled heads terminate on.
        let mut dcpu = cpu();
        dcpu.set_cpu_type(CpuType::M68040);
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        bus.write_word(0x0100, 0x4230);
        bus.write_word(0x0102, 0x2004); // (4,A0,D2.W)
        let byte_trace = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("CLR.B (d8,An,Xn) should decode");
        assert!(matches!(
            byte_trace.op,
            JitTraceOp::ClrMem {
                size: Size::Byte,
                dst: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
            }
        ));
        bus.write_word(0x0100, 0x4270);
        let word_trace = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("CLR.W (d8,An,Xn) should decode");
        assert!(matches!(
            word_trace.op,
            JitTraceOp::ClrMem {
                size: Size::Word,
                ..
            }
        ));

        // CLR sets Z, clears N/V/C, and leaves X alone.
        let mut mem = vec![0u8; 0x1000];
        mem[0x0100..0x0102].copy_from_slice(&0x4270u16.to_be_bytes());
        mem[0x0102..0x0104].copy_from_slice(&0x2004u16.to_be_bytes());
        mem[0x0306..0x0308].copy_from_slice(&0xCAFEu16.to_be_bytes());
        let mut c = cpu();
        c.set_cpu_type(CpuType::M68040);
        c.set_a(0, 0x0300);
        c.set_d(2, 2);
        c.set_ccr(0x1F); // all set: X must survive, NZVC must be rewritten
        attach_window(&mut c, &mut mem);
        let cycles = execute_portable_op(&mut c, word_trace, 0x0100, 0x0104)
            .expect("portable indexed CLR executes");
        assert!(cycles > 0);
        assert_eq!(&mem[0x0306..0x0308], &[0, 0], "destination cleared");
        assert_eq!(c.get_ccr(), 0x14, "Z set, N/V/C cleared, X preserved");
        assert_eq!(c.pc, 0x0104, "pc advanced past opcode and extension");
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_clr_to_indexed_destination_matches_portable_and_bails_atomically() {
        // Case 0: CLR.B with a negative word index. Case 1: CLR.W with a
        // scaled long index (68040 brief format).
        let cases: [(u16, u16, JitTraceOp); 2] = [
            (
                0x4230,
                0x20FC, // (-4,A0,D2.W)
                JitTraceOp::ClrMem {
                    size: Size::Byte,
                    dst: JitEa::Index {
                        base: 0,
                        index: JitDirectReg::Data(2),
                        index_long: false,
                        scale: 0,
                        displacement: -4,
                    },
                },
            ),
            (
                0x4270,
                0x3A08, // (8,A0,D3.L*2)
                JitTraceOp::ClrMem {
                    size: Size::Word,
                    dst: JitEa::Index {
                        base: 0,
                        index: JitDirectReg::Data(3),
                        index_long: true,
                        scale: 1,
                        displacement: 8,
                    },
                },
            ),
        ];
        for (case, (opcode, extension, op)) in cases.iter().enumerate() {
            let clr = TraceBuildOp {
                opcode: *opcode,
                extension: Some(*extension),
                extension2: None,
                pc: 0x0100,
                op: *op,
            };
            let branch = TraceBuildOp {
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
            };
            let ops = vec![clr, branch];
            let prepare = |mem: &mut Vec<u8>| {
                mem[0x0100..0x0102].copy_from_slice(&opcode.to_be_bytes());
                mem[0x0102..0x0104].copy_from_slice(&extension.to_be_bytes());
                mem[0x0300..0x0400].fill(0xAA);
                let mut c = cpu();
                c.set_cpu_type(CpuType::M68040);
                c.set_a(0, 0x0320);
                c.set_d(2, 0xFFFF_FF00); // word part -256
                c.set_d(3, 0x0000_0040); // scaled by 2 = 0x80
                c.set_ccr(0x10);
                attach_window(&mut c, mem);
                c
            };
            let mut emem = vec![0u8; 0x1000];
            let mut expected = prepare(&mut emem);
            let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0106);
            let mut amem = vec![0u8; 0x1000];
            let mut actual = prepare(&mut amem);
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops.clone(), Some(0x0100))
                .expect("indexed CLR loop should compile");
            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
            assert_eq!(
                actual_packed, expected_packed,
                "case {case}: cycles/retired"
            );
            assert_eq!(actual.dar, expected.dar, "case {case}: registers");
            assert_eq!(actual.get_ccr(), expected.get_ccr(), "case {case}: ccr");
            assert_eq!(amem, emem, "case {case}: memory");
            let cleared = if case == 0 {
                0x0320 - 256 - 4
            } else {
                0x0320 + 0x80 + 8
            };
            assert_eq!(amem[cleared], 0, "case {case}: destination byte cleared");

            if case == 0 {
                // A store aimed at the trace's own code must bail on both
                // paths with nothing committed.
                let overlap = |mem: &mut Vec<u8>| {
                    let mut c = prepare(mem);
                    // 0x0102 = base - 256 - 4  =>  base = 0x0206
                    c.set_a(0, 0x0206);
                    c
                };
                let mut pmem = vec![0u8; 0x1000];
                let mut pcpu = overlap(&mut pmem);
                let packed = execute_portable_trace(&mut pcpu, &ops, 0x0100, 0x0106);
                assert_eq!(packed, 0, "portable bails on a store into code");
                assert_eq!(pcpu.pc, 0x0100);
                assert_eq!(pcpu.get_ccr(), 0x10);
                assert_eq!(pmem[0x0102], 0x20, "extension word untouched");

                let mut nmem = vec![0u8; 0x1000];
                let mut ncpu = overlap(&mut nmem);
                let before = ncpu.dar;
                let packed = unsafe { compiled.call_native(&mut ncpu, 1) };
                assert_eq!(packed, 0, "native bails on a store into code");
                assert_eq!(ncpu.pc, 0x0100);
                assert_eq!(ncpu.dar, before);
                assert_eq!(ncpu.get_ccr(), 0x10);
                assert_eq!(nmem, pmem, "native bail commits nothing");
            }
        }
    }

    #[test]
    fn clr_to_predecrement_decodes_and_matches_the_interpreter() {
        // 42A7 = CLR.L -(SP): the top gameplay blocker. No extension word.
        let mut dcpu = cpu();
        dcpu.set_cpu_type(CpuType::M68040);
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        bus.write_word(0x0100, 0x42A7);
        let long_trace = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("CLR.L -(SP) should decode");
        assert!(matches!(
            long_trace.op,
            JitTraceOp::ClrMem {
                size: Size::Long,
                dst: JitEa::PreDec(7),
            }
        ));
        assert!(long_trace.extension.is_none(), "no phantom extension word");

        // CLR.B -(A7) keeps the stack pointer even: the step is 2, and the
        // byte lands at the decremented address. Checked against the
        // interpreter's decoded execution, not just our own model.
        bus.write_word(0x0100, 0x4227);
        let byte_trace = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("CLR.B -(A7) should decode");

        let prepare = |mem: &mut Vec<u8>| {
            let mut c = cpu();
            c.set_cpu_type(CpuType::M68040);
            c.set_a(7, 0x0800);
            mem[0x0100..0x0102].copy_from_slice(&0x4227u16.to_be_bytes());
            mem[0x07FE] = 0xAA;
            mem[0x07FF] = 0xBB;
            attach_window(&mut c, mem);
            c.pc = 0x0100;
            c
        };
        let mut imem = vec![0u8; 0x1000];
        let mut icpu = prepare(&mut imem);
        icpu.pc = 0x0102;
        assert!(super::super::mem_ops::execute_mem_op(
            &mut icpu,
            DecodedMemOp::Clr {
                size: Size::Byte,
                ea: FastEa::AnPreDec(7),
            },
        ));
        let mut pmem = vec![0u8; 0x1000];
        let mut pcpu = prepare(&mut pmem);
        execute_portable_op(&mut pcpu, byte_trace, 0x0100, 0x0102)
            .expect("portable byte predec CLR executes");
        assert_eq!(icpu.a(7), 0x07FE, "interpreter keeps SP even (step 2)");
        assert_eq!(pcpu.a(7), icpu.a(7), "portable matches interpreter A7");
        assert_eq!(pmem, imem, "memory matches interpreter exactly");
        assert_eq!(pcpu.get_ccr(), icpu.get_ccr(), "flags agree");
    }

    #[test]
    fn clr_to_absolute_decodes_with_exact_extents() {
        let dcpu = cpu();
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        // 42B8 = CLR.L (xxx).W with a negative address: sign-extends.
        bus.write_word(0x0100, 0x42B8);
        bus.write_word(0x0102, 0x8100);
        let trace = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("CLR.L (xxx).W should decode");
        assert!(matches!(
            trace.op,
            JitTraceOp::ClrMem {
                size: Size::Long,
                dst: JitEa::AbsWord(0xFFFF_8100),
            }
        ));
        assert_eq!(trace.extension, Some(0x8100));
        assert!(trace.extension2.is_none(), "abs.W carries one extension");

        // 4279 = CLR.W (xxx).L assembles the address from both words.
        bus.write_word(0x0100, 0x4279);
        bus.write_word(0x0102, 0x0001);
        bus.write_word(0x0104, 0x4208);
        let trace = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("CLR.W (xxx).L should decode");
        assert!(matches!(
            trace.op,
            JitTraceOp::ClrMem {
                size: Size::Word,
                dst: JitEa::AbsLong(0x0001_4208),
            }
        ));
        assert_eq!(trace.extension, Some(0x0001));
        assert_eq!(trace.extension2, Some(0x4208));
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_clr_to_absolute_matches_portable_and_bails_atomically() {
        // Case 0: CLR.W (xxx).W. Case 1: CLR.L (xxx).L. Case 2: CLR.B
        // (xxx).L — the gameplay census head (4239).
        let cases: [(u16, u16, Option<u16>, JitTraceOp, usize, usize); 3] = [
            (
                0x4278,
                0x0320,
                None,
                JitTraceOp::ClrMem {
                    size: Size::Word,
                    dst: JitEa::AbsWord(0x0320),
                },
                0x0320,
                2,
            ),
            (
                0x42B9,
                0x0000,
                Some(0x0328),
                JitTraceOp::ClrMem {
                    size: Size::Long,
                    dst: JitEa::AbsLong(0x0328),
                },
                0x0328,
                4,
            ),
            (
                0x4239,
                0x0000,
                Some(0x0327),
                JitTraceOp::ClrMem {
                    size: Size::Byte,
                    dst: JitEa::AbsLong(0x0327),
                },
                0x0327,
                1,
            ),
        ];
        for (case, (opcode, ext, ext2, op, cleared, len)) in cases.iter().enumerate() {
            let clr_len: u32 = if ext2.is_some() { 6 } else { 4 };
            let branch_pc = 0x0100 + clr_len;
            let displacement = -(clr_len as i32) - 2;
            let branch_opcode = 0x6000 | (displacement as u8 as u16);
            let clr = TraceBuildOp {
                opcode: *opcode,
                extension: Some(*ext),
                extension2: *ext2,
                pc: 0x0100,
                op: *op,
            };
            let branch = TraceBuildOp {
                opcode: branch_opcode,
                extension: None,
                extension2: None,
                pc: branch_pc,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement,
                    length: 2,
                    expected_taken: None,
                },
            };
            let ops = vec![clr, branch];
            let trace_end = branch_pc + 2;
            let prepare = |mem: &mut Vec<u8>| {
                mem[0x0100..0x0102].copy_from_slice(&opcode.to_be_bytes());
                mem[0x0102..0x0104].copy_from_slice(&ext.to_be_bytes());
                if let Some(ext2) = ext2 {
                    mem[0x0104..0x0106].copy_from_slice(&ext2.to_be_bytes());
                }
                mem[0x0300..0x0400].fill(0xAA);
                let mut c = cpu();
                c.set_cpu_type(CpuType::M68040);
                c.set_ccr(0x10);
                attach_window(&mut c, mem);
                c
            };
            let mut emem = vec![0u8; 0x1000];
            let mut expected = prepare(&mut emem);
            let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, trace_end);
            let mut amem = vec![0u8; 0x1000];
            let mut actual = prepare(&mut amem);
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops.clone(), Some(0x0100))
                .expect("absolute CLR loop should compile");
            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
            assert_eq!(
                actual_packed, expected_packed,
                "case {case}: cycles/retired"
            );
            assert_ne!(expected_packed, 0, "case {case}: the trace ran");
            assert_eq!(actual.dar, expected.dar, "case {case}: registers");
            assert_eq!(actual.get_ccr(), expected.get_ccr(), "case {case}: ccr");
            assert_eq!(
                actual.get_ccr() & 0x1F,
                0x14,
                "case {case}: Z set, NVC clear, X preserved"
            );
            assert_eq!(amem, emem, "case {case}: memory");
            for offset in 0..*len {
                assert_eq!(amem[cleared + offset], 0, "case {case}: byte cleared");
            }
            assert_eq!(amem[cleared - 1], 0xAA, "case {case}: preceding byte kept");
            assert_eq!(
                amem[cleared + len],
                0xAA,
                "case {case}: following byte kept"
            );
        }

        // A CLR aimed at the trace's own extension word must bail on both
        // paths with nothing committed.
        let clr = TraceBuildOp {
            opcode: 0x4278,
            extension: Some(0x0102),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::ClrMem {
                size: Size::Word,
                dst: JitEa::AbsWord(0x0102),
            },
        };
        let branch = TraceBuildOp {
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
        };
        let ops = vec![clr, branch];
        let prepare = |mem: &mut Vec<u8>| {
            mem[0x0100..0x0102].copy_from_slice(&0x4278u16.to_be_bytes());
            mem[0x0102..0x0104].copy_from_slice(&0x0102u16.to_be_bytes());
            let mut c = cpu();
            c.set_cpu_type(CpuType::M68040);
            c.set_ccr(0x10);
            attach_window(&mut c, mem);
            c
        };
        let mut pmem = vec![0u8; 0x1000];
        let mut pcpu = prepare(&mut pmem);
        let packed = execute_portable_trace(&mut pcpu, &ops, 0x0100, 0x0106);
        assert_eq!(packed, 0, "portable bails on a store into code");
        assert_eq!(pcpu.pc, 0x0100);
        assert_eq!(pcpu.get_ccr(), 0x10);
        assert_eq!(
            pmem[0x0102..0x0104],
            [0x01, 0x02],
            "extension word untouched"
        );
        let mut nmem = vec![0u8; 0x1000];
        let mut ncpu = prepare(&mut nmem);
        let before = ncpu.dar;
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&ncpu, 0x0100, CpuType::M68040, ops, Some(0x0100))
            .expect("self-aimed absolute CLR still compiles");
        let packed = unsafe { compiled.call_native(&mut ncpu, 1) };
        assert_eq!(packed, 0, "native bails on a store into code");
        assert_eq!(ncpu.pc, 0x0100);
        assert_eq!(ncpu.dar, before);
        assert_eq!(ncpu.get_ccr(), 0x10);
        assert_eq!(nmem, pmem, "native bail commits nothing");
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn clr_to_absolute_cycles_match_the_interpreter_on_a_68000() {
        // One loop pass over all three widths, compiled for a 68000,
        // against the step interpreter's charge for the same sequence.
        let ops = vec![
            TraceBuildOp {
                opcode: 0x4278,
                extension: Some(0x0320),
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::ClrMem {
                    size: Size::Word,
                    dst: JitEa::AbsWord(0x0320),
                },
            },
            TraceBuildOp {
                opcode: 0x42B9,
                extension: Some(0x0000),
                extension2: Some(0x0328),
                pc: 0x0104,
                op: JitTraceOp::ClrMem {
                    size: Size::Long,
                    dst: JitEa::AbsLong(0x0328),
                },
            },
            TraceBuildOp {
                opcode: 0x4239,
                extension: Some(0x0000),
                extension2: Some(0x0327),
                pc: 0x010A,
                op: JitTraceOp::ClrMem {
                    size: Size::Byte,
                    dst: JitEa::AbsLong(0x0327),
                },
            },
            TraceBuildOp {
                opcode: 0x60EE,
                extension: None,
                extension2: None,
                pc: 0x0110,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -18,
                    length: 2,
                    expected_taken: None,
                },
            },
        ];
        let words: [u16; 9] = [
            0x4278, 0x0320, 0x42B9, 0x0000, 0x0328, 0x4239, 0x0000, 0x0327, 0x60EE,
        ];
        let prepare = |mem: &mut Vec<u8>| {
            for (index, word) in words.iter().enumerate() {
                let at = 0x0100 + index * 2;
                mem[at..at + 2].copy_from_slice(&word.to_be_bytes());
            }
            mem[0x0300..0x0400].fill(0xAA);
            let mut c = cpu();
            c.set_cpu_type(CpuType::M68000);
            attach_window(&mut c, mem);
            c
        };
        let mut nmem = vec![0u8; 0x1000];
        let mut ncpu = prepare(&mut nmem);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&ncpu, 0x0100, CpuType::M68000, ops, Some(0x0100))
            .expect("absolute CLR sequence should compile for a 68000");
        let packed = unsafe { compiled.call_native(&mut ncpu, 1) };
        assert_eq!((packed >> 32) as u32, 4, "all four ops retired");
        let native_cycles = packed as u32;

        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        for (index, word) in words.iter().enumerate() {
            bus.write_word(0x0100 + index as u32 * 2, *word);
        }
        let mut scpu = CpuCore::new();
        scpu.set_cpu_type(CpuType::M68000);
        scpu.set_sr(0x2700);
        scpu.pc = 0x0100;
        let mut step_cycles: u32 = 0;
        for _ in 0..4 {
            match scpu.step(&mut bus) {
                crate::StepResult::Ok { cycles } => step_cycles += cycles as u32,
                other => panic!("unexpected step result {other:?}"),
            }
        }
        assert_eq!(scpu.pc, 0x0100, "step run wrapped back to the head");
        assert_eq!(
            native_cycles, step_cycles,
            "native charge equals the 68000 interpreter's"
        );
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_clr_to_predecrement_matches_portable_and_bails_without_moving_sp() {
        let clr = TraceBuildOp {
            opcode: 0x42A7,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::ClrMem {
                size: Size::Long,
                dst: JitEa::PreDec(7),
            },
        };
        let branch = TraceBuildOp {
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
        };
        let ops = vec![clr, branch];
        let prepare = |mem: &mut Vec<u8>| {
            mem[0x0100..0x0102].copy_from_slice(&0x42A7u16.to_be_bytes());
            mem[0x0700..0x0704].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
            let mut c = cpu();
            c.set_cpu_type(CpuType::M68040);
            c.set_a(7, 0x0704);
            c.set_ccr(0x10);
            attach_window(&mut c, mem);
            c
        };
        let mut emem = vec![0u8; 0x1000];
        let mut expected = prepare(&mut emem);
        let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0104);
        let mut amem = vec![0u8; 0x1000];
        let mut actual = prepare(&mut amem);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops.clone(), Some(0x0100))
            .expect("predec CLR loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(actual_packed, expected_packed, "cycles/retired");
        assert_eq!(actual.a(7), 0x0700, "SP decremented once");
        assert_eq!(actual.dar, expected.dar, "registers");
        assert_eq!(actual.get_ccr(), expected.get_ccr(), "Z set, X preserved");
        assert_eq!(&amem[0x0700..0x0704], &[0, 0, 0, 0], "cell cleared");
        assert_eq!(amem, emem, "memory");

        // A predec store aimed at the trace's own code must bail on both
        // paths with the stack pointer NOT decremented.
        let overlap = |mem: &mut Vec<u8>| {
            let mut c = prepare(mem);
            c.set_a(7, 0x0106); // 0x0106 - 4 = 0x0102, inside the trace
            c
        };
        let mut pmem = vec![0u8; 0x1000];
        let mut pcpu = overlap(&mut pmem);
        let packed = execute_portable_trace(&mut pcpu, &ops, 0x0100, 0x0104);
        assert_eq!(packed, 0, "portable bails on a store into code");
        assert_eq!(pcpu.a(7), 0x0106, "portable bail leaves SP untouched");
        assert_eq!(pcpu.get_ccr(), 0x10);
        let mut nmem = vec![0u8; 0x1000];
        let mut ncpu = overlap(&mut nmem);
        let packed = unsafe { compiled.call_native(&mut ncpu, 1) };
        assert_eq!(packed, 0, "native bails on a store into code");
        assert_eq!(ncpu.a(7), 0x0106, "native bail leaves SP untouched");
        assert_eq!(ncpu.get_ccr(), 0x10);
        assert_eq!(nmem, pmem, "neither path commits anything");
    }

    #[test]
    fn move_immediate_to_memory_decodes_within_the_extension_budget() {
        // 3F3C = MOVE.W #imm,-(SP): the 20.9M-dispatch gameplay blocker.
        let dcpu = cpu();
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        bus.write_word(0x0100, 0x3F3C);
        bus.write_word(0x0102, 0x1234);
        let t = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("MOVE.W #imm,-(SP) should decode");
        assert!(matches!(
            t.op,
            JitTraceOp::MoveImmMem {
                size: Size::Word,
                value: 0x1234,
                dst: JitEa::PreDec(7),
            }
        ));
        // 2F3C = MOVE.L #imm,-(SP): both extension words carry the value.
        bus.write_word(0x0100, 0x2F3C);
        bus.write_word(0x0102, 0xDEAD);
        bus.write_word(0x0104, 0xBEEF);
        let t = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("MOVE.L #imm,-(SP) should decode");
        assert!(matches!(
            t.op,
            JitTraceOp::MoveImmMem {
                size: Size::Long,
                value: 0xDEAD_BEEF,
                dst: JitEa::PreDec(7),
            }
        ));
        assert_eq!((t.extension, t.extension2), (Some(0xDEAD), Some(0xBEEF)));
        // Byte immediates use the extension word's low byte only.
        bus.write_word(0x0100, 0x1F3C);
        bus.write_word(0x0102, 0xAABB);
        let t = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("MOVE.B #imm,-(SP) should decode");
        assert!(matches!(
            t.op,
            JitTraceOp::MoveImmMem {
                size: Size::Byte,
                value: 0xBB,
                dst: JitEa::PreDec(7),
            }
        ));
        // MOVE.W #imm,(d16,An): the displacement rides in extension2.
        bus.write_word(0x0100, 0x3B7C); // MOVE.W #imm,(d16,A5)
        bus.write_word(0x0102, 0x0042);
        bus.write_word(0x0104, 0x0010);
        let t = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("MOVE.W #imm,(d16,An) should decode");
        assert!(matches!(
            t.op,
            JitTraceOp::MoveImmMem {
                size: Size::Word,
                value: 0x0042,
                dst: JitEa::Disp(5, 0x0010),
            }
        ));
        // MOVE.L #imm,(d16,An) needs three extension words: stays decoded.
        bus.write_word(0x0100, 0x2B7C);
        let t = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040);
        assert!(
            !matches!(t.map(|t| t.op), Some(JitTraceOp::MoveImmMem { .. })),
            "three-extension form must not decode as MoveImmMem"
        );
        // 31BC = MOVE.W #imm,(d8,A0,Xn): the immediate word plus the brief
        // extension fit the two-slot budget.
        bus.write_word(0x0100, 0x31BC);
        bus.write_word(0x0102, 0x0042);
        bus.write_word(0x0104, 0x2004); // (4,A0,D2.W)
        let t = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040)
            .expect("MOVE.W #imm,(d8,An,Xn) should decode");
        assert!(matches!(
            t.op,
            JitTraceOp::MoveImmMem {
                size: Size::Word,
                value: 0x0042,
                dst: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
            }
        ));
        // ...but the long form would need three words and stays decoded.
        bus.write_word(0x0100, 0x21BC);
        let t = decode_trace_op(&dcpu, &mut bus, 0x0100, CpuType::M68040);
        assert!(
            !matches!(t.map(|t| t.op), Some(JitTraceOp::MoveImmMem { .. })),
            "long immediate to an indexed destination must stay decoded"
        );
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_move_immediate_to_memory_matches_portable_and_bails_without_moving_sp() {
        // Case 0: MOVE.W #imm,-(SP). Case 1: MOVE.L #imm,-(SP).
        // Case 2: MOVE.W #imm,(d16,A5).
        let cases: [(u16, JitTraceOp, Option<u16>, Option<u16>); 4] = [
            (
                0x3F3C,
                JitTraceOp::MoveImmMem {
                    size: Size::Word,
                    value: 0x1234,
                    dst: JitEa::PreDec(7),
                },
                Some(0x1234),
                None,
            ),
            (
                0x2F3C,
                JitTraceOp::MoveImmMem {
                    size: Size::Long,
                    value: 0xDEAD_BEEF,
                    dst: JitEa::PreDec(7),
                },
                Some(0xDEAD),
                Some(0xBEEF),
            ),
            (
                0x3B7C,
                JitTraceOp::MoveImmMem {
                    size: Size::Word,
                    value: 0x8000, // negative word: N must set
                    dst: JitEa::Disp(5, 0x0010),
                },
                Some(0x8000),
                Some(0x0010),
            ),
            (
                0x31BC, // MOVE.W #imm,(4,A0,D2.W)
                JitTraceOp::MoveImmMem {
                    size: Size::Word,
                    value: 0x0042,
                    dst: JitEa::Index {
                        base: 0,
                        index: JitDirectReg::Data(2),
                        index_long: false,
                        scale: 0,
                        displacement: 4,
                    },
                },
                Some(0x0042),
                Some(0x2004),
            ),
        ];
        for (case, (opcode, op, ext, ext2)) in cases.iter().enumerate() {
            let mv = TraceBuildOp {
                opcode: *opcode,
                extension: *ext,
                extension2: *ext2,
                pc: 0x0100,
                op: *op,
            };
            let oplen = 2 + 2 * (ext.is_some() as u32 + ext2.is_some() as u32);
            let branch = TraceBuildOp {
                opcode: 0x60FE,
                extension: None,
                extension2: None,
                pc: 0x0100 + oplen,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -(oplen as i32) - 2,
                    length: 2,
                    expected_taken: None,
                },
            };
            let ops = vec![mv, branch];
            let end = 0x0100 + oplen + 2;
            let prepare = |mem: &mut Vec<u8>| {
                let mut c = cpu();
                c.set_cpu_type(CpuType::M68040);
                c.set_a(5, 0x0300);
                c.set_a(7, 0x0800);
                c.set_ccr(0x10); // X set: MOVE must preserve it
                mem[0x0700..0x0708].fill(0xAA);
                mem[0x0310..0x0312].fill(0xAA);
                attach_window(&mut c, mem);
                c
            };
            let mut emem = vec![0u8; 0x1000];
            let mut expected = prepare(&mut emem);
            let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, end);
            let mut amem = vec![0u8; 0x1000];
            let mut actual = prepare(&mut amem);
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops.clone(), Some(0x0100))
                .expect("immediate store loop should compile");
            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
            assert_eq!(
                actual_packed, expected_packed,
                "case {case}: cycles/retired"
            );
            assert_eq!(actual.dar, expected.dar, "case {case}: registers");
            assert_eq!(actual.get_ccr(), expected.get_ccr(), "case {case}: ccr");
            assert_eq!(amem, emem, "case {case}: memory");
            assert_ne!(actual.get_ccr() & 0x10, 0, "case {case}: X preserved");

            if case == 0 {
                // A predec store aimed at the trace's own code bails on
                // both paths with SP untouched.
                let overlap = |mem: &mut Vec<u8>| {
                    let mut c = prepare(mem);
                    c.set_a(7, 0x0104); // 0x0104 - 2 = 0x0102, inside the trace
                    c
                };
                let mut pmem = vec![0u8; 0x1000];
                let mut pcpu = overlap(&mut pmem);
                let packed = execute_portable_trace(&mut pcpu, &ops, 0x0100, end);
                assert_eq!(packed, 0, "portable bails on store into code");
                assert_eq!(pcpu.a(7), 0x0104, "portable bail leaves SP untouched");
                let mut nmem = vec![0u8; 0x1000];
                let mut ncpu = overlap(&mut nmem);
                let packed = unsafe { compiled.call_native(&mut ncpu, 1) };
                assert_eq!(packed, 0, "native bails on store into code");
                assert_eq!(ncpu.a(7), 0x0104, "native bail leaves SP untouched");
                assert_eq!(nmem, pmem, "neither path commits anything");
            }
        }
    }

    #[test]
    fn move_immediate_matches_the_interpreter_with_exact_68000_cycles() {
        // Native parity alone shares this feature's addressing, flag, and
        // max_cycles logic on both sides; a shared regression would pass
        // it. This differential pits the portable executor against the
        // interpreter's own step() on a 68000 for every admitted form,
        // asserting the exact register/memory/NZVCX state AND that the
        // trace's cycle charge equals the interpreter's true cycle count.
        let cases: [(&[u16], &str); 4] = [
            (&[0x3F3C, 0x8111], "MOVE.W #imm,-(SP)"),
            (&[0x2F3C, 0xDEAD, 0xBEEF], "MOVE.L #imm,-(SP)"),
            (&[0x3B7C, 0x0000, 0x0010], "MOVE.W #zero,(d16,A5)"),
            (&[0x31BC, 0x0042, 0x2004], "MOVE.W #imm,(4,A0,D2.W)"),
        ];
        for (words, label) in cases {
            let setup = |c: &mut CpuCore| {
                c.set_cpu_type(CpuType::M68000);
                c.set_a(0, 0x0300);
                c.set_a(5, 0x0400);
                c.set_a(7, 0x0800);
                c.set_d(2, 0x0006);
                c.set_ccr(0x1F); // X must survive; N/Z/V/C must be rewritten
                c.pc = 0x0100;
            };
            // Interpreter twin: true 68000 execution with true cycles,
            // reading and writing through its own bus.
            let mut ibus = super::super::memory::LinearMemoryBus::new(0x1000);
            for (index, word) in words.iter().enumerate() {
                ibus.write_word(0x0100 + index as u32 * 2, *word);
            }
            let mut icpu = cpu();
            setup(&mut icpu);
            let icycles = match icpu.step(&mut ibus) {
                super::super::types::StepResult::Ok { cycles } => cycles,
                other => panic!("{label}: interpreter step failed: {other:?}"),
            };
            // Portable twin: the trace op decoded from the same bytes,
            // executing through the attached window.
            let mut pmem = vec![0u8; 0x1000];
            for (index, word) in words.iter().enumerate() {
                pmem[0x0100 + index * 2..0x0102 + index * 2].copy_from_slice(&word.to_be_bytes());
            }
            let mut pcpu = cpu();
            setup(&mut pcpu);
            attach_window(&mut pcpu, &mut pmem);
            let t = decode_trace_op(&pcpu, &mut ibus, 0x0100, CpuType::M68000)
                .unwrap_or_else(|| panic!("{label}: should decode"));
            assert!(matches!(t.op, JitTraceOp::MoveImmMem { .. }), "{label}");
            let pcycles =
                execute_portable_op(&mut pcpu, t, 0x0100, 0x0100 + words.len() as u32 * 2)
                    .unwrap_or_else(|| panic!("{label}: portable executes"));
            assert_eq!(pcpu.dar, icpu.dar, "{label}: registers");
            assert_eq!(pcpu.get_ccr(), icpu.get_ccr(), "{label}: NZVCX");
            // The stored value, read from each twin's own memory at the
            // interpreter-computed destination.
            let (dst_addr, bytes) = match t.op {
                JitTraceOp::MoveImmMem {
                    size,
                    dst: JitEa::PreDec(r),
                    ..
                } => (icpu.a(r.into()), size.bytes()),
                JitTraceOp::MoveImmMem {
                    size,
                    dst: JitEa::Disp(r, d),
                    ..
                } => (icpu.a(r.into()).wrapping_add(d as i32 as u32), size.bytes()),
                JitTraceOp::MoveImmMem {
                    size,
                    dst:
                        JitEa::Index {
                            base, displacement, ..
                        },
                    ..
                } => (
                    icpu.a(base.into())
                        .wrapping_add(icpu.d(2) as u16 as i16 as i32 as u32)
                        .wrapping_add(displacement as i32 as u32),
                    size.bytes(),
                ),
                _ => unreachable!(),
            };
            for offset in 0..bytes {
                let a = dst_addr + offset;
                assert_eq!(
                    pmem[a as usize],
                    ibus.read_byte(a),
                    "{label}: stored byte at {a:#06x}"
                );
            }
            assert_eq!(
                pcycles, icycles,
                "{label}: the trace cycle charge must equal the 68000's"
            );
        }
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_cmp_indexed_matches_portable_and_bails_atomically() {
        let cmp = TraceBuildOp {
            opcode: 0xB270,
            extension: Some(0x2004),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Word,
                src: JitEa::Index {
                    base: 0,
                    index: JitDirectReg::Data(2),
                    index_long: false,
                    scale: 0,
                    displacement: 4,
                },
                dst: 1,
            },
        };
        let branch = TraceBuildOp {
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
        };
        let ops = vec![cmp, branch];

        let prepare = |mem: &mut [u8]| {
            mem[0x0102..0x0104].copy_from_slice(&0x2004u16.to_be_bytes());
            mem[0x0206..0x0208].copy_from_slice(&0x8000u16.to_be_bytes());
            let mut cpu = cpu();
            attach_window(&mut cpu, mem);
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_a(0, 0x0200);
            cpu.set_d(1, 0x1234_7FFF);
            cpu.set_d(2, 2);
            cpu.set_ccr(0x10);
            cpu
        };

        let mut expected_mem = vec![0u8; 0x1000];
        let mut expected = prepare(&mut expected_mem);
        let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0106);

        let mut actual_mem = vec![0u8; 0x1000];
        let mut actual = prepare(&mut actual_mem);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops, Some(0x0100))
            .expect("indexed CMP loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(actual_packed, expected_packed);
        assert_eq!(actual.pc, expected.pc);
        assert_eq!(actual.dar, expected.dar);
        assert_eq!(actual.get_ccr(), expected.get_ccr());

        actual.pc = 0x0100;
        actual.set_a(0, 0x0FFE); // indexed word read falls outside the window
        actual.set_d(1, 0x1234_7FFF);
        actual.set_d(2, 2);
        actual.set_ccr(0x10);
        let before = actual.dar;
        let packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(packed, 0, "bail retires no instructions or cycles");
        assert_eq!(actual.pc, 0x0100);
        assert_eq!(actual.dar, before);
        assert_eq!(actual.get_ccr(), 0x10);
    }

    #[test]
    fn portable_cmpa_word_memory_sign_extends_and_preserves_destination_and_x() {
        let mut cpu = cpu();
        let mut mem = vec![0u8; 0x1000];
        mem[0x0102..0x0104].copy_from_slice(&0x0010u16.to_be_bytes());
        mem[0x0210..0x0212].copy_from_slice(&0xFFFFu16.to_be_bytes());
        attach_window(&mut cpu, &mut mem);
        cpu.set_a(1, 0x0200);
        cpu.set_a(3, 0);
        cpu.set_ccr(0x10);

        let op = TraceBuildOp {
            opcode: 0xB6E9,
            extension: Some(0x0010),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AddrCmpMemToReg {
                size: Size::Word,
                src: JitEa::Disp(1, 0x0010),
                dst: 3,
            },
        };
        assert_eq!(execute_portable_op(&mut cpu, op, 0x0100, 0x0104), Some(24));
        assert_eq!(cpu.a(3), 0, "CMPA must not write its destination");
        assert_eq!(cpu.pc, 0x0104);
        assert_eq!(
            cpu.get_ccr(),
            0x11,
            "word source is sign-extended; X is preserved"
        );
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_cmpa_displacement_matches_portable_and_bails_atomically() {
        let cmp = TraceBuildOp {
            opcode: 0xB7E9,
            extension: Some(0x0010),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AddrCmpMemToReg {
                size: Size::Long,
                src: JitEa::Disp(1, 0x0010),
                dst: 3,
            },
        };
        let branch = TraceBuildOp {
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
        };
        let ops = vec![cmp, branch];

        let mut expected = cpu();
        let mut expected_mem = vec![0u8; 0x1000];
        expected_mem[0x0102..0x0104].copy_from_slice(&0x0010u16.to_be_bytes());
        expected_mem[0x0210..0x0214].copy_from_slice(&0x1234_5678u32.to_be_bytes());
        attach_window(&mut expected, &mut expected_mem);
        expected.set_cpu_type(CpuType::M68040);
        expected.set_a(1, 0x0200);
        expected.set_a(3, 0x1234_5678);
        expected.set_ccr(0x10);
        let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0106);

        let mut actual = cpu();
        let mut actual_mem = vec![0u8; 0x1000];
        actual_mem[0x0102..0x0104].copy_from_slice(&0x0010u16.to_be_bytes());
        actual_mem[0x0210..0x0214].copy_from_slice(&0x1234_5678u32.to_be_bytes());
        attach_window(&mut actual, &mut actual_mem);
        actual.set_cpu_type(CpuType::M68040);
        actual.set_a(1, 0x0200);
        actual.set_a(3, 0x1234_5678);
        actual.set_ccr(0x10);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops, Some(0x0100))
            .expect("CMPA loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

        assert_eq!(actual_packed, expected_packed);
        assert_eq!(actual.a(3), expected.a(3));
        assert_eq!(actual.get_ccr(), expected.get_ccr());
        assert_eq!(actual.pc, expected.pc);

        actual.set_a(1, 0x00FF_FFF8);
        actual.set_a(3, 0xABCD_EF01);
        actual.set_ccr(0x15);
        actual.pc = 0x0100;
        let packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(packed, 0);
        assert_eq!(actual.pc, 0x0100);
        assert_eq!(actual.a(3), 0xABCD_EF01);
        assert_eq!(actual.get_ccr(), 0x15);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_indirect_memory_addq_matches_portable_and_bails_on_code() {
        let addq = TraceBuildOp {
            opcode: 0x5497,
            extension: None,
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::MemAddqSubq {
                data: 2,
                size: Size::Long,
                dst: JitEa::Ind(7),
                is_sub: false,
            },
        };
        let branch = TraceBuildOp {
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
        };
        let ops = vec![addq, branch];

        let prepare = || {
            let mut cpu = cpu();
            let mut mem = vec![0u8; 0x1000];
            mem[0x0100..0x0102].copy_from_slice(&0x5497u16.to_be_bytes());
            mem[0x0102..0x0104].copy_from_slice(&0x60FCu16.to_be_bytes());
            mem[0x0300..0x0304].copy_from_slice(&0x7FFF_FFFFu32.to_be_bytes());
            attach_window(&mut cpu, &mut mem);
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_a(7, 0x0300);
            cpu.set_ccr(0);
            (cpu, mem)
        };

        let (mut expected, expected_mem) = prepare();
        let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0104);

        let (mut actual, actual_mem) = prepare();
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops, Some(0x0100))
            .expect("indirect memory ADDQ loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

        assert_eq!(actual_packed, expected_packed);
        assert_eq!(&actual_mem[0x0300..0x0304], &expected_mem[0x0300..0x0304]);
        assert_eq!(&actual_mem[0x0300..0x0304], &0x8000_0001u32.to_be_bytes());
        assert_eq!(actual.get_ccr(), expected.get_ccr());
        assert_eq!(actual.pc, 0x0100);

        actual.set_a(7, 0x0100);
        actual.set_ccr(0x15);
        actual.pc = 0x0100;
        let code_before = actual_mem[0x0100..0x0104].to_vec();
        let packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(packed, 0, "bail retires no instructions or cycles");
        assert_eq!(actual.pc, 0x0100);
        assert_eq!(actual.get_ccr(), 0x15);
        assert_eq!(&actual_mem[0x0100..0x0104], code_before.as_slice());
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_register_long_multiply_matches_portable() {
        for (signed, source, destination) in [
            (false, 0xFFFF_FFFFu32, 2u32),
            (true, 0xFFFF_FFFE, 3),
            (true, 0x4000_0000, 4),
        ] {
            let extension = (2 << 12) | if signed { 0x0800 } else { 0 };
            let multiply = TraceBuildOp {
                opcode: 0x4C03,
                extension: Some(extension),
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::MulLongDataReg {
                    src: 3,
                    dst: 2,
                    signed,
                },
            };
            let branch = TraceBuildOp {
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
            };
            let ops = vec![multiply, branch];

            let prepare = || {
                let mut cpu = cpu();
                let mut mem = vec![0u8; 0x1000];
                mem[0x0100..0x0102].copy_from_slice(&0x4C03u16.to_be_bytes());
                mem[0x0102..0x0104].copy_from_slice(&extension.to_be_bytes());
                mem[0x0104..0x0106].copy_from_slice(&0x60FAu16.to_be_bytes());
                attach_window(&mut cpu, &mut mem);
                cpu.set_cpu_type(CpuType::M68040);
                cpu.set_d(3, source);
                cpu.set_d(2, destination);
                cpu.set_ccr(0x1F);
                (cpu, mem)
            };

            let (mut expected, _expected_mem) = prepare();
            let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0106);

            let (mut actual, _actual_mem) = prepare();
            let mut jit = TraceJit::new();
            let compiled = jit
                .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops, Some(0x0100))
                .expect("register long multiply loop should compile");
            let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

            assert_eq!(actual_packed, expected_packed, "signed={signed}");
            assert_eq!(actual.d(2), expected.d(2), "signed={signed}");
            assert_eq!(actual.get_ccr(), expected.get_ccr(), "signed={signed}");
            assert_eq!(actual.get_ccr() & 0x10, 0x10, "X must be preserved");
            assert_eq!(actual.pc, 0x0100);
        }
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_immediate_cmpa_matches_portable_and_preserves_x() {
        let cmpa = TraceBuildOp {
            opcode: 0xBCFC,
            extension: Some(0xFFFF),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AddrCmpImmediate {
                immediate: 0xFFFF,
                dst: 6,
                size: Size::Word,
                cycles: 10,
            },
        };
        let branch = TraceBuildOp {
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
        };
        let ops = vec![cmpa, branch];

        let prepare = || {
            let mut cpu = cpu();
            let mut mem = vec![0u8; 0x1000];
            mem[0x0100..0x0102].copy_from_slice(&0xBCFCu16.to_be_bytes());
            mem[0x0102..0x0104].copy_from_slice(&0xFFFFu16.to_be_bytes());
            mem[0x0104..0x0106].copy_from_slice(&0x60FAu16.to_be_bytes());
            attach_window(&mut cpu, &mut mem);
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_a(6, 0);
            cpu.set_ccr(0x10);
            (cpu, mem)
        };

        let (mut expected, _expected_mem) = prepare();
        let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x0106);

        let (mut actual, _actual_mem) = prepare();
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops, Some(0x0100))
            .expect("immediate CMPA loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

        assert_eq!(actual_packed, expected_packed);
        assert_eq!(actual.a(6), 0, "CMPA must not write its destination");
        assert_eq!(actual.get_ccr(), expected.get_ccr());
        assert_eq!(actual.get_ccr(), 0x11, "X is preserved while C is updated");
        assert_eq!(actual.pc, 0x0100);
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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
                op: JitTraceOp::MemAddqSubq {
                    data: 1,
                    size: Size::Word,
                    dst: JitEa::Disp(5, 0x0100),
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_trace_accepts_supported_shift_forms_with_either_count() {
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

        // Register counts are admitted for the same three forms; the
        // distance, the shifted-out bit, and the cycle cost are computed in
        // the trace instead of folded at compile time.
        let register_asr = DecodedSimpleOp::decode(CpuType::M68040, 0xE267)
            .unwrap()
            .to_jit_trace_op();
        assert!(matches!(
            register_asr,
            Some(JitTraceOp::ShiftReg {
                reg: 7,
                size: Size::Word,
                count_or_reg: 1,
                count_is_register: true,
                direction: 0,
                op: 0,
            })
        ));

        // ASL still refuses with either count kind: its overflow flag is
        // set from whether the sign bit changed during the shift, which is
        // not yet lowered.
        let register_asl = DecodedSimpleOp::decode(CpuType::M68040, 0xE367)
            .unwrap()
            .to_jit_trace_op();
        assert!(register_asl.is_none());

        // Rotates likewise: ROXL/ROXR carry the extend bit through the
        // rotation, and ROL/ROR have their own carry rule.
        let register_rol = DecodedSimpleOp::decode(CpuType::M68040, 0xE37F)
            .unwrap()
            .to_jit_trace_op();
        assert!(register_rol.is_none());
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_immediate_lsr_matches_interpreter() {
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
                let opcode = 0xE008 | (u16::from(encoded_count) << 9) | (size_code << 6);
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
                            direction: 0,
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
                let (result, shift_cycles) = expected.exec_lsr(size, shift, initial & size.mask());
                expected.set_d(0, (initial & !size.mask()) | result);

                let mut actual = cpu();
                actual.set_cpu_type(cpu_type);
                actual.set_d(0, initial);
                actual.set_ccr(0x1F);
                let mut jit = TraceJit::new();
                let compiled = jit
                    .compile_decoded_ops(&actual, 0x0100, cpu_type, ops, Some(0x0100))
                    .expect("native LSR loop should compile");
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

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_fixed_point_update_loop_matches_portable_and_bails_atomically() {
        // A compiler can use this fixed-point state update to avoid division:
        //
        //   state->accumulator += state->step >> 8;
        //   state->cursor = state->base + 2 * (state->accumulator >> 8);
        //
        // The trace includes both newly admitted operations: LSR.L #8,D0 and
        // ADD.L D0,d16(A1). The remaining instructions were already admitted.
        let ops = vec![
            TraceBuildOp {
                opcode: 0x2029, // MOVE.L $0010(A1),D0
                extension: Some(0x0010),
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::MoveMem {
                    size: Size::Long,
                    src: JitEa::Disp(1, 0x0010),
                    dst: JitEa::Data(0),
                },
            },
            TraceBuildOp {
                opcode: 0xE088, // LSR.L #8,D0
                extension: None,
                extension2: None,
                pc: 0x0104,
                op: JitTraceOp::ShiftReg {
                    reg: 0,
                    size: Size::Long,
                    count_or_reg: 0,
                    count_is_register: false,
                    direction: 0,
                    op: 1,
                },
            },
            TraceBuildOp {
                opcode: 0xD1A9, // ADD.L D0,$0018(A1)
                extension: Some(0x0018),
                extension2: None,
                pc: 0x0106,
                op: JitTraceOp::AddRegToMem {
                    is_sub: false,
                    size: Size::Long,
                    src: 0,
                    dst: JitEa::Disp(1, 0x0018),
                },
            },
            TraceBuildOp {
                opcode: 0x2029, // MOVE.L $0018(A1),D0
                extension: Some(0x0018),
                extension2: None,
                pc: 0x010A,
                op: JitTraceOp::MoveMem {
                    size: Size::Long,
                    src: JitEa::Disp(1, 0x0018),
                    dst: JitEa::Data(0),
                },
            },
            TraceBuildOp {
                opcode: 0xE088, // LSR.L #8,D0
                extension: None,
                extension2: None,
                pc: 0x010E,
                op: JitTraceOp::ShiftReg {
                    reg: 0,
                    size: Size::Long,
                    count_or_reg: 0,
                    count_is_register: false,
                    direction: 0,
                    op: 1,
                },
            },
            TraceBuildOp {
                opcode: 0xD080, // ADD.L D0,D0
                extension: None,
                extension2: None,
                pc: 0x0110,
                op: JitTraceOp::BinaryDataReg {
                    op: JitBinaryOp::Add,
                    src: JitDirectReg::Data(0),
                    dst: 0,
                    size: Size::Long,
                    cycles: 6,
                },
            },
            TraceBuildOp {
                opcode: 0x2069, // MOVEA.L $0008(A1),A0
                extension: Some(0x0008),
                extension2: None,
                pc: 0x0112,
                op: JitTraceOp::MoveMem {
                    size: Size::Long,
                    src: JitEa::Disp(1, 0x0008),
                    dst: JitEa::Addr(0),
                },
            },
            TraceBuildOp {
                opcode: 0xD1C0, // ADDA.L D0,A0
                extension: None,
                extension2: None,
                pc: 0x0116,
                op: JitTraceOp::AddrDataReg {
                    op: JitAddrOp::Adda,
                    src: JitDirectReg::Data(0),
                    dst: 0,
                    size: Size::Long,
                },
            },
            TraceBuildOp {
                opcode: 0x2348, // MOVE.L A0,$0020(A1)
                extension: Some(0x0020),
                extension2: None,
                pc: 0x0118,
                op: JitTraceOp::MoveMem {
                    size: Size::Long,
                    src: JitEa::Addr(0),
                    dst: JitEa::Disp(1, 0x0020),
                },
            },
            TraceBuildOp {
                opcode: 0x60E2, // BRA.S $0100
                extension: None,
                extension2: None,
                pc: 0x011C,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -30,
                    length: 2,
                    expected_taken: None,
                },
            },
        ];

        let prepare = || {
            let mut cpu = cpu();
            let mut mem = vec![0u8; 0x1000];
            for op in &ops {
                let pc = op.pc as usize;
                mem[pc..pc + 2].copy_from_slice(&op.opcode.to_be_bytes());
                if let Some(extension) = op.extension {
                    mem[pc + 2..pc + 4].copy_from_slice(&extension.to_be_bytes());
                }
            }
            attach_window(&mut cpu, &mut mem);
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_a(1, 0x0300);
            mem[0x0308..0x030C].copy_from_slice(&0x0000_0500u32.to_be_bytes());
            mem[0x0310..0x0314].copy_from_slice(&0x0000_0200u32.to_be_bytes());
            mem[0x0318..0x031C].copy_from_slice(&0x0000_0100u32.to_be_bytes());
            (cpu, mem)
        };

        let (mut expected, expected_mem) = prepare();
        let expected_packed = execute_portable_trace(&mut expected, &ops, 0x0100, 0x011E);

        let (mut actual, actual_mem) = prepare();
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&actual, 0x0100, CpuType::M68040, ops, Some(0x0100))
            .expect("fixed-point update loop should compile");
        let actual_packed = unsafe { compiled.call_native(&mut actual, 1) };

        assert_eq!(actual_packed, expected_packed);
        assert_eq!(&actual_mem[0x0318..0x031C], &expected_mem[0x0318..0x031C]);
        assert_eq!(&actual_mem[0x0320..0x0324], &expected_mem[0x0320..0x0324]);
        assert_eq!(&actual_mem[0x0318..0x031C], &0x0000_0102u32.to_be_bytes());
        assert_eq!(&actual_mem[0x0320..0x0324], &0x0000_0502u32.to_be_bytes());
        assert_eq!(actual.d(0), expected.d(0));
        assert_eq!(actual.a(0), expected.a(0));
        assert_eq!(actual.get_ccr(), expected.get_ccr());
        assert_eq!(actual.pc, 0x0100);

        actual.set_a(1, 0x00FF_FFE8); // accumulator address wraps outside the window
        actual.set_d(0, 0xDEAD_BEEF);
        actual.set_ccr(0x15);
        actual.pc = 0x0100;
        let packed = unsafe { compiled.call_native(&mut actual, 1) };
        assert_eq!(packed, 0, "bail retires no instructions or cycles");
        assert_eq!(actual.pc, 0x0100);
        assert_eq!(actual.d(0), 0xDEAD_BEEF);
        assert_eq!(actual.get_ccr(), 0x15);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
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

    #[test]
    fn pure_poll_classification_requires_idempotence_and_a_memory_read() {
        let branch = TraceBuildOp {
            opcode: 0x67FA,
            extension: None,
            extension2: None,
            pc: 0x0104,
            op: JitTraceOp::Branch {
                condition: 7,
                displacement: -6,
                length: 2,
                expected_taken: None,
            },
        };
        let poll = TraceBuildOp {
            opcode: 0xB26D,
            extension: Some(0xFFF0),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Cmp,
                size: Size::Word,
                src: JitEa::Disp(5, -16),
                dst: 1,
            },
        };
        // A memory compare plus the loop branch: a pure poll.
        assert!(is_pure_poll_loop(&[poll, branch]));

        // A memory-tested loop is also a poll.
        let tst = TraceBuildOp {
            opcode: 0x4A6D,
            extension: Some(0xFFF0),
            extension2: None,
            pc: 0x0100,
            op: JitTraceOp::AnDispUnary {
                op: JitUnaryOp::Tst,
                size: Size::Word,
                reg: 5,
                displacement: -16,
            },
        };
        assert!(is_pure_poll_loop(&[tst, branch]));

        // No memory read: burning by intent (a calibration loop), not a poll.
        assert!(!is_pure_poll_loop(&[branch]));

        // Any register mutation makes progress self-driven.
        let counter = TraceBuildOp {
            opcode: 0x5280,
            extension: None,
            extension2: None,
            pc: 0x0102,
            op: JitTraceOp::AddqSubqReg {
                reg: 0,
                data: 1,
                size: Size::Long,
                is_sub: false,
            },
        };
        assert!(!is_pure_poll_loop(&[poll, counter, branch]));

        // A memory ALU that writes its destination register is not a poll.
        let add = TraceBuildOp {
            op: JitTraceOp::AluMemToReg {
                op: JitBinaryOp::Add,
                size: Size::Word,
                src: JitEa::Disp(5, -16),
                dst: 1,
            },
            ..poll
        };
        assert!(!is_pure_poll_loop(&[add, branch]));

        // Loading through a scratch register is register mutation, even
        // though the loop "only" polls: not classified (v1 is strict).
        let scratch_load = TraceBuildOp {
            op: JitTraceOp::MoveMem {
                size: Size::Word,
                src: JitEa::Disp(5, -16),
                dst: JitEa::Data(0),
            },
            ..poll
        };
        assert!(!is_pure_poll_loop(&[scratch_load, branch]));

        // Flag provenance: a bit test writes only Z, so a loop branching
        // on Z classifies…
        let btst = TraceBuildOp {
            opcode: 0x082D,
            extension: Some(0x0003),
            extension2: Some(0xFFF0),
            pc: 0x0100,
            op: JitTraceOp::AnDispBit {
                op: JitBitOp::Test,
                bit: JitBitSource::Imm(3),
                reg: 5,
                displacement: -16,
            },
        };
        let bne = TraceBuildOp {
            op: JitTraceOp::Branch {
                condition: 6,
                displacement: -8,
                length: 2,
                expected_taken: None,
            },
            ..branch
        };
        assert!(is_pure_poll_loop(&[btst, bne]));

        // …but the same loop branching on carry does not: BTST never
        // writes C, so the exit would depend on preserved pre-loop state.
        let bcs = TraceBuildOp {
            op: JitTraceOp::Branch {
                condition: 5,
                displacement: -8,
                length: 2,
                expected_taken: None,
            },
            ..branch
        };
        assert!(!is_pure_poll_loop(&[btst, bcs]));

        // A loop with only an unconditional branch has no memory-driven
        // exit and is not classified.
        let bra = TraceBuildOp {
            op: JitTraceOp::Branch {
                condition: 0,
                displacement: -6,
                length: 2,
                expected_taken: None,
            },
            ..branch
        };
        assert!(!is_pure_poll_loop(&[poll, bra]));

        // Regression for the multi-shape hazard: an interior guarded
        // branch means this head can record several dynamic paths (a poll
        // path and a mutating path), and the per-head wait flag would
        // erase the non-wait shapes' opportunity data. Such loops must
        // not classify; only a single terminal conditional branch keeps
        // the recorded path structurally unique.
        let guarded = TraceBuildOp {
            op: JitTraceOp::Branch {
                condition: 6,
                displacement: 4,
                length: 2,
                expected_taken: Some(false),
            },
            ..branch
        };
        assert!(!is_pure_poll_loop(&[poll, guarded, bra]));
        assert!(!is_pure_poll_loop(&[poll, guarded, branch]));
    }

    #[cfg(all(
        feature = "jit",
        feature = "trace-profile",
        not(target_family = "wasm")
    ))]
    #[test]
    fn blocker_then_eviction_then_wait_keeps_the_blocker_ranked() {
        // Ben's reverse-order reproduction: record the fall-through blocker
        // first, evict the slot, then classify the taken loop as a wait.
        // Both orderings must converge to the same reported state: the
        // blocker stays visible in the opportunity ranking, the wait shape
        // reports in the wait section, and wait-attributed volume cannot
        // inflate projected dispatches.
        super::super::trace_profile::reset();
        let mut bus = super::super::memory::LinearMemoryBus::new(0x4000);
        bus.write_word(0x0100, 0xB26D); // CMP.W (-0x10,A5),D1
        bus.write_word(0x0102, 0xFFF0);
        bus.write_word(0x0104, 0x67FA); // BEQ.S head
        bus.write_word(0x0106, 0xC3F0); // MULS.W (0,A0,D0.W),D1 -- untraceable
        bus.write_word(0x0108, 0x0800);
        bus.write_word(0x010A, 0x60F4); // BRA.S head
        bus.write_word(0x0300, 0x1234); // the polled word
        bus.write_word(0x2100, 0x5282); // ADDQ.L #1,D2
        bus.write_word(0x2102, 0x60FC); // BRA.S 0x2100

        let mut cpu = cpu();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(5, 0x0310);
        cpu.set_a(0, 0x0400);
        cpu.set_a(7, 0x0900);
        cpu.set_d(0, 0);

        // Phase 1: unequal compare -- fall-through records the blocker.
        cpu.pc = 0x0100;
        cpu.set_d(1, 0x5555);
        cpu.run_batch(&mut bus, 5_000, &[]);
        let mid = super::super::trace_profile::snapshot();
        assert_eq!(
            mid.rows
                .iter()
                .find(|row| row.start_pc == 0x0100)
                .expect("phase 1 head profiled")
                .blocker_pc,
            Some(0x0106),
            "phase 1 records the real blocker"
        );

        // Phase 2: evict the direct-mapped slot with the colliding head.
        cpu.pc = 0x2100;
        cpu.run_batch(&mut bus, 200, &[]);

        // Phase 3: equal compare -- the taken loop classifies as a wait.
        cpu.pc = 0x0100;
        cpu.set_d(1, 0x1234);
        cpu.run_batch(&mut bus, 5_000, &[]);

        let snapshot = super::super::trace_profile::snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0x0100)
            .expect("phase 3 head profiled");
        assert_eq!(
            row.blocker_pc,
            Some(0x0106),
            "the wait classification never disturbs blocker accounting"
        );
        assert!(row.wait_hits > 0, "phase-3 spins attribute as wait hits");
        assert!(
            snapshot
                .wait_shapes
                .iter()
                .any(|shape| shape.start_pc == 0x0100),
            "the wait shape reports independently"
        );
        assert_eq!(
            row.projected_dispatches(),
            row.rejected_hits
                .saturating_sub(row.wait_hits)
                .saturating_mul(u64::from(row.prefix_ops)),
            "wait volume is subtracted from the ranking metric"
        );

        let report = snapshot.report();
        let ranking = report
            .split("wait loops")
            .next()
            .expect("report has a ranking section");
        assert!(
            ranking.contains("00000100"),
            "the blocker keeps the head visible in the ranking: {ranking}"
        );
        assert!(report.contains("wait loops"));
    }

    #[cfg(all(
        feature = "jit",
        feature = "trace-profile",
        not(target_family = "wasm")
    ))]
    #[test]
    fn wait_then_eviction_then_fall_through_restores_the_blocker() {
        // The wait bit must reflect the most recent completed recording,
        // not the first. Sequence: classify a genuine wait; evict its
        // direct-mapped slot with a colliding head (0x0100 and 0x2100 both
        // map to slot 0x80); revisit with the compare unequal so the back
        // edge falls through into an unsupported operation. The second
        // recording finds a real blocker, and the head must return to the
        // opportunity ranking rather than staying hidden in the wait
        // section by the stale classification.
        super::super::trace_profile::reset();
        let mut bus = super::super::memory::LinearMemoryBus::new(0x4000);
        // Wait/fall-through head at 0x0100.
        bus.write_word(0x0100, 0xB26D); // CMP.W (-0x10,A5),D1
        bus.write_word(0x0102, 0xFFF0);
        bus.write_word(0x0104, 0x67FA); // BEQ.S head
        bus.write_word(0x0106, 0xC3F0); // MULS.W (0,A0,D0.W),D1 -- untraceable
        bus.write_word(0x0108, 0x0800);
        bus.write_word(0x010A, 0x60F4); // BRA.S head
        bus.write_word(0x0300, 0x1234); // the polled word
        // Colliding loop head at 0x2100.
        bus.write_word(0x2100, 0x5282); // ADDQ.L #1,D2
        bus.write_word(0x2102, 0x60FC); // BRA.S 0x2100

        let mut cpu = cpu();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(5, 0x0310);
        cpu.set_a(0, 0x0400);
        cpu.set_a(7, 0x0900);
        cpu.set_d(0, 0);

        // Phase 1: equal compare -- the loop spins and classifies as a wait.
        cpu.pc = 0x0100;
        cpu.set_d(1, 0x1234);
        cpu.run_batch(&mut bus, 5_000, &[]);
        let mid = super::super::trace_profile::snapshot();
        let row = mid
            .rows
            .iter()
            .find(|row| row.start_pc == 0x0100)
            .expect("phase 1 head profiled");
        assert!(
            mid.wait_shapes.iter().any(|shape| shape.start_pc == 0x0100),
            "phase 1 classifies the taken loop as a wait shape"
        );
        assert!(row.wait_hits > 0, "phase 1 spins attribute as wait hits");

        // Phase 2: a backward branch to the colliding head evicts the
        // rejected slot for 0x0100.
        cpu.pc = 0x2100;
        cpu.run_batch(&mut bus, 200, &[]);

        // Phase 3: unequal compare -- the back edge falls through into the
        // unsupported multiply and a fresh recording finds the blocker.
        cpu.pc = 0x0100;
        cpu.set_d(1, 0x5555);
        cpu.run_batch(&mut bus, 5_000, &[]);

        let snapshot = super::super::trace_profile::snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0x0100)
            .expect("phase 3 head profiled");
        assert_eq!(
            row.blocker_pc,
            Some(0x0106),
            "the real blocker is attributed"
        );
        // The fall-through's [CMP, BEQ] prefix compiles as a 2-op loop, so
        // fall-through volume flows through guard exits rather than
        // rejected hits: projected stays 0, and it is the recorded blocker
        // that keeps the head visible in the ranking.
        assert!(
            row.wait_hits > 0,
            "phase-1 spin volume stays wait-attributed"
        );
        assert_eq!(
            row.rejected_hits, row.wait_hits,
            "no non-wait rejected volume exists in this flow"
        );
        assert!(
            snapshot
                .wait_shapes
                .iter()
                .any(|shape| shape.start_pc == 0x0100),
            "the phase-1 wait shape survives independently"
        );

        let report = snapshot.report();
        let ranking = report
            .split("wait loops")
            .next()
            .expect("report has a ranking section");
        assert!(
            ranking.contains("00000100"),
            "the head returns to the opportunity ranking: {ranking}"
        );
    }

    #[cfg(all(
        feature = "jit",
        feature = "trace-profile",
        not(target_family = "wasm")
    ))]
    #[test]
    fn fall_through_past_a_back_edge_is_not_a_wait() {
        // The recorded operations here are byte-identical to a pure poll --
        // a memory compare and a conditional branch back to the head -- but
        // the branch was *not* taken and execution fell through into an
        // unsupported state-mutating operation. Classifying on the branch's
        // static target alone would call this a wait and drop its real
        // blocker out of the opportunity ranking.
        super::super::trace_profile::reset();
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        bus.write_word(0x0100, 0xB26D); // head: CMP.W (-0x10,A5),D1
        bus.write_word(0x0102, 0xFFF0);
        bus.write_word(0x0104, 0x67FA); // BEQ.S head  (never taken)
        bus.write_word(0x0106, 0xC3F0); // MULS.W (0,A0,D0.W),D1 -- untraceable
        bus.write_word(0x0108, 0x0800);
        bus.write_word(0x010A, 0x60F4); // BRA.S head
        bus.write_word(0x0300, 0x0001); // the polled word

        let mut cpu = cpu();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.pc = 0x0100;
        cpu.set_a(5, 0x0310);
        cpu.set_a(0, 0x0400);
        cpu.set_a(7, 0x0900);
        // Unequal, so the back edge falls through every iteration.
        cpu.set_d(1, 0x5555);
        cpu.set_d(0, 0);
        cpu.run_batch(&mut bus, 5_000, &[]);

        let snapshot = super::super::trace_profile::snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0x0100)
            .expect("the head was profiled");
        assert_eq!(
            row.wait_hits, 0,
            "a fall-through path is not a wait even though its recorded ops look like one"
        );
        assert!(
            snapshot
                .wait_shapes
                .iter()
                .all(|shape| shape.start_pc != 0x0100),
            "no wait shape is recorded for a fall-through"
        );
        assert_eq!(
            row.blocker_pc,
            Some(0x0106),
            "the real blocker is still attributed"
        );

        let report = snapshot.report();
        let ranking = report
            .split("wait loops")
            .next()
            .expect("report has a ranking section");
        assert!(
            ranking.contains("00000100"),
            "the head stays in the opportunity ranking: {ranking}"
        );
    }

    #[cfg(all(
        feature = "jit",
        feature = "trace-profile",
        not(target_family = "wasm")
    ))]
    #[test]
    fn pure_poll_loop_is_reported_as_wait_and_left_uncompiled() {
        super::super::trace_profile::reset();
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        bus.write_word(0x0100, 0xB26D); // CMP.W (-0x10,A5),D1
        bus.write_word(0x0102, 0xFFF0);
        bus.write_word(0x0104, 0x67FA); // BEQ.S back to 0x0100
        bus.write_word(0x0FF0, 0x1234); // the polled value

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = 0x0100;
        cpu.set_a(5, 0x1000);
        cpu.set_d(1, 0x1234); // equal → the poll loops until the budget ends

        let result = cpu.run_batch(&mut bus, 20_000, &[]);
        assert_eq!(result.instructions, 20_000, "the wait executes decoded");

        let snapshot = super::super::trace_profile::snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0x0100)
            .expect("the poll head was profiled");
        assert!(
            row.wait_hits > 0,
            "spins after classification attribute as wait hits"
        );
        assert_eq!(
            row.projected_dispatches(),
            0,
            "a pure wait carries no ranked opportunity"
        );
        assert!(
            snapshot
                .wait_shapes
                .iter()
                .any(|shape| shape.start_pc == 0x0100 && shape.recordings > 0),
            "the classified shape is recorded in the wait table"
        );
        // The chosen contract: the shape table (and the wait-loops report
        // section) is the sole representation of a classified wait. It is
        // not a silent rejection, so the reason field stays empty.
        assert_eq!(
            row.reject_reason, None,
            "a classified wait never masquerades as a silent-rejection reason"
        );
        assert!(
            !snapshot
                .compiled_shapes
                .iter()
                .any(|shape| shape.start_pc == 0x0100),
            "the wait was not compiled"
        );
        let report = snapshot.report();
        assert!(report.contains("wait loops"));
        assert!(
            !report
                .lines()
                .take_while(|line| !line.contains("wait loops"))
                .any(|line| line.contains("00000100")),
            "the wait head is excluded from the opportunity ranking"
        );

        // The veto routes through the reasoned rejection path added for
        // silent-rejection accounting, so assert it is attributed *as a
        // wait* there rather than as a generic compile-stage refusal or by
        // dropping out of the accounting entirely.
        assert!(
            !snapshot
                .silent_rejections
                .iter()
                .any(|entry| entry.start_pc == 0x0100),
            "a classified wait is not filed as a silent rejection"
        );
        assert!(
            !report.contains("backend"),
            "a classified wait is not attributed to the compiler backend"
        );
    }

    #[test]
    fn link_and_unlk_a7_forms_are_never_admitted() {
        let mut bus = super::super::memory::LinearMemoryBus::new(0x1000);
        bus.write_word_at(0x0100, 0x4E57); // LINK A7,#d16
        bus.write_word_at(0x0102, 0xFFF8);
        bus.write_word_at(0x0200, 0x4E5F); // UNLK A7
        let cpu = CpuCore::new();
        assert!(decode_link_unlk_trace_op(&cpu, &mut bus, 0x0100, 0x4E57).is_none());
        assert!(decode_link_unlk_trace_op(&cpu, &mut bus, 0x0200, 0x4E5F).is_none());
        // The A6 forms decode.
        bus.write_word_at(0x0300, 0x4E56);
        bus.write_word_at(0x0302, 0xFFF8);
        let link = decode_link_unlk_trace_op(&cpu, &mut bus, 0x0300, 0x4E56).unwrap();
        assert!(matches!(
            link.op,
            JitTraceOp::Link {
                reg: 6,
                displacement: -8
            }
        ));
        assert_eq!(link.length(), 4);
        let unlk = decode_link_unlk_trace_op(&cpu, &mut bus, 0x0400, 0x4E5E).unwrap();
        assert!(matches!(unlk.op, JitTraceOp::Unlk { reg: 6 }));
        assert_eq!(unlk.length(), 2);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    fn link_unlk_loop_ops() -> Vec<TraceBuildOp> {
        vec![
            TraceBuildOp {
                opcode: 0x4E56,
                extension: Some(0xFFF8),
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::Link {
                    reg: 6,
                    displacement: -8,
                },
            },
            TraceBuildOp {
                opcode: 0x4E5E,
                extension: None,
                extension2: None,
                pc: 0x0104,
                op: JitTraceOp::Unlk { reg: 6 },
            },
            TraceBuildOp {
                opcode: 0x60F8,
                extension: None,
                extension2: None,
                pc: 0x0106,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -8,
                    length: 2,
                    expected_taken: None,
                },
            },
        ]
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn native_link_unlk_matches_portable() {
        let ops = link_unlk_loop_ops();
        let mut mem = vec![0u8; 0x1000];
        let mut native = cpu();
        native.set_cpu_type(CpuType::M68040);
        native.set_a(6, 0x1111_2222);
        native.set_a(7, 0x0800);
        attach_window(&mut native, &mut mem);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&native, 0x0100, CpuType::M68040, ops.clone(), Some(0x0100))
            .expect("frame loop should compile");
        let packed = unsafe { compiled.call_native(&mut native, 1) };
        assert_eq!((packed >> 32) as u32, 3, "all ops retired");
        assert_eq!(native.a(6), 0x1111_2222, "frame pointer restored");
        assert_eq!(native.a(7), 0x0800, "stack balanced");
        assert_eq!(
            &mem[0x07FC..0x0800],
            &0x1111_2222u32.to_be_bytes(),
            "the old frame pointer was pushed"
        );

        let mut pmem = vec![0u8; 0x1000];
        let mut portable = cpu();
        portable.set_cpu_type(CpuType::M68040);
        portable.set_a(6, 0x1111_2222);
        portable.set_a(7, 0x0800);
        attach_window(&mut portable, &mut pmem);
        let ppacked = execute_portable_trace(&mut portable, &ops, 0x0100, 0x0108);
        assert_eq!(ppacked, packed, "retired count and cycles agree");
        assert_eq!(portable.a(6), native.a(6));
        assert_eq!(portable.a(7), native.a(7));
        assert_eq!(pmem, mem, "memory effects agree");
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn link_bails_atomically_on_window_and_own_span() {
        // Out of window: SP underflows the slab. Nothing commits.
        let ops = link_unlk_loop_ops();
        let mut mem = vec![0u8; 0x1000];
        let mut c = cpu();
        c.set_cpu_type(CpuType::M68040);
        c.set_a(6, 0x1111_2222);
        c.set_a(7, 0x0002);
        attach_window(&mut c, &mut mem);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&c, 0x0100, CpuType::M68040, ops.clone(), Some(0x0100))
            .expect("frame loop should compile");
        let packed = unsafe { compiled.call_native(&mut c, 1) };
        assert_eq!((packed >> 32) as u32, 0, "bails before the push");
        assert_eq!(c.a(6), 0x1111_2222);
        assert_eq!(c.a(7), 0x0002);
        assert_eq!(c.pc, 0x0100, "resume at the LINK for full dispatch");

        // Store into the trace's own code: in-window, aligned, guarded.
        let mut mem2 = vec![0u8; 0x1000];
        let mut c2 = cpu();
        c2.set_cpu_type(CpuType::M68040);
        c2.set_a(6, 0x1111_2222);
        c2.set_a(7, 0x0106);
        attach_window(&mut c2, &mut mem2);
        let packed2 = unsafe { compiled.call_native(&mut c2, 1) };
        assert_eq!((packed2 >> 32) as u32, 0, "own-span push bails");
        assert_eq!(c2.a(6), 0x1111_2222);
        assert_eq!(c2.a(7), 0x0106);

        // Portable parity for both bails.
        let mut c3 = cpu();
        c3.set_cpu_type(CpuType::M68040);
        c3.set_a(6, 0x1111_2222);
        c3.set_a(7, 0x0002);
        let mut mem3 = vec![0u8; 0x1000];
        attach_window(&mut c3, &mut mem3);
        assert_eq!(
            (execute_portable_trace(&mut c3, &ops, 0x0100, 0x0108) >> 32) as u32,
            0
        );
        assert_eq!(c3.a(7), 0x0002);
        c3.set_a(7, 0x0106);
        assert_eq!(
            (execute_portable_trace(&mut c3, &ops, 0x0100, 0x0108) >> 32) as u32,
            0
        );
        assert_eq!(c3.a(7), 0x0106);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn unlk_bails_atomically_on_misalignment() {
        // Pre-68020: an odd frame pointer must bail before either register
        // moves. Build UNLK-first ops so the bail is at op 0.
        let ops = vec![
            TraceBuildOp {
                opcode: 0x4E5E,
                extension: None,
                extension2: None,
                pc: 0x0100,
                op: JitTraceOp::Unlk { reg: 6 },
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
        let mut mem = vec![0u8; 0x1000];
        let mut c = cpu();
        c.set_cpu_type(CpuType::M68000);
        c.set_a(6, 0x0301);
        c.set_a(7, 0x0800);
        attach_window(&mut c, &mut mem);
        let mut jit = TraceJit::new();
        let compiled = jit
            .compile_decoded_ops(&c, 0x0100, CpuType::M68000, ops.clone(), Some(0x0100))
            .expect("UNLK loop should compile");
        let packed = unsafe { compiled.call_native(&mut c, 1) };
        assert_eq!((packed >> 32) as u32, 0, "odd frame pointer bails");
        assert_eq!(c.a(6), 0x0301, "frame register untouched");
        assert_eq!(c.a(7), 0x0800, "stack pointer untouched");

        let mut pmem = vec![0u8; 0x1000];
        let mut p = cpu();
        p.set_cpu_type(CpuType::M68000);
        p.set_a(6, 0x0301);
        p.set_a(7, 0x0800);
        attach_window(&mut p, &mut pmem);
        assert_eq!(
            (execute_portable_trace(&mut p, &ops, 0x0100, 0x0104) >> 32) as u32,
            0
        );
        assert_eq!(p.a(6), 0x0301);
        assert_eq!(p.a(7), 0x0800);
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn blocked_recording_salvages_the_prefix_through_the_last_branch() {
        // Nine admissible ops (the last a recorded interior branch), then
        // the A7 LINK/UNLK pair -- refused by documented design, so no
        // future coverage can admit this blocker -- then a loop tail. Without
        // salvage the whole head rejects; with it the prefix through the
        // branch compiles and the tail stays interpreted.
        const A: u32 = 0x0100;
        let words = [
            0x5282, // head: ADDQ.L #1,D2
            0x5283, // ADDQ.L #1,D3
            0x5284, // ADDQ.L #1,D4
            0x5285, // ADDQ.L #1,D5
            0x5286, // ADDQ.L #1,D6
            0x5287, // ADDQ.L #1,D7
            0x5281, // ADDQ.L #1,D1
            0x4A41, // TST.W D1
            0x6602, // BNE.S +2 (always taken: D1 counts up)
            0x4E71, // NOP (skipped)
            0x4A42, // TST.W D2 -- past the branch: no terminal to stop at
            0x4A42, // TST.W D2
            0x4E57, 0x0000, // LINK A7,#0 -- refused by design (A7 exclusion)
            0x4E5F, // UNLK A7 -- likewise; the pair nets zero stack motion
            0x51C8, 0xFFE0, // DBRA D0,head
            0x707F, // MOVEQ #127,D0
            0x60DA, // BRA.S head
        ];
        let mut bus = super::super::memory::LinearMemoryBus::new(0x10000);
        for (index, word) in words.iter().enumerate() {
            bus.write_word_at(A + index as u32 * 2, *word);
        }
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = A;
        cpu.set_a(6, 0x3000);
        cpu.set_a(7, 0x9000);
        cpu.set_d(0, 0x7F);
        let result = cpu.run_batch(&mut bus, 40_000, &[0]);
        assert_eq!(result.instructions, 40_000, "loop runs to budget");
        // Every iteration bumps D2..D7 and D1 exactly once each.
        assert!(cpu.d(2) > 2_000, "the loop actually iterated");
        assert!(
            cpu.d(2).abs_diff(cpu.d(3)) <= 1 && cpu.d(2).abs_diff(cpu.d(7)) <= 1,
            "counters advance in lockstep"
        );
        let salvaged = TRACE_JIT.with_borrow(|jit| match &jit.slots[trace_cache_index(A)] {
            TraceSlot::Compiled(trace) if trace.pc == A => Some((
                trace.ops.len(),
                matches!(
                    trace.ops.last().map(|op| op.op),
                    Some(JitTraceOp::Branch { .. })
                ),
                trace
                    .ops
                    .iter()
                    .all(|op| op.opcode != 0x4E57 && op.opcode != 0x4E5F),
            )),
            _ => None,
        });
        let (ops, ends_in_branch, excludes_blocked_tail) =
            salvaged.expect("the blocked head compiles its salvageable prefix");
        assert_eq!(ops, 9, "trimmed at the recorded branch");
        assert!(ends_in_branch, "the trimmed terminal is the branch");
        assert!(
            excludes_blocked_tail,
            "the unsupported tail is not recorded"
        );
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn short_blocked_recordings_still_reject() {
        // A five-op prefix whose last terminal sits at op four: below
        // SALVAGE_MIN_OPS the head must reject exactly as before. (The
        // blocker must not directly follow the branch: a recording whose
        // last op is already a terminal compiles today without salvage.)
        const A: u32 = 0x0100;
        let words = [
            0x5282, // head: ADDQ.L #1,D2
            0x5283, // ADDQ.L #1,D3
            0x4A42, // TST.W D2
            0x6602, // BNE.S +2 (always taken)
            0x4E71, // NOP (skipped)
            0x5284, // ADDQ.L #1,D4
            0x4E57, 0x0000, // LINK A7,#0 -- refused by design (A7 exclusion)
            0x4E5F, // UNLK A7 -- likewise; the pair nets zero stack motion
            0x51C8, 0xFFEC, // DBRA D0,head
            0x707F, // MOVEQ #127,D0
            0x60E6, // BRA.S head
        ];
        let mut bus = super::super::memory::LinearMemoryBus::new(0x10000);
        for (index, word) in words.iter().enumerate() {
            bus.write_word_at(A + index as u32 * 2, *word);
        }
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = A;
        cpu.set_a(6, 0x3000);
        cpu.set_a(7, 0x9000);
        cpu.set_d(0, 0x7F);
        cpu.run_batch(&mut bus, 40_000, &[0]);
        TRACE_JIT.with_borrow(|jit| {
            assert!(
                matches!(
                    &jit.slots[trace_cache_index(A)],
                    TraceSlot::Rejected { pc, .. } if *pc == A
                ),
                "a four-op prefix is below the salvage bar"
            );
        });
    }

    #[cfg(all(feature = "jit", not(target_family = "wasm")))]
    #[test]
    fn blocked_recording_without_a_branch_still_rejects() {
        // A straight-line prefix has no terminal to trim to; the head
        // must reject exactly as before regardless of length.
        const A: u32 = 0x0100;
        let words = [
            0x5282, 0x5283, 0x5284, 0x5285, 0x5286, 0x5287, 0x5281, 0x5280,
            0x4A41, // nine straight-line ops, no branch
            0x4E57, 0x0000, // LINK A7,#0 -- refused by design (A7 exclusion)
            0x4E5F, // UNLK A7 -- likewise; the pair nets zero stack motion
            0x51C8, 0xFFE6, // DBRA D0,head
            0x707F, // MOVEQ #127,D0
            0x60E0, // BRA.S head
        ];
        let mut bus = super::super::memory::LinearMemoryBus::new(0x10000);
        for (index, word) in words.iter().enumerate() {
            bus.write_word_at(A + index as u32 * 2, *word);
        }
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = A;
        cpu.set_a(6, 0x3000);
        cpu.set_a(7, 0x9000);
        cpu.set_d(0, 0x7F);
        cpu.run_batch(&mut bus, 40_000, &[0]);
        TRACE_JIT.with_borrow(|jit| {
            assert!(
                matches!(
                    &jit.slots[trace_cache_index(A)],
                    TraceSlot::Rejected { pc, .. } if *pc == A
                ),
                "no recorded branch means nothing to salvage"
            );
        });
    }
}

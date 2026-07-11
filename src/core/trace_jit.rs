//! Trace execution for hot simple-op loops.
//!
//! Native targets lower hot traces to Cranelift machine code. WebAssembly targets keep the same
//! trace detection and validation path, but execute the trace through a compact Rust micro-op loop.

#[cfg(not(target_family = "wasm"))]
use super::cpu::{CFLAG_SET, VFLAG_SET};
use super::cpu::{CpuCore, NFLAG_SET};
use super::execute::RUN_MODE_BERR_AERR_RESET;
use super::memory::AddressBus;
use super::op_cache::{CachedRunResult, DecodedSimpleOp};
use super::types::{CpuType, Size};
#[cfg(not(target_family = "wasm"))]
use cranelift_codegen::Context;
#[cfg(not(target_family = "wasm"))]
use cranelift_codegen::ir::{
    AbiParam, Block, Function, InstBuilder, MemFlags, Type, UserFuncName, Value, condcodes::IntCC,
    types,
};
#[cfg(not(target_family = "wasm"))]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(not(target_family = "wasm"))]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(not(target_family = "wasm"))]
use cranelift_module::{Linkage, Module, default_libcall_names};
use std::cell::RefCell;
use std::fmt;
#[cfg(not(target_family = "wasm"))]
use std::mem::{offset_of, size_of, transmute};
use std::sync::atomic::{AtomicBool, Ordering};

const TRACE_CACHE_SIZE: usize = 4096;
pub(crate) const TRACE_MAX_OPS: usize = 16;
const TRACE_HOT_THRESHOLD: u8 = 2;

/// Sentinel for `CpuCore::trace_record_skip` / `trace_probe_skip`: no PC.
pub(crate) const TRACE_PC_NONE: u32 = u32::MAX;

#[cfg(not(target_family = "wasm"))]
/// Compiled trace entry point. Returns `(ops_retired << 32) | cycles`:
/// a full pass retires every op; a mem-op bail retires a prefix and sets
/// `cpu.pc` to the first un-executed instruction.
type TraceFn = unsafe extern "C" fn(*mut CpuCore) -> u64;

static TRACE_JIT_HAS_CANDIDATES: AtomicBool = AtomicBool::new(false);

thread_local! {
    static TRACE_JIT: RefCell<TraceJit> = RefCell::new(TraceJit::new());
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JitDirectReg {
    Data(u8),
    Addr(u8),
}

/// Effective-address forms allowed in memory trace ops. All are one-word
/// (no extension), so the trace's code bytes stay contiguous and cheap to
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
}

impl JitEa {
    fn is_mem(self) -> bool {
        matches!(self, Self::Ind(_) | Self::PostInc(_) | Self::PreDec(_))
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
    },
    Dbcc {
        condition: u8,
        reg: u8,
        displacement: i16,
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
}

#[derive(Debug, Clone, Copy)]
struct TraceBuildOp {
    opcode: u16,
    extension: Option<u16>,
    pc: u32,
    op: JitTraceOp,
}

struct CompiledTrace {
    pc: u32,
    cpu_type: CpuType,
    ops: Vec<TraceBuildOp>,
    /// The exact instruction bytes the trace was compiled from (ops are
    /// contiguous from `pc`). Lets validation be a single compare against
    /// a fastmem window instead of per-op bus reads.
    code: Vec<u8>,
    max_cycles: i32,
    /// The final branch's taken-target is the trace head, so the trace is
    /// a whole loop iteration and can be re-run (budget permitting)
    /// without re-validating: trace stores that would touch code bail out
    /// before committing, and nothing observable happens between
    /// iterations.
    self_loop: bool,
    /// Contains `MoveMem` ops: only executable while a fastmem window is
    /// active (i.e. inside `run_batch`).
    needs_window: bool,
    /// Address-masked range of the trace's code bytes; trace stores into
    /// this range bail so self-modification is observed like the
    /// interpreter would. Baked into the compiled function on native
    /// targets; read at execution time by the portable path.
    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    code_start: u32,
    #[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
    code_end: u32,
    #[cfg(not(target_family = "wasm"))]
    func: TraceFn,
}

enum TraceSlot {
    Empty,
    Counting {
        pc: u32,
        cpu_type: CpuType,
        hits: u8,
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
    ) -> Option<(CachedRunResult, u32)> {
        #[cfg(not(target_family = "wasm"))]
        self.module.as_ref()?;

        if cpu.has_pmmu && cpu.pmmu_enabled || cpu.cycles_remaining <= 0 {
            return None;
        }

        let pc = cpu.pc;
        let idx = trace_cache_index(pc);

        if let TraceSlot::Compiled(trace) = &self.slots[idx]
            && trace.pc == pc
            && trace.cpu_type == cpu_type
        {
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
            if cpu.fm_len != 0 {
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
            loop {
                #[cfg(not(target_family = "wasm"))]
                let packed = unsafe { (trace.func)(cpu as *mut CpuCore) };
                #[cfg(target_family = "wasm")]
                let packed =
                    execute_portable_trace(cpu, &trace.ops, trace.code_start, trace.code_end);
                cycles_total += (packed as u32) as i64;
                let ops_done = (packed >> 32) as u32;
                retired += ops_done;
                if ops_done < ops_len {
                    // A mem op bailed mid-trace (window/alignment miss or a
                    // store into code). PC points at the un-executed op;
                    // the interpreter takes over from there.
                    break;
                }
                full_iters += 1;
                // Re-entering the (already validated) loop head is the only
                // continue condition; trace ops cannot fault or take
                // exceptions, and stores that would touch code bail before
                // committing, so nothing needs re-checking between
                // iterations.
                if full_iters >= max_iters || cpu.pc != pc {
                    break;
                }
            }
            cpu.cycles_remaining -= i32::try_from(cycles_total).unwrap_or(i32::MAX);
            if retired == 0 {
                // The very first op bailed: nothing executed. Fall back to
                // the interpreter so the offending instruction makes
                // progress through full dispatch.
                return None;
            }
            return Some((CachedRunResult::Ran, retired));
        }

        match &mut self.slots[idx] {
            TraceSlot::Counting {
                pc: counted_pc,
                cpu_type: counted_type,
                hits,
            } if *counted_pc == pc && *counted_type == cpu_type => {
                *hits = hits.saturating_add(1);
                if *hits < TRACE_HOT_THRESHOLD {
                    return None;
                }
            }
            TraceSlot::Rejected {
                pc: rejected_pc,
                cpu_type: rejected_type,
            } if *rejected_pc == pc && *rejected_type == cpu_type => {
                // Known-uncompilable target: tell the loop to stop probing
                // it (note_backward_branch consults this filter).
                push_probe_skip(cpu, pc);
                return None;
            }
            _ => {
                return None;
            }
        }

        let Some(trace) = self.compile_trace(cpu, bus, pc, cpu_type) else {
            self.slots[idx] = TraceSlot::Rejected { pc, cpu_type };
            push_probe_skip(cpu, pc);
            return None;
        };

        self.slots[idx] = TraceSlot::Compiled(trace);
        None
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
                };
                TRACE_JIT_HAS_CANDIDATES.store(true, Ordering::Relaxed);
            }
        }
    }

    fn compile_trace<B: AddressBus>(
        &mut self,
        cpu: &CpuCore,
        bus: &mut B,
        start_pc: u32,
        cpu_type: CpuType,
    ) -> Option<CompiledTrace> {
        let mut pc = start_pc;
        let mut ops = Vec::with_capacity(TRACE_MAX_OPS);
        let mut max_cycles = 0i32;

        for _ in 0..TRACE_MAX_OPS {
            let op = decode_trace_op(cpu, bus, pc, cpu_type)?;
            max_cycles += op.op.max_cycles();
            ops.push(op);

            let jit_op = op.op;
            pc = pc.wrapping_add(jit_op.length() as u32);
            if jit_op.ends_trace() {
                break;
            }
        }

        if ops.len() < 2 || !ops.last().is_some_and(|op| op.op.ends_trace()) {
            return None;
        }

        let mut code = Vec::with_capacity(ops.len() * 4);
        for op in &ops {
            code.extend_from_slice(&op.opcode.to_be_bytes());
            if let Some(extension) = op.extension {
                code.extend_from_slice(&extension.to_be_bytes());
            }
        }
        debug_assert_eq!(code.len() as u32, pc.wrapping_sub(start_pc));

        let self_loop = ops
            .last()
            .is_some_and(|op| op.op.taken_target(op.pc) == Some(start_pc));

        let needs_window = ops
            .iter()
            .any(|op| matches!(op.op, JitTraceOp::MoveMem { .. }));

        // Address-masked code range, used by the store-overlap (SMC) bail
        // checks. Reject the exotic case of a trace wrapping the address
        // space so the range stays a simple interval.
        let code_start = start_pc & cpu.address_mask;
        let code_end = code_start as u64 + code.len() as u64;
        if code_end > cpu.address_mask as u64 + 1 {
            return None;
        }
        let code_end = code_end as u32;

        self.compile_ops(CompileParams {
            start_pc,
            cpu_type,
            ops: &ops,
            code,
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
            max_cycles,
            self_loop,
            needs_window,
            code_start,
            code_end,
            aligned_only,
            address_mask,
        } = params;
        let module = self.module.as_mut()?;
        let ptr_ty = module.target_config().pointer_type();
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(ptr_ty));
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

            let mut bails: Vec<BailReq> = Vec::new();
            let mut cycles_before: i64 = 0;
            let mut cycles_value = builder.ins().iconst(types::I32, 0);
            for (index, op) in ops.iter().enumerate() {
                let op_cycles = if let JitTraceOp::MoveMem { size, src, dst } = op.op {
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
                        BailAt {
                            ops_before: index as u32,
                            cycles_before,
                        },
                    )
                } else {
                    emit_jit_op(&mut builder, cpu_ptr, *op, aligned_only)
                };
                cycles_value = builder.ins().iadd(cycles_value, op_cycles);
                // Exact for every op that can precede a bail (MoveMem and
                // register ops have constant cycles; only the closing
                // branch is dynamic, and nothing bails after it).
                cycles_before += op.op.max_cycles() as i64;
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

            // Success: all ops retired. Pack (ops_retired << 32) | cycles.
            let cycles64 = builder.ins().uextend(types::I64, cycles_value);
            let ops_len = builder.ins().iconst(types::I64, (ops.len() as i64) << 32);
            let packed = builder.ins().bor(cycles64, ops_len);
            builder.ins().return_(&[packed]);

            // Bail exits: set PC to the un-executed op, return the ops and
            // (constant) cycles retired before it.
            for bail in bails {
                builder.switch_to_block(bail.block);
                store_u32(&mut builder, cpu_ptr, offset_of!(CpuCore, pc), bail.pc);
                let packed = builder.ins().iconst(
                    types::I64,
                    ((bail.at.ops_before as i64) << 32) | bail.at.cycles_before,
                );
                builder.ins().return_(&[packed]);
            }

            builder.seal_all_blocks();
            builder.finalize();
        }

        module.define_function(func_id, &mut ctx).ok()?;
        module.clear_context(&mut ctx);
        module.finalize_definitions().ok()?;
        let ptr = module.get_finalized_function(func_id);
        let func = unsafe { transmute::<*const u8, TraceFn>(ptr) };

        Some(CompiledTrace {
            pc: start_pc,
            cpu_type,
            ops: ops.to_vec(),
            code,
            max_cycles,
            self_loop,
            needs_window,
            code_start,
            code_end,
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
            max_cycles: params.max_cycles,
            self_loop: params.self_loop,
            needs_window: params.needs_window,
            code_start: params.code_start,
            code_end: params.code_end,
        })
    }
}

/// Everything `compile_ops` needs, gathered by `compile_trace`.
struct CompileParams<'a> {
    start_pc: u32,
    cpu_type: CpuType,
    ops: &'a [TraceBuildOp],
    code: Vec<u8>,
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
) -> Option<(CachedRunResult, u32)> {
    if cpu.run_mode == RUN_MODE_BERR_AERR_RESET {
        return None;
    }

    TRACE_JIT.with_borrow_mut(|jit| jit.try_execute(cpu, bus, cpu_type, instr_budget, single_iter))
}

pub(crate) fn record_trace_target(pc: u32, cpu_type: CpuType) {
    TRACE_JIT.with_borrow_mut(|jit| jit.record_trace_target(pc, cpu_type));
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
            Self::Branch { .. } => 10,
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
                };
                let dst_c = if dst.is_mem() {
                    if long { 8 } else { 4 }
                } else {
                    0
                };
                4 + src_c + dst_c
            }
        }
    }

    fn ends_trace(self) -> bool {
        matches!(self, Self::Branch { .. } | Self::Dbcc { .. })
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

    fn length(self) -> u8 {
        match self {
            Self::Branch { length, .. } => length,
            Self::Dbcc { .. } => 4,
            _ => 2,
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
    if let Some(op) = decode_move_mem_trace_op(pc, opcode) {
        return Some(op);
    }

    let decoded = DecodedSimpleOp::decode(cpu_type, opcode)?;
    let op = decoded.to_jit_trace_op()?;
    Some(TraceBuildOp {
        opcode,
        extension: None,
        pc,
        op,
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
        pc,
        op: JitTraceOp::Branch {
            condition,
            displacement: extension as i16 as i32,
            length: 4,
        },
    })
}

/// MOVE/MOVEA (groups 1-3) where both EAs are register or register-indirect
/// forms — no extension words. At least one side must be memory
/// (register-to-register MOVEs are simple ops already).
fn decode_move_mem_trace_op(pc: u32, opcode: u16) -> Option<TraceBuildOp> {
    let size = match opcode >> 12 {
        1 => Size::Byte,
        2 => Size::Long,
        3 => Size::Word,
        _ => return None,
    };
    let src = decode_jit_ea((opcode >> 3) & 7, opcode & 7)?;
    let dst = decode_jit_ea((opcode >> 6) & 7, (opcode >> 9) & 7)?;
    if !src.is_mem() && !dst.is_mem() {
        return None;
    }
    // MOVEA.B does not exist, and An is not a legal byte source.
    if size == Size::Byte && (matches!(src, JitEa::Addr(_)) || matches!(dst, JitEa::Addr(_))) {
        return None;
    }
    Some(TraceBuildOp {
        opcode,
        extension: None,
        pc,
        op: JitTraceOp::MoveMem { size, src, dst },
    })
}

fn decode_jit_ea(mode: u16, reg: u16) -> Option<JitEa> {
    Some(match mode & 7 {
        0 => JitEa::Data(reg as u8),
        1 => JitEa::Addr(reg as u8),
        2 => JitEa::Ind(reg as u8),
        3 => JitEa::PostInc(reg as u8),
        4 => JitEa::PreDec(reg as u8),
        _ => return None,
    })
}

/// Interpreted trace execution (wasm and unit tests). Same contract as a
/// compiled [`TraceFn`]: returns `(ops_retired << 32) | cycles`, and a
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
            Some(c) => cycles += c,
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
    Some(execute_portable_reg_op(cpu, op))
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
        JitEa::Ind(r) | JitEa::PostInc(r) | JitEa::PreDec(r) => {
            let base = dst_base(cpu, r);
            let (addr, new_reg) = match dst {
                JitEa::Ind(_) => (base, None),
                JitEa::PostInc(_) => (base, Some(base.wrapping_add(jit_ea_step(size, r)))),
                JitEa::PreDec(_) => {
                    let a = base.wrapping_sub(jit_ea_step(size, r));
                    (a, Some(a))
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
        JitTraceOp::MoveMem { .. } => {
            unreachable!("MoveMem is handled by execute_portable_move_mem")
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
        JitTraceOp::ShiftReg { .. } => unreachable!("ShiftReg traces are wasm-only"),
        JitTraceOp::MoveMem { .. } => unreachable!("MoveMem is emitted by emit_move_mem"),
        JitTraceOp::Branch {
            condition,
            displacement,
            length,
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
    ops_before: u32,
    cycles_before: i64,
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
        JitEa::Ind(r) | JitEa::PostInc(r) | JitEa::PreDec(r) => {
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
    let value = mask_value(builder, value, size);
    let msb = iconst_u32(builder, size_msb(size));
    let sign_bits = builder.ins().band(value, msb);
    let n = flag_from_nonzero(builder, sign_bits, NFLAG_SET);
    store_value_u32(builder, cpu, offset_of!(CpuCore, n_flag), n);
    store_value_u32(builder, cpu, offset_of!(CpuCore, not_z_flag), value);
    store_u32(builder, cpu, offset_of!(CpuCore, v_flag), 0);
    store_u32(builder, cpu, offset_of!(CpuCore, c_flag), 0);
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

    #[test]
    fn portable_trace_executes_unconditional_loop_iteration() {
        let mut cpu = cpu();
        let ops = [
            TraceBuildOp {
                opcode: 0x5280,
                extension: None,
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
                pc: 0x0102,
                op: JitTraceOp::Branch {
                    condition: 0,
                    displacement: -4,
                    length: 2,
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

    #[test]
    fn portable_trace_uses_flags_for_conditional_branch() {
        let mut cpu = cpu();
        cpu.set_d(0, 1);
        let ops = [
            TraceBuildOp {
                opcode: 0x5340,
                extension: None,
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
                pc: 0x0102,
                op: JitTraceOp::Branch {
                    condition: 6,
                    displacement: -4,
                    length: 2,
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
    fn portable_trace_executes_register_shift() {
        let mut cpu = cpu();
        cpu.set_d(0, 0x8000_0001);
        let ops = [TraceBuildOp {
            opcode: 0xE188,
            extension: None,
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
}

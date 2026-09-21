//! Experimental bounded region combining existing checked-wrapper/body IR.
//! Region-only egraph optimization follows inlining; original eager-state
//! stores and callback semantics remain the input contract.
//! Snapshots and compilation are separate so an eventual profitable upgrade can
//! be compiled off-thread. This proof currently compiles at trace installation.

use super::*;
use cranelift_codegen::inline::{Inline, InlineCommand};
use cranelift_codegen::ir::{ExternalName, Inst, Opcode};
use cranelift_module::FuncId;
use std::borrow::Cow;
use std::sync::Arc;

#[path = "instruction_accounting.rs"]
mod instruction_accounting;
#[cfg(test)]
#[path = "native_region_math_tests.rs"]
mod math_tests;

/// `Public` is used only through instruction-budgeted run_batch, whose
/// documented contract clobbers cycles_remaining. Precise cycle APIs never
/// call this trace driver. The test-only Reference mode preserves the previous
/// private driver as an independent boundary oracle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RegionMode {
    Disabled,
    #[cfg(test)]
    Reference,
    Public,
}

impl RegionMode {
    // Read once when the thread-local JIT is created. Unknown values keep the
    // released execution path; promotion remains explicitly opt-in.
    pub(super) fn from_value(value: Option<&str>) -> Self {
        match value {
            Some("public" | "on" | "1") => Self::Public,
            _ => Self::Disabled,
        }
    }
}

#[cfg(test)]
mod configuration_tests {
    use super::{RegionMode, TraceJit};

    #[test]
    fn native_regions_require_explicit_runtime_opt_in() {
        for value in [
            None,
            Some(""),
            Some("off"),
            Some("0"),
            Some("unknown"),
            Some("reference"),
        ] {
            let mode = RegionMode::from_value(value);
            assert_eq!(mode, RegionMode::Disabled, "{value:?}");
            let jit = TraceJit::new_with_region_mode(mode);
            assert!(!jit.native_region_enabled, "{value:?}");
            assert!(!jit.native_region_public, "{value:?}");
            assert!(jit.native_region.is_none());
        }
        for value in ["public", "on", "1"] {
            let mode = RegionMode::from_value(Some(value));
            assert_eq!(mode, RegionMode::Public, "{value}");
            let jit = TraceJit::new_with_region_mode(mode);
            assert!(jit.native_region_enabled, "{value}");
            assert!(jit.native_region_public, "{value}");
        }
    }
}

type RegionFn = unsafe extern "C" fn(*mut CpuCore, u32, u32, *mut RegionExit);

#[derive(Clone, Debug, PartialEq, Eq)]
struct Head {
    pc: u32,
    cpu_type: CpuType,
    ops: u32,
    max_cycles: u32,
    native_loop: bool,
    needs_window: bool,
    tracked_writes: bool,
    checked: usize,
    body: usize,
}

/// Already-legalized code from the original checked wrapper and body. Keep
/// their module identities with the IR so inlining never guesses a callee.
pub(super) struct InlineBody {
    pub(super) body_id: FuncId,
    pub(super) body: Function,
    pub(super) checked_id: FuncId,
    pub(super) checked: Function,
}

#[derive(Clone)]
struct RegionInput {
    heads: Vec<Head>,
    bodies: Vec<Arc<InlineBody>>,
    public: bool,
}

impl PartialEq for RegionInput {
    fn eq(&self, other: &Self) -> bool {
        // A module never reuses finalized entry addresses while this JIT
        // lives. These identities also identify the captured immutable IR.
        self.public == other.public && self.heads == other.heads
    }
}

pub(super) struct NativeRegion {
    root_pc: u32,
    root_ops: u32,
    cpu_type: CpuType,
    input: RegionInput,
    entry: RegionFn,
    reported_first_execution: bool,
    inlined_calls: usize,
}

impl NativeRegion {
    #[inline(always)]
    pub(super) fn matches_root(&self, pc: u32, cpu_type: CpuType, budget: u32) -> bool {
        // Small budgets must retain the original validator-before-refusal
        // path. Declining here reaches that path without region preflight.
        self.root_pc == pc && self.cpu_type == cpu_type && budget >= self.root_ops
    }
}

const RESUME_HEAD: u32 = 1;
const FINISH_HEAD: u32 = 2;

/// A validation-needed child cannot use the old zero-retirement sentinel:
/// earlier heads have already run. Carry its exact pending continuation.
#[repr(C)]
#[derive(Default)]
struct RegionExit {
    kind: u32,
    index: u32,
    instr_budget: u32,
    chain_budget: u32,
    retired: u32,
    latest_retired: u32,
    packed: u64,
}

/// Raw test entry keeps callback-time CPU observation outside a live Rust
/// `&mut CpuCore`, just like the existing raw native-body oracle.
#[cfg(all(test, target_arch = "x86_64", not(feature = "trace-profile")))]
pub(super) struct NativeRegionTestExit {
    pub(super) kind: u32,
    pub(super) index: u32,
    pub(super) instr_budget: u32,
    pub(super) chain_budget: u32,
    pub(super) retired: u32,
    pub(super) latest_retired: u32,
    pub(super) packed: u64,
}

fn pointer(function: NativeTraceFn) -> usize {
    match function {
        NativeTraceFn::Once(function) => function as usize,
        NativeTraceFn::Loop(function) => function as usize,
    }
}

fn snapshot(trace: &CompiledTrace) -> Result<Head, &'static str> {
    if trace.adaptive_branch {
        return Err("adaptive trace");
    }
    if trace.seeded_exit {
        return Err("call/return seeded exit");
    }
    if trace.self_loop && !trace.native_loop {
        return Err("one-pass self loop");
    }
    if trace.ops.is_empty()
        || trace.max_cycles <= 0
        || trace.max_cycles as u32 > TRACE_RETURN_CYCLES_MASK
    {
        return Err("invalid fixed budget");
    }
    let checked = trace.checked_func.ok_or("no checked native entry")?;
    if matches!(checked, NativeTraceFn::Loop(_)) != trace.native_loop
        || matches!(trace.func, NativeTraceFn::Loop(_)) != trace.native_loop
    {
        return Err("native ABI mismatch");
    }
    Ok(Head {
        pc: trace.pc,
        cpu_type: trace.cpu_type,
        ops: trace.ops.len() as u32,
        max_cycles: trace.max_cycles as u32,
        native_loop: trace.native_loop,
        needs_window: trace.needs_window,
        tracked_writes: trace.tracked_writes,
        checked: pointer(checked),
        body: pointer(trace.func),
    })
}

fn matches_installed(head: &Head, trace: &CompiledTrace) -> bool {
    // Fixed metadata cannot change on an installed trace; finalized native
    // addresses are unique for this module lifetime. Adaptive policy is the
    // only mutable admission metadata and is checked explicitly.
    trace.pc == head.pc
        && trace.cpu_type == head.cpu_type
        && !trace.adaptive_branch
        && pointer(trace.func) == head.body
        && trace
            .checked_func
            .is_some_and(|entry| pointer(entry) == head.checked)
}

fn return_arm(trace: &CompiledTrace, root: u32) -> bool {
    if trace.ops.len() > 4 || trace.self_loop || trace.ops.is_empty() {
        return false;
    }
    let last = trace.ops.last().unwrap();
    if !matches!(last.op, JitTraceOp::Branch { condition: 0, .. })
        || last.op.taken_target(last.pc) != Some(root)
    {
        return false;
    }
    read_only_arm_ops(&trace.ops)
}

fn read_only_arm_ops(ops: &[TraceBuildOp]) -> bool {
    ops[..ops.len() - 1].iter().all(|op| {
        matches!(
            op.op,
            JitTraceOp::Moveq { .. }
                | JitTraceOp::MoveReg { .. }
                | JitTraceOp::AddrDataReg { .. }
                // Register arithmetic has no callbacks or memory faults; the
                // existing body still materializes all of its register/CCR effects.
                | JitTraceOp::AddqSubqReg { .. }
                | JitTraceOp::AddqSubqAddr { .. }
                | JitTraceOp::MoveMem { dst: JitEa::Data(_) | JitEa::Addr(_), .. }
        )
    })
}

pub(super) fn retain_inline_body(ops: &[TraceBuildOp], native_loop: bool) -> bool {
    (native_loop
        && ops.iter().any(|op| {
            matches!(
                op.op,
                JitTraceOp::IndirectJmp {
                    expected_target: Some(_),
                    ..
                }
            )
        }))
        || (!ops.is_empty()
            && ops.len() <= 4
            && matches!(
                ops.last().unwrap().op,
                JitTraceOp::Branch { condition: 0, .. }
            )
            && read_only_arm_ops(ops))
}

impl TraceJit {
    #[cfg(all(test, target_arch = "x86_64", not(feature = "trace-profile")))]
    pub(super) fn native_region_inlined_calls(&self) -> usize {
        self.native_region
            .as_ref()
            .map_or(0, |region| region.inlined_calls)
    }
    /// The caller must establish the same slot and watch proofs as preflight,
    /// and provide a valid CPU whose raw callback context remains valid.
    #[cfg(all(test, target_arch = "x86_64", not(feature = "trace-profile")))]
    pub(super) unsafe fn call_native_region_raw(
        &self,
        cpu: *mut CpuCore,
        instr_budget: u32,
        chain_budget: u8,
    ) -> NativeRegionTestExit {
        let region = self.native_region.as_ref().expect("compiled test region");
        let mut exit = RegionExit::default();
        unsafe { (region.entry)(cpu, instr_budget, u32::from(chain_budget), &mut exit) };
        NativeRegionTestExit {
            kind: exit.kind,
            index: exit.index,
            instr_budget: exit.instr_budget,
            chain_budget: exit.chain_budget,
            retired: exit.retired,
            latest_retired: exit.latest_retired,
            packed: exit.packed,
        }
    }

    /// Manual entry is also used by differential tests. Automatic discovery
    /// calls it only once all three already-compiled shapes are available.
    pub(super) fn compile_native_region(&mut self, pcs: &[u32]) -> Result<(), &'static str> {
        let result = self.prepare_native_region(pcs);
        self.native_region_status = match &result {
            Ok(()) => "compiled native region",
            Err(reason) => reason,
        };
        if std::env::var_os("M68K_NATIVE_REGION_DIAGNOSTICS").is_some() {
            eprintln!(
                "NATIVE_REGION pcs={pcs:x?} status={}",
                self.native_region_status
            );
        }
        result
    }

    fn prepare_native_region(&mut self, pcs: &[u32]) -> Result<(), &'static str> {
        if cfg!(feature = "trace-profile") {
            return Err("trace-profile retains original per-head instrumentation");
        }
        if !(2..=3).contains(&pcs.len()) {
            return Err("requires two or three heads");
        }
        let mut heads: Vec<Head> = Vec::with_capacity(pcs.len());
        let mut bodies = Vec::with_capacity(pcs.len());
        for (index, &pc) in pcs.iter().enumerate() {
            let TraceSlot::Compiled(trace) = &self.slots[trace_cache_index(pc)] else {
                return Err("head not compiled");
            };
            if trace.pc != pc || pcs[..index].contains(&pc) {
                return Err("head identity/duplicate");
            }
            if index == 0 {
                if !trace.native_loop
                    || !trace.ops.iter().any(|op| {
                        matches!(
                            op.op,
                            JitTraceOp::IndirectJmp {
                                expected_target: Some(_),
                                ..
                            }
                        )
                    })
                {
                    return Err("root is not a counted indirect dispatcher");
                }
            } else if !return_arm(trace, pcs[0]) {
                return Err("not a short read-only return arm");
            }
            let head = snapshot(trace)?;
            if let Some(first) = heads.first()
                && (head.cpu_type != first.cpu_type || head.tracked_writes != first.tracked_writes)
            {
                return Err("mixed CPU/write mode");
            }
            heads.push(head);
            let body = trace.region_ir.as_ref().ok_or("no retained inline IR")?;
            bodies.push(if self.native_region_public {
                Arc::new(InlineBody {
                    body_id: body.body_id,
                    body: instruction_accounting::without_synthetic_cycles(&body.body)
                        .ok_or("unrecognized cycle return packing")?,
                    checked_id: body.checked_id,
                    checked: body.checked.clone(),
                })
            } else {
                body.clone()
            });
        }
        let input = RegionInput {
            heads,
            bodies,
            public: self.native_region_public,
        };
        if self
            .native_region
            .as_ref()
            .is_some_and(|region| region.input == input)
        {
            return Ok(());
        }
        let ordinal = self.next_func;
        self.next_func = self.next_func.wrapping_add(1);
        let module = self.module.as_mut().ok_or("native JIT unavailable")?;
        let (entry, inlined_calls) = compile_snapshot(module, &mut self.func_ctx, ordinal, &input)
            .ok_or("region code generation failed")?;
        self.native_region = Some(NativeRegion {
            root_pc: input.heads[0].pc,
            root_ops: input.heads[0].ops,
            cpu_type: input.heads[0].cpu_type,
            input,
            entry,
            reported_first_execution: false,
            inlined_calls,
        });
        Ok(())
    }

    pub(super) fn maybe_compile_native_region(&mut self, installed_pc: u32) {
        if !self.native_region_enabled || cfg!(feature = "trace-profile") {
            return;
        }
        let TraceSlot::Compiled(installed) = &self.slots[trace_cache_index(installed_pc)] else {
            return;
        };
        if installed.pc != installed_pc {
            return;
        }
        let root = if installed.native_loop
            && installed.ops.iter().any(|op| {
                matches!(
                    op.op,
                    JitTraceOp::IndirectJmp {
                        expected_target: Some(_),
                        ..
                    }
                )
            }) {
            installed_pc
        } else if installed.ops.len() <= 4 {
            let Some(last) = installed.ops.last() else {
                return;
            };
            let Some(target) = last.op.taken_target(last.pc) else {
                return;
            };
            target
        } else {
            return;
        };
        // An unrelated/new short arm does not require rescanning the entire
        // cache when the existing selected group is still valid. Preserve
        // that group until an actual constituent changes.
        if self.native_region.as_ref().is_some_and(|region| {
            region.root_pc == root && region.input.public == self.native_region_public
                && region.input.heads.iter().all(|head| {
                    matches!(&self.slots[trace_cache_index(head.pc)], TraceSlot::Compiled(trace) if matches_installed(head, trace))
                })
        }) { return; }
        let TraceSlot::Compiled(trace) = &self.slots[trace_cache_index(root)] else {
            return;
        };
        if trace.pc != root
            || !trace.native_loop
            || !trace.ops.iter().any(|op| {
                matches!(
                    op.op,
                    JitTraceOp::IndirectJmp {
                        expected_target: Some(_),
                        ..
                    }
                )
            })
        {
            return;
        }
        if snapshot(trace).is_err() {
            self.native_region_status = "root not settled/eligible";
            return;
        }
        let mut pcs = vec![root];
        for slot in &self.slots {
            if let TraceSlot::Compiled(arm) = slot
                && arm.cpu_type == trace.cpu_type
                && return_arm(arm, root)
                && snapshot(arm).is_ok()
            {
                pcs.push(arm.pc);
                if pcs.len() == 3 {
                    break;
                }
            }
        }
        if pcs.len() == 3 {
            let _ = self.compile_native_region(&pcs);
        }
    }

    /// Outer None means ordinary execution should handle this entry. Inner
    /// None means the dispatcher ran but retired no instructions, like the
    /// existing trace result. No guest state is lazily held across callbacks.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_native_region<B: AddressBus>(
        &mut self,
        cpu: &mut CpuCore,
        bus: &mut B,
        cpu_type: CpuType,
        instr_budget: u32,
        single_iter: bool,
        watch_pcs: &[u32],
        chain_budget: u8,
    ) -> Option<Option<(CachedRunResult, u32)>> {
        if !self.native_region_enabled || single_iter {
            return None;
        }
        let region = self.native_region.as_ref()?;
        if region.input.public != self.native_region_public {
            return None;
        }
        // Only the root enters this specialized function; generated arm
        // transitions are its internal old-head boundaries.
        if region.input.heads[0].pc != cpu.pc || region.input.heads[0].cpu_type != cpu_type {
            return None;
        }
        for head in &region.input.heads {
            let TraceSlot::Compiled(trace) = &self.slots[trace_cache_index(head.pc)] else {
                self.native_region_status = "constituent evicted";
                self.native_region = None;
                return None;
            };
            if !matches_installed(head, trace) {
                self.native_region_status = "constituent replaced or changed mode";
                self.native_region = None;
                return None;
            }
            // Conservative group refusal preserves all watched old-head and
            // interior boundaries, including address aliases and single-step.
            if watch_pcs.iter().any(|&watch| {
                cpu.address(watch) == cpu.address(head.pc)
                    || trace.contains_interior_watch(cpu, watch)
            }) {
                self.native_region_status = "watched region head/interior";
                return None;
            }
        }
        let entry = region.entry;
        let pcs: [u32; 3] =
            std::array::from_fn(|index| region.input.heads.get(index).map_or(0, |head| head.pc));
        let report_execution = !region.reported_first_execution;
        let inlined_calls = region.inlined_calls;
        let mut exit = RegionExit::default();
        #[cfg(test)]
        {
            self.native_region_entries += 1;
            if region.input.public {
                self.native_region_public_entries += 1;
            }
        }
        unsafe { entry(cpu, instr_budget, u32::from(chain_budget), &mut exit) };
        #[cfg(test)]
        {
            // A RESUME exit has not run its pending head. A FINISH exit
            // counts that head only when it actually retired instructions.
            if !self.native_region_public {
                self.native_region_heads_run +=
                    u64::from(u32::from(chain_budget) - exit.chain_budget)
                        + u64::from(exit.kind == FINISH_HEAD && exit.latest_retired != 0);
            }
        }
        if report_execution {
            self.native_region
                .as_mut()
                .unwrap()
                .reported_first_execution = true;
            if std::env::var_os("M68K_NATIVE_REGION_DIAGNOSTICS").is_some() {
                eprintln!(
                    "NATIVE_REGION_ENTER pcs={pcs:x?} inlined_calls={inlined_calls} exit_kind={} depth_before={chain_budget} depth_after={} retired={}",
                    exit.kind, exit.chain_budget, exit.retired
                );
            }
        }
        // All already completed logical entries have charged their cycles.
        // Disable this upgrade only while servicing its precise cold exit.
        let enabled = self.native_region_enabled;
        self.native_region_enabled = false;
        // A bus-aware slow validator can unwind. Restore the upgrade switch
        // even if the embedder catches that panic and reuses this JIT.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if exit.kind == RESUME_HEAD {
                debug_assert_eq!(cpu.pc, pcs[exit.index as usize]);
                let continuation = self.try_execute(
                    cpu,
                    bus,
                    cpu_type,
                    exit.instr_budget,
                    false,
                    watch_pcs,
                    exit.chain_budget as u8,
                );
                add_prior(continuation, exit.retired)
            } else {
                debug_assert_eq!(exit.kind, FINISH_HEAD);
                let original_pc = pcs[exit.index as usize];
                self.finish_region_head(cpu, bus, cpu_type, watch_pcs, original_pc, &exit)
            }
        }));
        self.native_region_enabled = enabled;
        match result {
            Ok(result) => Some(result),
            Err(payload) => std::panic::resume_unwind(payload),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_region_head<B: AddressBus>(
        &mut self,
        cpu: &mut CpuCore,
        bus: &mut B,
        cpu_type: CpuType,
        watch_pcs: &[u32],
        original_pc: u32,
        exit: &RegionExit,
    ) -> Option<(CachedRunResult, u32)> {
        if exit.latest_retired == 0 {
            return add_prior(None, exit.retired);
        }
        let guarded = trace_return_guarded_branch_exit(exit.packed);
        let complete = trace_return_complete(exit.packed);
        let clean_link = !guarded && complete && self.compiled_head_at(cpu.pc, cpu_type);
        // The admitted snapshot excludes seeded calls/returns and adaptive
        // traces, so this is exactly the remaining ordinary exit policy.
        if (guarded || clean_link) && exit.chain_budget > 0 && cpu.pc != original_pc {
            let watched = watch_pcs
                .iter()
                .any(|&watch| cpu.address(watch) == cpu.address(cpu.pc));
            let seed = if clean_link || self.compiled_head_at(cpu.pc, cpu_type) {
                if watched {
                    ExitSeed::None
                } else {
                    ExitSeed::Chain
                }
            } else {
                self.note_trace_exit(cpu.pc, cpu_type, watched)
            };
            match seed {
                ExitSeed::Chain => {
                    let child = self.try_execute(
                        cpu,
                        bus,
                        cpu_type,
                        exit.instr_budget.saturating_sub(exit.latest_retired),
                        false,
                        watch_pcs,
                        (exit.chain_budget - 1) as u8,
                    );
                    return add_prior(child, exit.retired);
                }
                ExitSeed::StartRecording => cpu.trace_recording = true,
                ExitSeed::None => {}
            }
        }
        Some((CachedRunResult::Ran, exit.retired))
    }
}

fn add_prior(result: Option<(CachedRunResult, u32)>, prior: u32) -> Option<(CachedRunResult, u32)> {
    match result {
        Some((result, retired)) => Some((result, prior + retired)),
        None if prior != 0 => Some((CachedRunResult::Ran, prior)),
        None => None,
    }
}

/// Exact unsigned quotient for an immutable positive divisor. Ceil reciprocal
/// gives floor(x/d) or one above it; the product comparison corrects that one.
/// This does not change the instruction/cycle budget or its limiting formula.
fn constant_quotient(builder: &mut FunctionBuilder<'_>, value: Value, divisor: u32) -> Value {
    assert!(divisor != 0);
    if divisor == 1 {
        return value;
    }
    if divisor.is_power_of_two() {
        return builder
            .ins()
            .ushr_imm(value, i64::from(divisor.trailing_zeros()));
    }
    let wide = builder.ins().uextend(types::I64, value);
    let magic = (1u64 << 32).div_ceil(u64::from(divisor));
    let product = builder.ins().imul_imm(wide, magic as i64);
    let quotient = builder.ins().ushr_imm(product, 32);
    let check = builder.ins().imul_imm(quotient, i64::from(divisor));
    let excess = builder.ins().icmp(IntCC::UnsignedGreaterThan, check, wide);
    let previous = builder.ins().iadd_imm(quotient, -1);
    let exact = builder.ins().select(excess, previous, quotient);
    builder.ins().ireduce(types::I32, exact)
}

#[allow(clippy::too_many_arguments)]
fn emit_exit(
    builder: &mut FunctionBuilder<'_>,
    out: Value,
    kind: u32,
    index: usize,
    budget: Value,
    depth: Value,
    retired: Value,
    latest: Value,
    packed: Value,
) {
    store_u32(builder, out, offset_of!(RegionExit, kind), kind);
    store_u32(builder, out, offset_of!(RegionExit, index), index as u32);
    for (offset, value) in [
        (offset_of!(RegionExit, instr_budget), budget),
        (offset_of!(RegionExit, chain_budget), depth),
        (offset_of!(RegionExit, retired), retired),
        (offset_of!(RegionExit, latest_retired), latest),
        (offset_of!(RegionExit, packed), packed),
    ] {
        builder
            .ins()
            .store(MemFlags::trusted(), value, out, offset as i32);
    }
    builder.ins().return_(&[]);
}

struct RegionInliner<'a> {
    bodies: &'a [Arc<InlineBody>],
    calls: usize,
}

impl RegionInliner<'_> {
    fn callee<'a>(&'a self, caller: &Function, callee: FuncRef) -> Option<&'a Function> {
        let ExternalName::User(name) = caller.dfg.ext_funcs[callee].name else {
            return None;
        };
        let name = &caller.params.user_named_funcs()[name];
        if name.namespace != 0 {
            return None;
        }
        self.bodies.iter().find_map(|body| {
            if name.index == body.checked_id.as_u32() {
                Some(&body.checked)
            } else if name.index == body.body_id.as_u32() {
                Some(&body.body)
            } else {
                None
            }
        })
    }
}

impl Inline for RegionInliner<'_> {
    fn inline(
        &mut self,
        caller: &Function,
        _: Inst,
        _: Opcode,
        callee: FuncRef,
        _: &[Value],
    ) -> InlineCommand<'_> {
        // Increment before taking the returned borrow, then undo for calls
        // outside this exact snapshot. Bus hooks and runtime helpers remain
        // ordinary calls with their original memory/state visibility.
        self.calls += 1;
        if self.callee(caller, callee).is_none() {
            self.calls -= 1;
            return InlineCommand::KeepCall;
        }
        InlineCommand::Inline {
            callee: Cow::Borrowed(self.callee(caller, callee).unwrap()),
            visit_callee: true,
        }
    }
}

fn compile_snapshot(
    module: &mut JITModule,
    frontend: &mut FunctionBuilderContext,
    ordinal: u32,
    input: &RegionInput,
) -> Option<(RegionFn, usize)> {
    let ptr = module.target_config().pointer_type();
    let mut signature = module.make_signature();
    signature.params.extend([
        AbiParam::new(ptr),
        AbiParam::new(types::I32),
        AbiParam::new(types::I32),
        AbiParam::new(ptr),
    ]);
    let function = module
        .declare_function(
            &format!("m68k_native_region_{ordinal}"),
            Linkage::Local,
            &signature,
        )
        .ok()?;
    let mut context = Context::new();
    context.func =
        Function::with_name_signature(UserFuncName::user(0, function.as_u32()), signature);
    {
        let mut builder = FunctionBuilder::new(&mut context.func, frontend);
        let entry = builder.create_block();
        builder.switch_to_block(entry);
        builder.append_block_params_for_function_params(entry);
        let parameters = builder.block_params(entry).to_vec();
        let (cpu, budget, depth, out) =
            (parameters[0], parameters[1], parameters[2], parameters[3]);
        let zero = builder.ins().iconst(types::I32, 0);
        let zero64 = builder.ins().iconst(types::I64, 0);
        let watch_address_mask = load_u32(&mut builder, cpu, offset_of!(CpuCore, address_mask));
        let heads: Vec<Block> = input
            .heads
            .iter()
            .map(|_| {
                let block = builder.create_block();
                for _ in 0..3 {
                    builder.append_block_param(block, types::I32);
                }
                block
            })
            .collect();
        builder
            .ins()
            .jump(heads[0], &[budget.into(), depth.into(), zero.into()]);
        for (index, head) in input.heads.iter().enumerate() {
            builder.switch_to_block(heads[index]);
            let parameters = builder.block_params(heads[index]).to_vec();
            let (budget, depth, retired) = (parameters[0], parameters[1], parameters[2]);
            let resume = builder.create_block();
            let invoke = builder.create_block();
            let accounting = builder.create_block();
            builder.append_block_param(accounting, types::I64);
            let finish = builder.create_block();
            for ty in [types::I32, types::I32, types::I64] {
                builder.append_block_param(finish, ty);
            }

            let cycles = if input.public {
                None
            } else {
                Some(load_u32(
                    &mut builder,
                    cpu,
                    offset_of!(CpuCore, cycles_remaining),
                ))
            };
            let budget_low =
                builder
                    .ins()
                    .icmp_imm(IntCC::UnsignedLessThan, budget, i64::from(head.ops));
            let mut decline = budget_low;
            if let Some(cycles) = cycles {
                let cycles_low = builder.ins().icmp_imm(
                    IntCC::SignedLessThan,
                    cycles,
                    i64::from(head.max_cycles),
                );
                decline = builder.ins().bor(decline, cycles_low);
            }
            let has_pmmu = load_u8(&mut builder, cpu, offset_of!(CpuCore, has_pmmu));
            let enabled = load_u8(&mut builder, cpu, offset_of!(CpuCore, pmmu_enabled));
            let mmu = builder.ins().band(has_pmmu, enabled);
            let mmu = builder.ins().icmp_imm(IntCC::NotEqual, mmu, 0);
            decline = builder.ins().bor(decline, mmu);
            let recording = load_u8(&mut builder, cpu, offset_of!(CpuCore, trace_recording));
            let recording = builder.ins().icmp_imm(IntCC::NotEqual, recording, 0);
            decline = builder.ins().bor(decline, recording);
            if head.needs_window {
                let len = load_u32(&mut builder, cpu, offset_of!(CpuCore, fm_len));
                let missing = builder.ins().icmp_imm(IntCC::Equal, len, 0);
                decline = builder.ins().bor(decline, missing);
                let hook = builder.ins().load(
                    ptr,
                    MemFlags::trusted(),
                    cpu,
                    offset_of!(CpuCore, fm_write_hook) as i32,
                );
                let tracked = builder.ins().icmp_imm(IntCC::NotEqual, hook, 0);
                let bad_mode = builder.ins().icmp_imm(
                    IntCC::NotEqual,
                    tracked,
                    i64::from(head.tracked_writes),
                );
                decline = builder.ins().bor(decline, bad_mode);
            }
            builder.ins().brif(decline, resume, &[], invoke, &[]);
            builder.switch_to_block(invoke);
            let mut arguments = vec![cpu];
            if head.native_loop {
                let by_instructions = constant_quotient(&mut builder, budget, head.ops);
                let iterations = if let Some(cycles) = cycles {
                    let cap = builder
                        .ins()
                        .iconst(types::I32, i64::from(TRACE_RETURN_CYCLES_MASK));
                    let larger = builder.ins().icmp(IntCC::UnsignedGreaterThan, cycles, cap);
                    let payload = builder.ins().select(larger, cap, cycles);
                    let by_cycles = constant_quotient(&mut builder, payload, head.max_cycles);
                    let smaller =
                        builder
                            .ins()
                            .icmp(IntCC::UnsignedLessThan, by_instructions, by_cycles);
                    builder.ins().select(smaller, by_instructions, by_cycles)
                } else {
                    by_instructions
                };
                arguments.push(iterations);
            }
            let callee = module.declare_func_in_func(input.bodies[index].checked_id, builder.func);
            let call = builder.ins().call(callee, &arguments);
            let packed = builder.inst_results(call)[0];
            let invalid =
                builder
                    .ins()
                    .icmp_imm(IntCC::Equal, packed, TRACE_RETURN_VALIDATION_NEEDED as i64);
            builder
                .ins()
                .brif(invalid, resume, &[], accounting, &[packed.into()]);

            builder.switch_to_block(accounting);
            let packed = builder.block_params(accounting)[0];
            let low = builder.ins().ireduce(types::I32, packed);
            if !input.public {
                let used_cycles = builder
                    .ins()
                    .band_imm(low, i64::from(TRACE_RETURN_CYCLES_MASK));
                let current_cycles =
                    load_u32(&mut builder, cpu, offset_of!(CpuCore, cycles_remaining));
                let remaining_cycles = builder.ins().isub(current_cycles, used_cycles);
                builder.ins().store(
                    MemFlags::trusted(),
                    remaining_cycles,
                    cpu,
                    offset_of!(CpuCore, cycles_remaining) as i32,
                );
            }
            let high = builder.ins().ushr_imm(packed, 32);
            let latest = builder.ins().ireduce(types::I32, high);
            let total = builder.ins().iadd(retired, latest);
            let finish_args: [BlockArg; 3] = [total.into(), latest.into(), packed.into()];
            let mut stop = builder.ins().icmp_imm(IntCC::Equal, latest, 0);
            let pc = load_u32(&mut builder, cpu, offset_of!(CpuCore, pc));
            if !input.public {
                let exhausted = builder.ins().icmp_imm(IntCC::Equal, depth, 0);
                stop = builder.ins().bor(stop, exhausted);
                let same = builder.ins().icmp_imm(IntCC::Equal, pc, i64::from(head.pc));
                stop = builder.ins().bor(stop, same);
            }
            // Watch proofs were made for this exact mask. If a callback or
            // aliased write changes it, finish the parent through the host:
            // its watched-target policy must run before entering a child.
            // Merely resuming the child would skip that parent decision.
            let address_mask = load_u32(&mut builder, cpu, offset_of!(CpuCore, address_mask));
            let mask_changed =
                builder
                    .ins()
                    .icmp(IntCC::NotEqual, address_mask, watch_address_mask);
            stop = builder.ins().bor(stop, mask_changed);
            // COMPLETE or GUARDED permits chaining; ordinary partial memory
            // bails do not seed or enter a continuation.
            let metadata = builder.ins().band_imm(
                low,
                (TRACE_RETURN_COMPLETE | TRACE_RETURN_GUARDED_BRANCH_EXIT) as i64,
            );
            let memory_bail = builder.ins().icmp_imm(IntCC::Equal, metadata, 0);
            stop = builder.ins().bor(stop, memory_bail);
            let dispatch = builder.create_block();
            builder
                .ins()
                .brif(stop, finish, &finish_args, dispatch, &[]);
            builder.switch_to_block(dispatch);
            let next_budget = builder.ins().isub(budget, latest);
            // run_batch's public bound is remaining instructions, not the
            // old private Rust recursion cap. Every nonzero body retirement
            // reduces that exact budget before another known head can run.
            let next_depth = if input.public {
                depth
            } else {
                builder.ins().iadd_imm(depth, -1)
            };
            for (target_index, target) in input.heads.iter().enumerate() {
                let next = builder.create_block();
                let matches = builder
                    .ins()
                    .icmp_imm(IntCC::Equal, pc, i64::from(target.pc));
                builder.ins().brif(
                    matches,
                    heads[target_index],
                    &[next_budget.into(), next_depth.into(), total.into()],
                    next,
                    &[],
                );
                builder.switch_to_block(next);
            }
            builder.ins().jump(finish, &finish_args);

            builder.switch_to_block(resume);
            emit_exit(
                &mut builder,
                out,
                RESUME_HEAD,
                index,
                budget,
                depth,
                retired,
                zero,
                zero64,
            );
            builder.switch_to_block(finish);
            let values = builder.block_params(finish).to_vec();
            emit_exit(
                &mut builder,
                out,
                FINISH_HEAD,
                index,
                budget,
                depth,
                values[0],
                values[1],
                values[2],
            );
        }
        builder.seal_all_blocks();
        builder.finalize();
    }
    // Cranelift rewrites each callee return to the exact continuation of this
    // call site. Therefore every existing guard/fault/completion return flows
    // through the same packed accounting without new handwritten emitters.
    let mut inliner = RegionInliner {
        bodies: &input.bodies,
        calls: 0,
    };
    if !context.inline(&mut inliner).ok()? || inliner.calls != 2 * input.heads.len() {
        return None;
    }
    // This is an actual combined function, not a dispatcher that merely
    // replaced indirect calls with direct calls. Refuse an incomplete splice.
    for block in context.func.layout.blocks() {
        for inst in context.func.layout.block_insts(block) {
            if let cranelift_codegen::ir::InstructionData::Call { func_ref, .. } =
                context.func.dfg.insts[inst]
                && inliner.callee(&context.func, func_ref).is_some()
            {
                return None;
            }
        }
    }
    let inlined_calls = inliner.calls;
    // The ordinary first tier deliberately uses Cranelift's default
    // opt_level=None. Only this combined upgrade pays for DAG optimization
    // and alias-aware load forwarding. optimize() establishes legalization,
    // CFG and dominators; the explicit egraph pass then exploits the inlined
    // boundaries even though the module's global setting remains unchanged.
    // Guest RAM aliases and callback calls retain their original MemFlags
    // and side effects, so they remain barriers whenever forwarding is unsafe.
    let mut control_plane = Default::default();
    context.optimize(module.isa(), &mut control_plane).ok()?;
    context.egraph_pass(module.isa(), &mut control_plane).ok()?;
    module.define_function(function, &mut context).ok()?;
    module.clear_context(&mut context);
    module.finalize_definitions().ok()?;
    Some((
        unsafe { transmute::<*const u8, RegionFn>(module.get_finalized_function(function)) },
        inlined_calls,
    ))
}

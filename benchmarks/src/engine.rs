//! The common interface every benchmarked core adapts to.

use crate::workloads::Workload;

/// Architectural state sampled after a run, used to verify that all engines
/// executed the same instruction stream to the same effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineState {
    pub d: [u32; 8],
    pub mem_hash: u64,
}

pub trait Engine {
    fn name(&self) -> &'static str;

    /// Install a workload's memory image and put the CPU at its entry point.
    fn load_workload(&mut self, w: &Workload);

    /// Re-arm registers/PC for another run of the already-loaded workload.
    /// Guest memory is left as-is: every workload here is idempotent over
    /// memory (loads/stores rewrite the same values), so runs are identical.
    fn reset_run(&mut self, w: &Workload);

    /// Cycle-budgeted execution; returns cycles actually consumed (may
    /// overshoot the budget by up to one instruction).
    fn run_cycles(&mut self, budget: i64) -> i64;

    /// Instruction-budgeted execution; returns instructions retired.
    fn run_instructions(&mut self, budget: u64) -> u64;

    /// Whether this engine runs on a cycle budget (`run_cycles`) or an
    /// instruction budget (`run_instructions`).
    fn is_cycle_budgeted(&self) -> bool;

    fn state(&mut self) -> EngineState;
}

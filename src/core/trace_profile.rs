//! Opt-in trace and decoded-operation profiling.
//!
//! Enable the `trace-profile` Cargo feature and set `M68K_TRACE_PROFILE=1`
//! to print a report when the CPU thread exits. The normal build contains
//! none of this module or its hot-path hooks. Profiling works with both the
//! portable trace executor and the native `jit` backend.

use super::types::CpuType;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Debug, Clone, PartialEq, Eq)]
/// Per-trace-head profiling counters.
#[non_exhaustive]
pub struct TraceProfileRow {
    /// Guest program counter at which the trace starts.
    pub start_pc: u32,
    /// CPU model active for this trace.
    pub cpu_type: CpuType,
    /// Backward branches observed at the trace head.
    pub backward_hits: u64,
    /// Trace-entry attempts rejected by an unsupported operation.
    pub rejected_hits: u64,
    /// Number of trace-recording attempts.
    pub recording_attempts: u64,
    /// Longest supported prefix before the recorded blocker.
    pub prefix_ops: u32,
    /// Guest PC of the operation that blocked trace construction.
    pub blocker_pc: Option<u32>,
    /// Opcode word that blocked trace construction.
    pub blocker_opcode: Option<u16>,
    /// Most recent reason a recording at this head produced no trace
    /// without an unsupported opcode, if any. A head with this set and no
    /// `blocker_pc` carries no opcode-admission opportunity at all.
    pub reject_reason: Option<TraceRejectReason>,
    /// Number of operations in the current compiled/portable trace.
    pub compiled_ops: u32,
    /// Number of calls into the trace executor.
    ///
    /// The field name is retained for compatibility; portable trace calls
    /// are counted here as well.
    pub native_calls: u64,
    /// Total guest instructions retired by trace execution.
    pub jit_retired: u64,
    /// Exits caused by a guarded branch taking an unrecorded direction.
    pub guarded_branch_exits: u64,
    /// Trace re-recordings triggered by adaptive branch behavior.
    pub adaptive_rerecords: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Why replay-time decoding could not append an executed instruction.
pub enum TraceDecodeFailureReason {
    /// The blocker's memory could not be read side-effect-free: no fastmem
    /// window covers it, so the recorder cannot inspect it without adding
    /// observable bus transactions.
    OpcodeReadFailed,
    /// The opcode executed by the CPU no longer matches guest memory.
    OpcodeChanged,
    /// Memory still contains the executed opcode, but its form is not
    /// traceable or an operand-extension read failed.
    UnsupportedFormOrOperandRead,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Why a recording produced no trace even though decoding never refused an
/// instruction.
///
/// These outcomes carry no *opcode* opportunity: there is no blocker to
/// support, so heads whose recordings end this way are reported separately
/// and are absent from the opportunity ranking by construction.
#[non_exhaustive]
pub enum TraceRejectReason {
    /// A trap or exception surfaced mid-recording. The embedder handles it
    /// and may resume at an unrelated guest PC, so no instruction-set
    /// coverage extends a recording past this point: it is a structural
    /// bound on the head, not a missing feature.
    TrapOrException,
    /// The batch ended at a host boundary mid-recording — instruction
    /// budget exhausted, a watched PC, a stopped CPU, or a decoded
    /// fast-path miss. This says nothing about the head: it is an artifact
    /// of how the embedder sliced execution, and the same head may record
    /// further in a later batch.
    HostBoundary,
    /// The recorded prefix does not end in a trace-terminating operation.
    NoTraceTerminal,
    /// Fewer operations than the minimum for the region's kind.
    TooShort,
    /// An indirect-JSR region shorter than the minimum that amortizes the
    /// call boundary.
    IndirectJsrTooShort,
    /// A linear (non-loop) region carrying checked memory ALU operations,
    /// which does not amortize trace validation.
    LinearMemoryAlu,
    /// The region's code range wraps the address space.
    AddressWrap,
    /// The compiler backend declined the region or was unavailable.
    Backend,
}

impl TraceRejectReason {
    /// Short stable label used in the report.
    pub fn label(self) -> &'static str {
        match self {
            Self::TrapOrException => "trap-or-exception",
            Self::HostBoundary => "host-boundary",
            Self::NoTraceTerminal => "no-trace-terminal",
            Self::TooShort => "too-short",
            Self::IndirectJsrTooShort => "indirect-jsr-too-short",
            Self::LinearMemoryAlu => "linear-memory-alu",
            Self::AddressWrap => "address-wrap",
            Self::Backend => "backend",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// One instruction captured in a successfully compiled trace path.
#[non_exhaustive]
pub struct TraceShapeOp {
    /// Guest address at which the instruction executed.
    pub pc: u32,
    /// Primary opcode word.
    pub opcode: u16,
    /// First extension word captured by the trace decoder, if any.
    pub extension: Option<u16>,
    /// Second extension word captured by the trace decoder, if any.
    pub extension2: Option<u16>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TraceBlocker {
    pub(crate) pc: u32,
    pub(crate) executed_opcode: u16,
    pub(crate) memory_opcode: Option<u16>,
    pub(crate) next_word: Option<u16>,
    pub(crate) next_word2: Option<u16>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One distinct failed recording path for a trace head.
#[non_exhaustive]
pub struct FailedTraceShapeProfileRow {
    /// Guest program counter at which recording began.
    pub start_pc: u32,
    /// CPU model active while recording.
    pub cpu_type: CpuType,
    /// Successfully reconstructed operations before the failure.
    pub prefix_ops: u32,
    /// The exact reconstructed operations preceding the failure, in
    /// execution order.
    pub prefix: Vec<TraceShapeOp>,
    /// Guest address of the instruction whose reconstruction failed.
    pub blocker_pc: u32,
    /// Opcode retained in the CPU's instruction register after execution.
    pub executed_opcode: u16,
    /// Opcode reread from the fastmem window while reconstructing the
    /// trace, when the address is inside the window.
    pub memory_opcode: Option<u16>,
    /// First guest word after the blocker opcode, read through the fastmem
    /// window when available. For an extensionless blocker this is the next
    /// opcode, not an operand.
    pub next_word: Option<u16>,
    /// Second guest word after the blocker opcode, read through the fastmem
    /// window when available.
    pub next_word2: Option<u16>,
    /// Coarse classification of the reconstruction failure.
    pub reason: TraceDecodeFailureReason,
    /// Number of recording attempts with this exact shape.
    pub recordings: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One distinct recording that ended without an unsupported opcode.
///
/// Complements [`FailedTraceShapeProfileRow`]: that type describes
/// recordings stopped by an instruction the decoder refused, this one
/// describes recordings that ran out of decoded execution or were declined
/// by a compile-stage policy. Both were previously indistinguishable from
/// "this head was never hot".
#[non_exhaustive]
pub struct SilentRejectionProfileRow {
    /// Guest program counter at which recording began.
    pub start_pc: u32,
    /// CPU model active while recording.
    pub cpu_type: CpuType,
    /// Why the recording produced no trace.
    pub reason: TraceRejectReason,
    /// Guest address of the instruction that stopped the recording: the
    /// trapping instruction for [`TraceRejectReason::TrapOrException`], and
    /// the PC the CPU had reached otherwise. For a trap this is the bound
    /// on how far this head can ever record.
    pub exit_pc: u32,
    /// Operations successfully reconstructed before the recording ended, in
    /// execution order.
    pub prefix: Vec<TraceShapeOp>,
    /// Number of recording attempts with this exact shape and outcome.
    pub recordings: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// One distinct successfully compiled path for a trace head.
#[non_exhaustive]
pub struct CompiledTraceShapeProfileRow {
    /// Guest program counter at which recording began.
    pub start_pc: u32,
    /// CPU model active while recording.
    pub cpu_type: CpuType,
    /// Executed instructions in dynamic path order.
    pub ops: Vec<TraceShapeOp>,
    /// Number of times this exact path was compiled.
    pub recordings: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Execution count aggregated by decoded memory-operation opcode.
#[non_exhaustive]
pub struct DecodedMemProfileRow {
    /// Guest opcode word.
    pub opcode: u16,
    /// Number of executions through the decoded memory fast path.
    pub executions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Execution count for one decoded memory-operation site.
#[non_exhaustive]
pub struct DecodedMemSiteProfileRow {
    /// Guest program counter of the operation.
    pub pc: u32,
    /// Guest opcode word.
    pub opcode: u16,
    /// Number of executions through the decoded memory fast path.
    pub executions: u64,
}

impl TraceProfileRow {
    /// Approximate interpreter dispatches made eligible by supporting the
    /// blocker. This deliberately excludes the blocker itself: some control-
    /// flow instructions should terminate a trace rather than execute in it.
    pub fn projected_dispatches(&self) -> u64 {
        self.rejected_hits
            .saturating_mul(u64::from(self.prefix_ops))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// Snapshot of all trace and decoded-memory profiling counters.
#[non_exhaustive]
pub struct TraceProfileSnapshot {
    /// Per-trace-head counters.
    pub rows: Vec<TraceProfileRow>,
    /// Decoded memory counts aggregated by opcode.
    pub decoded_mem_ops: Vec<DecodedMemProfileRow>,
    /// Decoded memory counts split by guest PC and opcode.
    pub decoded_mem_sites: Vec<DecodedMemSiteProfileRow>,
    /// Failed paths kept separately from successful recordings at the same
    /// trace head. This prevents a historical blocker from being mistaken
    /// for the boundary of the currently compiled path.
    pub failed_shapes: Vec<FailedTraceShapeProfileRow>,
    /// Successfully compiled dynamic paths, including their exact executed
    /// guest instruction sequence.
    pub compiled_shapes: Vec<CompiledTraceShapeProfileRow>,
    /// Failure recordings dropped after the distinct failed-shape cap was
    /// reached. Zero unless a pathological workload produced thousands of
    /// distinct failed paths.
    pub failed_shape_overflow: u64,
    /// Recordings that ended without an unsupported opcode: control left
    /// the decoded path, or a compile-stage policy declined the region.
    pub silent_rejections: Vec<SilentRejectionProfileRow>,
    /// Silent rejections dropped after the distinct-shape cap was reached.
    pub silent_rejection_overflow: u64,
    /// Total observed backward branches.
    pub backward_hits: u64,
    /// Total rejected trace-entry opportunities.
    pub rejected_hits: u64,
    /// Total calls into a trace executor.
    pub native_calls: u64,
    /// Total instructions retired by trace executors.
    pub jit_retired: u64,
}

impl TraceProfileSnapshot {
    /// Format the snapshot as a ranked, human-readable profiling report.
    pub fn report(&self) -> String {
        let mut rows = self.rows.clone();
        rows.sort_unstable_by(|a, b| {
            b.projected_dispatches()
                .cmp(&a.projected_dispatches())
                .then_with(|| b.backward_hits.cmp(&a.backward_hits))
                .then_with(|| a.start_pc.cmp(&b.start_pc))
        });

        let average = if self.native_calls == 0 {
            0.0
        } else {
            self.jit_retired as f64 / self.native_calls as f64
        };
        let mut out = String::new();
        let _ = writeln!(out, "m68k trace opportunity profile");
        let _ = writeln!(
            out,
            "totals: backward_hits={} rejected_hits={} native_calls={} jit_retired={} avg_ops_per_native_call={average:.2}",
            self.backward_hits, self.rejected_hits, self.native_calls, self.jit_retired
        );
        let _ = writeln!(
            out,
            "note: head rows aggregate the process lifetime; failure and compiled columns may describe different recording paths. See the shape tables below."
        );
        let _ = writeln!(
            out,
            "rank  start_pc  hits       rejected   attempts prefix projected   blocker_pc opcode  compiled calls      retired"
        );
        for (rank, row) in rows.iter().take(40).enumerate() {
            let blocker_pc = row
                .blocker_pc
                .map_or_else(|| "--------".to_owned(), |pc| format!("{pc:08X}"));
            let blocker_opcode = row
                .blocker_opcode
                .map_or_else(|| "----".to_owned(), |opcode| format!("{opcode:04X}"));
            let _ = writeln!(
                out,
                "{:>4}  {:08X}  {:>10}  {:>10}  {:>8} {:>6} {:>10}   {}  {}  {:>8} {:>10} {:>12}",
                rank + 1,
                row.start_pc,
                row.backward_hits,
                row.rejected_hits,
                row.recording_attempts,
                row.prefix_ops,
                row.projected_dispatches(),
                blocker_pc,
                blocker_opcode,
                row.compiled_ops,
                row.native_calls,
                row.jit_retired
            );
        }

        let mut compiled_rows: Vec<_> = self
            .rows
            .iter()
            .filter(|row| row.native_calls != 0)
            .collect();
        compiled_rows.sort_unstable_by(|a, b| {
            b.jit_retired
                .cmp(&a.jit_retired)
                .then_with(|| b.native_calls.cmp(&a.native_calls))
                .then_with(|| a.start_pc.cmp(&b.start_pc))
        });
        let _ = writeln!(out, "compiled traces by retired instructions");
        let _ = writeln!(
            out,
            "rank  start_pc  ops      calls      retired avg_ops guard_exits rerecords"
        );
        for (rank, row) in compiled_rows.iter().take(40).enumerate() {
            let average = row.jit_retired as f64 / row.native_calls as f64;
            let _ = writeln!(
                out,
                "{:>4}  {:08X}  {:>3} {:>10} {:>12} {:>7.2} {:>11} {:>9}",
                rank + 1,
                row.start_pc,
                row.compiled_ops,
                row.native_calls,
                row.jit_retired,
                average,
                row.guarded_branch_exits,
                row.adaptive_rerecords
            );
        }

        let mut failed_shapes = self.failed_shapes.clone();
        failed_shapes.sort_unstable_by(|a, b| {
            b.recordings
                .cmp(&a.recordings)
                .then_with(|| b.prefix_ops.cmp(&a.prefix_ops))
                .then_with(|| a.start_pc.cmp(&b.start_pc))
        });
        let _ = writeln!(out, "failed recording shapes");
        if self.failed_shape_overflow > 0 {
            let _ = writeln!(
                out,
                "note: {} failure recordings dropped past the {}-shape cap",
                self.failed_shape_overflow, FAILED_SHAPE_CAP
            );
        }
        let _ = writeln!(
            out,
            "rank  start_pc  records prefix blocker_pc ir   memory next1 next2 reason"
        );
        for (rank, shape) in failed_shapes.iter().take(40).enumerate() {
            let memory = shape
                .memory_opcode
                .map_or_else(|| "----".to_owned(), |value| format!("{value:04X}"));
            let next_word = shape
                .next_word
                .map_or_else(|| "----".to_owned(), |value| format!("{value:04X}"));
            let next_word2 = shape
                .next_word2
                .map_or_else(|| "----".to_owned(), |value| format!("{value:04X}"));
            let reason = match shape.reason {
                TraceDecodeFailureReason::OpcodeReadFailed => "opcode-read-failed",
                TraceDecodeFailureReason::OpcodeChanged => "opcode-changed",
                TraceDecodeFailureReason::UnsupportedFormOrOperandRead => {
                    "unsupported-form-or-operand-read"
                }
            };
            let _ = writeln!(
                out,
                "{:>4}  {:08X} {:>8} {:>6}   {:08X} {:04X} {} {:>5} {:>5} {}",
                rank + 1,
                shape.start_pc,
                shape.recordings,
                shape.prefix_ops,
                shape.blocker_pc,
                shape.executed_opcode,
                memory,
                next_word,
                next_word2,
                reason
            );
            let _ = write!(out, "      prefix:");
            if shape.prefix.is_empty() {
                let _ = write!(out, " (none)");
            }
            for op in &shape.prefix {
                let _ = write!(out, " {:08X}:{:04X}", op.pc, op.opcode);
                if let Some(extension) = op.extension {
                    let _ = write!(out, "/{extension:04X}");
                }
                if let Some(extension) = op.extension2 {
                    let _ = write!(out, "/{extension:04X}");
                }
            }
            let _ = writeln!(out);
        }

        let mut silent_rejections = self.silent_rejections.clone();
        silent_rejections.sort_unstable_by(|a, b| {
            b.recordings
                .cmp(&a.recordings)
                .then_with(|| b.prefix.len().cmp(&a.prefix.len()))
                .then_with(|| a.start_pc.cmp(&b.start_pc))
        });
        let _ = writeln!(
            out,
            "silent rejections (recordings that produced no trace with no unsupported opcode)"
        );
        let _ = writeln!(
            out,
            "note: these heads have no blocker to support, so they carry no opcode-admission opportunity and are absent from the ranking above. trap-or-exception is a structural bound — no instruction coverage extends a recording past it; host-boundary is only an artifact of batch slicing."
        );
        if self.silent_rejection_overflow > 0 {
            let _ = writeln!(
                out,
                "note: {} silent rejections dropped past the {}-shape cap",
                self.silent_rejection_overflow, SILENT_REJECTION_CAP
            );
        }
        let _ = writeln!(out, "rank  start_pc  records stranded exit_pc  reason");
        for (rank, shape) in silent_rejections.iter().take(40).enumerate() {
            let _ = writeln!(
                out,
                "{:>4}  {:08X} {:>8} {:>8}   {:08X} {}",
                rank + 1,
                shape.start_pc,
                shape.recordings,
                shape.prefix.len(),
                shape.exit_pc,
                shape.reason.label()
            );
            let _ = write!(out, "      prefix:");
            if shape.prefix.is_empty() {
                let _ = write!(out, " (none)");
            }
            for op in &shape.prefix {
                let _ = write!(out, " {:08X}:{:04X}", op.pc, op.opcode);
                if let Some(extension) = op.extension {
                    let _ = write!(out, "/{extension:04X}");
                }
                if let Some(extension) = op.extension2 {
                    let _ = write!(out, "/{extension:04X}");
                }
            }
            let _ = writeln!(out);
        }

        let mut compiled_shapes = self.compiled_shapes.clone();
        compiled_shapes.sort_unstable_by(|a, b| {
            b.recordings
                .cmp(&a.recordings)
                .then_with(|| b.ops.len().cmp(&a.ops.len()))
                .then_with(|| a.start_pc.cmp(&b.start_pc))
        });
        let _ = writeln!(out, "compiled recording shapes");
        for (rank, shape) in compiled_shapes.iter().take(40).enumerate() {
            let _ = writeln!(
                out,
                "{:>4}  {:08X} records={} ops={}",
                rank + 1,
                shape.start_pc,
                shape.recordings,
                shape.ops.len()
            );
            let _ = write!(out, "      path:");
            for op in &shape.ops {
                let _ = write!(out, " {:08X}:{:04X}", op.pc, op.opcode);
                if let Some(extension) = op.extension {
                    let _ = write!(out, "/{extension:04X}");
                }
                if let Some(extension) = op.extension2 {
                    let _ = write!(out, "/{extension:04X}");
                }
            }
            let _ = writeln!(out);
        }

        let mut decoded_mem_ops = self.decoded_mem_ops.clone();
        decoded_mem_ops.sort_unstable_by(|a, b| {
            b.executions
                .cmp(&a.executions)
                .then_with(|| a.opcode.cmp(&b.opcode))
        });
        let decoded_mem_total: u64 = decoded_mem_ops.iter().map(|row| row.executions).sum();
        let _ = writeln!(
            out,
            "decoded memory operations: total={decoded_mem_total} distinct_opcodes={}",
            decoded_mem_ops.len()
        );
        let _ = writeln!(out, "rank  opcode  executions percent");
        for (rank, row) in decoded_mem_ops.iter().take(40).enumerate() {
            let percent = if decoded_mem_total == 0 {
                0.0
            } else {
                row.executions as f64 * 100.0 / decoded_mem_total as f64
            };
            let _ = writeln!(
                out,
                "{:>4}  {:04X} {:>11} {:>6.2}%",
                rank + 1,
                row.opcode,
                row.executions,
                percent
            );
        }

        let mut decoded_mem_sites = self.decoded_mem_sites.clone();
        decoded_mem_sites.sort_unstable_by(|a, b| {
            b.executions
                .cmp(&a.executions)
                .then_with(|| a.pc.cmp(&b.pc))
                .then_with(|| a.opcode.cmp(&b.opcode))
        });
        let _ = writeln!(out, "decoded memory sites by execution count");
        let _ = writeln!(out, "rank  pc        opcode  executions");
        for (rank, row) in decoded_mem_sites.iter().take(60).enumerate() {
            let _ = writeln!(
                out,
                "{:>4}  {:08X}  {:04X} {:>11}",
                rank + 1,
                row.pc,
                row.opcode,
                row.executions
            );
        }
        out
    }
}

#[derive(Default)]
struct Row {
    cpu_type: u32,
    backward_hits: u64,
    rejected_hits: u64,
    recording_attempts: u64,
    prefix_ops: u32,
    blocker_pc: Option<u32>,
    blocker_opcode: Option<u16>,
    reject_reason: Option<TraceRejectReason>,
    compiled_ops: u32,
    native_calls: u64,
    jit_retired: u64,
    guarded_branch_exits: u64,
    adaptive_rerecords: u64,
}

/// Upper bound on distinct failed shapes kept per process. Each entry holds
/// a full prefix sequence, so the map must not grow for the whole session;
/// drops past the cap are counted and reported.
const FAILED_SHAPE_CAP: usize = 4096;

/// Upper bound on distinct silent-rejection shapes kept per process, for
/// the same reason as [`FAILED_SHAPE_CAP`].
const SILENT_REJECTION_CAP: usize = 4096;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SilentRejectionKey {
    start_pc: u32,
    cpu_type: u32,
    reason: TraceRejectReason,
    exit_pc: u32,
    prefix: Vec<TraceShapeOp>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct FailedShapeKey {
    start_pc: u32,
    cpu_type: u32,
    prefix: Vec<TraceShapeOp>,
    blocker_pc: u32,
    executed_opcode: u16,
    memory_opcode: Option<u16>,
    next_word: Option<u16>,
    next_word2: Option<u16>,
    reason: TraceDecodeFailureReason,
}

type CompiledShapeKey = (u32, u32, Vec<TraceShapeOp>);

/// The site key is already a uniformly useful `(pc << 16) | opcode` integer,
/// so hashing it again only adds overhead to this hot, feature-only profiler.
#[derive(Default)]
struct IdentityHasher(u64);

impl Hasher for IdentityHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for &byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        self.0 = hash;
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }
}

type SiteCounts = HashMap<u64, u64, BuildHasherDefault<IdentityHasher>>;

struct Profile {
    rows: BTreeMap<(u32, u32), Row>,
    failed_shapes: BTreeMap<FailedShapeKey, u64>,
    /// Failure recordings dropped after `FAILED_SHAPE_CAP` distinct shapes.
    /// The prefix sequences make each entry meaningfully sized, so the map
    /// is bounded for the process lifetime and the report states what was
    /// dropped instead of growing without limit.
    failed_shape_overflow: u64,
    /// Recordings that ended without an unsupported opcode, bounded by
    /// `SILENT_REJECTION_CAP` on the same rationale.
    silent_rejections: BTreeMap<SilentRejectionKey, u64>,
    silent_rejection_overflow: u64,
    compiled_shapes: BTreeMap<CompiledShapeKey, u64>,
    decoded_mem_counts: Box<[u64]>,
    decoded_mem_site_counts: SiteCounts,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            rows: BTreeMap::new(),
            failed_shapes: BTreeMap::new(),
            compiled_shapes: BTreeMap::new(),
            decoded_mem_counts: vec![0; super::op_cache::DECODE_TABLE_SIZE].into_boxed_slice(),
            decoded_mem_site_counts: SiteCounts::default(),
            failed_shape_overflow: 0,
            silent_rejections: BTreeMap::new(),
            silent_rejection_overflow: 0,
        }
    }
}

impl Profile {
    fn row(&mut self, pc: u32, cpu_type: CpuType) -> &mut Row {
        self.rows
            .entry((pc, cpu_type as u32))
            .or_insert_with(|| Row {
                cpu_type: cpu_type as u32,
                ..Row::default()
            })
    }

    fn snapshot(&self) -> TraceProfileSnapshot {
        let rows: Vec<_> = self
            .rows
            .iter()
            .map(|(&(start_pc, _), row)| TraceProfileRow {
                start_pc,
                cpu_type: cpu_type_from_repr(row.cpu_type),
                backward_hits: row.backward_hits,
                rejected_hits: row.rejected_hits,
                recording_attempts: row.recording_attempts,
                prefix_ops: row.prefix_ops,
                blocker_pc: row.blocker_pc,
                blocker_opcode: row.blocker_opcode,
                reject_reason: row.reject_reason,
                compiled_ops: row.compiled_ops,
                native_calls: row.native_calls,
                jit_retired: row.jit_retired,
                guarded_branch_exits: row.guarded_branch_exits,
                adaptive_rerecords: row.adaptive_rerecords,
            })
            .collect();
        let decoded_mem_ops = self
            .decoded_mem_counts
            .iter()
            .enumerate()
            .filter_map(|(opcode, &executions)| {
                (executions != 0).then_some(DecodedMemProfileRow {
                    opcode: opcode as u16,
                    executions,
                })
            })
            .collect();
        let decoded_mem_sites = self
            .decoded_mem_site_counts
            .iter()
            .map(|(&key, &executions)| DecodedMemSiteProfileRow {
                pc: (key >> 16) as u32,
                opcode: key as u16,
                executions,
            })
            .collect();
        let failed_shapes = self
            .failed_shapes
            .iter()
            .map(|(key, &recordings)| FailedTraceShapeProfileRow {
                start_pc: key.start_pc,
                cpu_type: cpu_type_from_repr(key.cpu_type),
                prefix_ops: key.prefix.len() as u32,
                prefix: key.prefix.clone(),
                blocker_pc: key.blocker_pc,
                executed_opcode: key.executed_opcode,
                memory_opcode: key.memory_opcode,
                next_word: key.next_word,
                next_word2: key.next_word2,
                reason: key.reason,
                recordings,
            })
            .collect();
        let silent_rejections = self
            .silent_rejections
            .iter()
            .map(|(key, &recordings)| SilentRejectionProfileRow {
                start_pc: key.start_pc,
                cpu_type: cpu_type_from_repr(key.cpu_type),
                reason: key.reason,
                exit_pc: key.exit_pc,
                prefix: key.prefix.clone(),
                recordings,
            })
            .collect();
        let compiled_shapes = self
            .compiled_shapes
            .iter()
            .map(
                |((start_pc, cpu_type, ops), &recordings)| CompiledTraceShapeProfileRow {
                    start_pc: *start_pc,
                    cpu_type: cpu_type_from_repr(*cpu_type),
                    ops: ops.clone(),
                    recordings,
                },
            )
            .collect();
        TraceProfileSnapshot {
            backward_hits: rows.iter().map(|row| row.backward_hits).sum(),
            rejected_hits: rows.iter().map(|row| row.rejected_hits).sum(),
            native_calls: rows.iter().map(|row| row.native_calls).sum(),
            jit_retired: rows.iter().map(|row| row.jit_retired).sum(),
            rows,
            decoded_mem_ops,
            decoded_mem_sites,
            failed_shapes,
            compiled_shapes,
            failed_shape_overflow: self.failed_shape_overflow,
            silent_rejections,
            silent_rejection_overflow: self.silent_rejection_overflow,
        }
    }
}

struct ProfileState(Profile);

impl Drop for ProfileState {
    fn drop(&mut self) {
        if std::env::var_os("M68K_TRACE_PROFILE").is_some() {
            eprintln!("{}", self.0.snapshot().report());
        }
    }
}

thread_local! {
    static PROFILE: RefCell<ProfileState> = RefCell::new(ProfileState(Profile::default()));
}

/// Clear every counter in the current thread's profiler.
pub fn reset() {
    PROFILE.with_borrow_mut(|profile| profile.0 = Profile::default());
}

/// Capture the current thread's profiling counters without resetting them.
pub fn snapshot() -> TraceProfileSnapshot {
    PROFILE.with_borrow(|profile| profile.0.snapshot())
}

pub(crate) fn note_decoded_mem(pc: u32, opcode: u16) {
    PROFILE.with_borrow_mut(|profile| {
        let count = &mut profile.0.decoded_mem_counts[usize::from(opcode)];
        *count = count.saturating_add(1);
        let site_key = (u64::from(pc) << 16) | u64::from(opcode);
        let site_count = profile
            .0
            .decoded_mem_site_counts
            .entry(site_key)
            .or_default();
        *site_count = site_count.saturating_add(1);
    });
}

pub(crate) fn note_backward_edge(pc: u32, cpu_type: CpuType, rejected: bool) {
    PROFILE.with_borrow_mut(|profile| {
        let row = profile.0.row(pc, cpu_type);
        row.backward_hits = row.backward_hits.saturating_add(1);
        if rejected {
            row.rejected_hits = row.rejected_hits.saturating_add(1);
        }
    });
}

pub(crate) fn note_recording(pc: u32, cpu_type: CpuType) {
    PROFILE.with_borrow_mut(|profile| {
        let row = profile.0.row(pc, cpu_type);
        row.recording_attempts = row.recording_attempts.saturating_add(1);
    });
}

pub(crate) fn note_blocker(
    start_pc: u32,
    cpu_type: CpuType,
    prefix: Vec<TraceShapeOp>,
    blocker: TraceBlocker,
) {
    PROFILE.with_borrow_mut(|profile| {
        let row = profile.0.row(start_pc, cpu_type);
        // Keep the longest observed prefix for this trace head. It is the
        // conservative amount of already-supported work stranded behind the
        // blocker; path variation is visible through the shape table.
        if prefix.len() as u32 >= row.prefix_ops {
            row.prefix_ops = prefix.len() as u32;
            row.blocker_pc = Some(blocker.pc);
            row.blocker_opcode = Some(blocker.executed_opcode);
        }
        let reason = match blocker.memory_opcode {
            None => TraceDecodeFailureReason::OpcodeReadFailed,
            Some(opcode) if opcode != blocker.executed_opcode => {
                TraceDecodeFailureReason::OpcodeChanged
            }
            Some(_) => TraceDecodeFailureReason::UnsupportedFormOrOperandRead,
        };
        let key = FailedShapeKey {
            start_pc,
            cpu_type: cpu_type as u32,
            prefix,
            blocker_pc: blocker.pc,
            executed_opcode: blocker.executed_opcode,
            memory_opcode: blocker.memory_opcode,
            next_word: blocker.next_word,
            next_word2: blocker.next_word2,
            reason,
        };
        if profile.0.failed_shapes.len() >= FAILED_SHAPE_CAP
            && !profile.0.failed_shapes.contains_key(&key)
        {
            profile.0.failed_shape_overflow = profile.0.failed_shape_overflow.saturating_add(1);
            return;
        }
        let recordings = profile.0.failed_shapes.entry(key).or_default();
        *recordings = recordings.saturating_add(1);
    });
}

/// Record a recording that produced no trace without any unsupported
/// opcode. Without this the head keeps its backward-branch counters but
/// gains no prefix, blocker, or shape entry, so it is indistinguishable in
/// the report from a head that was never hot — and it silently leaves the
/// opportunity ranking, which orders by `rejected_hits * prefix_ops`.
pub(crate) fn note_silent_rejection(
    start_pc: u32,
    cpu_type: CpuType,
    exit_pc: u32,
    prefix: Vec<TraceShapeOp>,
    reason: TraceRejectReason,
) {
    PROFILE.with_borrow_mut(|profile| {
        profile.0.row(start_pc, cpu_type).reject_reason = Some(reason);
        let key = SilentRejectionKey {
            start_pc,
            cpu_type: cpu_type as u32,
            reason,
            exit_pc,
            prefix,
        };
        if profile.0.silent_rejections.len() >= SILENT_REJECTION_CAP
            && !profile.0.silent_rejections.contains_key(&key)
        {
            profile.0.silent_rejection_overflow =
                profile.0.silent_rejection_overflow.saturating_add(1);
            return;
        }
        let recordings = profile.0.silent_rejections.entry(key).or_default();
        *recordings = recordings.saturating_add(1);
    });
}

pub(crate) fn note_compiled(pc: u32, cpu_type: CpuType, ops: Vec<TraceShapeOp>) {
    PROFILE.with_borrow_mut(|profile| {
        profile.0.row(pc, cpu_type).compiled_ops = ops.len() as u32;
        let recordings = profile
            .0
            .compiled_shapes
            .entry((pc, cpu_type as u32, ops))
            .or_default();
        *recordings = recordings.saturating_add(1);
    });
}

pub(crate) fn note_native_call(pc: u32, cpu_type: CpuType, retired: u32) {
    PROFILE.with_borrow_mut(|profile| {
        let row = profile.0.row(pc, cpu_type);
        row.native_calls = row.native_calls.saturating_add(1);
        row.jit_retired = row.jit_retired.saturating_add(u64::from(retired));
    });
}

pub(crate) fn note_guarded_branch_exit(pc: u32, cpu_type: CpuType) {
    PROFILE.with_borrow_mut(|profile| {
        let row = profile.0.row(pc, cpu_type);
        row.guarded_branch_exits = row.guarded_branch_exits.saturating_add(1);
    });
}

pub(crate) fn note_adaptive_rerecord(pc: u32, cpu_type: CpuType) {
    PROFILE.with_borrow_mut(|profile| {
        let row = profile.0.row(pc, cpu_type);
        row.adaptive_rerecords = row.adaptive_rerecords.saturating_add(1);
    });
}

fn cpu_type_from_repr(value: u32) -> CpuType {
    match value {
        1 => CpuType::M68000,
        2 => CpuType::M68010,
        3 => CpuType::M68EC020,
        4 => CpuType::M68020,
        5 => CpuType::M68EC030,
        6 => CpuType::M68030,
        7 => CpuType::M68EC040,
        8 => CpuType::M68LC040,
        9 => CpuType::M68040,
        10 => CpuType::SCC68070,
        _ => CpuType::Invalid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AddressBus, BatchExit, CpuCore, LinearMemoryBus};

    /// A `count`-op prefix of single-word instructions starting at `start`.
    fn dummy_prefix(start: u32, count: u32) -> Vec<TraceShapeOp> {
        (0..count)
            .map(|index| TraceShapeOp {
                pc: start + index * 2,
                opcode: 0x7000 + index as u16,
                extension: None,
                extension2: None,
            })
            .collect()
    }

    #[test]
    fn report_ranks_stranded_dispatches_not_raw_hits() {
        reset();
        note_backward_edge(0x100, CpuType::M68040, true);
        note_blocker(
            0x100,
            CpuType::M68040,
            dummy_prefix(0x100, 2),
            TraceBlocker {
                pc: 0x104,
                executed_opcode: 0x4ead,
                memory_opcode: Some(0x4ead),
                next_word: None,
                next_word2: None,
            },
        );
        for _ in 0..3 {
            note_backward_edge(0x200, CpuType::M68040, true);
        }
        note_blocker(
            0x200,
            CpuType::M68040,
            dummy_prefix(0x200, 1),
            TraceBlocker {
                pc: 0x202,
                executed_opcode: 0x486d,
                memory_opcode: Some(0x486d),
                next_word: None,
                next_word2: None,
            },
        );

        let report = snapshot().report();
        assert!(report.find("00000200").unwrap() < report.find("00000100").unwrap());
    }

    #[test]
    fn failed_shape_cap_counts_overflow() {
        reset();
        let blocker = TraceBlocker {
            pc: 0x9000,
            executed_opcode: 0x4c01,
            memory_opcode: Some(0x4c01),
            next_word: None,
            next_word2: None,
        };
        for index in 0..(FAILED_SHAPE_CAP as u32 + 5) {
            note_blocker(
                0x100 + index * 2,
                CpuType::M68040,
                dummy_prefix(0x100, 1),
                blocker,
            );
        }
        let snapshot = snapshot();
        assert_eq!(snapshot.failed_shapes.len(), FAILED_SHAPE_CAP);
        assert_eq!(snapshot.failed_shape_overflow, 5);
        assert!(
            snapshot
                .report()
                .contains("5 failure recordings dropped past the 4096-shape cap")
        );
    }

    #[test]
    fn distinct_equal_length_prefixes_stay_distinct() {
        reset();
        let blocker = TraceBlocker {
            pc: 0x108,
            executed_opcode: 0x4c01,
            memory_opcode: Some(0x4c01),
            next_word: None,
            next_word2: None,
        };
        let mut path_a = dummy_prefix(0x100, 4);
        let mut path_b = dummy_prefix(0x100, 4);
        // Same head, same length, same blocker; one interior operation
        // differs, as when a guarded branch takes another recorded route.
        path_a[2].opcode = 0x5280;
        path_b[2].opcode = 0x5281;
        note_blocker(0x100, CpuType::M68040, path_a.clone(), blocker);
        note_blocker(0x100, CpuType::M68040, path_b.clone(), blocker);
        note_blocker(0x100, CpuType::M68040, path_b, blocker);

        let snapshot = snapshot();
        let rows: Vec<_> = snapshot
            .failed_shapes
            .iter()
            .filter(|row| row.start_pc == 0x100)
            .collect();
        assert_eq!(
            rows.len(),
            2,
            "equal-length distinct prefixes must not merge"
        );
        assert!(rows.iter().all(|row| row.prefix_ops == 4));
        let mut recordings: Vec<u64> = rows.iter().map(|row| row.recordings).collect();
        recordings.sort_unstable();
        assert_eq!(recordings, vec![1, 2]);
        assert_ne!(rows[0].prefix, rows[1].prefix);

        let report = snapshot.report();
        assert!(report.contains("prefix: 00000100:7000 00000102:7001 00000104:5280 00000106:7003"));
        assert!(report.contains("prefix: 00000100:7000 00000102:7001 00000104:5281 00000106:7003"));
    }

    #[test]
    fn report_separates_failed_and_compiled_shapes_at_one_head() {
        reset();
        note_blocker(
            0x100,
            CpuType::M68040,
            dummy_prefix(0x100, 20),
            TraceBlocker {
                pc: 0x140,
                executed_opcode: 0x4c01,
                memory_opcode: Some(0x4c01),
                next_word: Some(0x0800),
                next_word2: Some(0xee80),
            },
        );
        note_compiled(
            0x100,
            CpuType::M68040,
            vec![
                TraceShapeOp {
                    pc: 0x100,
                    opcode: 0x7000,
                    extension: None,
                    extension2: None,
                },
                TraceShapeOp {
                    pc: 0x102,
                    opcode: 0x60fc,
                    extension: None,
                    extension2: None,
                },
            ],
        );

        let snapshot = snapshot();
        assert_eq!(snapshot.failed_shapes.len(), 1);
        assert_eq!(snapshot.compiled_shapes.len(), 1);
        assert_eq!(snapshot.failed_shapes[0].prefix_ops, 20);
        assert_eq!(snapshot.compiled_shapes[0].ops.len(), 2);
        let report = snapshot.report();
        assert!(report.contains("failure and compiled columns may describe different"));
        assert!(report.contains("00000140 4C01 4C01  0800  EE80"));
        assert!(report.contains("path: 00000100:7000 00000102:60FC"));
    }

    #[test]
    fn rejected_trace_keeps_counting_dynamic_backward_edges() {
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        bus.write_word(0, 0x5280); // ADDQ.L #1,D0: traceable prefix
        bus.write_word(2, 0x4AC0); // TAS D0: untraceable blocker
        bus.write_word(4, 0x60FA); // BRA.S $0000

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.pc = 0;
        let result = cpu.run_batch(&mut bus, 120, &[]);
        assert_eq!(result.instructions, 120);

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0)
            .expect("loop head was profiled");
        assert_eq!(row.backward_hits, 40);
        assert_eq!(row.rejected_hits, 39);
        assert_eq!(row.recording_attempts, 1);
        assert_eq!(row.prefix_ops, 1);
        assert_eq!(row.blocker_pc, Some(2));
        assert_eq!(row.blocker_opcode, Some(0x4AC0));
        assert_eq!(row.projected_dispatches(), row.rejected_hits);
    }

    #[test]
    fn two_op_self_loop_is_compiled_and_runs_natively() {
        reset();
        let mut bus = LinearMemoryBus::new(0x4000);
        bus.write_word(0, 0x22D8); // MOVE.L (A0)+,(A1)+
        bus.write_word(2, 0x51C8); // DBRA D0,$0000
        bus.write_word(4, 0xFFFC);

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(0, 0x1000);
        cpu.set_a(1, 0x2000);
        cpu.set_d(0, 1000);
        cpu.pc = 0;
        let result = cpu.run_batch(&mut bus, 120, &[]);
        assert_eq!(result.instructions, 120);

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0)
            .expect("two-op loop head was profiled");
        assert_eq!(row.compiled_ops, 2);
        #[cfg(all(feature = "jit", not(target_family = "wasm")))]
        assert!(
            row.native_calls > 1,
            "two-op read/write loops retain the measured faster one-pass path"
        );
        #[cfg(any(not(feature = "jit"), target_family = "wasm"))]
        assert!(row.native_calls > 0);
        assert!(row.jit_retired > 0);
    }

    #[test]
    fn full_dispatch_instruction_can_complete_a_recorded_loop() {
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        bus.write_word(0, 0x4C98); // MOVEM.W (A0)+,D1 (full dispatcher)
        bus.write_word(2, 0x0002);
        bus.write_word(4, 0x60FA); // BRA.S $0000

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(0, 0x0100);
        cpu.pc = 0;
        let result = cpu.run_batch(&mut bus, 120, &[]);
        assert_eq!(result.instructions, 120);

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0)
            .expect("MOVEM loop head was profiled");
        assert_eq!(row.compiled_ops, 2);
        assert_eq!(row.blocker_pc, None);
        assert!(row.jit_retired > 0);
    }

    fn warm_full_dispatch_recording(cpu: &mut CpuCore, bus: &mut LinearMemoryBus) {
        bus.write_word(0, 0x4C98); // MOVEM.W (A0)+,D1 (full dispatcher)
        bus.write_word(2, 0x0002);
        bus.write_word(4, 0x60FA); // BRA.S $0000

        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_a(0, 0x0100);
        cpu.pc = 0;
        let result = cpu.run_batch(bus, 4, &[]);
        assert_eq!(result.exit, BatchExit::BudgetExhausted);
        assert_eq!(result.instructions, 4);
        assert_eq!(cpu.pc, 0);
        assert!(!cpu.trace_recording);
    }

    #[test]
    fn budget_exit_ends_recording_after_full_dispatch_instruction() {
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        let mut cpu = CpuCore::new();
        warm_full_dispatch_recording(&mut cpu, &mut bus);

        let result = cpu.run_batch(&mut bus, 1, &[]);

        assert_eq!(result.exit, BatchExit::BudgetExhausted);
        assert_eq!(result.instructions, 1);
        assert_eq!(cpu.pc, 4);
        assert!(!cpu.trace_recording);
    }

    #[test]
    fn watched_exit_ends_recording_after_full_dispatch_instruction() {
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        let mut cpu = CpuCore::new();
        warm_full_dispatch_recording(&mut cpu, &mut bus);

        let result = cpu.run_batch(&mut bus, 10, &[4]);

        assert_eq!(result.exit, BatchExit::WatchedPc { pc: 4 });
        assert_eq!(result.instructions, 1);
        assert_eq!(cpu.pc, 4);
        assert!(!cpu.trace_recording);
    }

    #[test]
    fn surfaced_trap_ends_full_dispatch_recording() {
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        bus.write_word(0, 0x5280); // ADDQ.L #1,D0
        bus.write_word(2, 0x0C80); // CMPI.L #3,D0
        bus.write_word(4, 0x0000);
        bus.write_word(6, 0x0003);
        bus.write_word(8, 0x66F6); // BNE.S $0000 (taken twice)
        bus.write_word(10, 0xA123); // surfaced A-line trap

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.pc = 0;
        let result = cpu.run_batch(&mut bus, 100, &[]);

        assert_eq!(result.exit, BatchExit::AlineTrap { opcode: 0xA123 });
        assert_eq!(cpu.d(0), 3);
        assert!(!cpu.trace_recording);
    }

    #[test]
    fn cheap_self_loop_iterations_stay_in_one_native_call() {
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        bus.write_word(0, 0x5280); // ADDQ.L #1,D0
        bus.write_word(2, 0x60FC); // BRA.S $0000

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.pc = 0;
        let result = cpu.run_batch(&mut bus, 120, &[]);
        assert_eq!(result.instructions, 120);

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0)
            .expect("cheap loop head was profiled");
        assert_eq!(row.compiled_ops, 2);
        #[cfg(all(feature = "jit", not(target_family = "wasm")))]
        assert_eq!(row.native_calls, 1);
        #[cfg(any(not(feature = "jit"), target_family = "wasm"))]
        assert!(row.native_calls > 1);
        assert!(row.jit_retired > 0);
    }

    #[test]
    fn dominant_guard_side_exit_is_rerecorded() {
        reset();
        const HEAD: u32 = 0x6000;
        let words = [
            0xB210, // CMP.B (A0),D1
            0x6606, // BNE.S outer
            0x10DC, // common: MOVE.B (A4)+,(A0)+
            0x51C8, 0xFFF8, // DBRA D0,head
            0x2042, // outer: MOVEA.L D2,A0
            0x2843, // MOVEA.L D3,A4
            0x707F, // MOVEQ #127,D0
            0x5884, // ADDQ.L #4,D4
            0x60EC, // BRA.S head
        ];
        let mut bus = LinearMemoryBus::new(0x1_0000);
        for (index, word) in words.iter().enumerate() {
            bus.write_word(HEAD + index as u32 * 2, *word);
        }

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = HEAD;
        cpu.set_a(0, 0x4000);
        cpu.set_a(4, 0x5000);
        cpu.set_d(0, 127);
        cpu.set_d(1, 1);
        cpu.set_d(2, 0x4000);
        cpu.set_d(3, 0x5000);

        // Record the uncommon seven-op BNE path, then make the four-op
        // fallthrough loop dominant long enough to trigger adaptation.
        assert_eq!(cpu.run_batch(&mut bus, 14, &[0]).instructions, 14);
        cpu.set_d(1, 0);
        assert_eq!(cpu.run_batch(&mut bus, 100_000, &[0]).instructions, 100_000);

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == HEAD)
            .expect("biased loop head was profiled");
        assert_eq!(row.recording_attempts, 2);
        assert_eq!(row.adaptive_rerecords, 1);
        assert_eq!(row.guarded_branch_exits, 64);
        assert_eq!(row.compiled_ops, 4);
        assert!(row.jit_retired > 90_000);
    }

    #[test]
    fn alternating_guard_paths_are_not_rerecorded() {
        reset();
        const HEAD: u32 = 0x7000;
        let mut bus = LinearMemoryBus::new(0x1_0000);
        let words = [
            0x4600, // NOT.B D0: alternates Z every iteration
            0x6602, // BNE.S skip
            0x4E71, // opposite-path NOP
            0x5281, // skip: ADDQ.L #1,D1
            0x60F6, // BRA.S head
        ];
        for (index, word) in words.iter().enumerate() {
            bus.write_word(HEAD + index as u32 * 2, *word);
        }

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = HEAD;
        assert_eq!(cpu.run_batch(&mut bus, 100_000, &[0]).instructions, 100_000);

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == HEAD)
            .expect("alternating loop head was profiled");
        assert_eq!(row.recording_attempts, 1);
        assert_eq!(row.adaptive_rerecords, 0);
        assert_eq!(row.compiled_ops, 5);
        assert!(row.guarded_branch_exits > 1_000);
    }

    #[test]
    fn rare_non_self_loop_guard_exit_is_not_rerecorded() {
        reset();
        const HEAD: u32 = 0x8000;
        let mut bus = LinearMemoryBus::new(0x1_0000);
        let words = [
            0x5340, // SUBQ.W #1,D0
            0x6602, // BNE.S common (taken about 99% of entries)
            0x7063, // rare: MOVEQ #99,D0
            0x5281, // common: ADDQ.L #1,D1
            0x51CF, 0x0004, // DBF D7,outer
            0x4E71, // unreachable padding
            0x4E71, // unreachable padding
            0x7E01, // outer: MOVEQ #1,D7
            0x60EC, // BRA.S head
        ];
        for (index, word) in words.iter().enumerate() {
            bus.write_word(HEAD + index as u32 * 2, *word);
        }

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.set_sr(0x2700);
        cpu.pc = HEAD;
        cpu.set_d(0, 100);
        cpu.set_d(7, 1);
        assert_eq!(cpu.run_batch(&mut bus, 50_000, &[0]).instructions, 50_000);

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == HEAD)
            .expect("rare-exit loop head was profiled");
        assert_eq!(row.recording_attempts, 1);
        assert_eq!(row.adaptive_rerecords, 0);
        assert_eq!(row.compiled_ops, 4);
        assert!(row.guarded_branch_exits > 64);
    }

    #[test]
    fn report_ranks_decoded_memory_opcodes_by_execution_count() {
        reset();
        note_decoded_mem(0x1000, 0x20d9);
        note_decoded_mem(0x1002, 0x10dc);
        note_decoded_mem(0x1000, 0x20d9);

        let snapshot = snapshot();
        assert_eq!(snapshot.decoded_mem_ops.len(), 2);
        let report = snapshot.report();
        assert!(report.contains("decoded memory operations: total=3 distinct_opcodes=2"));
        assert!(report.find("20D9").unwrap() < report.find("10DC").unwrap());
        assert!(report.contains("00001000  20D9           2"));
    }

    #[test]
    fn silently_rejected_head_is_reported_instead_of_vanishing() {
        reset();
        for _ in 0..4 {
            note_backward_edge(0x900, CpuType::M68040, true);
        }
        note_silent_rejection(
            0x900,
            CpuType::M68040,
            0x914,
            dummy_prefix(0x900, 9),
            TraceRejectReason::TrapOrException,
        );

        let snapshot = snapshot();
        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0x900)
            .expect("head keeps its row");
        assert_eq!(row.reject_reason, Some(TraceRejectReason::TrapOrException));
        // No blocker means no opcode to support: the head must not acquire
        // projected dispatches it can never deliver.
        assert_eq!(row.blocker_pc, None);
        assert_eq!(row.prefix_ops, 0);
        assert_eq!(row.projected_dispatches(), 0);

        let report = snapshot.report();
        assert!(report.contains("silent rejections"));
        assert!(report.contains("00000900        1        9   00000914 trap-or-exception"));
        // The stranded prefix is visible, which is what makes the exit pc
        // actionable.
        assert!(report.contains("00000900:7000"));
    }

    #[test]
    fn trap_inside_a_loop_body_is_attributed_not_silently_dropped() {
        // The shape that motivated this change: a hot loop whose body
        // reaches a trap, so recording ends with a prefix too short to
        // compile. Before this, the head kept its backward-branch counters
        // but gained no prefix, blocker, or shape entry — indistinguishable
        // in the report from a head that was never hot.
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        bus.write_word(0, 0x4A80); // TST.L D0          <- loop head
        bus.write_word(2, 0x6704); // BEQ.S $0008
        bus.write_word(4, 0x5380); // SUBQ.L #1,D0
        bus.write_word(6, 0x60F8); // BRA.S $0000       (back edge)
        bus.write_word(8, 0xA123); // A-line trap in the loop's exit path

        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        cpu.pc = 0;
        // One iteration makes the head hot; the second is the recording
        // pass, and it is the one that reaches the trap — so the recording
        // never closes the loop.
        cpu.set_d(0, 1);
        let result = cpu.run_batch(&mut bus, 100, &[]);

        assert_eq!(result.exit, BatchExit::AlineTrap { opcode: 0xA123 });
        assert!(!cpu.trace_recording);

        let snapshot = snapshot();
        let rejection = snapshot
            .silent_rejections
            .iter()
            .find(|row| row.start_pc == 0)
            .expect("the trap-ended recording is attributed");
        // The trap outranks the compile-stage reason: it is the bound on
        // how far this head can ever record.
        assert_eq!(rejection.reason, TraceRejectReason::TrapOrException);
        assert_eq!(rejection.exit_pc, 8);
        assert_eq!(rejection.prefix.len(), 2);
        assert_eq!(rejection.prefix[0].opcode, 0x4A80);

        let row = snapshot
            .rows
            .iter()
            .find(|row| row.start_pc == 0)
            .expect("head row");
        assert_eq!(row.reject_reason, Some(TraceRejectReason::TrapOrException));
        assert_eq!(row.compiled_ops, 0);
        assert!(snapshot.report().contains("trap-or-exception"));
    }

    #[test]
    fn silent_rejection_reasons_are_distinguished() {
        reset();
        note_silent_rejection(
            0x100,
            CpuType::M68040,
            0x110,
            dummy_prefix(0x100, 2),
            TraceRejectReason::TooShort,
        );
        note_silent_rejection(
            0x200,
            CpuType::M68040,
            0x230,
            dummy_prefix(0x200, 3),
            TraceRejectReason::LinearMemoryAlu,
        );
        // Same head, same prefix, different outcome: distinct entries.
        note_silent_rejection(
            0x200,
            CpuType::M68040,
            0x230,
            dummy_prefix(0x200, 3),
            TraceRejectReason::LinearMemoryAlu,
        );

        let snapshot = snapshot();
        assert_eq!(snapshot.silent_rejections.len(), 2);
        let linear = snapshot
            .silent_rejections
            .iter()
            .find(|row| row.reason == TraceRejectReason::LinearMemoryAlu)
            .expect("linear-memory-alu entry");
        assert_eq!(linear.recordings, 2);
        assert_eq!(linear.exit_pc, 0x230);
        let report = snapshot.report();
        assert!(report.contains("too-short"));
        assert!(report.contains("linear-memory-alu"));
    }

    #[test]
    fn silent_rejections_are_capped_and_overflow_is_reported() {
        reset();
        for index in 0..(SILENT_REJECTION_CAP as u32 + 3) {
            note_silent_rejection(
                0x100 + index * 2,
                CpuType::M68040,
                0x100 + index * 2 + 8,
                dummy_prefix(0x100, 1),
                TraceRejectReason::Backend,
            );
        }
        let snapshot = snapshot();
        assert_eq!(snapshot.silent_rejections.len(), SILENT_REJECTION_CAP);
        assert_eq!(snapshot.silent_rejection_overflow, 3);
        assert!(
            snapshot
                .report()
                .contains("3 silent rejections dropped past the 4096-shape cap")
        );
    }
}

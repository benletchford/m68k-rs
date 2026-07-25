//! Opt-in trace-JIT opportunity profiling.
//!
//! Enable the `trace-profile` Cargo feature and set `M68K_TRACE_PROFILE=1`
//! to print a report when the CPU thread exits. The normal build contains
//! none of this module or its hot-path hooks.

use super::types::CpuType;
use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;
use std::hash::{BuildHasherDefault, Hasher};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceProfileRow {
    pub start_pc: u32,
    pub cpu_type: CpuType,
    pub backward_hits: u64,
    pub rejected_hits: u64,
    pub recording_attempts: u64,
    pub prefix_ops: u32,
    pub blocker_pc: Option<u32>,
    pub blocker_opcode: Option<u16>,
    pub compiled_ops: u32,
    pub native_calls: u64,
    pub jit_retired: u64,
    pub guarded_branch_exits: u64,
    pub adaptive_rerecords: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedMemProfileRow {
    pub opcode: u16,
    pub executions: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodedMemSiteProfileRow {
    pub pc: u32,
    pub opcode: u16,
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
pub struct TraceProfileSnapshot {
    pub rows: Vec<TraceProfileRow>,
    pub decoded_mem_ops: Vec<DecodedMemProfileRow>,
    pub decoded_mem_sites: Vec<DecodedMemSiteProfileRow>,
    pub backward_hits: u64,
    pub rejected_hits: u64,
    pub native_calls: u64,
    pub jit_retired: u64,
}

impl TraceProfileSnapshot {
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
    compiled_ops: u32,
    native_calls: u64,
    jit_retired: u64,
    guarded_branch_exits: u64,
    adaptive_rerecords: u64,
}

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
    decoded_mem_counts: Box<[u64]>,
    decoded_mem_site_counts: SiteCounts,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            rows: BTreeMap::new(),
            decoded_mem_counts: vec![0; super::op_cache::DECODE_TABLE_SIZE].into_boxed_slice(),
            decoded_mem_site_counts: SiteCounts::default(),
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
        TraceProfileSnapshot {
            backward_hits: rows.iter().map(|row| row.backward_hits).sum(),
            rejected_hits: rows.iter().map(|row| row.rejected_hits).sum(),
            native_calls: rows.iter().map(|row| row.native_calls).sum(),
            jit_retired: rows.iter().map(|row| row.jit_retired).sum(),
            rows,
            decoded_mem_ops,
            decoded_mem_sites,
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

pub fn reset() {
    PROFILE.with_borrow_mut(|profile| profile.0 = Profile::default());
}

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
    prefix_ops: usize,
    blocker_pc: u32,
    blocker_opcode: u16,
) {
    PROFILE.with_borrow_mut(|profile| {
        let row = profile.0.row(start_pc, cpu_type);
        // Keep the longest observed prefix for this trace head. It is the
        // conservative amount of already-supported work stranded behind the
        // blocker; path variation is visible through repeated recordings.
        if prefix_ops as u32 >= row.prefix_ops {
            row.prefix_ops = prefix_ops as u32;
            row.blocker_pc = Some(blocker_pc);
            row.blocker_opcode = Some(blocker_opcode);
        }
    });
}

pub(crate) fn note_compiled(pc: u32, cpu_type: CpuType, ops: usize) {
    PROFILE.with_borrow_mut(|profile| {
        profile.0.row(pc, cpu_type).compiled_ops = ops as u32;
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
    use crate::{AddressBus, CpuCore, LinearMemoryBus};

    #[test]
    fn report_ranks_stranded_dispatches_not_raw_hits() {
        reset();
        note_backward_edge(0x100, CpuType::M68040, true);
        note_blocker(0x100, CpuType::M68040, 2, 0x104, 0x4ead);
        for _ in 0..3 {
            note_backward_edge(0x200, CpuType::M68040, true);
        }
        note_blocker(0x200, CpuType::M68040, 1, 0x202, 0x486d);

        let report = snapshot().report();
        assert!(report.find("00000200").unwrap() < report.find("00000100").unwrap());
    }

    #[test]
    fn rejected_trace_keeps_counting_dynamic_backward_edges() {
        reset();
        let mut bus = LinearMemoryBus::new(0x1000);
        bus.write_word(0, 0x5280); // ADDQ.L #1,D0: traceable prefix
        bus.write_word(2, 0x0640); // ADDI.W #1,D0: untraceable blocker
        bus.write_word(4, 0x0001);
        bus.write_word(6, 0x60F8); // BRA.S $0000

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
        assert_eq!(row.blocker_opcode, Some(0x0640));
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
        #[cfg(not(target_family = "wasm"))]
        assert!(
            row.native_calls > 1,
            "two-op read/write loops retain the measured faster one-pass path"
        );
        #[cfg(target_family = "wasm")]
        assert!(row.native_calls > 0);
        assert!(row.jit_retired > 0);
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
        #[cfg(not(target_family = "wasm"))]
        assert_eq!(row.native_calls, 1);
        #[cfg(target_family = "wasm")]
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
}

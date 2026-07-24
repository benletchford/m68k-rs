//! Opt-in trace-JIT opportunity profiling.
//!
//! Enable the `trace-profile` Cargo feature and set `M68K_TRACE_PROFILE=1`
//! to print a report when the CPU thread exits. The normal build contains
//! none of this module or its hot-path hooks.

use super::types::CpuType;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::fmt::Write;

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
        let _ = writeln!(out, "rank  start_pc  ops      calls      retired avg_ops");
        for (rank, row) in compiled_rows.iter().take(40).enumerate() {
            let average = row.jit_retired as f64 / row.native_calls as f64;
            let _ = writeln!(
                out,
                "{:>4}  {:08X}  {:>3} {:>10} {:>12} {:>7.2}",
                rank + 1,
                row.start_pc,
                row.compiled_ops,
                row.native_calls,
                row.jit_retired,
                average
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
}

#[derive(Default)]
struct Profile {
    rows: BTreeMap<(u32, u32), Row>,
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
            })
            .collect();
        TraceProfileSnapshot {
            backward_hits: rows.iter().map(|row| row.backward_hits).sum(),
            rejected_hits: rows.iter().map(|row| row.rejected_hits).sum(),
            native_calls: rows.iter().map(|row| row.native_calls).sum(),
            jit_retired: rows.iter().map(|row| row.jit_retired).sum(),
            rows,
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
}

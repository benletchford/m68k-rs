//! Head-to-head 68k core comparison: m68k-rs vs Musashi.
//!
//! Methodology
//! -----------
//! Every engine runs the same hand-encoded 68000 workloads over the same flat
//! 16MB mask-and-index memory. Cycle-budgeted engines (Musashi's
//! `m68k_execute`, m68k-rs's cycle-accurate `execute`) get an identical cycle
//! budget; the instruction-budgeted m68k-rs `run_batch` fast path gets the
//! equivalent instruction budget, so all engines retire the same instruction
//! stream.
//!
//! Rather than hardcoding 68000 cycle timings, the harness *calibrates*: it
//! single-steps one loop iteration on each cycle-budgeted engine and records
//! the cycles it charges. Calibration mismatches between engines are reported
//! (they indicate a cycle-accuracy divergence, itself useful data) and exact
//! per-engine instruction counts are recovered from counter registers where
//! the workload provides one.
//!
//! After the timed runs, data registers and a hash of guest memory are
//! compared across engines that retired the same number of instructions; a
//! mismatch fails the run, since numbers from cores that computed different
//! things are meaningless.

mod engine;
mod musashi;
mod native;
mod workloads;

use engine::{Engine, EngineState};
use std::io::Write as _;
use std::time::Instant;
use workloads::{CODE_BASE, Code, Workload};

const WARMUP_DIVISOR: u64 = 20;

struct Measurement {
    engine: &'static str,
    /// Instructions retired in the timed run (exact where the engine or a
    /// counter register provides it, else derived from calibrated cycles).
    instrs: u64,
    instrs_exact: bool,
    median_secs: f64,
    /// Cycles consumed, for cycle-budgeted engines.
    used_cycles: Option<i64>,
    /// Calibrated cycles per workload iteration, for cycle-budgeted engines.
    cycles_per_iter: Option<i64>,
    state: EngineState,
}

impl Measurement {
    fn mips(&self) -> f64 {
        self.instrs as f64 / self.median_secs / 1e6
    }

    fn emulated_mhz(&self) -> Option<f64> {
        self.used_cycles.map(|c| c as f64 / self.median_secs / 1e6)
    }
}

/// Single-step one iteration of the workload and return the cycles the engine
/// charges for it. Runs two iterations and keeps the second, so decode caches
/// are warm and the loop is in steady state.
fn calibrate_cycles_per_iter(e: &mut dyn Engine, w: &Workload) -> i64 {
    e.reset_run(w);
    let mut per_iter = 0;
    for _ in 0..2 {
        per_iter = 0;
        for _ in 0..w.instrs_per_iter {
            per_iter += e.run_cycles(1);
        }
    }
    per_iter
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

fn exact_instrs_from_counter(w: &Workload, state: &EngineState) -> Option<u64> {
    let cr = w.count_reg.as_ref()?;
    let init = w
        .d_init
        .iter()
        .find(|(r, _)| *r == cr.reg)
        .map(|&(_, v)| v)
        .unwrap_or(0);
    let iters = state.d[cr.reg].wrapping_sub(init) as u64;
    Some(iters * w.instrs_per_iter as u64)
}

fn measure(e: &mut dyn Engine, w: &Workload, cycle_budget: i64, reps: usize) -> Measurement {
    e.load_workload(w);

    let (cycles_per_iter, warmup_budget) = if e.is_cycle_budgeted() {
        (
            Some(calibrate_cycles_per_iter(e, w)),
            cycle_budget as u64 / WARMUP_DIVISOR,
        )
    } else {
        (None, w.target_instrs() / WARMUP_DIVISOR)
    };

    // Warmup: fills decode caches and lets the trace JIT compile hot loops.
    e.reset_run(w);
    if e.is_cycle_budgeted() {
        e.run_cycles(warmup_budget.max(400) as i64);
    } else {
        e.run_instructions(warmup_budget.max(100));
    }

    let mut times = Vec::with_capacity(reps);
    let mut used_cycles = None;
    let mut retired = None;
    for _ in 0..reps {
        e.reset_run(w);
        let start = Instant::now();
        if e.is_cycle_budgeted() {
            used_cycles = Some(e.run_cycles(cycle_budget));
        } else {
            retired = Some(e.run_instructions(w.target_instrs()));
        }
        times.push(start.elapsed().as_secs_f64());
    }

    let state = e.state();

    // Best available instruction count for the timed run.
    let (instrs, instrs_exact) = if let Some(n) = retired {
        (n, true)
    } else if let Some(n) = exact_instrs_from_counter(w, &state) {
        (n, true)
    } else {
        let cpi = cycles_per_iter.unwrap() as f64 / w.instrs_per_iter as f64;
        ((used_cycles.unwrap() as f64 / cpi).round() as u64, false)
    };

    Measurement {
        engine: e.name(),
        instrs,
        instrs_exact,
        median_secs: median(times),
        used_cycles,
        cycles_per_iter,
        state,
    }
}

/// Compare final CPU/memory state across engines. Data registers are only
/// comparable between engines that retired the same instruction count;
/// guest memory must match for everyone (all workloads write fixed values
/// to fixed addresses, if they write at all).
fn verify_states(results: &[Measurement]) -> Result<(), String> {
    let base = &results[0];
    for r in &results[1..] {
        if r.state.mem_hash != base.state.mem_hash {
            return Err(format!(
                "memory hash mismatch: {} {:#018x} vs {} {:#018x}",
                base.engine, base.state.mem_hash, r.engine, r.state.mem_hash
            ));
        }
        if r.instrs == base.instrs && r.state.d != base.state.d {
            return Err(format!(
                "data register mismatch after {} instructions: {} {:08x?} vs {} {:08x?}",
                base.instrs, base.engine, base.state.d, r.engine, r.state.d
            ));
        }
    }
    Ok(())
}

fn print_disassembly(w: &Workload) {
    println!("\n{}:", w.name);
    match w.code {
        Code::Fill(op) => {
            let (text, _) = m68k::dasm::disassemble(0, op, m68k::CpuType::M68000);
            println!("  ${op:04X} x 8M    {text}   (fills all 16MB)");
        }
        Code::Words(words) => {
            let mut i = 0;
            while i < words.len() {
                let pc = CODE_BASE + (i as u32) * 2;
                let (text, size) = m68k::dasm::disassemble(pc, words[i], m68k::CpuType::M68000);
                let nwords = (size as usize / 2).max(1).min(words.len() - i);
                let raw: Vec<String> = words[i..i + nwords]
                    .iter()
                    .map(|w| format!("{w:04X}"))
                    .collect();
                println!("  ${pc:06X}  {:<14}  {text}", raw.join(" "));
                i += nwords;
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let disasm = args.iter().any(|a| a == "--disasm");
    let reps = args
        .iter()
        .position(|a| a == "--reps")
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(3usize);

    let workloads = workloads::all();

    if disasm {
        for w in &workloads {
            print_disassembly(w);
        }
        return;
    }

    println!("m68k-rs vs Musashi (68000, flat 16MB memory, median of {reps} runs)\n");
    println!(
        "{:<15} {:<19} {:>9} {:>12} {:>10}  state",
        "workload", "engine", "MIPS", "vs Musashi", "emul. MHz"
    );

    let mut all_ok = true;
    let mut summary: Vec<(String, Vec<(String, f64)>)> = Vec::new();

    for w in &workloads {
        // Musashi is the timing reference: its calibrated cycles/iteration
        // fixes the shared cycle budget.
        let mut musashi = musashi::Musashi::new();
        musashi.load_workload(w);
        let ref_cpi = calibrate_cycles_per_iter(&mut musashi, w);
        let cycle_budget = ref_cpi * w.target_iters as i64;
        assert!(
            cycle_budget <= i32::MAX as i64,
            "cycle budget for {} exceeds i32",
            w.name
        );

        let mut interp = native::NativeInterp::new();
        let mut batch = native::NativeBatch::new();
        let engines: Vec<&mut dyn Engine> = vec![&mut musashi, &mut interp, &mut batch];

        let mut results = Vec::new();
        for e in engines {
            let m = measure(e, w, cycle_budget, reps);
            print!(".");
            std::io::stdout().flush().unwrap();
            results.push(m);
        }
        print!("\r");

        let gate = verify_states(&results);
        if let Err(ref msg) = gate {
            all_ok = false;
            eprintln!("STATE MISMATCH in {}: {msg}", w.name);
        }

        let musashi_mips = results[0].mips();
        let mut row = Vec::new();
        for (i, r) in results.iter().enumerate() {
            let ratio = if i == 0 {
                "1.00x".to_string()
            } else {
                format!("{:.2}x", r.mips() / musashi_mips)
            };
            let mhz = r
                .emulated_mhz()
                .map(|m| format!("{m:.0}"))
                .unwrap_or_else(|| "-".into());
            let approx = if r.instrs_exact { "" } else { "~" };
            println!(
                "{:<15} {:<19} {approx:>1}{:>8.1} {:>12} {:>10}  {}",
                if i == 0 { w.name } else { "" },
                r.engine,
                r.mips(),
                ratio,
                mhz,
                if gate.is_ok() { "ok" } else { "MISMATCH" },
            );
            row.push((r.engine.to_string(), r.mips()));
        }

        // Surface cycle-accounting divergence between the two cycle-budgeted
        // cores: same budget + different cycles/iteration = different timing
        // model, and the MIPS ratio above still holds but iteration counts
        // differ slightly.
        let cpis: Vec<(&str, i64)> = results
            .iter()
            .filter_map(|r| r.cycles_per_iter.map(|c| (r.engine, c)))
            .collect();
        if cpis.windows(2).any(|p| p[0].1 != p[1].1) {
            let detail: Vec<String> = cpis.iter().map(|(n, c)| format!("{n}: {c}")).collect();
            println!(
                "                  note: cycles/iteration differ ({})",
                detail.join(", ")
            );
        }

        summary.push((w.name.to_string(), row));
    }

    // Markdown summary for pasting into docs.
    println!("\n---\n");
    let engine_names: Vec<&str> = summary[0].1.iter().map(|(n, _)| n.as_str()).collect();
    println!(
        "| workload | {} |",
        engine_names
            .iter()
            .map(|n| format!("{n} (MIPS)"))
            .collect::<Vec<_>>()
            .join(" | ")
    );
    println!("|---|{}", "---:|".repeat(engine_names.len()));
    for (name, row) in &summary {
        println!(
            "| {name} | {} |",
            row.iter()
                .map(|(_, mips)| format!("{mips:.1}"))
                .collect::<Vec<_>>()
                .join(" | ")
        );
    }

    if !all_ok {
        eprintln!("\nFAILED: engines disagreed on final state (see above)");
        std::process::exit(1);
    }
}

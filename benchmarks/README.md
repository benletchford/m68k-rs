# m68k-rs benchmarks

Head-to-head comparison of m68k-rs against [Musashi](https://github.com/kstenerud/Musashi)
(Karl Stenerud's C core, used by MAME and many Genesis/arcade emulators — the
de facto reference 68k emulator). Musashi 4.10 is vendored under
`vendor/musashi/` (see `VENDOR.md`) and built by `build.rs` exactly as its own
Makefile does, including running the `m68kmake` opcode generator.

This crate is standalone (its own workspace) and is not part of the published
`m68k` package.

## Running

```sh
cd benchmarks
cargo run --release                 # the comparison (median of 3 runs each)
cargo run --release -- --reps 5     # more repetitions
cargo run --release -- --disasm     # print the workloads' disassembly and exit
```

## Methodology

- **Same memory model everywhere.** All engines run over a flat 16MB
  mask-and-index buffer (the 68000's 24-bit address space). Musashi reads it
  through its usual extern-function memory interface (`shim/bench_shim.c`) —
  the same call-per-access shape it has in real deployments. m68k-rs runs in
  two configurations:
  - `m68k-rs execute` — the cycle-accurate interpreter on a plain
    `AddressBus` with no fastmem window, the closest analogue to Musashi's
    `m68k_execute`;
  - `m68k-rs run_batch` — the instruction-budgeted fast path
    (decoded-op cache + trace JIT) on a `LinearMemoryBus` fastmem window.
- **Same work everywhere.** Cycle-budgeted engines get an identical cycle
  budget; `run_batch` gets the equivalent instruction budget. Workloads are
  hand-encoded 68000 loops (see `src/workloads.rs`; verify with `--disasm`).
- **Calibrated, not assumed, timings.** The harness single-steps one loop
  iteration on each cycle-budgeted engine to learn the cycles it charges,
  instead of hardcoding a 68000 timing table. When the two cores charge
  different cycle counts for the same iteration the run prints a
  `cycles/iteration differ` note — that's a cycle-accuracy divergence worth
  investigating in its own right. Instruction counts are recovered exactly
  from counter registers where the workload has one (marked `~` in the output
  when they had to be estimated from cycles instead).
- **Verified before trusted.** After the timed runs, data registers (for
  engines that retired the same instruction count) and an FNV-1a hash of all
  guest memory are compared across engines. Any mismatch fails the run:
  numbers from cores that computed different things are meaningless.
- MIPS = retired 68000 instructions per wall-clock second. "emul. MHz" is the
  emulated-cycle rate — how fast a real 68000 would have to be clocked to
  match (a stock Amiga 500/Sega Genesis 68000 runs at ~7.6 MHz).

## Example results

One machine, one snapshot — run it on your own hardware. These numbers are
from a shared cloud container (x86-64, gcc 13 `-O3` for Musashi, rustc
`opt-level=3` for m68k-rs), median of 3 runs:

| workload | Musashi (MIPS) | m68k-rs execute (MIPS) | m68k-rs run_batch (MIPS) |
|---|---:|---:|---:|
| linear NOP | 291.3 | 233.0 | 231.9 |
| linear MOVEQ | 115.8 | 190.3 | 201.7 |
| linear ADDQ.L | 120.0 | 117.2 | 116.8 |
| loop ADDQ/BRA | 172.7 | 503.7 | 500.9 |
| loop TST/BNE | 196.0 | 538.2 | 542.6 |
| loop reg mix | 147.1 | 777.2 | 775.5 |
| memcpy 4KB | 110.1 | 298.9 | 301.7 |
| call/return | 145.2 | 582.1 | 584.2 |

Hot loops — memory copies and call/return included — run 2.7–5.3x Musashi
(the trace JIT compiles them, memory operands and one level of BSR/RTS
included, and iterates natively); dense linear code is at parity or better,
with pure-NOP dispatch the one remaining synthetic gap (0.8x). When this
harness was first built the picture was very different (m68k-rs at
0.25–0.5x on nearly everything) — the harness directly drove the
optimizations that closed the gap, so keep it honest when changing the fast
paths: the state gates catch divergence immediately.

The run also cross-checks cycle accounting between the two cores. This
originally exposed large 68000 timing gaps in m68k-rs (e.g. `ADDQ.L #1,Dn`
charged at 4 cycles instead of 8, memory effective-address time not charged
at all); those have since been fixed against the SingleStepTests
real-hardware cycle counts, which the test suite now enforces. The remaining
`cycles/iteration differ` note is Musashi undercharging `ADD.L Dn,Dn` at 6
cycles where the M68000UM and SingleStepTests say 8.

## Extending

- **More emulators.** The `Engine` trait (`src/engine.rs`) is the only
  integration surface: adapt a core, add it to the engine list in `main.rs`.
  Natural candidates: Moira (C++, cycle-exact, vAmiga), the `r68k` and
  `m68000` Rust crates, the UAE core.
- **Real programs.** With an `m68k-elf-gcc` cross-compiler available,
  Dhrystone/CoreMark images (run until a magic TRAP) would make good headline
  workloads alongside these microbenchmarks.
- **FPU/68020+ workloads** — currently everything runs as a 68000.

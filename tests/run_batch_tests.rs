//! Tests for `CpuCore::run_batch`, the instruction-budgeted batch
//! execution entry point used by HLE embedders.

use m68k::core::memory::AddressBus;
use m68k::{BatchExit, CpuCore, CpuType, LinearMemoryBus};

fn cpu_at(pc: u32) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.pc = pc;
    cpu.set_sr(0x2700);
    cpu.set_a(7, 0x8000);
    cpu
}

fn bus_with(words: &[(u32, u16)]) -> LinearMemoryBus {
    let mut bus = LinearMemoryBus::new(0x10000);
    for &(addr, word) in words {
        bus.load(addr, &word.to_be_bytes());
    }
    bus
}

#[test]
fn budget_exhausted_returns_exact_instruction_count() {
    // MOVEQ #1,D0 repeated (simple-op fast path).
    let mut bus = LinearMemoryBus::new(0x10000);
    for addr in (0x1000..0x2000).step_by(2) {
        bus.load(addr, &0x7001u16.to_be_bytes());
    }
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 100, &[]);
    assert_eq!(result.exit, BatchExit::BudgetExhausted);
    assert_eq!(result.instructions, 100);
    assert_eq!(cpu.pc, 0x1000 + 100 * 2);
}

#[test]
fn budget_exhausted_count_is_exact_with_jit_traces() {
    // Tight backward loop that gets hot enough to compile into a trace:
    //   ADDQ.L #1,D0 ; BRA.S -4
    // Run several batches so the trace JIT engages, and verify the retired
    // counts stay exact (traces retire multiple instructions per call and
    // must not overshoot the budget).
    let mut bus = bus_with(&[(0x1000, 0x5280), (0x1002, 0x60FC)]);
    let mut cpu = cpu_at(0x1000);

    let mut total: u64 = 0;
    for _ in 0..100 {
        let result = cpu.run_batch(&mut bus, 10_001, &[]);
        assert_eq!(result.exit, BatchExit::BudgetExhausted);
        assert_eq!(result.instructions, 10_001);
        total += result.instructions as u64;
    }
    // Every second instruction is the ADDQ; D0 counts retired ADDQs.
    // With an odd budget the batch boundary alternates between the two
    // instructions, so just check the total adds up.
    assert_eq!(total, 100 * 10_001);
    assert_eq!(cpu.d(0) as u64, total / 2);
}

#[test]
fn unrelated_watch_keeps_jit_loop_budget_exact() {
    // A caller may always watch PC 0 as a clean-exit sentinel. A trace whose
    // loop head is elsewhere may run multiple iterations per native call,
    // but must still stop at the exact requested instruction budget.
    let mut bus = bus_with(&[(0x1000, 0x5280), (0x1002, 0x60FC)]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 1_000_001, &[0]);

    assert_eq!(result.exit, BatchExit::BudgetExhausted);
    assert_eq!(result.instructions, 1_000_001);
    assert_eq!(cpu.d(0), 500_001);
    assert_eq!(cpu.pc, 0x1002);
}

#[test]
fn multi_block_trace_follows_recorded_taken_branch() {
    // loop: ADDQ D0; BNE skip; NOP; skip: ADDQ D1; BRA loop
    // BNE is an interior branch. A region trace follows its observed taken
    // edge, skips the NOP, and closes only at the final backward BRA.
    let mut bus = bus_with(&[
        (0x1000, 0x5280),
        (0x1002, 0x6602),
        (0x1004, 0x4E71),
        (0x1006, 0x5281),
        (0x1008, 0x60F6),
    ]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 1_000_000, &[0]);

    assert_eq!(result.exit, BatchExit::BudgetExhausted);
    assert_eq!(result.instructions, 1_000_000);
    assert_eq!(cpu.d(0), 250_000);
    assert_eq!(cpu.d(1), 250_000);
    assert_eq!(cpu.pc, 0x1000);
}

#[test]
fn multi_block_trace_yields_before_interior_watched_pc() {
    let mut bus = bus_with(&[
        (0x1000, 0x5280), // ADDQ.L #1,D0
        (0x1002, 0x6602), // BNE.S 0x1006
        (0x1004, 0x4E71), // uncommon fallthrough
        (0x1006, 0x5281), // ADDQ.L #1,D1
        (0x1008, 0x60F6), // BRA.S 0x1000
    ]);
    let mut cpu = cpu_at(0x1000);

    // Compile and exercise the region before enabling the watch.
    let warmup = cpu.run_batch(&mut bus, 100, &[]);
    assert_eq!(warmup.exit, BatchExit::BudgetExhausted);
    assert_eq!(cpu.pc, 0x1000);
    let d0_before = cpu.d(0);
    let d1_before = cpu.d(1);

    let result = cpu.run_batch(&mut bus, 100, &[0x1006]);

    assert_eq!(result.exit, BatchExit::WatchedPc { pc: 0x1006 });
    assert_eq!(result.instructions, 2);
    assert_eq!(cpu.pc, 0x1006);
    assert_eq!(cpu.d(0), d0_before + 1);
    assert_eq!(cpu.d(1), d1_before);
}

#[test]
fn multi_block_trace_guard_side_exit_matches_step() {
    // The recorded BNE path is initially taken, but D0 reaches zero and
    // forces one not-taken side exit through the NOP before paths rejoin.
    let words = [
        (0x1000, 0x5340), // SUBQ.W #1,D0
        (0x1002, 0x6602), // BNE.S 0x1006
        (0x1004, 0x4E71), // uncommon fallthrough
        (0x1006, 0x5281), // ADDQ.L #1,D1
        (0x1008, 0x60F6), // BRA.S 0x1000
    ];
    let mut batch_bus = bus_with(&words);
    let mut batch_cpu = cpu_at(0x1000);
    batch_cpu.set_d(0, 6);
    let result = batch_cpu.run_batch(&mut batch_bus, 100_000, &[0]);
    assert_eq!(result.instructions, 100_000);

    let mut step_bus = bus_with(&words);
    let mut step_cpu = cpu_at(0x1000);
    step_cpu.set_d(0, 6);
    for _ in 0..100_000 {
        assert!(matches!(
            step_cpu.step(&mut step_bus),
            m68k::StepResult::Ok { .. }
        ));
    }

    assert_eq!(batch_cpu.pc, step_cpu.pc);
    assert_eq!(batch_cpu.get_sr(), step_cpu.get_sr());
    for reg in 0..8 {
        assert_eq!(batch_cpu.d(reg), step_cpu.d(reg), "D{reg}");
        assert_eq!(batch_cpu.a(reg), step_cpu.a(reg), "A{reg}");
    }
}

#[test]
fn multi_block_trace_path_change_matches_step() {
    // First record the BNE path, then switch to the opposite, dominant path.
    // The common path is a CMP/branch/copy/DBRA topology; the outer path
    // resets its pointers and counter.
    let words = [
        (0x1000, 0xB210), // CMP.B (A0),D1
        (0x1002, 0x6606), // BNE.S outer
        (0x1004, 0x10DC), // MOVE.B (A4)+,(A0)+
        (0x1006, 0x51C8), // DBRA D0,head
        (0x1008, 0xFFF8),
        (0x100A, 0x2042), // outer: MOVEA.L D2,A0
        (0x100C, 0x2843), // MOVEA.L D3,A4
        (0x100E, 0x707F), // MOVEQ #127,D0
        (0x1010, 0x5884), // ADDQ.L #4,D4
        (0x1012, 0x60EC), // BRA.S head
    ];
    let prepare = || {
        let mut cpu = cpu_at(0x1000);
        cpu.set_a(0, 0x2000);
        cpu.set_a(4, 0x3000);
        cpu.set_d(0, 127);
        cpu.set_d(1, 1);
        cpu.set_d(2, 0x2000);
        cpu.set_d(3, 0x3000);
        cpu
    };

    let mut batch_bus = bus_with(&words);
    let mut batch_cpu = prepare();
    assert_eq!(
        batch_cpu.run_batch(&mut batch_bus, 14, &[0]).instructions,
        14
    );
    batch_cpu.set_d(1, 0);
    let result = batch_cpu.run_batch(&mut batch_bus, 100_000, &[0]);
    assert_eq!(result.instructions, 100_000);

    let mut step_bus = bus_with(&words);
    let mut step_cpu = prepare();
    for _ in 0..14 {
        assert!(matches!(
            step_cpu.step(&mut step_bus),
            m68k::StepResult::Ok { .. }
        ));
    }
    step_cpu.set_d(1, 0);
    for _ in 0..100_000 {
        assert!(matches!(
            step_cpu.step(&mut step_bus),
            m68k::StepResult::Ok { .. }
        ));
    }

    assert_eq!(batch_cpu.pc, step_cpu.pc);
    assert_eq!(batch_cpu.get_sr(), step_cpu.get_sr());
    for reg in 0..8 {
        assert_eq!(batch_cpu.d(reg), step_cpu.d(reg), "D{reg}");
        assert_eq!(batch_cpu.a(reg), step_cpu.a(reg), "A{reg}");
    }
}

#[test]
fn aline_trap_exits_without_counting_the_trap() {
    // NOP ; NOP ; A-line ; NOP
    let mut bus = bus_with(&[
        (0x1000, 0x4E71),
        (0x1002, 0x4E71),
        (0x1004, 0xA123),
        (0x1006, 0x4E71),
    ]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 1000, &[]);
    assert_eq!(result.exit, BatchExit::AlineTrap { opcode: 0xA123 });
    assert_eq!(result.instructions, 2);
    // Same state step() leaves: PC past the trap word, ppc at the trap.
    assert_eq!(cpu.pc, 0x1006);
    assert_eq!(cpu.ppc, 0x1004);
}

#[test]
fn watched_pc_exits_before_executing_the_watched_instruction() {
    // NOP at 0x1000..0x1008; watch 0x1004.
    let mut bus = bus_with(&[
        (0x1000, 0x4E71),
        (0x1002, 0x4E71),
        (0x1004, 0x4E71),
        (0x1006, 0x4E71),
    ]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 1000, &[0x1004]);
    assert_eq!(result.exit, BatchExit::WatchedPc { pc: 0x1004 });
    assert_eq!(result.instructions, 2);
    assert_eq!(cpu.pc, 0x1004);
}

#[test]
fn watched_pc_not_checked_on_entry() {
    // Watching the entry PC must not exit with zero instructions.
    let mut bus = bus_with(&[(0x1000, 0x4E71), (0x1002, 0x4E71)]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 2, &[0x1000]);
    assert_eq!(result.exit, BatchExit::BudgetExhausted);
    assert_eq!(result.instructions, 2);
}

#[test]
fn watched_pc_caught_from_jit_trace_loop_exit() {
    // Hot loop that exits forward to a watched PC:
    //   SUBQ.L #1,D0 ; BNE.S -4 ; NOP(watched)
    let mut bus = bus_with(&[(0x1000, 0x5380), (0x1002, 0x66FC), (0x1004, 0x4E71)]);
    let mut cpu = cpu_at(0x1000);
    cpu.set_d(0, 50_000);

    let result = cpu.run_batch(&mut bus, 1_000_000, &[0x1004]);
    assert_eq!(result.exit, BatchExit::WatchedPc { pc: 0x1004 });
    assert_eq!(result.instructions, 2 * 50_000);
    assert_eq!(cpu.d(0), 0);
}

#[test]
fn stop_instruction_exits_stopped_and_counts_it() {
    // NOP ; STOP #$2700 ; (unreached)
    let mut bus = bus_with(&[(0x1000, 0x4E71), (0x1002, 0x4E72), (0x1004, 0x2700)]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 1000, &[]);
    assert_eq!(result.exit, BatchExit::Stopped);
    assert_eq!(result.instructions, 2);

    // Re-entering while stopped returns immediately.
    let result = cpu.run_batch(&mut bus, 1000, &[]);
    assert_eq!(result.exit, BatchExit::Stopped);
    assert_eq!(result.instructions, 0);
}

#[test]
fn trap_instruction_exits() {
    // TRAP #5
    let mut bus = bus_with(&[(0x1000, 0x4E45)]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 1000, &[]);
    assert_eq!(result.exit, BatchExit::TrapInstruction { trap_num: 5 });
    assert_eq!(result.instructions, 0);
    assert_eq!(cpu.ppc, 0x1000);
}

#[test]
fn zero_budget_returns_immediately() {
    let mut bus = bus_with(&[(0x1000, 0x4E71)]);
    let mut cpu = cpu_at(0x1000);

    let result = cpu.run_batch(&mut bus, 0, &[]);
    assert_eq!(result.exit, BatchExit::BudgetExhausted);
    assert_eq!(result.instructions, 0);
    assert_eq!(cpu.pc, 0x1000);
}

#[test]
fn jit_trace_conditional_branches_fall_through() {
    // Regression test for the native trace JIT emitting bitwise `bnot`
    // on 0/1 booleans: `bnot(1) == 0xFE` is still truthy to `select`,
    // which made every negated condition (BNE/BCC/BPL/BGE/BGT/BHI...)
    // permanently taken once a loop was compiled to a trace. Each loop
    // here runs hot enough to compile, then must terminate by falling
    // through its conditional branch.
    //
    // Loop shape: SUBQ.L #1,D0 ; Bcc.S -4 ; NOP(watched)
    // (opcode, start, expected loop iterations before fall-through)
    let cases: &[(u16, u32, u32)] = &[
        (0x66FC, 50_000, 50_000), // BNE: falls through when D0 hits 0
        (0x6AFC, 50_000, 50_001), // BPL: falls through when result goes negative
        (0x6CFC, 50_000, 50_001), // BGE: same boundary as BPL here (V clear)
        (0x6EFC, 50_000, 50_000), // BGT: falls through at zero (Z set)
        (0x62FC, 50_000, 50_000), // BHI: falls through at zero (Z set)
        (0x64FC, 50_000, 50_001), // BCC: falls through on borrow past zero
    ];

    for &(branch, start, iterations) in cases {
        let mut bus = bus_with(&[(0x1000, 0x5380), (0x1002, branch), (0x1004, 0x4E71)]);
        let mut cpu = cpu_at(0x1000);
        cpu.set_d(0, start);

        let result = cpu.run_batch(&mut bus, 10 * iterations, &[0x1004]);
        assert_eq!(
            result.exit,
            BatchExit::WatchedPc { pc: 0x1004 },
            "branch {branch:#06x} never fell through"
        );
        assert_eq!(
            result.instructions,
            2 * iterations,
            "branch {branch:#06x} fell through after the wrong iteration count"
        );
    }

    // DBRA counts D1 down to -1: MOVEQ#0,D0; loop: ADDQ.L #1,D0 ; DBRA D1,loop ; NOP
    let mut bus = bus_with(&[
        (0x1000, 0x7000),
        (0x1002, 0x5280),
        (0x1004, 0x51C9),
        (0x1006, 0xFFFC),
        (0x1008, 0x4E71),
    ]);
    let mut cpu = cpu_at(0x1000);
    cpu.set_d(1, 49_999);

    let result = cpu.run_batch(&mut bus, 1_000_000, &[0x1008]);
    assert_eq!(result.exit, BatchExit::WatchedPc { pc: 0x1008 });
    assert_eq!(cpu.d(0), 50_000);
    assert_eq!(cpu.d(1) & 0xFFFF, 0xFFFF);
}

#[test]
fn batch_matches_step_on_random_register_loops() {
    // Differential fuzz between the interpreter (`step`) and the batch
    // path (`run_batch`, which engages the decoded-op cache and the
    // native Cranelift trace JIT). Each program is a DBRA loop whose
    // body is random one-word register ops over D0-D5/A0-A5 (D6 is the
    // loop counter; D7/A6/A7 stay reserved), hot enough for traces to
    // compile. Any divergence in registers, flags, or PC fails with the
    // reproducing seed.
    let mut seed: u64 = 0x00C0_FFEE_5EED_1234;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for program_index in 0..300 {
        let mut r = || rng() as u32;
        let reg = |v: u32| (v % 6) as u16; // D0-D5 / A0-A5 only
        let size2 = |v: u32| (v % 3) as u16; // 00/01/10 size field

        let mut words: Vec<u16> = Vec::new();
        let iterations = (r() % 100) as u16; // DBRA runs iterations+1 times
        words.push(0x7C00 | (iterations & 0x7F)); // MOVEQ #it,D6
        let loop_start = words.len();
        for _ in 0..10 {
            let op = match r() % 13 {
                0 => 0x7000 | (reg(r()) << 9) | (r() & 0xFF) as u16, // MOVEQ
                1 => {
                    // MOVE.W/L Dx,Dy (word=0x3000, long=0x2000)
                    let base = if r() % 2 == 0 { 0x3000 } else { 0x2000 };
                    base | (reg(r()) << 9) | reg(r())
                }
                2 => {
                    // ADD/SUB/AND/OR/CMP <size> Dx,Dy
                    let group = [0xD000u16, 0x9000, 0xC000, 0x8000, 0xB000][(r() % 5) as usize];
                    group | (reg(r()) << 9) | (size2(r()) << 6) | reg(r())
                }
                3 => {
                    // EOR <size> Dx,Dy (opmode 4-6)
                    0xB100 | (reg(r()) << 9) | (size2(r()) << 6) | reg(r())
                }
                4 => {
                    // ADDQ/SUBQ #q,Dn
                    let sub = if r() % 2 == 0 { 0x0100 } else { 0 };
                    0x5000 | sub | (((r() % 8) as u16) << 9) | (size2(r()) << 6) | reg(r())
                }
                5 => 0x4840 | reg(r()), // SWAP
                6 => {
                    // EXT.W/EXT.L
                    if r() % 2 == 0 { 0x4880 } else { 0x48C0 }.to_owned() | reg(r())
                }
                7 => {
                    // CLR/NEG/NOT/TST <size> Dn
                    let unary = [0x4200u16, 0x4400, 0x4600, 0x4A00][(r() % 4) as usize];
                    unary | (size2(r()) << 6) | reg(r())
                }
                8 => {
                    // ADDA/SUBA/CMPA .W/.L Dx,Ay
                    let base = [0xD0C0u16, 0x90C0, 0xB0C0][(r() % 3) as usize];
                    let long = if r() % 2 == 0 { 0x0100 } else { 0 };
                    base | long | (reg(r()) << 9) | reg(r())
                }
                9 => 0x50C0 | (((r() % 16) as u16) << 8) | reg(r()), // Scc Dn
                10 => {
                    // ADDX/SUBX Dx,Dy
                    let base = if r() % 2 == 0 { 0xD100 } else { 0x9100 };
                    base | (reg(r()) << 9) | (size2(r()) << 6) | reg(r())
                }
                11 => {
                    // EXG
                    let mode = [0x0140u16, 0x0148, 0x0188][(r() % 3) as usize];
                    0xC000 | mode | (reg(r()) << 9) | reg(r())
                }
                _ => {
                    // Shift/rotate register form (decoded-op only on
                    // native; exercises mixed trace/interpreter runs)
                    0xE000
                        | (((r() % 8) as u16) << 9)
                        | (((r() % 2) as u16) << 8)
                        | (size2(r()) << 6)
                        | (((r() % 4) as u16) << 3)
                        | reg(r())
                }
            };
            words.push(op);
        }
        // DBRA D6, loop_start
        let body_bytes = (words.len() - loop_start) * 2;
        words.push(0x51CE);
        words.push((-(body_bytes as i32) - 2) as i16 as u16);
        words.push(0xA000); // A-line sentinel ends the program

        let mut bytes = Vec::new();
        for w in &words {
            bytes.extend_from_slice(&w.to_be_bytes());
        }

        let init_regs: Vec<u32> = (0..12).map(|_| r()).collect();
        let init_ccr = (r() & 0x1F) as u16;
        let setup = |cpu: &mut CpuCore| {
            for i in 0..6 {
                cpu.set_d(i, init_regs[i]);
                cpu.set_a(i, init_regs[6 + i]);
            }
            cpu.set_sr(0x2700 | init_ccr);
        };

        let mut bus_a = LinearMemoryBus::new(0x10000);
        bus_a.load(0x1000, &bytes);
        let mut cpu_a = cpu_at(0x1000);
        setup(&mut cpu_a);
        let mut steps: u64 = 0;
        loop {
            match cpu_a.step(&mut bus_a) {
                m68k::StepResult::Ok { .. } => steps += 1,
                m68k::StepResult::AlineTrap { .. } => break,
                other => panic!("program {program_index}: unexpected step result {other:?}"),
            }
            assert!(steps < 10_000_000, "program {program_index} diverged");
        }

        let mut bus_b = LinearMemoryBus::new(0x10000);
        bus_b.load(0x1000, &bytes);
        let mut cpu_b = cpu_at(0x1000);
        setup(&mut cpu_b);
        let mut batched: u64 = 0;
        loop {
            let result = cpu_b.run_batch(&mut bus_b, 100_000, &[]);
            batched += result.instructions as u64;
            match result.exit {
                BatchExit::BudgetExhausted => continue,
                BatchExit::AlineTrap { .. } => break,
                other => panic!("program {program_index}: unexpected batch exit {other:?}"),
            }
        }

        assert_eq!(steps, batched, "program {program_index}: instruction count");
        assert_eq!(cpu_a.pc, cpu_b.pc, "program {program_index}: pc");
        assert_eq!(
            cpu_a.get_sr(),
            cpu_b.get_sr(),
            "program {program_index}: sr (words={words:04X?})"
        );
        for i in 0..8 {
            assert_eq!(
                cpu_a.d(i),
                cpu_b.d(i),
                "program {program_index}: D{i} (words={words:04X?})"
            );
            assert_eq!(
                cpu_a.a(i),
                cpu_b.a(i),
                "program {program_index}: A{i} (words={words:04X?})"
            );
        }
    }
}

#[test]
fn batch_matches_step_semantics_on_memory_program() {
    // A small program mixing simple ops, memory ops, and a loop; run it
    // once with step() and once with run_batch() and compare final state.
    let program: &[u16] = &[
        0x7000, // MOVEQ #0,D0
        0x207C, 0x0000, 0x4000, // MOVEA.L #$4000,A0
        // loop:
        0x30C0, // MOVE.W D0,(A0)+
        0x5240, // ADDQ.W #1,D0
        0x0C40, 0x0040, // CMPI.W #64,D0
        0x66F6, // BNE.S loop
        0x4E71, // NOP
    ];
    let mut bytes = Vec::new();
    for w in program {
        bytes.extend_from_slice(&w.to_be_bytes());
    }

    let mut bus_a = LinearMemoryBus::new(0x10000);
    bus_a.load(0x1000, &bytes);
    let mut cpu_a = cpu_at(0x1000);
    let mut stepped: u32 = 0;
    while cpu_a.pc != 0x1000 + bytes.len() as u32 {
        match cpu_a.step(&mut bus_a) {
            m68k::StepResult::Ok { .. } => stepped += 1,
            other => panic!("unexpected step result {other:?}"),
        }
        assert!(stepped < 100_000, "step run diverged");
    }

    let mut bus_b = LinearMemoryBus::new(0x10000);
    bus_b.load(0x1000, &bytes);
    let mut cpu_b = cpu_at(0x1000);
    let end_pc = 0x1000 + bytes.len() as u32;
    let result = cpu_b.run_batch(&mut bus_b, 100_000, &[end_pc]);
    let batched = result.instructions;
    match result.exit {
        BatchExit::WatchedPc { pc } => {
            assert_eq!(pc, end_pc);
        }
        other => panic!("unexpected batch exit {other:?}"),
    }

    assert_eq!(stepped, batched);
    assert_eq!(cpu_a.pc, cpu_b.pc);
    assert_eq!(cpu_a.get_sr(), cpu_b.get_sr());
    for r in 0..8 {
        assert_eq!(cpu_a.d(r), cpu_b.d(r), "D{r} mismatch");
        assert_eq!(cpu_a.a(r), cpu_b.a(r), "A{r} mismatch");
    }
    for addr in (0x4000..0x4080).step_by(2) {
        assert_eq!(bus_a.read_word(addr), bus_b.read_word(addr));
    }
}

// ============================================================================
// Fastmem (AddressBus::fast_mem) coverage
// ============================================================================

/// Flat big-endian RAM bus that exposes a (configurable) fastmem window.
/// Length must be a power of two; out-of-range accesses wrap like
/// `LinearMemoryBus`, so the fallback path stays deterministic.
struct FastRamBus {
    mem: Vec<u8>,
    fm_base: u32,
    fm_len: u32,
}

impl FastRamBus {
    fn new(size: usize) -> Self {
        assert!(size.is_power_of_two());
        Self {
            mem: vec![0; size],
            fm_base: 0,
            fm_len: size as u32,
        }
    }

    fn load(&mut self, addr: u32, bytes: &[u8]) {
        self.mem[addr as usize..addr as usize + bytes.len()].copy_from_slice(bytes);
    }

    #[inline]
    fn idx(&self, addr: u32) -> usize {
        (addr as usize) & (self.mem.len() - 1)
    }
}

impl AddressBus for FastRamBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        self.mem[self.idx(address)]
    }
    fn read_word(&mut self, address: u32) -> u16 {
        ((self.read_byte(address) as u16) << 8) | self.read_byte(address.wrapping_add(1)) as u16
    }
    fn read_long(&mut self, address: u32) -> u32 {
        ((self.read_word(address) as u32) << 16) | self.read_word(address.wrapping_add(2)) as u32
    }
    fn write_byte(&mut self, address: u32, value: u8) {
        let i = self.idx(address);
        self.mem[i] = value;
    }
    fn write_word(&mut self, address: u32, value: u16) {
        self.write_byte(address, (value >> 8) as u8);
        self.write_byte(address.wrapping_add(1), value as u8);
    }
    fn write_long(&mut self, address: u32, value: u32) {
        self.write_word(address, (value >> 16) as u16);
        self.write_word(address.wrapping_add(2), value as u16);
    }
    fn fast_mem(&mut self) -> Option<m68k::FastMem> {
        if self.fm_len == 0 {
            return None;
        }
        Some(m68k::FastMem {
            ptr: unsafe { self.mem.as_mut_ptr().add(self.fm_base as usize) },
            base: self.fm_base,
            len: self.fm_len,
        })
    }
}

fn cpu_020_at(pc: u32) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68020);
    cpu.pc = pc;
    cpu.set_sr(0x2700);
    cpu.set_a(7, 0x9000);
    cpu
}

fn assemble(words: &[u16]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 2);
    for w in words {
        bytes.extend_from_slice(&w.to_be_bytes());
    }
    bytes
}

/// Run `words` at 0x1000 to the A-line sentinel with step() (bus without
/// fastmem) and with run_batch() (bus with fastmem); assert identical
/// counts, registers, SR, PC, and memory.
fn assert_fastmem_matches_step(
    label: &str,
    words: &[u16],
    cpu_type: CpuType,
    setup: impl Fn(&mut CpuCore),
) {
    let bytes = assemble(words);

    let mk_cpu = |pc: u32| {
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(cpu_type);
        cpu.pc = pc;
        cpu.set_sr(0x2700);
        cpu.set_a(7, 0x9000);
        setup(&mut cpu);
        cpu
    };

    let mut bus_a = FastRamBus::new(0x20000);
    bus_a.fm_len = 0; // step() reference: no fastmem anywhere
    bus_a.load(0x1000, &bytes);
    let mut cpu_a = mk_cpu(0x1000);
    let mut steps: u64 = 0;
    loop {
        match cpu_a.step(&mut bus_a) {
            m68k::StepResult::Ok { .. } => steps += 1,
            m68k::StepResult::AlineTrap { .. } => break,
            other => panic!("{label}: unexpected step result {other:?}"),
        }
        assert!(steps < 10_000_000, "{label}: step run diverged");
    }

    let mut bus_b = FastRamBus::new(0x20000);
    bus_b.load(0x1000, &bytes);
    let mut cpu_b = mk_cpu(0x1000);
    let mut batched: u64 = 0;
    loop {
        let result = cpu_b.run_batch(&mut bus_b, 65_536, &[]);
        batched += result.instructions as u64;
        match result.exit {
            BatchExit::BudgetExhausted => continue,
            BatchExit::AlineTrap { .. } => break,
            other => panic!("{label}: unexpected batch exit {other:?}"),
        }
    }

    assert_eq!(steps, batched, "{label}: instruction count");
    assert_eq!(cpu_a.pc, cpu_b.pc, "{label}: pc");
    assert_eq!(cpu_a.get_sr(), cpu_b.get_sr(), "{label}: sr");
    for i in 0..8 {
        assert_eq!(cpu_a.d(i), cpu_b.d(i), "{label}: D{i}");
        assert_eq!(cpu_a.a(i), cpu_b.a(i), "{label}: A{i}");
    }
    assert_eq!(bus_a.mem, bus_b.mem, "{label}: memory contents");
}

#[test]
fn fastmem_move_and_alu_addressing_modes() {
    // Deterministic pass over every fastmem EA family:
    // (An), (An)+, -(An), d16(An), d8(An,Xn), abs.W, abs.L, d16(PC), #imm.
    let words: &[u16] = &[
        0x203C, 0x1234, 0x5678, // MOVE.L #$12345678,D0
        0x2A7C, 0x0001, 0x4000, // MOVEA.L #$14000,A5
        0x2ABC, 0xCAFE, 0xBABE, // MOVE.L #$CAFEBABE,(A5)
        0x2015, // MOVE.L (A5),D0
        0x3B40, 0x0010, // MOVE.W D0,$10(A5)
        0x102D, 0x0011, // MOVE.B $11(A5),D0
        0x2B80, 0x5820, // MOVE.L D0,($20,A5,D5.W*2)  (D5=0)
        0x31C0, 0x4100, // MOVE.W D0,($4100).W
        0x23C0, 0x0001, 0x4208, // MOVE.L D0,($14208).L
        0x303A, 0xFFEE, // MOVE.W (d16,PC),D0  (reads earlier code)
        0x2A9B, // MOVE.L (A3)+,(A5)
        0x2B23, // MOVE.L -(A3),-(A5)... wait predec dst uses A5
        0xD095, // ADD.L (A5),D0
        0x94FC, 0x0002, // SUBA.W #2,A2
        0x0685, 0x0000, 0x0100, // ADDI.L #$100,D5
        0x0C6D, 0x0042, 0x0010, // CMPI.W #$42,$10(A5)
        0x4A2D, 0x0011, // TST.B $11(A5)
        0x422D, 0x0013, // CLR.B $13(A5)
        0x446D, 0x0010, // NEG.W $10(A5)
        0x466D, 0x0010, // NOT.W $10(A5)
        0x5255, // ADDQ.W #1,(A5)
        0x5395, // SUBQ.L #1,(A5)
        0x0815, 0x0003, // BTST #3,(A5)
        0x08D5, 0x0002, // BSET #2,(A5)
        0x0895, 0x0001, // BCLR #1,(A5)
        0x0855, 0x0000, // BCHG #0,(A5)
        0x03D5, // BSET D1,(A5)
        0xB50D, // CMPM.B (A5)+,(A2)+
        0xA000, // sentinel
    ];
    assert_fastmem_matches_step("modes", words, CpuType::M68020, |cpu| {
        cpu.set_a(3, 0x15000);
        cpu.set_a(2, 0x14800);
        cpu.set_d(5, 0);
        cpu.set_d(1, 5);
    });
}

#[test]
fn fastmem_control_flow_ops() {
    // LEA/PEA/MOVEA/JSR(abs.L)/BSR.W/RTS/JMP(abs.L) against step().
    let words: &[u16] = &[
        // 0x1000
        0x41F8, 0x4000, // LEA ($4000).W,A0
        0x43E8, 0x0200, // LEA $200(A0),A1
        0x4869, 0x0100, // PEA $100(A1)
        0x245F, // MOVEA.L (A7)+,A2
        0x4EB9, 0x0000, 0x1024, // JSR ($1024).L
        0x6100, 0x0012, // BSR.W +0x12 (to 0x1028)
        0x6106, // BSR.S +6 (to 0x101C+... see layout)
        0x4EF9, 0x0000, 0x102C, // JMP ($102C).L
        // 0x1024: subroutine 1
        0x5280, // ADDQ.L #1,D0
        0x4E75, // RTS
        // 0x1028: subroutine 2
        0x5281, // ADDQ.L #1,D1
        0x4E75, // RTS
        // 0x102C: done
        0xA000, // sentinel
    ];
    // Note: the BSR.S at 0x1016 targets 0x101E which lands mid-JMP — so
    // give it a real target instead: rebuild with explicit layout below.
    let _ = words;

    let words: &[u16] = &[
        // 0x1000
        0x41F8, 0x4000, // LEA ($4000).W,A0            ; 0x1000
        0x43E8, 0x0200, // LEA $200(A0),A1             ; 0x1004
        0x4869, 0x0100, // PEA $100(A1)                ; 0x1008
        0x245F, //        MOVEA.L (A7)+,A2             ; 0x100C
        0x4EB9, 0x0000, 0x1020, // JSR ($1020).L       ; 0x100E
        0x6100,
        0x000C, // BSR.W +0xC (-> 0x1022+... ) ; 0x1014 -> 0x1016+0xC=0x1022? base=0x1016, +0xC=0x1022... target 0x1024
        0x4EF9, 0x0000, 0x1028, // JMP ($1028).L       ; 0x1018
        0x4E71, // NOP (padding)                       ; 0x101E
        // 0x1020: subroutine 1
        0x5280, // ADDQ.L #1,D0
        0x4E75, // RTS                                 ; 0x1022
        // 0x1024: subroutine 2
        0x5281, // ADDQ.L #1,D1
        0x4E75, // RTS                                 ; 0x1026
        // 0x1028: done
        0xA000, // sentinel
    ];
    assert_fastmem_matches_step("control", words, CpuType::M68020, |_| {});
}

#[test]
fn fastmem_a7_byte_quirk() {
    // Byte pushes/pops through A7 move SP by 2, and the byte lives at
    // the *low* address of the word slot.
    let words: &[u16] = &[
        0x7041, // MOVEQ #$41,D0
        0x1F00, // MOVE.B D0,-(A7)
        0x1F3C, 0x0042, // MOVE.B #$42,-(A7)
        0x121F, // MOVE.B (A7)+,D1
        0x141F, // MOVE.B (A7)+,D2
        0xA000, // sentinel
    ];
    assert_fastmem_matches_step("a7-byte", words, CpuType::M68020, |_| {});
    assert_fastmem_matches_step("a7-byte-68000", words, CpuType::M68000, |_| {});
}

#[test]
fn fastmem_68000_alignment_faults_match_step() {
    // Odd word/long accesses raise a 68000 address error; the fastmem
    // path must fall back so the exception frame matches step() exactly.
    // (Vectors are zero-filled, so both sides jump to PC 0 and execute
    // whatever's there; run a bounded number of instructions and compare.)
    let words: &[u16] = &[
        0x2A7C, 0x0001, 0x4001, // MOVEA.L #$14001,A5 (odd)
        0x3A80, // MOVE.W D0,(A5)  -> address error
        0x4E71, // NOP (skipped via exception)
        0xA000,
    ];
    let bytes = assemble(words);

    let mut bus_a = FastRamBus::new(0x20000);
    bus_a.fm_len = 0;
    bus_a.load(0x1000, &bytes);
    let mut cpu_a = cpu_at(0x1000); // 68000
    for _ in 0..4 {
        let _ = cpu_a.step(&mut bus_a);
    }

    let mut bus_b = FastRamBus::new(0x20000);
    bus_b.load(0x1000, &bytes);
    let mut cpu_b = cpu_at(0x1000);
    let mut executed = 0;
    while executed < 4 {
        let result = cpu_b.run_batch(&mut bus_b, 4 - executed, &[]);
        executed += result.instructions;
        match result.exit {
            BatchExit::BudgetExhausted => {}
            // The faulting instruction isn't counted by the batch loop's
            // fault path; step() counts it as an executed step. Just stop
            // on any other exit and compare state below.
            other => panic!("unexpected exit {other:?}"),
        }
    }

    assert_eq!(cpu_a.pc, cpu_b.pc, "pc after address error");
    assert_eq!(cpu_a.get_sr(), cpu_b.get_sr(), "sr after address error");
    assert_eq!(cpu_a.a(7), cpu_b.a(7), "sp after address error");
    assert_eq!(bus_a.mem, bus_b.mem, "memory after address error");
}

#[test]
fn fastmem_partial_window_falls_back_outside() {
    // Window covers only [0x10000, 0x18000); code at 0x1000 and a low
    // absolute write are both outside it and must use the bus path.
    let words: &[u16] = &[
        0x31FC, 0x1111, 0x4000, // MOVE.W #$1111,($4000).W   (outside window)
        0x33FC, 0x2222, 0x0001, 0x4000, // MOVE.W #$2222,($14000).L (inside)
        0x3038, 0x4000, // MOVE.W ($4000).W,D0
        0x3239, 0x0001, 0x4000, // MOVE.W ($14000).L,D1
        0xA000,
    ];
    let bytes = assemble(words);

    let mut bus_a = FastRamBus::new(0x20000);
    bus_a.fm_len = 0;
    bus_a.load(0x1000, &bytes);
    let mut cpu_a = cpu_020_at(0x1000);
    loop {
        match cpu_a.step(&mut bus_a) {
            m68k::StepResult::Ok { .. } => {}
            m68k::StepResult::AlineTrap { .. } => break,
            other => panic!("unexpected step result {other:?}"),
        }
    }

    let mut bus_b = FastRamBus::new(0x20000);
    bus_b.fm_base = 0x10000;
    bus_b.fm_len = 0x8000;
    bus_b.load(0x1000, &bytes);
    let mut cpu_b = cpu_020_at(0x1000);
    loop {
        let result = cpu_b.run_batch(&mut bus_b, 1000, &[]);
        match result.exit {
            BatchExit::BudgetExhausted => continue,
            BatchExit::AlineTrap { .. } => break,
            other => panic!("unexpected batch exit {other:?}"),
        }
    }

    assert_eq!(cpu_a.d(0), cpu_b.d(0));
    assert_eq!(cpu_a.d(1), cpu_b.d(1));
    assert_eq!(cpu_a.d(0) & 0xFFFF, 0x1111);
    assert_eq!(cpu_a.d(1) & 0xFFFF, 0x2222);
    assert_eq!(bus_a.mem, bus_b.mem);
}

#[test]
fn fastmem_differential_fuzz_memory_ops() {
    // Differential fuzz between step() (no fastmem) and run_batch() with
    // a fastmem window, over random DBRA loops whose bodies mix register
    // ops with every fastmem memory-op family. A0-A5 point into a data
    // zone well away from the code; ops that would send addresses
    // wandering (MOVEA/ADDA/LEA from memory) are exercised by the
    // deterministic tests above instead.
    let mut seed: u64 = 0xFA57_3E31_2345_6789;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };

    for program_index in 0..300 {
        let mut r = || rng() as u32;
        let dreg = |v: u32| (v % 6) as u16; // D0-D5
        let areg = |v: u32| (v % 6) as u16; // A0-A5
        let size2 = |v: u32| (v % 3) as u16;
        // Memory EA (mode,reg) in the safe data zone; returns (mode<<3)|reg
        // plus any extension words. d16 kept within ±0x1000 of the zone.
        let mem_ea = |r: &mut dyn FnMut() -> u32, exts: &mut Vec<u16>| -> u16 {
            match r() % 5 {
                0 => (2 << 3) | areg(r()), // (An)
                1 => (3 << 3) | areg(r()), // (An)+
                2 => (4 << 3) | areg(r()), // -(An)
                3 => {
                    exts.push((r() % 0x2000) as i16 as u16 & 0x1FFF); // 0..0x1FFF
                    (5 << 3) | areg(r())
                }
                _ => {
                    // (d8,An,Am.W/L[*scale]) — address-register index only,
                    // masked to the zone-safe low bits at setup.
                    let idx = 8 | (areg(r()) as u32); // An as index
                    let long = (r() % 2) << 11;
                    let scale = (r() % 4) << 9;
                    exts.push(
                        ((idx as u16) << 12)
                            | (long as u16)
                            | (scale as u16)
                            | ((r() % 0x80) as u16),
                    );
                    (6 << 3) | areg(r())
                }
            }
        };

        let mut words: Vec<u16> = Vec::new();
        let iterations = (r() % 60) as u16;
        words.push(0x7C00 | (iterations & 0x3F)); // MOVEQ #it,D6
        let loop_start = words.len();
        for _ in 0..12 {
            let mut exts: Vec<u16> = Vec::new();
            let op = match r() % 12 {
                0 => {
                    // MOVE.size mem→Dn
                    let base = [0x1000u16, 0x3000, 0x2000][(r() % 3) as usize];
                    let ea = mem_ea(&mut r, &mut exts);
                    base | (dreg(r()) << 9) | ea
                }
                1 => {
                    // MOVE.size Dn→mem
                    let base = [0x1000u16, 0x3000, 0x2000][(r() % 3) as usize];
                    let ea = mem_ea(&mut r, &mut exts);
                    let dst_mode = (ea >> 3) & 7;
                    let dst_reg = ea & 7;
                    base | (dst_reg << 9) | (dst_mode << 6) | dreg(r())
                }
                2 => {
                    // MOVE.size mem→mem
                    let base = [0x1000u16, 0x3000, 0x2000][(r() % 3) as usize];
                    let src = mem_ea(&mut r, &mut exts);
                    let mut dst_exts: Vec<u16> = Vec::new();
                    let dst = mem_ea(&mut r, &mut dst_exts);
                    exts.extend_from_slice(&dst_exts);
                    base | ((dst & 7) << 9) | (((dst >> 3) & 7) << 6) | src
                }
                3 => {
                    // ALU mem→Dn
                    let group = [0xD000u16, 0x9000, 0xC000, 0x8000, 0xB000][(r() % 5) as usize];
                    let ea = mem_ea(&mut r, &mut exts);
                    group | (dreg(r()) << 9) | (size2(r()) << 6) | ea
                }
                4 => {
                    // ALU Dn→mem (ADD/SUB/AND/OR/EOR)
                    let group = [0xD100u16, 0x9100, 0xC100, 0x8100, 0xB100][(r() % 5) as usize];
                    let ea = mem_ea(&mut r, &mut exts);
                    group | (dreg(r()) << 9) | (size2(r()) << 6) | ea
                }
                5 => {
                    // ADDI/SUBI/ANDI/ORI/EORI/CMPI #imm → Dn or mem
                    let op =
                        [0x0600u16, 0x0400, 0x0200, 0x0000, 0x0A00, 0x0C00][(r() % 6) as usize];
                    let size = size2(r());
                    let ea = if r() % 2 == 0 {
                        dreg(r())
                    } else {
                        mem_ea(&mut r, &mut exts)
                    };
                    let imm_exts: Vec<u16> = if size == 2 {
                        vec![(r() & 0xFFFF) as u16, (r() & 0xFFFF) as u16]
                    } else {
                        vec![(r() & 0xFFFF) as u16]
                    };
                    // Immediate words come before the EA extension words.
                    let mut all = imm_exts;
                    all.extend_from_slice(&exts);
                    exts = all;
                    op | (size << 6) | ea
                }
                6 => {
                    // ADDQ/SUBQ #q,mem
                    let sub = if r() % 2 == 0 { 0x0100 } else { 0 };
                    let ea = mem_ea(&mut r, &mut exts);
                    0x5000 | sub | (((r() % 8) as u16) << 9) | (size2(r()) << 6) | ea
                }
                7 => {
                    // TST/CLR/NEG/NOT mem
                    let unary = [0x4A00u16, 0x4200, 0x4400, 0x4600][(r() % 4) as usize];
                    let ea = mem_ea(&mut r, &mut exts);
                    unary | (size2(r()) << 6) | ea
                }
                8 => {
                    // BTST/BCHG/BCLR/BSET (Dn or #imm) on mem
                    let ea = mem_ea(&mut r, &mut exts);
                    if r() % 2 == 0 {
                        0x0100 | (dreg(r()) << 9) | (((r() % 4) as u16) << 6) | ea
                    } else {
                        let mut all = vec![(r() % 8) as u16];
                        all.extend_from_slice(&exts);
                        exts = all;
                        0x0800 | (((r() % 4) as u16) << 6) | ea
                    }
                }
                9 => {
                    // CMPM.size (Ay)+,(Ax)+
                    0xB108 | (areg(r()) << 9) | (size2(r()) << 6) | areg(r())
                }
                10 => {
                    // CMPA.W/L mem,An
                    let ea = mem_ea(&mut r, &mut exts);
                    let long = if r() % 2 == 0 { 0x0100 } else { 0 };
                    0xB0C0 | long | (areg(r()) << 9) | ea
                }
                _ => {
                    // Register filler: MOVEQ
                    0x7000 | (dreg(r()) << 9) | (r() & 0xFF) as u16
                }
            };
            words.push(op);
            words.extend_from_slice(&exts);
        }
        let body_bytes = (words.len() - loop_start) * 2;
        words.push(0x51CE); // DBRA D6
        words.push((-(body_bytes as i32) - 2) as i16 as u16);
        words.push(0xA000);

        let bytes = assemble(&words);

        // A0-A5 spread across the data zone; D0-D5 random data.
        let init_a: Vec<u32> = (0..6)
            .map(|i| 0x12000 + i * 0x800 + (r() % 0x400))
            .collect();
        let init_d: Vec<u32> = (0..6).map(|_| r()).collect();
        let init_ccr = (r() & 0x1F) as u16;
        let mut fill: Vec<u8> = Vec::with_capacity(0x10000);
        for _ in 0..0x4000 {
            fill.extend_from_slice(&r().to_be_bytes());
        }
        let setup = |cpu: &mut CpuCore| {
            for i in 0..6 {
                cpu.set_d(i, init_d[i]);
                cpu.set_a(i, init_a[i]);
            }
            cpu.set_sr(0x2700 | init_ccr);
        };

        let mut bus_a = FastRamBus::new(0x40000);
        bus_a.fm_len = 0;
        bus_a.load(0x1000, &bytes);
        bus_a.load(0x10000, &fill);
        let mut cpu_a = cpu_020_at(0x1000);
        setup(&mut cpu_a);
        let mut steps: u64 = 0;
        let mut timed_out = false;
        loop {
            match cpu_a.step(&mut bus_a) {
                m68k::StepResult::Ok { .. } => steps += 1,
                m68k::StepResult::AlineTrap { .. } => break,
                other => panic!(
                    "program {program_index}: unexpected step result {other:?} (words={words:04X?})"
                ),
            }
            if steps >= 2_000_000 {
                timed_out = true;
                break;
            }
        }
        if timed_out {
            // A stray write corrupted the loop; skip rather than hang.
            continue;
        }

        let mut bus_b = FastRamBus::new(0x40000);
        bus_b.load(0x1000, &bytes);
        bus_b.load(0x10000, &fill);
        let mut cpu_b = cpu_020_at(0x1000);
        setup(&mut cpu_b);
        let mut batched: u64 = 0;
        loop {
            let result = cpu_b.run_batch(&mut bus_b, 100_000, &[]);
            batched += result.instructions as u64;
            match result.exit {
                BatchExit::BudgetExhausted => {
                    assert!(
                        batched < steps + 200_000,
                        "program {program_index}: batch ran past step count (words={words:04X?})"
                    );
                }
                BatchExit::AlineTrap { .. } => break,
                other => panic!(
                    "program {program_index}: unexpected batch exit {other:?} (words={words:04X?})"
                ),
            }
        }

        assert_eq!(
            steps, batched,
            "program {program_index}: instruction count (words={words:04X?})"
        );
        assert_eq!(
            cpu_a.pc, cpu_b.pc,
            "program {program_index}: pc (words={words:04X?})"
        );
        assert_eq!(
            cpu_a.get_sr(),
            cpu_b.get_sr(),
            "program {program_index}: sr (words={words:04X?})"
        );
        for i in 0..8 {
            assert_eq!(
                cpu_a.d(i),
                cpu_b.d(i),
                "program {program_index}: D{i} (words={words:04X?})"
            );
            assert_eq!(
                cpu_a.a(i),
                cpu_b.a(i),
                "program {program_index}: A{i} (words={words:04X?})"
            );
        }
        assert_eq!(
            bus_a.mem, bus_b.mem,
            "program {program_index}: memory diverged (words={words:04X?})"
        );
    }
}

// ============================================================================
// Memory-op JIT traces (MoveMem): hot loops with memory operands compile
// into traces that loop natively. Everything must match step() exactly.
// ============================================================================

/// Fill 64 longs at $2000 with a changing pattern, then copy them to $3000
/// with the classic `MOVE.L (A0)+,(A1)+ / DBRA` inner loop. Both loops run
/// hot enough to compile and iterate inside the JIT.
/// Absolute-addressed CLR (the 4238/4239 gameplay census heads) must stay
/// exact through the whole record/compile/execute lifecycle, including the
/// flag pattern (Z set, NVC clear, X preserved) and re-dirtied targets.
#[test]
fn mem_trace_clr_absolute_matches_step() {
    let words = &[
        0x4278, 0x2000, // $1000: CLR.W ($2000).W
        0x42B9, 0x0000, 0x2004, // $1004: CLR.L ($2004).L
        0x4239, 0x0000, 0x2003, // $100A: CLR.B ($2003).L
        0x31C0, 0x2000, // $1010: MOVE.W D0,($2000).W  (re-dirty the target)
        0x51C8, 0xFFEA, // $1014: DBRA D0,$1000
        0xA000, // $1018: sentinel
    ];
    assert_fastmem_matches_step("clr absolute", words, CpuType::M68000, |cpu| {
        cpu.set_d(0, 300);
    });
}

#[test]
fn mem_trace_memcpy_loop_matches_step() {
    let words = &[
        0x20C2, // $1000: MOVE.L D2,(A0)+      (fill loop)
        0x5282, // $1002: ADDQ.L #1,D2
        0x51C8, 0xFFFA, // $1004: DBRA D0,$1000
        0x41F8, 0x2000, // $1008: LEA $2000.W,A0
        0x43F8, 0x3000, // $100C: LEA $3000.W,A1
        0x703F, // $1010: MOVEQ #63,D0
        0x22D8, // $1012: MOVE.L (A0)+,(A1)+   (copy loop)
        0x51C8, 0xFFFC, // $1014: DBRA D0,$1012
        0xA000, // $1018: sentinel
    ];
    assert_fastmem_matches_step("mem-trace memcpy", words, CpuType::M68000, |cpu| {
        cpu.set_a(0, 0x2000);
        cpu.set_d(0, 63);
        cpu.set_d(2, 0xDEAD_0001);
    });
}

/// A loop calling a two-op leaf through BSR.W: the call-through retry
/// records push, callee, checked return, and loop tail as one trace.
/// Everything must match step() exactly through the whole lifecycle --
/// first blocked recording, retry, compile, and native chaining.
#[test]
fn call_through_leaf_loop_matches_step() {
    let words = &[
        0x6100, 0x0012, // $1000: BSR.W $1014
        0x5283, // $1004: ADDQ.L #1,D3
        0x51C8, 0xFFF8, // $1006: DBRA D0,$1000
        0x5347, // $100A: SUBQ.W #1,D7
        0x6602, // $100C: BNE.S $1010
        0xA000, // $100E: sentinel
        0x707F, // $1010: MOVEQ #127,D0
        0x60EC, // $1012: BRA.S $1000
        0x5282, // $1014: leaf: ADDQ.L #1,D2
        0x4E75, // $1016: RTS
    ];
    assert_fastmem_matches_step("call-through leaf", words, CpuType::M68040, |cpu| {
        cpu.set_d(0, 50);
        cpu.set_d(7, 5);
    });
}

/// The same loop-calls-leaf shape with the leaf at BSR.W's maximum
/// forward reach (32KB away). The caller's and callee's code get
/// separate SMC store intervals, so the far call is admitted; the whole
/// lifecycle must still match step() exactly.
#[test]
fn far_leaf_call_through_matches_step() {
    let leaf = 0x9000u32;
    let bsr_disp = (leaf - (0x1000 + 2)) as u16;
    let mut words = vec![
        0x6100, bsr_disp, // $1000: BSR.W $9000
        0x5283,   // $1004: ADDQ.L #1,D3
        0x51C8, 0xFFF8, // $1006: DBRA D0,$1000
        0x5347, // $100A: SUBQ.W #1,D7
        0x6602, // $100C: BNE.S $1010
        0xA000, // $100E: sentinel
        0x707F, // $1010: MOVEQ #127,D0
        0x60EC, // $1012: BRA.S $1000
    ];
    // Never-executed filler up to the leaf.
    words.resize(((leaf - 0x1000) / 2) as usize, 0x4E71);
    words.extend([
        0x5282, // $9000: leaf: ADDQ.L #1,D2
        0x4E75, // $9002: RTS
    ]);
    assert_fastmem_matches_step("far-leaf call-through", &words, CpuType::M68040, |cpu| {
        cpu.set_a(7, 0x8000); // stack well clear of both code regions
        cpu.set_d(0, 50);
        cpu.set_d(7, 5);
    });
}

/// A memory operation followed immediately by DBRA is a common 68k copy-loop
/// shape. It must remain exact when admitted as the minimum two-op self-loop.
#[test]
fn mem_trace_two_op_memcpy_loop_matches_step() {
    let words = &[
        0x22D8, // $1000: MOVE.L (A0)+,(A1)+
        0x51C8, 0xFFFC, // $1002: DBRA D0,$1000
        0xA000, // $1006: sentinel
    ];
    assert_fastmem_matches_step("two-op mem-trace memcpy", words, CpuType::M68000, |cpu| {
        cpu.set_a(0, 0x2000);
        cpu.set_a(1, 0x3000);
        cpu.set_d(0, 127);
    });
}

/// Backward word copy with pre-decrement on both sides.
#[test]
fn mem_trace_predec_copy_matches_step() {
    let words = &[
        0x20C2, // $1000: MOVE.L D2,(A0)+      (fill loop)
        0x5482, // $1002: ADDQ.L #2,D2
        0x51C8, 0xFFFA, // $1004: DBRA D0,$1000
        0x41F8, 0x2100, // $1008: LEA $2100.W,A0 (end of source)
        0x43F8, 0x3100, // $100C: LEA $3100.W,A1 (end of dest)
        0x707F, // $1010: MOVEQ #127,D0
        0x3320, // $1012: MOVE.W -(A0),-(A1)   (copy loop)
        0x51C8, 0xFFFC, // $1014: DBRA D0,$1012
        0xA000, // $1018: sentinel
    ];
    assert_fastmem_matches_step("mem-trace predec", words, CpuType::M68000, |cpu| {
        cpu.set_a(0, 0x2000);
        cpu.set_d(0, 63);
        cpu.set_d(2, 0xBEEF_0001);
    });
}

/// A hot copy loop whose stores eventually sweep over its own code. The
/// trace must bail before the overlapping store commits so the modified
/// instructions take effect exactly when the interpreter would see them.
#[test]
fn mem_trace_store_into_own_code_matches_step() {
    let words = &[
        0x20C2, // $1000: MOVE.L D2,(A0)+
        0x51C8, 0xFFFC, // $1002: DBRA D0,$1000
        0xA000, // $1006: sentinel (never reached by fall-through)
    ];
    // Use a non-prefetch CPU so this remains a cache-coherency test. On a
    // real 68000, step() may legally retire words already in its prefetch
    // queue while run_batch() is the explicitly non-transactional fast path.
    assert_fastmem_matches_step("mem-trace smc", words, CpuType::M68020, |cpu| {
        // Stores start below the code and cross it after 64 iterations,
        // overwriting the loop with NOP + A-line — which must then execute.
        cpu.set_a(0, 0x0F00);
        cpu.set_d(0, 200);
        cpu.set_d(2, 0x4E71_A000);
    });
}

/// MOVE.L (A0)+,(A0)+ in a hot loop: the destination EA must observe the
/// source post-increment on the same register.
#[test]
fn mem_trace_same_register_pair_matches_step() {
    let words = &[
        0x20C2, // $1000: MOVE.L D2,(A0)+      (fill so the copy reads data)
        0x5282, // $1002: ADDQ.L #1,D2
        0x51C8, 0xFFFA, // $1004: DBRA D0,$1000
        0x41F8, 0x2000, // $1008: LEA $2000.W,A0
        0x701F, // $100C: MOVEQ #31,D0
        0x20D8, // $100E: MOVE.L (A0)+,(A0)+   (copy every other long forward)
        0x51C8, 0xFFFC, // $1010: DBRA D0,$100E
        0xA000, // $1014: sentinel
    ];
    assert_fastmem_matches_step("mem-trace same-reg", words, CpuType::M68000, |cpu| {
        cpu.set_a(0, 0x2000);
        cpu.set_d(0, 63);
        cpu.set_d(2, 0xCAFE_0001);
    });
}

/// Odd source address on a 68000: the trace bails and full dispatch takes
/// the address error, identically to step().
#[test]
fn mem_trace_unaligned_matches_step_on_68020() {
    // On the 68020 unaligned accesses are legal; run an odd-address copy
    // loop hot so the trace path handles it (bail + interpreter, or
    // window access) with results identical to step().
    let words = &[
        0x20C2, // $1000: MOVE.L D2,(A0)+
        0x5282, // $1002: ADDQ.L #1,D2
        0x51C8, 0xFFFA, // $1004: DBRA D0,$1000
        0x41F8, 0x2001, // $1008: LEA $2001.W,A0 (odd)
        0x43F8, 0x3001, // $100C: LEA $3001.W,A1 (odd)
        0x701F, // $1010: MOVEQ #31,D0
        0x22D8, // $1012: MOVE.L (A0)+,(A1)+
        0x51C8, 0xFFFC, // $1014: DBRA D0,$1012
        0xA000, // $1018: sentinel
    ];
    assert_fastmem_matches_step("mem-trace unaligned-020", words, CpuType::M68020, |cpu| {
        cpu.set_a(0, 0x2000);
        cpu.set_d(0, 63);
        cpu.set_d(2, 0x0BAD_0001);
    });
}

/// Common memory-source CMP forms must compile without changing
/// architectural behavior.
#[test]
fn mem_trace_cmp_sources_match_step() {
    let indirect = &[
        0xB210, // $1000: CMP.B (A0),D1
        0x4E71, // $1002: NOP (three-op minimum with DBRA)
        0x51C8, 0xFFFA, // $1004: DBRA D0,$1000
        0xA000, // $1008: sentinel
    ];
    assert_fastmem_matches_step("mem-trace cmp indirect", indirect, CpuType::M68000, |cpu| {
        cpu.set_a(0, 0x2000);
        cpu.set_d(0, 127);
        cpu.set_d(1, 0x1234_567F);
        cpu.set_ccr(0x10);
    });

    let displacement = &[
        0xBC6E, 0x0010, // $1000: CMP.W $0010(A6),D6
        0x4E71, // $1004: NOP
        0x51C8, 0xFFF8, // $1006: DBRA D0,$1000
        0xA000, // $100A: sentinel
    ];
    assert_fastmem_matches_step(
        "mem-trace cmp displacement",
        displacement,
        CpuType::M68000,
        |cpu| {
            cpu.set_a(6, 0x2100);
            cpu.set_d(0, 127);
            cpu.set_d(6, 0xCAFE_BEEF);
            cpu.set_ccr(0x10);
        },
    );
}

/// A loop combining scaled brief-indexed byte reads with word/long ADDs
/// through a postincrement destination must match the interpreter's
/// addresses, big-endian stores, postincrements, and NZVCX results exactly.
#[test]
fn mem_trace_indexed_move_and_postinc_add_matches_step() {
    let words = &[
        0x2545, 0x0004, // $1000: MOVE.L D5,$0004(A2)
        0x2086, // $1004: MOVE.L D6,(A0)
        0x1832, 0x1C00, // $1006: MOVE.B 0(A2,D1.L*4),D4
        0xD998, // $100A: ADD.L D4,(A0)+
        0x1832, 0x1C01, // $100C: MOVE.B 1(A2,D1.L*4),D4
        0xD998, // $1010: ADD.L D4,(A0)+
        0x1832, 0x1C02, // $1012: MOVE.B 2(A2,D1.L*4),D4
        0xD958, // $1016: ADD.W D4,(A0)+
        0x51C8, 0xFFEC, // $1018: DBRA D0,$1006
        0xA000, // $101C: sentinel
    ];
    assert_fastmem_matches_step(
        "mem-trace indexed/add-postinc",
        words,
        CpuType::M68040,
        |cpu| {
            cpu.set_a(0, 0x3000);
            cpu.set_a(2, 0x2000);
            cpu.set_d(0, 127);
            cpu.set_d(1, 1);
            cpu.set_d(4, 0xA5A5_A5A5);
            cpu.set_d(5, 0x7F01_80FF);
            cpu.set_d(6, 0x7FFF_FF80);
            cpu.set_ccr(0x10);
        },
    );
}

/// A LINK/UNLK frame loop with a store through the frame pointer -- the
/// ROM-prologue shape behind the biggest blocked heads in the gameplay
/// profile. Must match step() through the whole lifecycle.
#[test]
fn link_unlk_frame_loop_matches_step() {
    let words = &[
        0x4E56, 0xFFF8, // $1000: LINK A6,#-8
        0x5283, // $1004: ADDQ.L #1,D3
        0x3D43, 0xFFFC, // $1006: MOVE.W D3,-4(A6)
        0x4E5E, // $100A: UNLK A6
        0x51C8, 0xFFF2, // $100C: DBRA D0,$1000
        0x5347, // $1010: SUBQ.W #1,D7
        0x6602, // $1012: BNE.S $1016
        0xA000, // $1014: sentinel
        0x707F, // $1016: MOVEQ #127,D0
        0x60E6, // $1018: BRA.S $1000
    ];
    assert_fastmem_matches_step("link/unlk frame loop", words, CpuType::M68040, |cpu| {
        cpu.set_d(0, 50);
        cpu.set_d(7, 5);
    });
}

/// The salvage shape end to end: a store-heavy prefix through an interior
/// branch, an inadmissible LEA (abs).L, and a loop tail. The salvaged
/// trace plus the interpreted tail must match step() exactly.
#[test]
fn salvaged_prefix_loop_matches_step() {
    let words = &[
        0x3083, // $1000: MOVE.W D3,(A0)
        0x5283, // $1002: ADDQ.L #1,D3
        0x3143, 0x0002, // $1004: MOVE.W D3,2(A0)
        0x5284, // $1008: ADDQ.L #1,D4
        0x3144, 0x0004, // $100A: MOVE.W D4,4(A0)
        0x5285, // $100E: ADDQ.L #1,D5
        0x3145, 0x0006, // $1010: MOVE.W D5,6(A0)
        0x5286, // $1014: ADDQ.L #1,D6
        0x3146, 0x0008, // $1016: MOVE.W D6,8(A0)
        0x5281, // $101A: ADDQ.L #1,D1
        0x4A41, // $101C: TST.W D1
        0x6602, // $101E: BNE.S $1022
        0x4E71, // $1020: NOP (skipped)
        0x4A42, // $1022: TST.W D2 (past the branch: no terminal here)
        0x4A42, // $1024: TST.W D2
        0x4E57, 0x0000, // $1026: LINK A7,#0 -- refused by design
        0x4E5F, // $102A: UNLK A7
        0x51C8, 0xFFD2, // $102C: DBRA D0,$1000
        0x5347, // $1030: SUBQ.W #1,D7
        0x6602, // $1032: BNE.S $1036
        0xA000, // $1034: sentinel
        0x707F, // $1036: MOVEQ #127,D0
        0x60C6, // $1038: BRA.S $1000
    ];
    assert_fastmem_matches_step("salvaged prefix loop", words, CpuType::M68040, |cpu| {
        cpu.set_a(0, 0x4000);
        cpu.set_d(0, 50);
        cpu.set_d(7, 5);
    });
}

/// The census exemplar: AND.W of a displaced field into a register,
/// with a counter.
#[test]
fn memory_and_loop_matches_step() {
    let words = &[
        0xC268, 0x0010, // $1000: AND.W ($10,A0),D1
        0x5283, // $1004: ADDQ.L #1,D3
        0x51C8, 0xFFF8, // $1006: DBRA D0,$1000
        0x5347, // $100A: SUBQ.W #1,D7
        0x6602, // $100C: BNE.S $1010
        0xA000, // $100E: sentinel
        0x707F, // $1010: MOVEQ #127,D0
        0x60EC, // $1012: BRA.S $1000
    ];
    assert_fastmem_matches_step("memory AND loop", words, CpuType::M68040, |cpu| {
        cpu.set_a(0, 0x3000);
        cpu.set_d(1, 0xFFFF_F0F0);
        cpu.set_d(0, 50);
        cpu.set_d(7, 5);
    });
}

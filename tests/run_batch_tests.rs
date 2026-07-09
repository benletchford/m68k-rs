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
    let mut seed: u64 = 0x00C0FFEE_5EED_1234;
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
    let mut batched: u32 = 0;
    loop {
        let result = cpu_b.run_batch(&mut bus_b, 100_000, &[end_pc]);
        batched += result.instructions;
        match result.exit {
            BatchExit::WatchedPc { pc } => {
                assert_eq!(pc, end_pc);
                break;
            }
            other => panic!("unexpected batch exit {other:?}"),
        }
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

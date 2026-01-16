//! Integration tests from Musashi.
//!
//! These tests load precompiled 68k binaries and run them through the emulator.
//! Tests signal pass/fail via memory-mapped test device registers.

mod common;

use common::TestBus;
use m68k::{AddressBus, CpuCore, CpuType, NoOpHleHandler, StepResult};

/// Run a test binary and return the result.
fn run_test(cpu_type: CpuType, binary: &[u8], max_cycles: i32) -> common::TestResult {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);

    let mut bus = TestBus::new();
    bus.load_rom(binary);
    bus.setup_boot();

    // Reset the CPU (loads SP/PC from vectors)
    cpu.reset(&mut bus);
    // Override PC to start at test code
    cpu.pc = 0x10000;
    cpu.set_sr(0x2700);

    // Debug: keep a rolling trace of the last N instructions to make late failures debuggable.
    let mut history: Vec<String> = Vec::new();
    const HISTORY_MAX: usize = 800;

    // Some Musashi-derived tests (notably BCD) intentionally run large nested loops.
    // Keep this high enough that "no pass/fail signaled" usually indicates a real emulator bug.
    let mut hle = NoOpHleHandler;
    for i in 0..1_000_000 {
        // Wire the test-device interrupt line into the CPU's interrupt input (level-sensitive).
        cpu.int_level = bus.test_device.interrupt_level.unwrap_or(0) as u32;

        // Allow pending interrupts to be serviced *between* instructions (matching 68k timing).
        // `execute(0)` will service interrupts (if any) without running additional instructions.
        let _ = cpu.execute(&mut bus, 0);

        let pc = cpu.pc;
        let opcode = bus.read_word(pc);

        // For instruction 19 (CMPM), show memory contents
        let m0 = bus.read_byte(cpu.a(0));
        let m1 = bus.read_byte(cpu.a(1));
        let entry = format!(
            "#{:05}: PC={:#06X} Op={:#06X} [A0]={:#04X} [A1]={:#04X} SR={:#06X} A0={:#010X} A6={:#010X} A7={:#010X}",
            i,
            pc,
            opcode,
            m0,
            m1,
            cpu.get_sr(),
            cpu.a(0),
            cpu.a(6),
            cpu.a(7)
        );
        history.push(entry);
        if history.len() > HISTORY_MAX {
            history.remove(0);
        }

        // Execute one instruction. HLE handler falls back to exceptions.
        match cpu.step_with_hle_handler(&mut bus, &mut hle) {
            StepResult::Ok { .. } => {}
            StepResult::Stopped => {
                eprintln!("=== CPU STOPPED at instruction #{} ===", i);
                break;
            }
            _ => {}
        }
        if cpu.stopped != 0 {
            eprintln!("=== CPU STOPPED at instruction #{} ===", i);
            break;
        }
        if bus.test_device.fail_count > 0 {
            eprintln!(
                "*** FAIL SIGNALED (value={:#X}, count={}) ***",
                0, bus.test_device.fail_count
            );
            eprintln!("=== FAIL at instruction #{}, D5 change history: ===", i);
            for h in &history {
                eprintln!("{}", h);
            }
            eprintln!(
                "Regs: D0={:#010X} D1={:#010X} D2={:#010X} D3={:#010X} D4={:#010X} D5={:#010X}",
                cpu.d(0),
                cpu.d(1),
                cpu.d(2),
                cpu.d(3),
                cpu.d(4),
                cpu.d(5)
            );
            eprintln!(
                "      A0={:#010X} A1={:#010X} A2={:#010X} A3={:#010X} A4={:#010X} A5={:#010X} A6/A7={:#010X}/{:#010X}",
                cpu.a(0),
                cpu.a(1),
                cpu.a(2),
                cpu.a(3),
                cpu.a(4),
                cpu.a(5),
                cpu.a(6),
                cpu.a(7)
            );
            // Extra context for interrupt fixture debugging: it checks words on the stack.
            eprintln!(
                "Mem: [A0].w={:#06X} [A6].w={:#06X} [A7].w={:#06X}",
                bus.read_word(cpu.a(0)),
                bus.read_word(cpu.a(6)),
                bus.read_word(cpu.a(7))
            );
            let a6 = cpu.a(6);
            eprintln!("Mem dump @A6={:#010X}:", a6);
            for off in (0..16u32).step_by(2) {
                eprintln!(
                    "  [A6+{:#04X}] = {:#06X}",
                    off,
                    bus.read_word(a6.wrapping_add(off))
                );
            }
            break;
        }
    }

    // Execute remainder
    cpu.execute(&mut bus, max_cycles);

    eprintln!(
        "Final: passes={}, fails={}",
        bus.test_device.pass_count, bus.test_device.fail_count
    );
    if bus.test_device.pass_count == 0 && bus.test_device.fail_count == 0 {
        eprintln!("Trace (first {} instructions):", history.len());
        for h in &history {
            eprintln!("{}", h);
        }
        eprintln!(
            "No pass/fail signaled. PC={:#08X} SR={:#06X} D0={:#010X} D1={:#010X} A0={:#010X}",
            cpu.pc,
            cpu.get_sr(),
            cpu.d(0),
            cpu.d(1),
            cpu.a(0)
        );
    }
    bus.result()
}

// ============================================================================
// MC68000 Tests
// ============================================================================

macro_rules! test_mc68000 {
    ($name:ident, $file:literal) => {
        #[test]
        fn $name() {
            // Use the Musashi test binaries via submodule at tests/fixtures/Musashi/.
            let binary = include_bytes!(concat!("fixtures/Musashi/test/mc68000/", $file));
            let result = run_test(CpuType::M68000, binary, 0x100000);
            assert!(
                result.pass_count > 0 && result.fail_count == 0,
                "Test failed: passes={}, fails={}",
                result.pass_count,
                result.fail_count
            );
        }
    };
}

test_mc68000!(test_mc68000_abcd, "abcd.bin");
test_mc68000!(test_mc68000_add, "add.bin");
test_mc68000!(test_mc68000_add_i, "add_i.bin");
test_mc68000!(test_mc68000_adda, "adda.bin");
test_mc68000!(test_mc68000_addq, "addq.bin");
test_mc68000!(test_mc68000_addx, "addx.bin");
test_mc68000!(test_mc68000_and, "and.bin");
test_mc68000!(test_mc68000_andi_to_ccr, "andi_to_ccr.bin");
test_mc68000!(test_mc68000_andi_to_sr, "andi_to_sr.bin");
test_mc68000!(test_mc68000_bcc, "bcc.bin");
test_mc68000!(test_mc68000_bchg, "bchg.bin");
test_mc68000!(test_mc68000_bclr, "bclr.bin");
test_mc68000!(test_mc68000_bool_i, "bool_i.bin");
test_mc68000!(test_mc68000_bset, "bset.bin");
test_mc68000!(test_mc68000_bsr, "bsr.bin");
test_mc68000!(test_mc68000_btst, "btst.bin");
test_mc68000!(test_mc68000_chk, "chk.bin");
test_mc68000!(test_mc68000_cmp, "cmp.bin");
test_mc68000!(test_mc68000_cmpa, "cmpa.bin");
test_mc68000!(test_mc68000_cmpm, "cmpm.bin");
test_mc68000!(test_mc68000_dbcc, "dbcc.bin");
test_mc68000!(test_mc68000_divs, "divs.bin");
test_mc68000!(test_mc68000_divu, "divu.bin");
test_mc68000!(test_mc68000_eor, "eor.bin");
test_mc68000!(test_mc68000_eori_to_ccr, "eori_to_ccr.bin");
test_mc68000!(test_mc68000_eori_to_sr, "eori_to_sr.bin");
test_mc68000!(test_mc68000_exg, "exg.bin");
test_mc68000!(test_mc68000_ext, "ext.bin");
test_mc68000!(test_mc68000_lea_pea, "lea_pea.bin");
test_mc68000!(test_mc68000_lea_tas, "lea_tas.bin");
test_mc68000!(test_mc68000_lea_tst, "lea_tst.bin");
test_mc68000!(test_mc68000_links, "links.bin");
test_mc68000!(test_mc68000_move, "move.bin");
test_mc68000!(test_mc68000_move_usp, "move_usp.bin");
test_mc68000!(test_mc68000_move_xxx_flags, "move_xxx_flags.bin");
test_mc68000!(test_mc68000_movem, "movem.bin");
test_mc68000!(test_mc68000_movep, "movep.bin");
test_mc68000!(test_mc68000_moveq, "moveq.bin");
test_mc68000!(test_mc68000_muls, "muls.bin");
test_mc68000!(test_mc68000_mulu, "mulu.bin");
test_mc68000!(test_mc68000_nbcd, "nbcd.bin");
test_mc68000!(test_mc68000_negs, "negs.bin");
test_mc68000!(test_mc68000_op_cmp_i, "op_cmp_i.bin");
test_mc68000!(test_mc68000_or, "or.bin");
test_mc68000!(test_mc68000_ori_to_ccr, "ori_to_ccr.bin");
test_mc68000!(test_mc68000_ori_to_sr, "ori_to_sr.bin");
test_mc68000!(test_mc68000_rox, "rox.bin");
test_mc68000!(test_mc68000_roxx, "roxx.bin");
test_mc68000!(test_mc68000_rtr, "rtr.bin");
test_mc68000!(test_mc68000_sbcd, "sbcd.bin");
test_mc68000!(test_mc68000_scc, "scc.bin");
test_mc68000!(test_mc68000_shifts, "shifts.bin");
test_mc68000!(test_mc68000_shifts2, "shifts2.bin");
test_mc68000!(test_mc68000_sub, "sub.bin");
test_mc68000!(test_mc68000_sub_i, "sub_i.bin");
test_mc68000!(test_mc68000_suba, "suba.bin");
test_mc68000!(test_mc68000_subq, "subq.bin");
test_mc68000!(test_mc68000_subx, "subx.bin");
test_mc68000!(test_mc68000_swap, "swap.bin");
test_mc68000!(test_mc68000_trapv, "trapv.bin");

// ============================================================================
// MC68040 Tests (68020+ features)
// ============================================================================

macro_rules! test_mc68040 {
    ($name:ident, $file:literal) => {
        #[test]
        fn $name() {
            // Use the Musashi test binaries via submodule at tests/fixtures/Musashi/.
            let binary = include_bytes!(concat!("fixtures/Musashi/test/mc68040/", $file));
            let result = run_test(CpuType::M68040, binary, 0x100000);
            assert!(
                result.pass_count > 0 && result.fail_count == 0,
                "Test failed: passes={}, fails={}",
                result.pass_count,
                result.fail_count
            );
        }
    };
}

test_mc68040!(test_mc68040_bfchg, "bfchg.bin");
test_mc68040!(test_mc68040_bfclr, "bfclr.bin");
test_mc68040!(test_mc68040_bfext, "bfext.bin");
test_mc68040!(test_mc68040_bfffo, "bfffo.bin");
test_mc68040!(test_mc68040_bfins, "bfins.bin");
test_mc68040!(test_mc68040_bfset, "bfset.bin");
test_mc68040!(test_mc68040_bftst, "bftst.bin");
test_mc68040!(test_mc68040_cas, "cas.bin");
test_mc68040!(test_mc68040_chk2, "chk2.bin");
test_mc68040!(test_mc68040_cmp2, "cmp2.bin");
test_mc68040!(test_mc68040_divs_long, "divs_long.bin");
test_mc68040!(test_mc68040_divu_long, "divu_long.bin");
test_mc68040!(test_mc68040_interrupt, "interrupt.bin");
test_mc68040!(test_mc68040_jmp, "jmp.bin");
test_mc68040!(test_mc68040_mul_long, "mul_long.bin");
test_mc68040!(test_mc68040_rtd, "rtd.bin");
test_mc68040!(test_mc68040_shifts3, "shifts3.bin");
test_mc68040!(test_mc68040_trapcc, "trapcc.bin");

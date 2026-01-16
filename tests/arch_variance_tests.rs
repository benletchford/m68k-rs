mod common;
use common::TestBus;
use m68k::NoOpHleHandler;
use m68k::core::cpu::CpuCore;
use m68k::core::types::CpuType;

fn run_test_inspect(cpu_type: CpuType, binary: &[u8], max_instructions: i32) -> u32 {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);

    let mut bus = TestBus::new();
    bus.load_rom(binary);
    bus.setup_boot();

    cpu.reset(&mut bus);
    cpu.pc = 0x10000;
    cpu.set_sr(0x2700);

    let mut hle = NoOpHleHandler;
    for _ in 0..max_instructions {
        if cpu.is_stopped() || cpu.is_halted() {
            break;
        }
        cpu.step_with_hle_handler(&mut bus, &mut hle);

        // `is_halted()` surfaces double-fault halts.
    }

    // Return D0 register value
    cpu.dar[0]
}

// Architecture Variance Tests
// Tests behaviors that differ between CPU models

#[test]
fn test_unaligned_68000() {
    let binary = include_bytes!("fixtures/extra/m68040/bin/arch_unaligned.bin");
    // 68000: Unaligned access -> Address Error -> D0=1
    let d0 = run_test_inspect(CpuType::M68000, binary, 2000);
    assert_eq!(
        d0, 1,
        "68000 should trigger Address Error on unaligned access (D0=1)"
    );
}

#[test]
fn test_unaligned_68020() {
    let binary = include_bytes!("fixtures/extra/m68040/bin/arch_unaligned.bin");
    // 68020+: Unaligned access -> Success -> D0=0
    let d0 = run_test_inspect(CpuType::M68020, binary, 2000);
    assert_eq!(d0, 0, "68020+ should handle unaligned access (D0=0)");
}

#[test]
fn test_fpu_lc040() {
    let binary = include_bytes!("fixtures/extra/m68040/bin/arch_fpu_trap.bin");
    // LC040: No FPU -> Line-F Trap -> D0=1
    let d0 = run_test_inspect(CpuType::M68LC040, binary, 2000);
    assert_eq!(
        d0, 1,
        "LC040 should trigger Line-F on FPU instruction (D0=1)"
    );
}

#[test]
fn test_fpu_040() {
    let binary = include_bytes!("fixtures/extra/m68040/bin/arch_fpu_trap.bin");
    // 68040: FPU present -> Success -> D0=0
    let d0 = run_test_inspect(CpuType::M68040, binary, 2000);
    assert_eq!(
        d0, 0,
        "68040 should execute FPU instruction successfully (D0=0)"
    );
}

#[test]
fn test_callm_68020() {
    let binary = include_bytes!("fixtures/extra/m68020/bin/callm_020.bin");
    // 68020: CALLM executes -> Jumps to module -> RTM returns -> D0=0
    let d0 = run_test_inspect(CpuType::M68020, binary, 2000);
    assert_eq!(d0, 0, "68020 should execute CALLM without trap (D0=0)");
}

#[test]
fn test_callm_68040() {
    let binary = include_bytes!("fixtures/extra/m68020/bin/callm_020.bin");
    // 68040: CALLM -> Line-F Trap -> D0=1
    let d0 = run_test_inspect(CpuType::M68040, binary, 2000);
    assert_eq!(d0, 1, "68040 should trigger Line-F on CALLM (D0=1)");
}

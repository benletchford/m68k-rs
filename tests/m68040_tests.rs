//! Extra 68040 Test Fixtures
//!
//! Tests 68020+ features not covered by the original Musashi test suite.

mod common;
use m68k::core::types::CpuType;

// ============================================================================
// Addressing Mode Extensions (68020+)
// ============================================================================
test_fixture!(
    test_mc68040_32bit_disp,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/32bit_disp.bin"
);
test_fixture!(
    test_mc68040_scale_index,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/scale_index.bin"
);
test_fixture!(
    test_mc68040_mem_indirect,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mem_indirect.bin"
);
test_fixture!(
    test_mc68040_pc_indirect,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/pc_indirect.bin"
);

// ============================================================================
// Data Movement Extensions
// ============================================================================
test_fixture!(
    test_mc68040_extb,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/extb.bin"
);
test_fixture!(
    test_mc68040_link32,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/link32.bin"
);
test_fixture!(
    test_mc68040_move16,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/move16.bin"
);
test_fixture!(
    test_mc68040_moves,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/moves.bin"
);
test_fixture!(
    test_mc68040_tas,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/tas.bin"
);

// ============================================================================
// BCD Extensions
// ============================================================================
test_fixture!(
    test_mc68040_pack,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/pack.bin"
);
test_fixture!(
    test_mc68040_unpk,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/unpk.bin"
);

// ============================================================================
// System Control
// ============================================================================
test_fixture!(
    test_mc68040_movec,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/movec.bin"
);
test_fixture!(
    test_mc68040_cache_ops,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/cache_ops.bin"
);
test_fixture!(
    test_mc68040_pflush,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/pflush.bin"
);
test_fixture!(
    test_mc68040_reset,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/reset.bin"
);

// ============================================================================
// Exception Handling
// ============================================================================
test_fixture!(
    test_mc68040_bkpt,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/bkpt.bin"
);
test_fixture!(
    test_mc68040_illegal,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/illegal.bin"
);
test_fixture!(
    test_mc68040_trace_modes,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/trace_modes.bin"
);
test_fixture!(
    test_mc68040_exception_frames,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/exception_frames.bin"
);

// ============================================================================
// Arithmetic and Control Flow
// ============================================================================
test_fixture!(
    test_mc68040_multiprec,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/multiprec.bin"
);
test_fixture!(
    test_mc68040_rtd,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/rtd.bin"
);
test_fixture!(
    test_mc68040_trapcc,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/trapcc.bin"
);

// ============================================================================
// FPU Operations
// ============================================================================
test_fixture!(
    test_mc68040_fpu_arith,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_arith.bin"
);
test_fixture!(
    test_mc68040_fpu_move,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_move.bin"
);
test_fixture!(
    test_mc68040_fpu_branch,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_branch.bin"
);
test_fixture!(
    test_mc68040_fpu_trans,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_trans.bin"
);
test_fixture!(
    test_mc68040_fpu_ctrl,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_ctrl.bin"
);
test_fixture!(
    test_mc68040_fpu_double,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_double.bin"
);
test_fixture!(
    test_mc68040_fpu_unimp,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_unimp.bin"
);

// ============================================================================
// MMU Operations
// ============================================================================
test_fixture!(
    test_mc68040_mmu_ptest,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_ptest.bin"
);
test_fixture!(
    test_mc68040_mmu_regs,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_regs.bin"
);
test_fixture!(
    test_mc68040_mmu_ttr,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_ttr.bin"
);
test_fixture!(
    test_mc68040_mmu_atc,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_atc.bin"
);
test_fixture!(
    test_mc68040_mmu_tc,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_tc.bin"
);

// ============================================================================
// Interrupt Handling
// ============================================================================
test_fixture!(
    test_mc68040_interrupts,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/interrupts.bin"
);

// ============================================================================
// Edge Cases
// ============================================================================
test_fixture!(
    test_mc68040_fpu_except,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_except.bin"
);
test_fixture!(
    test_mc68040_callm_rtm,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/callm_rtm.bin"
);

// ============================================================================
// General Stress Tests
// ============================================================================
test_fixture!(
    test_mc68040_smc_test,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/smc_test.bin"
);
test_fixture!(
    test_mc68040_double_fault,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/double_fault.bin"
);
test_fixture!(
    test_mc68040_address_stress,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/address_stress.bin"
);

// ============================================================================
// Architecture Variance Tests (EC040/LC040)
// ============================================================================
test_fixture!(
    test_mc68040_arch_unaligned,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/arch_unaligned.bin"
);
test_fixture!(
    test_mc68040_arch_fpu_trap,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/arch_fpu_trap.bin"
);
test_fixture!(
    test_mc68040_ec040_mmu_trap,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/ec040_mmu_trap.bin"
);
test_fixture!(
    test_mc68040_lc040_mmu_positive,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/lc040_mmu_positive.bin"
);
test_fixture!(
    test_mc68040_ec040_positive,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/ec040_positive.bin"
);

// ============================================================================
// Gap Tests: FPU Transcendental Functions
// ============================================================================
test_fixture!(
    test_mc68040_fpu_transcendental,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_transcendental.bin"
);
test_fixture!(
    test_mc68040_fpu_transcendental2,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_transcendental2.bin"
);
test_fixture!(
    test_mc68040_fpu_exp_log,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_exp_log.bin"
);

// ============================================================================
// Gap Tests: MMU Permissions and Exception Frames
// ============================================================================
test_fixture!(
    test_mc68040_mmu_permissions,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_permissions.bin"
);
test_fixture!(
    test_mc68040_exception_frame_format,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/exception_frame_format.bin"
);

// ============================================================================
// Gap Tests: Additional FPU Operations
// ============================================================================
test_fixture!(
    test_mc68040_fpu_basic_ops,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_basic_ops.bin"
);
test_fixture!(
    test_mc68040_fpu_sincos,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_sincos.bin"
);
test_fixture!(
    test_mc68040_fpu_rounding,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_rounding.bin"
);
test_fixture!(
    test_mc68040_fpu_constants,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_constants.bin"
);
test_fixture!(
    test_mc68040_fpu_remainder,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_remainder.bin"
);
test_fixture!(
    test_mc68040_fpu_scale,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/fpu_scale.bin"
);

// ============================================================================
// Gap Tests: System and Timing
// ============================================================================
test_fixture!(
    test_mc68040_movec_040,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/movec_040.bin"
);
test_fixture!(
    test_mc68040_cycle_timing,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/cycle_timing.bin"
);
test_fixture!(
    test_mc68040_bus_error,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/bus_error.bin"
);

// ============================================================================
// Gap Tests: Cycle Timing (expanded)
// ============================================================================
test_fixture!(
    test_mc68040_cycle_exception,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/cycle_exception.bin"
);
test_fixture!(
    test_mc68040_cycle_addressing,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/cycle_addressing.bin"
);

// ============================================================================
// Gap Tests: MMU Table Walk / ATC
// ============================================================================
test_fixture!(
    test_mc68040_mmu_table_walk,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_table_walk.bin"
);
test_fixture!(
    test_mc68040_mmu_atc_test,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_atc_test.bin"
);
test_fixture!(
    test_mc68040_mmu_ttr_config,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/mmu_ttr_config.bin"
);

// ============================================================================
// Gap Tests: Disassembler Coverage
// ============================================================================
test_fixture!(
    test_mc68040_dasm_data_move,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/dasm_data_move.bin"
);
test_fixture!(
    test_mc68040_dasm_arithmetic,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/dasm_arithmetic.bin"
);
test_fixture!(
    test_mc68040_dasm_logical,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/dasm_logical.bin"
);
test_fixture!(
    test_mc68040_dasm_shift,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/dasm_shift.bin"
);
test_fixture!(
    test_mc68040_dasm_branch,
    CpuType::M68040,
    "fixtures/extra/m68040/bin/dasm_branch.bin"
);

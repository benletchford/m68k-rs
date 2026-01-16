mod common;
use m68k::core::types::CpuType;

// Systematic instruction coverage tests on M68040 (should work on all CPUs)

// Bit manipulation
test_fixture!(
    test_bset_bclr,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/bset_bclr.bin"
);
test_fixture!(
    test_bchg_btst,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/bchg_btst.bin"
);
test_fixture!(
    test_bit_edge_cases,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/bit_edge_cases.bin"
);
test_fixture!(
    test_bit_memory_ea,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/bit_memory_ea.bin"
);

// Shift/rotate
test_fixture!(
    test_asl_asr,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/asl_asr.bin"
);
test_fixture!(
    test_lsl_lsr,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/lsl_lsr.bin"
);
test_fixture!(
    test_rol_ror,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/rol_ror.bin"
);
test_fixture!(
    test_roxl_roxr,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/roxl_roxr.bin"
);
test_fixture!(
    test_shift_count_zero,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/shift_count_zero.bin"
);
test_fixture!(
    test_shift_count_large,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/shift_count_large.bin"
);
test_fixture!(
    test_shift_memory,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/shift_memory.bin"
);
test_fixture!(
    test_shift_flags,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/shift_flags.bin"
);

// Condition codes
test_fixture!(
    test_bcc_all,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/bcc_all.bin"
);
test_fixture!(
    test_dbcc_all,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/dbcc_all.bin"
);
test_fixture!(
    test_scc_all,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/scc_all.bin"
);
test_fixture!(
    test_trapcc,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/trapcc.bin"
);
test_fixture!(
    test_condition_edge,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/condition_edge.bin"
);

// Multiply/divide
test_fixture!(
    test_mulu_edge,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/mulu_edge.bin"
);
test_fixture!(
    test_muls_edge,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/muls_edge.bin"
);
test_fixture!(
    test_divu_divs_edge,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/divu_divs_edge.bin"
);

// String/block operations
test_fixture!(
    test_cmpm_all,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/cmpm_all.bin"
);
test_fixture!(
    test_movem_patterns,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/movem_patterns.bin"
);
test_fixture!(
    test_movem_stack,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/movem_stack.bin"
);

// Miscellaneous
test_fixture!(
    test_exg_variants,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/exg_variants.bin"
);
test_fixture!(
    test_swap_ext,
    CpuType::M68040,
    "fixtures/extra/coverage/bin/swap_ext.bin"
);

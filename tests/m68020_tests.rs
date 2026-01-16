mod common;
use m68k::core::types::CpuType;

// 68020+ Standard Features
test_fixture!(
    test_m68020_bitfield,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/bitfield_ops.bin"
);
test_fixture!(
    test_m68020_cas,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/cas.bin"
);
test_fixture!(
    test_m68020_chk2_cmp2,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/chk2_cmp2.bin"
);
test_fixture!(
    test_m68020_long_muldiv,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/long_muldiv.bin"
);
test_fixture!(
    test_m68020_boundary_edge,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/boundary_edge.bin"
);
test_fixture!(
    test_m68020_msp_test,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/msp_test.bin"
);
test_fixture!(
    test_m68020_trace_t0,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/trace_t0.bin"
);

// CALLM 020 (should pass on 020)
test_fixture!(
    test_m68020_callm,
    CpuType::M68020,
    "fixtures/extra/m68020/bin/callm_020.bin"
);

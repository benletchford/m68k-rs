mod common;
use m68k::core::types::CpuType;

// M68010-specific features
test_fixture!(
    test_m68010_vbr,
    CpuType::M68010,
    "fixtures/extra/m68010/bin/vbr_test.bin"
);
test_fixture!(
    test_m68010_move_ccr,
    CpuType::M68010,
    "fixtures/extra/m68010/bin/move_ccr.bin"
);
test_fixture!(
    test_m68010_rte,
    CpuType::M68010,
    "fixtures/extra/m68010/bin/rte_010.bin"
);
test_fixture!(
    test_m68010_loop_mode,
    CpuType::M68010,
    "fixtures/extra/m68010/bin/loop_mode.bin"
);

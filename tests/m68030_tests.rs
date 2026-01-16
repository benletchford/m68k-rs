mod common;
use m68k::core::types::CpuType;

// 68030-specific features
test_fixture!(
    test_m68030_mmu_tc,
    CpuType::M68030,
    "fixtures/extra/m68030/bin/mmu_030_tc.bin"
);
test_fixture!(
    test_m68030_cache,
    CpuType::M68030,
    "fixtures/extra/m68030/bin/cache_030.bin"
);
test_fixture!(
    test_m68030_move16,
    CpuType::M68030,
    "fixtures/extra/m68030/bin/move16_030.bin"
);

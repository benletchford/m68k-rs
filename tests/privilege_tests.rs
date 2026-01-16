mod common;
use m68k::core::types::CpuType;

// Test privilege violations on M68040 (should behave identically on all CPUs)

test_fixture!(
    test_user_movec,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/user_movec.bin"
);
test_fixture!(
    test_user_reset,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/user_reset.bin"
);
test_fixture!(
    test_user_stop,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/user_stop.bin"
);
test_fixture!(
    test_user_rte,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/user_rte.bin"
);
test_fixture!(
    test_usp_ssp_switch,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/usp_ssp_switch.bin"
);
test_fixture!(
    test_move_usp,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/move_usp.bin"
);
test_fixture!(
    test_trap_privilege,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/trap_privilege.bin"
);
test_fixture!(
    test_exception_privilege,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/exception_privilege.bin"
);
test_fixture!(
    test_rte_privilege,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/rte_privilege.bin"
);
test_fixture!(
    test_privilege_boundary,
    CpuType::M68040,
    "fixtures/extra/privilege/bin/privilege_boundary.bin"
);

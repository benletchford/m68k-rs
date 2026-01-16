mod common;
use m68k::core::types::CpuType;

// =============================================================================
// Exception Handling Tests
// =============================================================================
// Tests are organized by exception type and CPU compatibility.

// -----------------------------------------------------------------------------
// Address Error Tests (M68000 only)
// -----------------------------------------------------------------------------
// These tests rely on Address Error exceptions triggered by misaligned word
// accesses. 68020+ CPUs handle misaligned accesses in hardware without
// generating exceptions, so these tests must run on M68000/M68010.
//
// 68000 Address Error stack frame layout (14 bytes):
//   SP+0:  Status Word (2)
//   SP+2:  Access Address (4)
//   SP+6:  Instruction Register (2)
//   SP+8:  SR (2)
//   SP+10: PC (4)

test_fixture!(
    test_double_exception,
    CpuType::M68000,
    "fixtures/extra/exceptions/bin/double_exception.bin"
);
test_fixture!(
    test_exception_priority,
    CpuType::M68000,
    "fixtures/extra/exceptions/bin/exception_priority.bin"
);
test_fixture!(
    test_rte_validation,
    CpuType::M68000,
    "fixtures/extra/exceptions/bin/rte_validation.bin"
);

// -----------------------------------------------------------------------------
// Interrupt Tests (All CPUs)
// -----------------------------------------------------------------------------
// These tests use interrupt vectors and TRAP instructions which work
// correctly on all CPU types.

test_fixture!(
    test_interrupt_nesting,
    CpuType::M68040,
    "fixtures/extra/exceptions/bin/interrupt_nesting.bin"
);
test_fixture!(
    test_spurious_interrupt,
    CpuType::M68040,
    "fixtures/extra/exceptions/bin/spurious_interrupt.bin"
);

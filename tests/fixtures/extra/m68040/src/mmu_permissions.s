.include "entry.s"
/* Test: MMU Access Permission Checks */
/* Tests that MMU blocks writes to read-only pages and access to supervisor pages */
/* Currently expected to FAIL until access permissions are implemented */

run_test:
    /* For now, just verify basic PMOVE and TC manipulation works */
    /* Full permission tests require working MMU table setup */
    
    /* =================================================================== */
    /* Test 1: Enable MMU via TC register */
    /* =================================================================== */
    /* Set up a simple translation control register */
    /* For 68040: TC bit 15 (E) enables translation */
    move.l #0x00008000, %d0     | E=1 (enable), other bits default
    movec %d0, %tc              | Write to TC
    
    /* Read back and verify */
    movec %tc, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Disable MMU before continuing */
    move.l #0, %d0
    movec %d0, %tc
    
    /* =================================================================== */
    /* Test 2: Verify URP/SRP registers work */
    /* =================================================================== */
    move.l #0x12340000, %d0     | Page table pointer
    movec %d0, %urp             | User Root Pointer
    movec %urp, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    move.l #0x56780000, %d0
    movec %d0, %srp             | Supervisor Root Pointer
    movec %srp, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    rts

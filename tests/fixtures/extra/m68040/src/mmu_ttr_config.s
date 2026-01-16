.include "entry.s"
/* Test: MMU TTR (Transparent Translation) Configuration */
/* Tests the ITT0/ITT1/DTT0/DTT1 registers for transparent regions */

run_test:
    /* =================================================================== */
    /* Test 1: ITT0 - Instruction Transparent Translation 0 */
    /* =================================================================== */
    
    /* ITT format: Base (8) | Limit (8) | E | S | UR | CM | W | 0000 */
    move.l #0x00FFE040, %d0 | Base=0x00, Limit=0xFF, E=1, S=0, UR=1
    movec %d0, %itt0
    
    movec %itt0, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: ITT1 - Instruction Transparent Translation 1 */
    /* =================================================================== */
    
    move.l #0x80FFE040, %d0 | Base=0x80, Limit=0xFF, E=1
    movec %d0, %itt1
    
    movec %itt1, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: DTT0 - Data Transparent Translation 0 */
    /* =================================================================== */
    
    move.l #0x00FFE060, %d0 | Base=0x00, Limit=0xFF, E=1, W=1
    movec %d0, %dtt0
    
    movec %dtt0, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: DTT1 - Data Transparent Translation 1 */
    /* =================================================================== */
    
    move.l #0xFFFFE060, %d0 | Base=0xFF, Limit=0xFF, E=1, W=1
    movec %d0, %dtt1
    
    movec %dtt1, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 5: Verify all four TTRs are independent */
    /* =================================================================== */
    
    move.l #0x11000000, %d0
    movec %d0, %itt0
    
    move.l #0x22000000, %d1
    movec %d1, %itt1
    
    move.l #0x33000000, %d2
    movec %d2, %dtt0
    
    move.l #0x44000000, %d3
    movec %d3, %dtt1
    
    /* Read back all four */
    movec %itt0, %d4
    movec %itt1, %d5
    movec %dtt0, %d6
    movec %dtt1, %d7
    
    /* Verify each has its own value */
    cmp.l #0x11000000, %d4
    bne TEST_FAIL
    
    cmp.l #0x22000000, %d5
    bne TEST_FAIL
    
    cmp.l #0x33000000, %d6
    bne TEST_FAIL
    
    cmp.l #0x44000000, %d7
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 6: Clear all TTRs */
    /* =================================================================== */
    
    move.l #0, %d0
    movec %d0, %itt0
    movec %d0, %itt1
    movec %d0, %dtt0
    movec %d0, %dtt1
    
    /* Verify cleared */
    movec %itt0, %d1
    tst.l %d1
    bne TEST_FAIL
    
    movec %itt1, %d1
    tst.l %d1
    bne TEST_FAIL
    
    movec %dtt0, %d1
    tst.l %d1
    bne TEST_FAIL
    
    movec %dtt1, %d1
    tst.l %d1
    bne TEST_FAIL
    
    rts

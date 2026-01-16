.include "entry.s"
/* Test: MMU Transparent Translation (TTR) */

run_test:
    /* Test 1: Configure DTT0 for RAM region */
    /* DTT0: Base=0x00, Mask=0xFF, E=1, S=0, CM=00, W=0 */
    move.l #0x00FFC000, %d0
    movec %d0, %dtt0
    
    /* Read back and verify */
    movec %dtt0, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Test 2: Configure DTT1 */
    move.l #0x10FFC000, %d0
    movec %d0, %dtt1
    movec %dtt1, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Test 3: Configure ITT0 */
    move.l #0x00FFC000, %d0
    movec %d0, %itt0
    movec %itt0, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Test 4: Disable all TTRs */
    move.l #0, %d0
    movec %d0, %dtt0
    movec %d0, %dtt1
    movec %d0, %itt0
    movec %d0, %itt1
    
    rts

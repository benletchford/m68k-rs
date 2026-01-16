.include "entry.s"
/* Test: MOVEC - Move Control Register (68010+) */

run_test:
    /* Test 1: Read VBR */
    movec %vbr, %d0
    /* Just checking it doesn't crash */
    
    /* Test 2: Write and read VBR */
    move.l #0x1000, %d0
    movec %d0, %vbr
    movec %vbr, %d1
    cmp.l #0x1000, %d1
    bne TEST_FAIL
    
    /* Restore VBR to 0 */
    move.l #0, %d0
    movec %d0, %vbr
    
    /* Test 3: Read SFC */
    movec %sfc, %d0
    
    /* Test 4: Write SFC */
    move.l #5, %d0
    movec %d0, %sfc
    movec %sfc, %d1
    and.l #7, %d1
    cmp.l #5, %d1
    bne TEST_FAIL
    
    /* Test 5: Write DFC */
    move.l #5, %d0
    movec %d0, %dfc
    movec %dfc, %d1
    and.l #7, %d1
    cmp.l #5, %d1
    bne TEST_FAIL
    
    /* Test 6: CACR (68020+) */
    move.l #0, %d0
    movec %d0, %cacr
    movec %cacr, %d1
    /* Value may be masked, just check no crash */
    
    rts

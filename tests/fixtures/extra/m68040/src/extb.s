.include "entry.s"
/* Test: EXTB.L - Sign Extend Byte to Long (68020+) */

run_test:
    /* Test 1: Positive byte to long */
    move.l #0xFFFFFF7F, %d0
    extb.l %d0
    cmp.l #0x0000007F, %d0
    bne TEST_FAIL
    
    /* Test 2: Negative byte to long */
    move.l #0x00000080, %d0
    extb.l %d0
    cmp.l #0xFFFFFF80, %d0
    bne TEST_FAIL
    
    /* Test 3: Zero */
    move.l #0xFFFFFF00, %d0
    extb.l %d0
    cmp.l #0x00000000, %d0
    bne TEST_FAIL
    
    /* Test 4: -1 */
    move.l #0x000000FF, %d0
    extb.l %d0
    cmp.l #0xFFFFFFFF, %d0
    bne TEST_FAIL
    
    /* Test 5: 0x55 (positive) */
    move.l #0x12345655, %d0
    extb.l %d0
    cmp.l #0x00000055, %d0
    bne TEST_FAIL
    
    /* Test 6: 0xAA (negative) */
    move.l #0x123456AA, %d0
    extb.l %d0
    cmp.l #0xFFFFFFAA, %d0
    bne TEST_FAIL
    
    /* Test 7: Check Z flag on zero */
    move.l #0xFFFFFF00, %d0
    extb.l %d0
    bne TEST_FAIL
    
    /* Test 8: Check N flag on negative */
    move.l #0x00000080, %d0
    extb.l %d0
    bpl TEST_FAIL
    
    rts

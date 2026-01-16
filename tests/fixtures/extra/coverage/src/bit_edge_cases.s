.include "entry.s"
/* Test: Bit operations at boundaries */

run_test:
    clr.l %d0
    
    /* Test bit 0 */
    clr.l %d1
    bset #0, %d1
    cmp.l #1, %d1
    bne TEST_FAIL
    
    /* Test bit 31 on long */
    clr.l %d2
    bset #31, %d2
    cmp.l #0x80000000, %d2
    bne TEST_FAIL
    
    /* Test bit 7 on byte */
    move.l #0x2000, %a0
    clr.b (%a0)
    bset #7, (%a0)
    move.b (%a0), %d3
    cmp.b #0x80, %d3
    bne TEST_FAIL
    
    /* Test BTST beyond byte boundary (should wrap mod 8 for memory) */
    move.b #1, (%a0)
    btst #0, (%a0)
    bne 1f
    bra TEST_FAIL
1:
    
    /* Test all bits in sequence */
    clr.l %d4
    moveq #0, %d5
2:  bset %d5, %d4
    addq.l #1, %d5
    cmp.l #32, %d5
    blt 2b
    
    cmp.l #0xFFFFFFFF, %d4
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

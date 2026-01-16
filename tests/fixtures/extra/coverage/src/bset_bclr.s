.include "entry.s"
/* Test: BSET and BCLR on data registers and memory */

run_test:
    clr.l %d0
    
    /* Test BSET on data register */
    clr.l %d1
    bset #7, %d1
    cmp.l #0x80, %d1
    bne TEST_FAIL
    
    bset #0, %d1
    cmp.l #0x81, %d1
    bne TEST_FAIL
    
    /* Test BCLR on data register */
    move.l #0xFF, %d2
    bclr #3, %d2
    cmp.l #0xF7, %d2
    bne TEST_FAIL
    
    /* Test BSET on memory */
    move.l #0x2000, %a0
    clr.b (%a0)
    bset #5, (%a0)
    move.b (%a0), %d3
    cmp.b #0x20, %d3
    bne TEST_FAIL
    
    /* Test BCLR on memory */
    move.b #0xFF, (%a0)
    bclr #2, (%a0)
    move.b (%a0), %d3
    cmp.b #0xFB, %d3
    bne TEST_FAIL
    
    /* Test dynamic bit number */
    move.l #4, %d4
    clr.l %d5
    bset %d4, %d5
    cmp.l #0x10, %d5
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

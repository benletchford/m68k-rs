.include "entry.s"
/* Test: MOVEM with various register masks */

run_test:
    clr.l %d0
    
    /* Setup test values */
    move.l #0x11111111, %d1
    move.l #0x22222222, %d2
    move.l #0x33333333, %d3
    move.l #0x1000, %a2
    move.l #0x2000, %a3
    
    /* MOVEM to memory */
    move.l #0x3000, %a0
    movem.l %d1-%d3/%a2-%a3, (%a0)
    
    /* Clear registers */
    clr.l %d1
    clr.l %d2
    clr.l %d3
    suba.l %a2, %a2          /* clr doesn't work on address regs */
    suba.l %a3, %a3
    
    /* MOVEM from memory */
    movem.l (%a0), %d1-%d3/%a2-%a3
    
    /* Verify */
    cmp.l #0x11111111, %d1
    bne TEST_FAIL
    cmp.l #0x22222222, %d2
    bne TEST_FAIL
    cmp.l #0x33333333, %d3
    bne TEST_FAIL
    cmp.l #0x1000, %a2
    bne TEST_FAIL
    cmp.l #0x2000, %a3
    bne TEST_FAIL
    
    /* Test single register */
    move.l #0x3100, %a1
    move.l #0xAAAAAAAA, %d4
    movem.l %d4, (%a1)
    clr.l %d4
    movem.l (%a1), %d4
    cmp.l #0xAAAAAAAA, %d4
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

.include "entry.s"
/* Test: MOVEM with -(An) and (An)+ modes */

run_test:
    clr.l %d0
    
    /* MOVEM with -(An) - predecrement */
    move.l #0x12345678, %d1
    move.l #0x9ABCDEF0, %d2
    
    move.l #0x3020, %a0     | Point past end
    movem.l %d1-%d2, -(%a0)
    
    /* a0 should be decremented */
    cmp.l #0x3018, %a0
    bne TEST_FAIL
    
    /* Clear and restore */
    clr.l %d1
    clr.l %d2
    
    movem.l (%a0)+, %d1-%d2
    
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    cmp.l #0x9ABCDEF0, %d2
    bne TEST_FAIL
    cmp.l #0x3020, %a0
    bne TEST_FAIL
    
    /* Test address registers */
    move.l #0x5000, %a2
    move.l #0x6000, %a3
    
    move.l #0x3100, %a4
    movem.l %a2-%a3, -(%a4)
    
    suba.l %a2, %a2          /* clr doesn't work on address regs */
    suba.l %a3, %a3
    
    /* A4 is already at $30F8 after MOVEM predecrement, no need to subtract */
    movem.l (%a4)+, %a2-%a3
    
    cmp.l #0x5000, %a2
    bne TEST_FAIL
    cmp.l #0x6000, %a3
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

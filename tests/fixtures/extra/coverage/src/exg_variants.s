.include "entry.s"
/* Test: EXG all mode combinations */

run_test:
    clr.l %d0
    
    /* EXG Dn, Dn */
    move.l #0x11111111, %d1
    move.l #0x22222222, %d2
    exg %d1, %d2
    cmp.l #0x22222222, %d1
    bne TEST_FAIL
    cmp.l #0x11111111, %d2
    bne TEST_FAIL
    
    /* EXG An, An */
    move.l #0x33333333, %a1
    move.l #0x44444444, %a2
    exg %a1, %a2
    cmp.l #0x44444444, %a1
    bne TEST_FAIL
    cmp.l #0x33333333, %a2
    bne TEST_FAIL
    
    /* EXG Dn, An */
    move.l #0x55555555, %d3
    move.l #0x66666666, %a3
    exg %d3, %a3
    cmp.l #0x66666666, %d3
    bne TEST_FAIL
    cmp.l #0x55555555, %a3
    bne TEST_FAIL
    
    /* EXG with same register (should be no-op) */
    move.l #0x77777777, %d4
    exg %d4, %d4
    cmp.l #0x77777777, %d4
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

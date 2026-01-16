.include "entry.s"
/* Test: TRAPcc - Trap on Condition (68020+) */

.set TRAPV_VECTOR, 0x1C

run_test:
    move.l TRAPV_VECTOR, %d7
    lea trap_handler, %a0
    move.l %a0, TRAPV_VECTOR
    clr.l %d6
    
    /* Test 1: TRAPF - never trap */
    trapf
    tst.l %d6
    bne TEST_FAIL
    
    /* Test 2: TRAPT - always trap */
    trapt
    cmp.l #1, %d6
    bne TEST_FAIL
    
    /* Test 3: TRAPEQ when Z set */
    clr.l %d6
    move.l #0, %d0
    tst.l %d0
    trapeq
    cmp.l #1, %d6
    bne TEST_FAIL
    
    /* Test 4: TRAPEQ when Z clear - no trap */
    clr.l %d6
    move.l #1, %d0
    tst.l %d0
    trapeq
    tst.l %d6
    bne TEST_FAIL
    
    move.l %d7, TRAPV_VECTOR
    rts

trap_handler:
    addq.l #1, %d6
    /* TRAPcc exception frame pushes PC of next instruction, so no adjustment needed */
    rte


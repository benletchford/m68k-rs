.include "entry.s"
/* Test: MOVE to/from USP in different privilege modes */

run_test:
    clr.l %d0
    
    /* Test 1: MOVE USP in supervisor mode (should work) */
    move.l #0x12345678, %a0
    move.l %a0, %usp
    move.l %usp, %a1
    cmp.l %a0, %a1
    bne TEST_FAIL
    
    /* Test 2: MOVE USP in user mode (should trap) */
    lea priv_handler, %a2
    move.l %a2, 0x20
    
    /* Switch to user mode */
    andi.w #0xDFFF, %sr
    
    /* Attempt MOVE USP in user mode */
    move.l %usp, %a3        | Should trap
    
    /* Should never reach here */
    rts

priv_handler:
    move.l #1, %d0
    move.w 0(%sp), %d1
    ori.w #0x2000, %d1
    move.w %d1, 0(%sp)
    addq.l #2, 2(%sp)
    rte

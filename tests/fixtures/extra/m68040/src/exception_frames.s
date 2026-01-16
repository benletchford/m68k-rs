.include "entry.s"
/* Test: Exception Frame Formats */

.set TRAP_1_VECTOR, 0x84

run_test:
    move.l TRAP_1_VECTOR, %d7
    lea trap_handler, %a0
    move.l %a0, TRAP_1_VECTOR
    clr.l %d6
    
    trap #1
    
    cmp.l #1, %d6
    bne TEST_FAIL
    
    move.l %d7, TRAP_1_VECTOR
    rts

trap_handler:
    move.w 6(%sp), %d0
    lsr.w #8, %d0
    lsr.w #4, %d0
    and.w #0xF, %d0
    cmp.w #0, %d0
    bne trap_fail
    move.l #1, %d6
    rte
trap_fail:
    clr.l %d6
    rte

.include "entry.s"
/* Test: ILLEGAL Instruction Exception */

.set ILLEGAL_VECTOR, 0x10

run_test:
    move.l ILLEGAL_VECTOR, %d7
    lea illegal_handler, %a0
    move.l %a0, ILLEGAL_VECTOR
    clr.l %d6
    
    illegal
    
    cmp.l #1, %d6
    bne TEST_FAIL
    
    move.l %d7, ILLEGAL_VECTOR
    rts

illegal_handler:
    move.l #1, %d6
    addq.l #2, 2(%sp)
    rte

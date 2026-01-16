.include "entry.s"
/* Test: BKPT - Breakpoint Instruction (68010+) */
/* BKPT triggers Illegal Instruction exception (vector 4) on 68000 family */

.set BKPT_VECTOR, 0x10   | Vector 4 (Illegal Instruction) = address 4*4 = 0x10

run_test:
    move.l BKPT_VECTOR, %d7
    lea bkpt_handler, %a0
    move.l %a0, BKPT_VECTOR
    clr.l %d6
    
    /* Test: BKPT #0 */
    bkpt #0
    cmp.l #1, %d6
    bne TEST_FAIL
    
    /* Test: BKPT #3 */
    clr.l %d6
    bkpt #3
    cmp.l #4, %d6
    bne TEST_FAIL
    
    move.l %d7, BKPT_VECTOR
    rts

bkpt_handler:
    move.l 2(%sp), %a0
    move.w (%a0), %d0
    and.l #7, %d0
    addq.l #1, %d0
    move.l %d0, %d6
    addq.l #2, 2(%sp)
    rte

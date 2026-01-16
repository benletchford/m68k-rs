.include "entry.s"
/* Test: Nested interrupt handling */

run_test:
    clr.l %d0
    
    /* Install interrupt handlers */
    lea level3_handler, %a0
    move.l %a0, 0x6C        | Level 3 autovector
    
    lea level5_handler, %a1
    move.l %a1, 0x74        | Level 5 autovector
    
    /* Set IPL to 2 (allow level 3+) */
    move.w #0x2200, %sr
    
    /* Simulate level 3 interrupt */
    move.l #3, INTERRUPT_REG
    
    /* Wait for interrupt */
    nop
    nop
    nop
    
    /* Verify handler executed */
    cmp.l #1, %d0
    bne TEST_FAIL
    
    rts

level3_handler:
    addq.l #1, %d0
    rte

level5_handler:
    move.l #99, %d0         | Should not execute
    rte

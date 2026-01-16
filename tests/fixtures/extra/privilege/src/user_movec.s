.include "entry.s"
/* Test: User mode attempting MOVEC (should trigger privilege violation) */

run_test:
    /* Install privilege violation handler */
    lea priv_handler, %a0
    move.l %a0, 0x20        | Privilege violation vector
    
    clr.l %d0               | D0 = 0 (no trap)
    
    /* Switch to user mode */
    andi.w #0xDFFF, %sr     | Clear S bit
    
    /* Attempt MOVEC in user mode - should trap */
    movec %vbr, %d1
    
    /* If we get here, no trap occurred (FAIL) */
    rts

priv_handler:
    move.l #1, %d0          | Mark privilege violation occurred
    /* Return to supervisor mode */
    move.w 0(%sp), %d1      | Get SR from stack
    ori.w #0x2000, %d1      | Set S bit
    move.w %d1, 0(%sp)
    /* Skip MOVEC instruction */
    addq.l #4, 2(%sp)
    rte

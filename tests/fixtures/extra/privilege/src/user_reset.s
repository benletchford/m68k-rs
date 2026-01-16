.include "entry.s"
/* Test: User mode attempting RESET (should trigger privilege violation) */

run_test:
    lea priv_handler, %a0
    move.l %a0, 0x20
    
    clr.l %d0
    
    /* Switch to user mode */
    andi.w #0xDFFF, %sr
    
    /* Attempt RESET in user mode */
    reset
    
    rts

priv_handler:
    move.l #1, %d0
    move.w 0(%sp), %d1
    ori.w #0x2000, %d1
    move.w %d1, 0(%sp)
    addq.l #2, 2(%sp)       | RESET is 2 bytes
    rte

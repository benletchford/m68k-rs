.include "entry.s"
/* Test: User mode attempting STOP (should trigger privilege violation) */

run_test:
    lea priv_handler, %a0
    move.l %a0, 0x20
    
    clr.l %d0
    
    /* Switch to user mode */
    andi.w #0xDFFF, %sr
    
    /* Attempt STOP in user mode */
    stop #0x2000
    
    rts

priv_handler:
    move.l #1, %d0
    move.w 0(%sp), %d1
    ori.w #0x2000, %d1
    move.w %d1, 0(%sp)
    addq.l #4, 2(%sp)       | STOP is 4 bytes (opcode + immediate)
    rte

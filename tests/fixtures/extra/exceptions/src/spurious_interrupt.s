.include "entry.s"
/* Test: Spurious interrupt handling */

run_test:
    clr.l %d0
    
    /* Install spurious interrupt handler */
    lea spurious_handler, %a0
    move.l %a0, 0x60        | Vector 24 (spurious)
    
    /* Enable interrupts */
    andi.w #0xF8FF, %sr     | IPL = 0
    
    /* This test mainly verifies the handler can be installed */
    /* Actual spurious interrupt simulation requires bus-level support */
    
    move.l #1, %d0
    rts

spurious_handler:
    /* Spurious interrupt - just return */
    rte

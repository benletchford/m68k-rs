.include "entry.s"
/* Test: Verify RTE restores SR correctly */

run_test:
    clr.l %d0
    
    /* Install TRAP handler */
    lea trap_handler, %a0
    move.l %a0, 0x80
    
    /* Set SR to specific value in user mode */
    move.w #0x001F, %sr     | User mode, all CCR flags set
    
    /* Trigger TRAP */
    trap #0
    
    /* Verify SR restored (should be user mode again) */
    move.w %sr, %d1
    and.w #0x201F, %d1      | Mask S bit and CCR
    cmp.w #0x001F, %d1
    bne TEST_FAIL
    
    rts

trap_handler:
    /* Modify SR in handler */
    move.w #0x2700, %sr     | Supervisor, mask interrupts
    
    move.l #1, %d0
    /* RTE should restore original SR from stack */
    rte

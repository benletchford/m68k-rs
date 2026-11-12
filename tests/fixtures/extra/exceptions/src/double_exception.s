.include "entry.s"
/* Test: Exception during exception handler (double exception) */

run_test:
    clr.l %d0
    
    /* Install TRAP handler */
    lea trap_handler, %a0
    move.l %a0, 0x80
    
    /* Install Address Error handler */
    lea addr_error_handler, %a1
    move.l %a1, 0x0C
    
    /* Trigger TRAP */
    trap #0
    
    /* Verify both handlers executed */
    cmp.l #2, %d0
    bne TEST_FAIL
    
    rts

trap_handler:
    addq.l #1, %d0          | Increment counter (D0 becomes 1)
    
    /* Trigger Address Error inside TRAP handler */
    lea after_addr_error, %a6
    move.w 0x1001, %d1      | Unaligned access

    /* After returning from Address Error handler, we continue here.
       RTE back to main. */
after_addr_error:
    rte

addr_error_handler:
    addq.l #1, %d0          | Increment counter (D0 becomes 2)

    /* 68000 group 0 frame (14 bytes): status word (SP+0), access address
       (SP+2), instruction register (SP+6), SR (SP+8), PC (SP+10). The
       stacked PC is only loosely defined (2 to 10 bytes past the opcode),
       so resume at the continuation address the test left in A6: replace
       the frame with a three-word one and RTE. */
    cmp.l #0x1001, 2(%sp)   | Access address is the odd operand address
    bne TEST_FAIL
    move.w 8(%sp), %d7      | Stacked SR
    lea 14(%sp), %sp        | Pop the group 0 frame
    move.l %a6, -(%sp)      | Continuation PC: trap_handler's RTE
    move.w %d7, -(%sp)      | SR
    rte

.include "entry.s"
/* Test: Exception priority (Address Error vs Trace) */

run_test:
    clr.l %d0
    
    /* Install Address Error handler */
    lea addr_error_handler, %a0
    move.l %a0, 0x0C
    
    /* Install Trace handler */
    lea trace_handler, %a1
    move.l %a1, 0x24
    
    /* Continuation address for the handler, set before tracing starts */
    lea after_addr_error, %a6

    /* Enable trace mode */
    ori.w #0x8000, %sr      | Set T bit

    /* Trigger Address Error (should take priority over trace) */
    move.w 0x1001, %d1
after_addr_error:

    /* Verify Address Error handled first */
    cmp.l #1, %d0
    bne TEST_FAIL
    
    rts

addr_error_handler:
    move.l #1, %d0

    /* 68000 group 0 frame (14 bytes): status word (SP+0), access address
       (SP+2), instruction register (SP+6), SR (SP+8), PC (SP+10). The
       stacked PC is only loosely defined (2 to 10 bytes past the opcode),
       so resume at the continuation address the test left in A6: replace
       the frame with a three-word one and RTE. */
    cmp.l #0x1001, 2(%sp)   | Access address is the odd operand address
    bne TEST_FAIL
    move.w 8(%sp), %d7      | Stacked SR
    andi.w #0x7FFF, %d7     | Disable trace in the SR that will be restored
    lea 14(%sp), %sp        | Pop the group 0 frame
    move.l %a6, -(%sp)      | Continuation PC
    move.w %d7, -(%sp)      | SR
    rte

trace_handler:
    move.l #2, %d0
    rte

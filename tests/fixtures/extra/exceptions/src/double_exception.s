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
    move.w 0x1001, %d1      | Unaligned access
    
    /* After returning from Address Error handler, we continue here.
       RTE back to main. */
    rte

addr_error_handler:
    addq.l #1, %d0          | Increment counter (D0 becomes 2)
    
    /* 68000 Address Error frame (14 bytes) - pop and return manually */
    move.l (%sp), %a0       | Get return PC
    addq.l #4, %a0          | Skip faulting instruction
    move.w 4(%sp), %d7      | Get SR
    lea 14(%sp), %sp        | Pop 14-byte frame
    move.w %d7, %sr         | Restore SR
    jmp (%a0)               | Return to trap_handler's RTE instruction

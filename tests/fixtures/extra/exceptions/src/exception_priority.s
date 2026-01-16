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
    
    /* Enable trace mode */
    ori.w #0x8000, %sr      | Set T bit
    
    /* Trigger Address Error (should take priority over trace) */
    move.w 0x1001, %d1
    
    /* Verify Address Error handled first */
    cmp.l #1, %d0
    bne TEST_FAIL
    
    rts

addr_error_handler:
    move.l #1, %d0
    
    /* 68000 Address Error frame (14 bytes):
       SP+0:  PC (4), SP+4: SR (2), SP+6..13: extra data
       
       Need to:
       1. Disable trace in the SR that will be restored
       2. Skip past faulting instruction
       3. Restructure stack for RTE (which only pops 6 bytes) */
    
    /* Modify the saved SR at SP+4 to clear trace bit */
    andi.w #0x7FFF, 4(%sp)
    
    /* Skip faulting instruction */
    addq.l #4, (%sp)
    
    /* Restructure stack for RTE */
    move.l (%sp), %a0       | Save modified PC
    move.w 4(%sp), %d7      | Save modified SR
    adda.l #8, %sp          | Skip 8 extra bytes
    move.l %a0, 2(%sp)      | Store PC at SP+2
    move.w %d7, (%sp)       | Store SR at SP
    rte

trace_handler:
    move.l #2, %d0
    rte

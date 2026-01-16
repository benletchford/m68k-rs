.include "entry.s"
/* Test: Trace Modes (68020+) */

.set TRACE_VECTOR, 0x24

run_test:
    move.l TRACE_VECTOR, %d7
    lea trace_handler, %a0
    move.l %a0, TRACE_VECTOR
    clr.l %d6
    
    /* Enable T1 trace */
    move.w #0xA700, %sr
    nop
    nop
    nop
    move.w #0x2700, %sr
    
    cmp.l #4, %d6
    blt TEST_FAIL
    
    move.l %d7, TRACE_VECTOR
    rts

trace_handler:
    addq.l #1, %d6
    rte

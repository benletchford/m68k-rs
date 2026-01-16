.include "entry.s"
/* Test: TRAPcc with various conditions (68020+) */

run_test:
    clr.l %d0
    
    /* Install TRAP handler */
    lea trap_handler, %a0
    move.l %a0, 0x1C        | TRAPV vector
    
    /* TRAPV with V=0 - should not trap */
    andi.w #0xFFFD, %sr     | Clear V
    trapv
    
    /* TRAPV with V=1 - should trap */
    clr.l %d7               | Clear marker
    ori.w #2, %sr           | Set V  
    trapv                   | Should jump to handler
    
    /* Check that handler was called (D7 should be 1) */
    cmp.l #1, %d7
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

trap_handler:
    move.l #1, %d7          | Mark that handler was called
    | For 68010+, stacked PC already points past TRAPV, so just return
    rte

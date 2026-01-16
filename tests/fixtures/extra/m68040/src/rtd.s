.include "entry.s"
/* Test: RTD - Return and Deallocate (68010+) */

run_test:
    move.l %sp, %d7
    
    /* Push return address and call function */
    jsr test_func
    
    /* After RTD, SP should be original + 8 (RTD pops extra) */
    /* Actually we need to set it up properly */
    
    /* Setup: Push extra words then call */
    move.l #0x12345678, -(%sp)
    move.l #0xABCDEF01, -(%sp)
    jsr rtd_func
    
    /* Should return here, SP should be back to d7 */
    cmp.l %d7, %sp
    bne TEST_FAIL
    
    rts

test_func:
    rts

rtd_func:
    rtd #8

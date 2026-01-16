.include "entry.s"
/* Test: MULU edge cases */

run_test:
    clr.l %d0
    
    /* MULU by zero */
    move.l #0xFFFF, %d1
    mulu.w #0, %d1
    cmp.l #0, %d1
    bne TEST_FAIL
    
    /* MULU max values */
    move.l #0xFFFF, %d2
    mulu.w #0xFFFF, %d2
    cmp.l #0xFFFE0001, %d2
    bne TEST_FAIL
    
    /* MULU result fits in word */
    move.w #100, %d3
    mulu.w #200, %d3
    cmp.l #20000, %d3
    bne TEST_FAIL
    
    /* MULU result needs long */
    move.w #0x8000, %d4
    mulu.w #0x8000, %d4
    cmp.l #0x40000000, %d4
    bne TEST_FAIL
    
    /* Verify flags: Z and N should be set based on result */
    move.w #0, %d5
    mulu.w #5, %d5
    bne TEST_FAIL           | Should be zero
    
    move.l #1, %d0
    rts

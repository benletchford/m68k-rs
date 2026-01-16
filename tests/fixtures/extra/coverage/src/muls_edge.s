.include "entry.s"
/* Test: MULS edge cases */

run_test:
    clr.l %d0
    
    /* MULS by zero */
    move.l #0xFFFF, %d1
    muls.w #0, %d1
    cmp.l #0, %d1
    bne TEST_FAIL
    
    /* MULS negative * negative */
    move.w #-10, %d2
    muls.w #-20, %d2
    cmp.l #200, %d2
    bne TEST_FAIL
    
    /* MULS positive * negative */
    move.w #100, %d3
    muls.w #-5, %d3
    cmp.l #-500, %d3
    bne TEST_FAIL
    
    /* MULS max negative */
    move.w #-32768, %d4
    muls.w #2, %d4
    cmp.l #-65536, %d4
    bne TEST_FAIL
    
    /* MULS overflow detection */
    move.w #-32768, %d5
    muls.w #-32768, %d5
    cmp.l #0x40000000, %d5
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

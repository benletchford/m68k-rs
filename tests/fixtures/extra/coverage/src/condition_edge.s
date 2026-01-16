.include "entry.s"
/* Test: Condition code edge cases */

run_test:
    clr.l %d0
    
    /* Test overflow detection */
    move.l #0x7FFFFFFF, %d1
    addq.l #1, %d1
    bvs 1f
    bra TEST_FAIL
1:
    
    /* Test carry propagation */
    move.l #0xFFFFFFFF, %d2
    addq.l #1, %d2
    bcs 2f
    bra TEST_FAIL
2:  bne TEST_FAIL           | Should also be zero
    
    /* Test negative zero */
    moveq #0, %d3
    tst.l %d3
    bmi TEST_FAIL           | Zero is not negative
    
    /* Test BGT vs BHI difference */
    /* BGT: Z=0 and N=V */
    andi.w #0xFFF0, %sr
    ori.w #8, %sr           | N=1, V=0 (N!=V)
    moveq #-1, %d4
    bgt TEST_FAIL           | Should not branch
    
    /* BHI: C=0 and Z=0 */
    andi.w #0xFFF0, %sr     | Clear all
    moveq #1, %d5
    bhi 3f                  | Should branch
    bra TEST_FAIL
3:
    
    move.l #1, %d0
    rts

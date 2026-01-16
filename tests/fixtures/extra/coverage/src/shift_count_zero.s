.include "entry.s"
/* Test: Shift count = 0 edge case */

run_test:
    clr.l %d0              /* d0 = 0 for use as shift count */
    
    /* ASL with count 0 - should not change value */
    move.l #0x12345678, %d1
    asl.l %d0, %d1         /* use register form for count 0 */
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    
    /* LSR with count 0 */
    move.l #0xABCDEF01, %d2
    lsr.l %d0, %d2         /* use register form for count 0 */
    cmp.l #0xABCDEF01, %d2
    bne TEST_FAIL
    
    /* ROL with count 0 */
    move.l #0x55AA55AA, %d3
    rol.l %d0, %d3         /* use register form for count 0 */
    cmp.l #0x55AA55AA, %d3
    bne TEST_FAIL
    
    /* Additional shift tests with register count = 0 */
    moveq #0, %d4
    move.l #0xFFFFFFFF, %d5
    asl.l %d4, %d5
    cmp.l #0xFFFFFFFF, %d5
    bne TEST_FAIL
    
    lsr.l %d4, %d5
    cmp.l #0xFFFFFFFF, %d5
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

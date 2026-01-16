.include "entry.s"
/* Test: Arithmetic shifts ASL/ASR with all sizes */

run_test:
    clr.l %d0
    
    /* ASL.b immediate */
    move.l #0x11, %d1
    asl.b #2, %d1
    cmp.b #0x44, %d1
    bne TEST_FAIL
    
    /* ASL.w immediate */
    move.l #0x1234, %d2
    asl.w #4, %d2
    cmp.w #0x2340, %d2
    bne TEST_FAIL
    
    /* ASL.l immediate */
    move.l #0x12345678, %d3
    asl.l #8, %d3
    cmp.l #0x34567800, %d3
    bne TEST_FAIL
    
    /* ASR.b immediate */
    move.l #0x84, %d4
    asr.b #2, %d4
    cmp.b #0xE1, %d4        | Sign-extended
    bne TEST_FAIL
    
    /* ASR.w with register count */
    move.l #3, %d5
    move.l #0x8000, %d6
    asr.w %d5, %d6
    cmp.w #0xF000, %d6      | Sign-extended
    bne TEST_FAIL
    
    /* ASL with CCR flags */
    move.l #0x60000000, %d7 | 0x60000000 << 1 = 0xC0000000 (overflow + negative)
    asl.l #1, %d7
    bvs 1f                  | Overflow should be set (MSB changed 0->1->1)
    bra TEST_FAIL
1:  bmi 2f                  | Negative should be set (result is 0xC0000000)
    bra TEST_FAIL
2:
    
    move.l #1, %d0
    rts

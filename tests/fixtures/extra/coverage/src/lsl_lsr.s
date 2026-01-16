.include "entry.s"
/* Test: Logical shifts LSL/LSR with all sizes */

run_test:
    clr.l %d0
    
    /* LSL.b immediate */
    move.l #0x33, %d1
    lsl.b #3, %d1
    cmp.b #0x98, %d1
    bne TEST_FAIL
    
    /* LSL.w immediate */
    move.l #0xABCD, %d2
    lsl.w #4, %d2
    cmp.w #0xBCD0, %d2
    bne TEST_FAIL
    
    /* LSL.l with register count */
    move.l #8, %d3
    move.l #0x12345678, %d4
    lsl.l %d3, %d4
    cmp.l #0x34567800, %d4
    bne TEST_FAIL
    
    /* LSR.b immediate (no sign extension) */
    move.l #0x84, %d5
    lsr.b #2, %d5
    cmp.b #0x21, %d5        | Zero-filled
    bne TEST_FAIL
    
    /* LSR.w with register */
    move.l #4, %d6
    move.l #0x8000, %d7
    lsr.w %d6, %d7
    cmp.w #0x0800, %d7      | Zero-filled
    bne TEST_FAIL
    
    /* LSR.l immediate */
    move.l #0x80000000, %d1
    lsr.l #1, %d1
    cmp.l #0x40000000, %d1
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

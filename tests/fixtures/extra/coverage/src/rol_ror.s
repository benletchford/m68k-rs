.include "entry.s"
/* Test: Rotates ROL/ROR with all sizes */

run_test:
    clr.l %d0
    
    /* ROL.b immediate */
    move.l #0x81, %d1
    rol.b #1, %d1
    cmp.b #0x03, %d1
    bne TEST_FAIL
    
    /* ROL.w immediate */
    move.l #0x8001, %d2
    rol.w #4, %d2
    cmp.w #0x0018, %d2
    bne TEST_FAIL
    
    /* ROL.l with register count */
    move.l #8, %d3
    move.l #0x12345678, %d4
    rol.l %d3, %d4
    cmp.l #0x34567812, %d4
    bne TEST_FAIL
    
    /* ROR.b immediate */
    move.l #0x81, %d5
    ror.b #1, %d5
    cmp.b #0xC0, %d5
    bne TEST_FAIL
    
    /* ROR.w with register */
    move.l #4, %d6
    move.l #0x1234, %d7
    ror.w %d6, %d7
    cmp.w #0x4123, %d7
    bne TEST_FAIL
    
    /* ROR.l immediate */
    move.l #0x12345678, %d1
    ror.l #8, %d1
    cmp.l #0x78123456, %d1
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

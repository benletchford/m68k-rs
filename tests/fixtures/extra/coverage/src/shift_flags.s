.include "entry.s"
/* Test: CCR flags for shift operations */

run_test:
    clr.l %d0
    
    /* Test Z flag */
    move.l #1, %d1
    lsl.l #1, %d1
    beq TEST_FAIL           | Should not be zero
    
    moveq #30, %d6
    lsl.l %d6, %d1           /* use register for count > 8 */
    beq TEST_FAIL
    lsl.l #1, %d1
    bne TEST_FAIL           | Should be zero now
    
    /* Test N flag */
    move.l #1, %d2
    moveq #31, %d7
    lsl.l %d7, %d2           /* use register for count > 8 */
    bpl TEST_FAIL           | Should be negative
    
    /* Test C flag (carry out) */
    move.l #0x80000000, %d3
    lsl.l #1, %d3
    bcc TEST_FAIL           | Carry should be set
    
    /* Test X flag (same as C for shifts) */
    move.l #0x80000000, %d4
    asl.l #1, %d4
    bcc TEST_FAIL           | X should be set
    
    /* Test V flag for ASL */
    move.l #0x40000000, %d5
    asl.l #1, %d5
    bvc TEST_FAIL           | Overflow should be set
    
    /* LSR should clear V */
    lsr.l #1, %d5
    bvs TEST_FAIL           | V should be clear
    
    move.l #1, %d0
    rts

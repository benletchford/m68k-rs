.include "entry.s"
/* Test: TAS - Test and Set (atomic operation) */

run_test:
    /* Test 1: TAS on zero - sets Z, clears N, sets bit 7 */
    move.b #0x00, STACK2_BASE
    tas STACK2_BASE
    bne TEST_FAIL
    move.b STACK2_BASE, %d0
    cmp.b #0x80, %d0
    bne TEST_FAIL
    
    /* Test 2: TAS on value with bit 7 set - clears Z, sets N */
    move.b #0x80, STACK2_BASE+1
    tas STACK2_BASE+1
    bpl TEST_FAIL
    move.b STACK2_BASE+1, %d0
    cmp.b #0x80, %d0
    bne TEST_FAIL
    
    /* Test 3: TAS on 0x55 - clears Z and N, sets bit 7 */
    move.b #0x55, STACK2_BASE+2
    tas STACK2_BASE+2
    beq TEST_FAIL
    bmi TEST_FAIL
    move.b STACK2_BASE+2, %d0
    cmp.b #0xD5, %d0
    bne TEST_FAIL
    
    /* Test 4: TAS on 0x7F - clears Z and N, result 0xFF */
    move.b #0x7F, STACK2_BASE+3
    tas STACK2_BASE+3
    move.b STACK2_BASE+3, %d0
    cmp.b #0xFF, %d0
    bne TEST_FAIL
    
    /* Test 5: TAS on register */
    move.l #0, %d1
    tas %d1
    bne TEST_FAIL
    cmp.b #0x80, %d1
    bne TEST_FAIL
    
    rts

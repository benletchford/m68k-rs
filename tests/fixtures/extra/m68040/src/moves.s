.include "entry.s"
/* Test: MOVES - Move to/from Address Space (68010+) */

run_test:
    /* Setup test data */
    move.l #0x12345678, STACK2_BASE
    
    /* Set SFC to 5 (supervisor data) */
    move.l #5, %d0
    movec %d0, %sfc
    
    /* Test 1: MOVES.L read */
    lea STACK2_BASE, %a0
    moves.l (%a0), %d1
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    
    /* Set DFC to 5 */
    move.l #5, %d0
    movec %d0, %dfc
    
    /* Test 2: MOVES.L write */
    move.l #0xDEADBEEF, %d0
    lea STACK2_BASE+4, %a0
    moves.l %d0, (%a0)
    
    /* Verify */
    move.l STACK2_BASE+4, %d1
    cmp.l #0xDEADBEEF, %d1
    bne TEST_FAIL
    
    /* Test 3: MOVES.W */
    move.w #0xABCD, %d0
    lea STACK2_BASE+8, %a0
    moves.w %d0, (%a0)
    move.w STACK2_BASE+8, %d1
    cmp.w #0xABCD, %d1
    bne TEST_FAIL
    
    /* Test 4: MOVES.B */
    move.b #0x42, %d0
    lea STACK2_BASE+10, %a0
    moves.b %d0, (%a0)
    move.b STACK2_BASE+10, %d1
    cmp.b #0x42, %d1
    bne TEST_FAIL
    
    rts

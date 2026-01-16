.include "entry.s"
/* Test: LINK.L - Link with 32-bit Displacement (68020+) */

run_test:
    move.l %sp, %d7
    
    /* Test 1: LINK with large negative displacement */
    link.l %a6, #-0x10000
    
    /* Verify A6 points to saved A6 on stack */
    cmp.l %sp, %a6
    beq TEST_FAIL
    
    /* Verify stack moved by displacement + 4 */
    move.l %d7, %d0
    sub.l #4, %d0
    sub.l #0x10000, %d0
    cmp.l %sp, %d0
    bne TEST_FAIL
    
    /* Unlink */
    unlk %a6
    
    /* Test 2: LINK with zero displacement */
    link.l %a5, #0
    
    /* Stack should only have saved A5 */
    move.l %d7, %d0
    sub.l #4, %d0
    cmp.l %sp, %d0
    bne TEST_FAIL
    
    unlk %a5
    
    /* Test 3: Verify original SP restored */
    cmp.l %d7, %sp
    bne TEST_FAIL
    
    rts

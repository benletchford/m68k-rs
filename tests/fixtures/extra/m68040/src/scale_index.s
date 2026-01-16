.include "entry.s"
/* Test: Scaled Index Addressing Modes (68020+) */

.set DATA_LOC, STACK2_BASE

run_test:
    /* Setup: array of longs */
    lea DATA_LOC, %a0
    move.l #0x11111111, (%a0)+
    move.l #0x22222222, (%a0)+
    move.l #0x33333333, (%a0)+
    move.l #0x44444444, (%a0)+
    move.l #0x55555555, (%a0)+
    move.l #0x66666666, (%a0)+
    move.l #0x77777777, (%a0)+
    move.l #0x88888888, (%a0)+
    
    /* Test 1: Scale *1 (no scaling) */
    lea DATA_LOC, %a0
    move.l #4, %d0
    move.l (0,%a0,%d0.l*1), %d1
    cmp.l #0x22222222, %d1
    bne TEST_FAIL
    
    /* Test 2: Scale *2 */
    move.l #2, %d0
    move.l (0,%a0,%d0.l*2), %d1
    cmp.l #0x22222222, %d1
    bne TEST_FAIL
    
    /* Test 3: Scale *4 */
    move.l #1, %d0
    move.l (0,%a0,%d0.l*4), %d1
    cmp.l #0x22222222, %d1
    bne TEST_FAIL
    
    /* Test 4: Scale *8 */
    lea DATA_LOC, %a0
    move.l #1, %d0
    move.l (0,%a0,%d0.l*8), %d1
    cmp.l #0x33333333, %d1
    bne TEST_FAIL
    
    /* Test 5: Scale *4 with base displacement */
    move.l #2, %d0
    move.l (4,%a0,%d0.l*4), %d1
    cmp.l #0x44444444, %d1
    bne TEST_FAIL
    
    /* Test 6: Word index with scaling */
    move.w #3, %d0
    move.l (0,%a0,%d0.w*4), %d1
    cmp.l #0x44444444, %d1
    bne TEST_FAIL
    
    /* Test 7: Negative index with scale */
    lea DATA_LOC+16, %a0
    move.l #-2, %d0
    move.l (0,%a0,%d0.l*4), %d1
    cmp.l #0x33333333, %d1
    bne TEST_FAIL
    
    rts

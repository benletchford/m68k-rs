.include "entry.s"
/* Test: MOVE16 - 16-byte Aligned Block Move (68030/68040; this fixture targets 68040) */

.set SRC_LOC, STACK2_BASE
.set DST_LOC, STACK2_BASE + 0x100

run_test:
    /* Setup: aligned source data (16 bytes) */
    lea SRC_LOC, %a0
    move.l #0x11111111, (%a0)+
    move.l #0x22222222, (%a0)+
    move.l #0x33333333, (%a0)+
    move.l #0x44444444, (%a0)+
    
    /* Clear destination */
    lea DST_LOC, %a0
    clr.l (%a0)+
    clr.l (%a0)+
    clr.l (%a0)+
    clr.l (%a0)+
    
    /* Test 1: MOVE16 (Ax)+,(Ay)+ */
    lea SRC_LOC, %a0
    lea DST_LOC, %a1
    move16 (%a0)+, (%a1)+
    
    /* Verify 16 bytes copied */
    move.l DST_LOC, %d0
    cmp.l #0x11111111, %d0
    bne TEST_FAIL
    
    move.l DST_LOC+4, %d0
    cmp.l #0x22222222, %d0
    bne TEST_FAIL
    
    move.l DST_LOC+8, %d0
    cmp.l #0x33333333, %d0
    bne TEST_FAIL
    
    move.l DST_LOC+12, %d0
    cmp.l #0x44444444, %d0
    bne TEST_FAIL
    
    /* Verify address registers advanced by 16 */
    lea SRC_LOC+16, %a2
    cmp.l %a2, %a0
    bne TEST_FAIL
    
    lea DST_LOC+16, %a2
    cmp.l %a2, %a1
    bne TEST_FAIL
    
    rts

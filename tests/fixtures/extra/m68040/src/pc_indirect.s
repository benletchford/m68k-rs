.include "entry.s"
/* Test: PC-Relative Memory Indirect (68020+) */

.set DATA_LOC, STACK2_BASE

run_test:
    lea DATA_LOC, %a0
    move.l #0xAAAA1111, (%a0)+
    move.l #0xBBBB2222, (%a0)+
    move.l #0xCCCC3333, (%a0)+
    
    /* Test 1: PC indirect post-indexed */
    move.l #0, %d0
    move.l ([pc_ptr1,%pc],%d0.l,0), %d1
    cmp.l #0xAAAA1111, %d1
    bne TEST_FAIL
    bra test2
pc_ptr1:
    .long DATA_LOC
    
test2:
    /* Test 2: With outer displacement */
    move.l ([pc_ptr2,%pc],%d0.l,4), %d1
    cmp.l #0xBBBB2222, %d1
    bne TEST_FAIL
    bra test3
pc_ptr2:
    .long DATA_LOC
    
test3:
    /* Test 3: With scaled index */
    move.l #1, %d0
    move.l ([pc_ptr3,%pc],%d0.l*4,0), %d1
    cmp.l #0xBBBB2222, %d1
    bne TEST_FAIL
    bra test_done
pc_ptr3:
    .long DATA_LOC

test_done:
    rts

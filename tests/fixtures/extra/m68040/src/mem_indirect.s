.include "entry.s"
/* Test: Memory Indirect Addressing (68020+) */

.set DATA_LOC, STACK2_BASE
.set PTR_LOC, STACK2_BASE + 0x100

run_test:
    /* Setup: data values */
    lea DATA_LOC, %a0
    move.l #0xAAAA1111, (%a0)+
    move.l #0xBBBB2222, (%a0)+
    move.l #0xCCCC3333, (%a0)+
    move.l #0xDDDD4444, (%a0)+
    
    /* Setup: pointer table */
    move.l #DATA_LOC, PTR_LOC
    move.l #DATA_LOC+4, PTR_LOC+4
    move.l #DATA_LOC+8, PTR_LOC+8
    
    /* Test 1: Memory indirect post-indexed ([An],Xn) */
    lea PTR_LOC, %a0
    move.l #0, %d0
    move.l ([%a0],%d0.l,0), %d1
    cmp.l #0xAAAA1111, %d1
    bne TEST_FAIL
    
    /* Test 2: With outer displacement */
    move.l ([%a0],%d0.l,4), %d1
    cmp.l #0xBBBB2222, %d1
    bne TEST_FAIL
    
    /* Test 3: Pre-indexed ([An,Xn]) */
    move.l #4, %d0
    move.l ([%a0,%d0.l],0), %d1
    cmp.l #0xBBBB2222, %d1
    bne TEST_FAIL
    
    /* Test 4: Post-indexed with scaled index */
    move.l #1, %d0
    move.l ([%a0],%d0.l*4,0), %d1
    cmp.l #0xBBBB2222, %d1
    bne TEST_FAIL
    
    /* Test 5: Pre-indexed with outer displacement */
    move.l #4, %d0
    move.l ([%a0,%d0.l],4), %d1
    cmp.l #0xCCCC3333, %d1
    bne TEST_FAIL
    
    rts

.include "entry.s"
/* Test: 68010 Loop Mode */
/* Verifies DBcc loop optimization behavior */

run_test:
    /* Test 1: Basic DBcc loop */
    move.w #10, %d0
    clr.l %d1
loop1:
    addq.l #1, %d1
    dbra %d0, loop1
    
    /* D1 should be 11 (loop executes until D0 = -1) */
    cmp.l #11, %d1
    bne TEST_FAIL
    
    /* Test 2: DBcc with condition true (exits early) */
    move.w #10, %d2
    clr.l %d3
loop2:
    addq.l #1, %d3
    cmp.l #5, %d3
    dbeq %d2, loop2         | Exit if Z=1
    
    /* D3 should be 5 (exits when condition met) */
    cmp.l #5, %d3
    bne TEST_FAIL
    
    /* Test 3: Nested DBcc loops */
    move.w #3, %d4          | Outer counter
    clr.l %d5               | Accumulator
outer_loop:
    move.w #2, %d6          | Inner counter
inner_loop:
    addq.l #1, %d5
    dbra %d6, inner_loop
    dbra %d4, outer_loop
    
    /* D5 should be 12 (4 outer × 3 inner) */
    cmp.l #12, %d5
    bne TEST_FAIL
    
    rts

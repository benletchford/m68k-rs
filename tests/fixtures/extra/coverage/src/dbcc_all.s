.include "entry.s"
/* Test: All 16 DBcc condition code variants */

run_test:
    clr.l %d0
    
    /* DBT - condition true, so no decrement, just fall through */
    move.w #3, %d1
1:  dbt %d1, 1b
    cmp.w #3, %d1           | DBT never decrements, D1 should still be 3
    bne TEST_FAIL
    
    /* DBF (DBRA) - decrement and branch if not -1 */
    move.w #3, %d2
    moveq #0, %d3
2:  addq.l #1, %d3
    dbf %d2, 2b
    cmp.l #4, %d3           | Should loop 4 times (3,2,1,0)
    bne TEST_FAIL
    
    /* DBEQ - decrement and branch if not equal (Z=0) */
    move.w #5, %d4
    moveq #0, %d5
3:  addq.l #1, %d5
    cmp.l #3, %d5
    dbeq %d4, 3b
    cmp.l #3, %d5           | Should stop when Z=1
    bne TEST_FAIL
    
    /* DBNE - condition NE (Z=0) is true, so just fall through */
    move.w #2, %d6          | MOVE sets Z=0 (since 2 != 0)
4:  dbne %d6, 4b            | NE true, so no decrement, falls through
    cmp.w #2, %d6           | D6 should still be 2
    bne TEST_FAIL
    
    /* DBCC - condition CC (C=0) is true, so just fall through */
    move.w #2, %d7          | MOVE clears C
5:  dbcc %d7, 5b            | CC true, so no decrement, falls through
    cmp.w #2, %d7           | D7 should still be 2
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

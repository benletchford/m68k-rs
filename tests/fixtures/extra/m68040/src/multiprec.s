.include "entry.s"
/* Test: Multi-precision Arithmetic (ADDX/SUBX/NEGX) */

run_test:
    /* Test 1: 64-bit addition */
    move.l #0xFFFFFFFF, %d0
    move.l #0x00000001, %d1
    move.l #0x00000001, %d2
    move.l #0x00000000, %d3
    move #0, %ccr
    add.l %d2, %d0
    addx.l %d3, %d1
    cmp.l #0x00000000, %d0
    bne TEST_FAIL
    cmp.l #0x00000002, %d1
    bne TEST_FAIL
    
    /* Test 2: 64-bit subtraction */
    move.l #0x00000000, %d0
    move.l #0x00000002, %d1
    move.l #0x00000001, %d2
    move.l #0x00000000, %d3
    move #0, %ccr
    sub.l %d2, %d0
    subx.l %d3, %d1
    cmp.l #0xFFFFFFFF, %d0
    bne TEST_FAIL
    cmp.l #0x00000001, %d1
    bne TEST_FAIL
    
    /* Test 3: 96-bit addition */
    move.l #0xFFFFFFFF, %d0
    move.l #0xFFFFFFFF, %d1
    move.l #0x00000001, %d2
    move.l #0x00000001, %d3
    move.l #0x00000000, %d4
    move.l #0x00000000, %d5
    move #0, %ccr
    add.l %d3, %d0
    addx.l %d4, %d1
    addx.l %d5, %d2
    cmp.l #0x00000000, %d0
    bne TEST_FAIL
    cmp.l #0x00000000, %d1
    bne TEST_FAIL
    cmp.l #0x00000002, %d2
    bne TEST_FAIL
    
    rts

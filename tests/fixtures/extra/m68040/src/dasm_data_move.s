.include "entry.s"
/* Test: Disassembler Coverage - Data Movement Instructions */
/* Exercises instructions that should be disassembled correctly */

run_test:
    /* =================================================================== */
    /* MOVE variations */
    /* =================================================================== */
    move.b #0x55, %d0
    move.w #0x1234, %d1
    move.l #0x12345678, %d2
    
    move.b %d0, %d3
    move.w %d1, %d4
    move.l %d2, %d5
    
    /* Verify data integrity */
    cmp.b #0x55, %d3
    bne TEST_FAIL
    cmp.w #0x1234, %d4
    bne TEST_FAIL
    cmp.l #0x12345678, %d5
    bne TEST_FAIL
    
    /* =================================================================== */
    /* MOVEA - move to address register */
    /* =================================================================== */
    move.l #0xDEADBEEF, %a0
    movea.l %a0, %a1
    
    cmp.l %a0, %a1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* MOVEQ - quick move */
    /* =================================================================== */
    moveq #127, %d0
    cmp.l #127, %d0
    bne TEST_FAIL
    
    moveq #-1, %d1
    cmp.l #-1, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* LEA - load effective address */
    /* =================================================================== */
    lea test_data, %a0
    lea 4(%a0), %a1
    
    move.l (%a0), %d0
    cmp.l #0xAAAAAAAA, %d0
    bne TEST_FAIL
    
    move.l (%a1), %d0
    cmp.l #0xBBBBBBBB, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* PEA - push effective address */
    /* =================================================================== */
    pea test_data
    move.l (%sp)+, %a2
    
    move.l (%a2), %d0
    cmp.l #0xAAAAAAAA, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* EXG - exchange registers */
    /* =================================================================== */
    move.l #0x11111111, %d0
    move.l #0x22222222, %d1
    exg %d0, %d1
    
    cmp.l #0x22222222, %d0
    bne TEST_FAIL
    cmp.l #0x11111111, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* SWAP - swap register halves */
    /* =================================================================== */
    move.l #0x12345678, %d0
    swap %d0
    
    cmp.l #0x56781234, %d0
    bne TEST_FAIL
    
    rts

test_data:
    .long 0xAAAAAAAA
    .long 0xBBBBBBBB

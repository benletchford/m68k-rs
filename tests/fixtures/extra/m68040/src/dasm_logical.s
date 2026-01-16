.include "entry.s"
/* Test: Disassembler Coverage - Logical and Bit Operations */

run_test:
    /* =================================================================== */
    /* AND variations */
    /* =================================================================== */
    move.l #0xFF00FF00, %d0
    move.l #0x0F0F0F0F, %d1
    and.l %d1, %d0
    
    cmp.l #0x0F000F00, %d0
    bne TEST_FAIL
    
    /* ANDI */
    move.l #0xFFFFFFFF, %d0
    andi.l #0x0000FFFF, %d0
    cmp.l #0x0000FFFF, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* OR variations */
    /* =================================================================== */
    move.l #0x00FF0000, %d0
    move.l #0x0000FF00, %d1
    or.l %d1, %d0
    
    cmp.l #0x00FFFF00, %d0
    bne TEST_FAIL
    
    /* ORI */
    move.l #0x00000000, %d0
    ori.l #0x12345678, %d0
    cmp.l #0x12345678, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* EOR variations */
    /* =================================================================== */
    move.l #0xAAAAAAAA, %d0
    move.l #0x55555555, %d1
    eor.l %d1, %d0
    
    cmp.l #0xFFFFFFFF, %d0
    bne TEST_FAIL
    
    /* EORI */
    eori.l #0xFFFFFFFF, %d0
    tst.l %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* NOT */
    /* =================================================================== */
    move.l #0x00000000, %d0
    not.l %d0
    cmp.l #0xFFFFFFFF, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Bit manipulation - BTST/BSET/BCLR/BCHG */
    /* =================================================================== */
    move.l #0x00000000, %d0
    
    bset #0, %d0            | Set bit 0
    cmp.l #0x00000001, %d0
    bne TEST_FAIL
    
    bset #7, %d0            | Set bit 7
    cmp.l #0x00000081, %d0
    bne TEST_FAIL
    
    bclr #0, %d0            | Clear bit 0
    cmp.l #0x00000080, %d0
    bne TEST_FAIL
    
    bchg #7, %d0            | Toggle bit 7
    tst.l %d0
    bne TEST_FAIL
    
    /* BTST - test doesn't modify */
    move.l #0x00000080, %d0
    btst #7, %d0
    beq TEST_FAIL           | Bit 7 should be set
    
    btst #0, %d0
    bne TEST_FAIL           | Bit 0 should be clear
    
    rts

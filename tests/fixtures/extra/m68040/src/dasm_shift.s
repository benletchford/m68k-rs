.include "entry.s"
/* Test: Disassembler Coverage - Shift and Rotate Instructions */

run_test:
    /* =================================================================== */
    /* ASL/ASR - Arithmetic shifts */
    /* =================================================================== */
    move.l #0x00000010, %d0
    asl.l #4, %d0
    cmp.l #0x00000100, %d0
    bne TEST_FAIL
    
    move.l #0x80000000, %d0
    asr.l #4, %d0
    cmp.l #0xF8000000, %d0  | Sign extends
    bne TEST_FAIL
    
    /* =================================================================== */
    /* LSL/LSR - Logical shifts */
    /* =================================================================== */
    move.l #0x00000001, %d0
    lsl.l #8, %d0
    cmp.l #0x00000100, %d0
    bne TEST_FAIL
    
    move.l #0x80000000, %d0
    lsr.l #4, %d0
    cmp.l #0x08000000, %d0  | Zero fills
    bne TEST_FAIL
    
    /* =================================================================== */
    /* ROL/ROR - Rotate through register */
    /* =================================================================== */
    move.l #0x80000001, %d0
    rol.l #1, %d0
    cmp.l #0x00000003, %d0
    bne TEST_FAIL
    
    move.l #0x80000001, %d0
    ror.l #1, %d0
    cmp.l #0xC0000000, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* ROXL/ROXR - Rotate through X bit */
    /* =================================================================== */
    move #0, %ccr           | Clear X
    move.l #0x80000000, %d0
    roxl.l #1, %d0
    cmp.l #0x00000000, %d0
    bne TEST_FAIL
    /* X flag should now be set */
    
    /* =================================================================== */
    /* Register-specified shift count */
    /* =================================================================== */
    move.l #4, %d1          | Shift count in D1
    move.l #0x00000001, %d0
    lsl.l %d1, %d0
    cmp.l #0x00000010, %d0
    bne TEST_FAIL
    
    rts

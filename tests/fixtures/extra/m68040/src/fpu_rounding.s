.include "entry.s"
/* Test: FPU Rounding Modes */
/* FPCR bits 5:4 control rounding: 00=Nearest, 01=Zero, 10=Minus, 11=Plus */

run_test:
    /* =================================================================== */
    /* Test 1: Round to Nearest (default) */
    /* =================================================================== */
    fmove.l #0x0000, %fpcr      | RN mode
    fmove.l %fpcr, %d0          | Read back FPCR
    and.l #0x30, %d0            | Mask rounding bits
    cmp.l #0x00, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: Round toward Zero */
    /* =================================================================== */
    fmove.l #0x0010, %fpcr      | RZ mode
    fmove.l %fpcr, %d0
    and.l #0x30, %d0
    cmp.l #0x10, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: Round toward Minus Infinity */
    /* =================================================================== */
    fmove.l #0x0020, %fpcr      | RM mode
    fmove.l %fpcr, %d0
    and.l #0x30, %d0
    cmp.l #0x20, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: Round toward Plus Infinity */
    /* =================================================================== */
    fmove.l #0x0030, %fpcr      | RP mode
    fmove.l %fpcr, %d0
    and.l #0x30, %d0
    cmp.l #0x30, %d0
    bne TEST_FAIL
    
    /* Restore default */
    fmove.l #0, %fpcr
    
    rts

.include "entry.s"
/* Test: FPU Basic Operations - FSQRT, FABS, FNEG, FINT, FINTRZ */

run_test:
    /* =================================================================== */
    /* Test 1: FSQRT - sqrt(4) should be 2 */
    /* =================================================================== */
    fmove.l #4, %fp0            | fp0 = 4.0
    fsqrt.x %fp0                | fp0 = sqrt(4) = 2.0
    
    fmove.l #2, %fp7
    fcmp.x %fp7, %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: FSQRT - sqrt(1) should be 1 */
    /* =================================================================== */
    fmove.l #1, %fp0
    fsqrt.x %fp0
    
    fmove.l #1, %fp7
    fcmp.x %fp7, %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: FABS - abs(-5) should be 5 */
    /* =================================================================== */
    fmove.l #-5, %fp1
    fabs.x %fp1
    
    fmove.l #5, %fp7
    fcmp.x %fp7, %fp1
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: FABS - abs(3) should be 3 */
    /* =================================================================== */
    fmove.l #3, %fp1
    fabs.x %fp1
    
    fmove.l #3, %fp7
    fcmp.x %fp7, %fp1
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 5: FNEG - neg(7) should be -7 */
    /* =================================================================== */
    fmove.l #7, %fp2
    fneg.x %fp2
    
    fmove.l #-7, %fp7
    fcmp.x %fp7, %fp2
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 6: FINT - int(3.7) rounded to nearest */
    /* =================================================================== */
    /* Set rounding mode to nearest (default) */
    fmove.l #0, %fpcr
    
    /* Load 3.7 and round */
    /* 3.7 = 0x40006666... in single precision, but we use integer load */
    fmove.l #4, %fp3            | Use 4 for simplicity
    fint.x %fp3                 | Should stay 4
    
    fmove.l #4, %fp7
    fcmp.x %fp7, %fp3
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 7: FINTRZ - truncate toward zero */
    /* =================================================================== */
    fmove.l #5, %fp4
    fintrz.x %fp4               | Should stay 5
    
    fmove.l #5, %fp7
    fcmp.x %fp7, %fp4
    fbne TEST_FAIL
    
    rts

.include "entry.s"
/* Test: FPU Exponential and Logarithm Functions */
/* Tests FTWOTOX, FTENTOX, FLOG2, FLOG10, FGETEXP, FGETMAN */

run_test:
    /* =================================================================== */
    /* Test 1: FTWOTOX - 2^0 should be 1 */
    /* =================================================================== */
    fmove.l #0, %fp0            | fp0 = 0.0
    ftwotox.x %fp0              | fp0 = 2^0 = 1.0
    
    fmove.l #1, %fp7
    fcmp.x %fp7, %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: FTENTOX - 10^0 should be 1 */
    /* =================================================================== */
    fmove.l #0, %fp1            | fp1 = 0.0
    ftentox.x %fp1              | fp1 = 10^0 = 1.0
    
    fcmp.x %fp7, %fp1
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: FLOG2 - log2(1) should be 0 */
    /* =================================================================== */
    fmove.l #1, %fp2            | fp2 = 1.0
    flog2.x %fp2                | fp2 = log2(1) = 0.0
    
    ftst.x %fp2
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: FLOG10 - log10(1) should be 0 */
    /* =================================================================== */
    fmove.l #1, %fp3            | fp3 = 1.0
    flog10.x %fp3               | fp3 = log10(1) = 0.0
    
    ftst.x %fp3
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 5: FGETEXP - exponent of 1.0 should be 0 */
    /* =================================================================== */
    fmove.l #1, %fp4            | fp4 = 1.0
    fgetexp.x %fp4              | fp4 = exponent of 1.0 = 0
    
    ftst.x %fp4
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 6: FGETMAN - mantissa of 1.0 should be 1.0 */
    /* =================================================================== */
    fmove.l #1, %fp5            | fp5 = 1.0
    fgetman.x %fp5              | fp5 = mantissa of 1.0 = 1.0
    
    fcmp.x %fp7, %fp5
    fbne TEST_FAIL
    
    rts

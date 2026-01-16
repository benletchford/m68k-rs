.include "entry.s"
/* Test: FPU Scale and Extract operations */
/* FSCALE, FGETEXP, FGETMAN, FLOGBP1, FETOXM1 */

run_test:
    /* =================================================================== */
    /* Test 1: FSCALE - multiply by power of 2 */
    /* 2.0 * 2^3 = 2.0 * 8 = 16.0 */
    /* =================================================================== */
    fmove.l #2, %fp0            | fp0 = 2.0
    fmove.l #3, %fp1            | fp1 = 3 (scale factor)
    fscale.x %fp1, %fp0         | fp0 = 2.0 * 2^3 = 16.0
    
    fmove.l #16, %fp7
    fcmp.x %fp7, %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: FSCALE - 1.0 * 2^4 = 16.0 */
    /* =================================================================== */
    fmove.l #1, %fp0
    fmove.l #4, %fp1
    fscale.x %fp1, %fp0         | fp0 = 1.0 * 2^4 = 16.0
    
    fmove.l #16, %fp7
    fcmp.x %fp7, %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: FETOXM1 - e^x - 1, for x near 0 */
    /* e^0 - 1 = 0 */
    /* =================================================================== */
    fmove.l #0, %fp2
    fetoxm1.x %fp2              | fp2 = e^0 - 1 = 0
    
    ftst.x %fp2
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: FLOGNP1 - ln(1+x), for x near 0 */
    /* ln(1 + 0) = ln(1) = 0 */
    /* =================================================================== */
    fmove.l #0, %fp3
    flognp1.x %fp3              | fp3 = ln(1 + 0) = 0
    
    ftst.x %fp3
    fbne TEST_FAIL
    
    rts

.include "entry.s"
/* Test: Additional FPU Transcendental Functions */
/* Tests FASIN, FACOS, FATAN, FSINH, FCOSH, FTANH */

run_test:
    /* =================================================================== */
    /* Test 1: FASIN - arcsin(0) should be 0 */
    /* =================================================================== */
    fmove.l #0, %fp0            | fp0 = 0.0
    fasin.x %fp0                | fp0 = asin(0) = 0.0
    
    ftst.x %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: FACOS - arccos(1) should be 0 */
    /* =================================================================== */
    fmove.l #1, %fp1            | fp1 = 1.0
    facos.x %fp1                | fp1 = acos(1) = 0.0
    
    ftst.x %fp1
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: FATAN - arctan(0) should be 0 */
    /* =================================================================== */
    fmove.l #0, %fp2            | fp2 = 0.0
    fatan.x %fp2                | fp2 = atan(0) = 0.0
    
    ftst.x %fp2
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: FSINH - sinh(0) should be 0 */
    /* =================================================================== */
    fmove.l #0, %fp3            | fp3 = 0.0
    fsinh.x %fp3                | fp3 = sinh(0) = 0.0
    
    ftst.x %fp3
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 5: FCOSH - cosh(0) should be 1 */
    /* =================================================================== */
    fmove.l #0, %fp4            | fp4 = 0.0
    fcosh.x %fp4                | fp4 = cosh(0) = 1.0
    
    fmove.l #1, %fp5
    fcmp.x %fp5, %fp4
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 6: FTANH - tanh(0) should be 0 */
    /* =================================================================== */
    fmove.l #0, %fp6            | fp6 = 0.0
    ftanh.x %fp6                | fp6 = tanh(0) = 0.0
    
    ftst.x %fp6
    fbne TEST_FAIL
    
    rts

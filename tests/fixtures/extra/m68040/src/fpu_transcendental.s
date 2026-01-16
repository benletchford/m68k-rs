.include "entry.s"
/* Test: FPU Transcendental Functions */
/* Tests FSIN, FCOS, FTAN, FETOX, FLOGN */
/* Currently expected to FAIL until transcendentals are implemented */

.set FPU_DATA, STACK2_BASE

run_test:
    /* =================================================================== */
    /* Test 1: FSIN - Sine of 0 should be 0 */
    /* =================================================================== */
    fmove.l #0, %fp0            | fp0 = 0.0
    fsin.x %fp0                 | fp0 = sin(0) = 0.0
    
    /* Check if result is zero */
    ftst.x %fp0
    fbne TEST_FAIL              | Should be zero
    
    /* =================================================================== */
    /* Test 2: FCOS - Cosine of 0 should be 1 */
    /* =================================================================== */
    fmove.l #0, %fp1            | fp1 = 0.0
    fcos.x %fp1                 | fp1 = cos(0) = 1.0
    
    /* Compare with 1.0 */
    fmove.l #1, %fp2
    fcmp.x %fp2, %fp1
    fbne TEST_FAIL              | Should be 1.0
    
    /* =================================================================== */
    /* Test 3: FETOX - e^0 should be 1 */
    /* =================================================================== */
    fmove.l #0, %fp3            | fp3 = 0.0
    fetox.x %fp3                | fp3 = e^0 = 1.0
    
    fmove.l #1, %fp4
    fcmp.x %fp4, %fp3
    fbne TEST_FAIL              | Should be 1.0
    
    /* =================================================================== */
    /* Test 4: FLOGN - ln(1) should be 0 */
    /* =================================================================== */
    fmove.l #1, %fp5            | fp5 = 1.0
    flogn.x %fp5                | fp5 = ln(1) = 0.0
    
    ftst.x %fp5
    fbne TEST_FAIL              | Should be zero
    
    /* =================================================================== */
    /* Test 5: FTAN - tan(0) should be 0 */
    /* =================================================================== */
    fmove.l #0, %fp6            | fp6 = 0.0
    ftan.x %fp6                 | fp6 = tan(0) = 0.0
    
    ftst.x %fp6
    fbne TEST_FAIL              | Should be zero
    
    rts

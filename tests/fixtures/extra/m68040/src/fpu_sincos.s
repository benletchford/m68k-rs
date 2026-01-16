.include "entry.s"
/* Test: FSINCOS - Combined sine and cosine */

run_test:
    /* =================================================================== */
    /* Test 1: FSINCOS(0) - sin(0)=0, cos(0)=1 */
    /* =================================================================== */
    fmove.l #0, %fp0            | fp0 = 0.0
    fsincos.x %fp0, %fp1:%fp0   | fp0 = sin(0), fp1 = cos(0)
    
    /* Check sin(0) = 0 */
    ftst.x %fp0
    fbne TEST_FAIL
    
    /* Check cos(0) = 1 */
    fmove.l #1, %fp7
    fcmp.x %fp7, %fp1
    fbne TEST_FAIL
    
    rts

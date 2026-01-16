.include "entry.s"
/* Test: FPU FREM and FMOD - remainder operations */

run_test:
    /* =================================================================== */
    /* Test 1: FMOD - 7 mod 3 should be 1 */
    /* =================================================================== */
    fmove.l #7, %fp0            | fp0 = 7
    fmove.l #3, %fp1            | fp1 = 3
    fmod.x %fp1, %fp0           | fp0 = 7 mod 3 = 1
    
    fmove.l #1, %fp7
    fcmp.x %fp7, %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: FMOD - 10 mod 4 should be 2 */
    /* =================================================================== */
    fmove.l #10, %fp0
    fmove.l #4, %fp1
    fmod.x %fp1, %fp0           | fp0 = 10 mod 4 = 2
    
    fmove.l #2, %fp7
    fcmp.x %fp7, %fp0
    fbne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: FREM - IEEE remainder of 7/3 */
    /* FREM uses round-to-nearest, so 7/3 = 2.33... rounds to 2 */
    /* remainder = 7 - 3*2 = 1 */
    /* =================================================================== */
    fmove.l #7, %fp2
    fmove.l #3, %fp3
    frem.x %fp3, %fp2           | fp2 = IEEE rem(7, 3) = 1
    
    fmove.l #1, %fp7
    fcmp.x %fp7, %fp2
    fbne TEST_FAIL
    
    rts

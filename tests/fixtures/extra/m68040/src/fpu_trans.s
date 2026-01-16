.include "entry.s"
/* Test: FPU Transcendental Functions */

run_test:
    /* Test 1: FNEG */
    fmove.s const_5, %fp0
    fneg.x %fp0
    fcmp.s const_neg5, %fp0
    fbne TEST_FAIL
    
    /* Test 2: FABS */
    fmove.s const_neg7, %fp0
    fabs.x %fp0
    fcmp.s const_7, %fp0
    fbne TEST_FAIL
    
    /* Test 3: FSQRT */
    fmove.s const_9, %fp0
    fsqrt.x %fp0
    fcmp.s const_3, %fp0
    fbne TEST_FAIL
    
    /* Test 4: FTST - test for zero */
    fmove.s const_0, %fp0
    ftst.x %fp0
    fbeq t4_ok
    bra TEST_FAIL
t4_ok:

    /* Test 5: FTST - test for negative */
    fmove.s const_neg5, %fp0
    ftst.x %fp0
    fblt t5_ok
    bra TEST_FAIL
t5_ok:

    /* Test 6: FINT - round to integer */
    fmove.s const_3_7, %fp0
    fint.x %fp0
    fcmp.s const_4, %fp0
    fbne TEST_FAIL
    
    rts

    .align 4
const_0:    .float 0.0
const_3:    .float 3.0
const_3_7:  .float 3.7
const_4:    .float 4.0
const_5:    .float 5.0
const_7:    .float 7.0
const_9:    .float 9.0
const_neg5: .float -5.0
const_neg7: .float -7.0

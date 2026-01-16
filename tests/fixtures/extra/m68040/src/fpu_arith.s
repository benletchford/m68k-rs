.include "entry.s"
/* Test: FPU Basic Operations (68040 FPU) */
/* Uses memory constants since gas doesn't support float immediates */

.set FPU_DATA, STACK2_BASE

run_test:
    /* Test 1: FMOVE from memory */
    fmove.s const_1, %fp0
    
    /* Test 2: FMOVE between registers */
    fmove.x %fp0, %fp1
    
    /* Test 3: FADD */
    fmove.s const_1, %fp0
    fmove.s const_2, %fp1
    fadd.x %fp1, %fp0
    /* FP0 should be 3.0 */
    fcmp.s const_3, %fp0
    fbne TEST_FAIL
    
    /* Test 4: FSUB */
    fmove.s const_5, %fp0
    fmove.s const_3, %fp1
    fsub.x %fp1, %fp0
    /* FP0 should be 2.0 */
    fcmp.s const_2, %fp0
    fbne TEST_FAIL
    
    /* Test 5: FMUL */
    fmove.s const_4, %fp0
    fmove.s const_3, %fp1
    fmul.x %fp1, %fp0
    /* FP0 should be 12.0 */
    fcmp.s const_12, %fp0
    fbne TEST_FAIL
    
    /* Test 6: FDIV */
    fmove.s const_12, %fp0
    fmove.s const_4, %fp1
    fdiv.x %fp1, %fp0
    /* FP0 should be 3.0 */
    fcmp.s const_3, %fp0
    fbne TEST_FAIL
    
    rts

/* Float constants in memory */
    .align 4
const_1:    .float 1.0
const_2:    .float 2.0
const_3:    .float 3.0
const_4:    .float 4.0
const_5:    .float 5.0
const_12:   .float 12.0

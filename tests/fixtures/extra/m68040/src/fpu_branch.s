.include "entry.s"
/* Test: FPU Branches and Conditions (FBcc, FScc) */

run_test:
    /* Test 1: FBEQ - branch if equal */
    fmove.s const_5, %fp0
    fcmp.s const_5, %fp0
    fbeq t1_ok
    bra TEST_FAIL
t1_ok:

    /* Test 2: FBNE - branch if not equal */
    fmove.s const_5, %fp0
    fcmp.s const_3, %fp0
    fbne t2_ok
    bra TEST_FAIL
t2_ok:

    /* Test 3: FBGT - branch if greater than */
    fmove.s const_7, %fp0
    fcmp.s const_3, %fp0
    fbgt t3_ok
    bra TEST_FAIL
t3_ok:

    /* Test 4: FBLT - branch if less than */
    fmove.s const_2, %fp0
    fcmp.s const_5, %fp0
    fblt t4_ok
    bra TEST_FAIL
t4_ok:

    /* Test 5: FBGE - branch if greater or equal */
    fmove.s const_5, %fp0
    fcmp.s const_5, %fp0
    fbge t5_ok
    bra TEST_FAIL
t5_ok:

    /* Test 6: FBLE - branch if less or equal */
    fmove.s const_3, %fp0
    fcmp.s const_5, %fp0
    fble t6_ok
    bra TEST_FAIL
t6_ok:

    /* Test 7: FSEQ - set if equal */
    fmove.s const_5, %fp0
    fcmp.s const_3, %fp0
    fseq %d0
    tst.b %d0
    bne TEST_FAIL
    
    /* Test 8: FSNE - set if not equal */
    fsne %d0
    cmp.b #0xFF, %d0
    bne TEST_FAIL
    
    rts

    .align 4
const_2:    .float 2.0
const_3:    .float 3.0
const_5:    .float 5.0
const_7:    .float 7.0

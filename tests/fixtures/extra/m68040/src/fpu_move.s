.include "entry.s"
/* Test: FPU Data Movement (FMOVEM, FMOVE to/from) */

run_test:
    /* Test 1: FMOVE from memory (single) */
    fmove.s const_1, %fp0
    fmove.s const_2, %fp1
    fmove.s const_3, %fp2
    fmove.s const_4, %fp3
    
    /* Test 2: FMOVEM save to memory */
    fmovem.x %fp0-%fp3, STACK2_BASE
    
    /* Test 3: Clear and restore */
    fmove.s const_0, %fp0
    fmove.s const_0, %fp1
    fmove.s const_0, %fp2
    fmove.s const_0, %fp3
    
    /* Restore */
    fmovem.x STACK2_BASE, %fp0-%fp3
    
    /* Verify FP0 = 1.0 */
    fcmp.s const_1, %fp0
    fbne TEST_FAIL
    
    /* Verify FP3 = 4.0 */
    fcmp.s const_4, %fp3
    fbne TEST_FAIL
    
    /* Test 4: FMOVE to integer register */
    fmove.s const_42, %fp0
    fmove.l %fp0, %d0
    cmp.l #42, %d0
    bne TEST_FAIL
    
    /* Test 5: FMOVE from integer */
    move.l #100, %d0
    fmove.l %d0, %fp0
    fcmp.s const_100, %fp0
    fbne TEST_FAIL
    
    rts

    .align 4
const_0:    .float 0.0
const_1:    .float 1.0
const_2:    .float 2.0
const_3:    .float 3.0
const_4:    .float 4.0
const_42:   .float 42.0
const_100:  .float 100.0

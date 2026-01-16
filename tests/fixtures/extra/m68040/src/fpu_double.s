.include "entry.s"
/* Test: FPU Double Precision Operations */

run_test:
    /* Test 1: FMOVE.D from memory */
    fmove.d const_d1, %fp0
    fcmp.d const_d1, %fp0
    fbne TEST_FAIL
    
    /* Test 2: FADD.D */
    fmove.d const_d1, %fp0
    fmove.d const_d2, %fp1
    fadd.x %fp1, %fp0
    fcmp.d const_d3, %fp0
    fbne TEST_FAIL
    
    /* Test 3: FMUL.D */
    fmove.d const_d2, %fp0
    fmove.d const_d3, %fp1
    fmul.x %fp1, %fp0
    fcmp.d const_d6, %fp0
    fbne TEST_FAIL
    
    /* Test 4: Extended precision */
    fmove.x const_x1, %fp0
    fmove.x const_x2, %fp1
    fadd.x %fp1, %fp0
    /* Just verify no crash for extended */
    
    rts

    .align 8
const_d1:   .double 1.0
const_d2:   .double 2.0
const_d3:   .double 3.0
const_d6:   .double 6.0

    .align 4
const_x1:   .dc.w 0x3FFF, 0x8000, 0x0000, 0x0000, 0x0000, 0x0000  /* 1.0 extended */
const_x2:   .dc.w 0x4000, 0x8000, 0x0000, 0x0000, 0x0000, 0x0000  /* 2.0 extended */

.include "entry.s"
/* Test: FPU Control Registers (FPCR, FPSR, FPIAR) */

run_test:
    /* Test 1: FMOVE to/from FPCR */
    move.l #0x00000010, %d0
    fmove.l %d0, %fpcr
    fmove.l %fpcr, %d1
    cmp.l #0x00000010, %d1
    bne TEST_FAIL
    
    /* Reset FPCR */
    moveq #0, %d0
    fmove.l %d0, %fpcr
    
    /* Test 2: FMOVE to/from FPSR */
    move.l #0, %d0
    fmove.l %d0, %fpsr
    fmove.l %fpsr, %d1
    /* Just verify no crash */
    
    /* Test 3: FMOVE to/from FPIAR */
    move.l #0x12345000, %d0
    fmove.l %d0, %fpiar
    fmove.l %fpiar, %d1
    cmp.l #0x12345000, %d1
    bne TEST_FAIL
    
    /* Test 4: FMOVEM control registers */
    fmovem.l %fpcr/%fpsr/%fpiar, STACK2_BASE
    
    /* Test 5: Restore control registers */
    fmovem.l STACK2_BASE, %fpcr/%fpsr/%fpiar
    
    rts

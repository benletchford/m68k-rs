.include "entry.s"
/* Test: Disassembler Coverage - Arithmetic Instructions */

run_test:
    /* =================================================================== */
    /* ADD variations */
    /* =================================================================== */
    move.l #100, %d0
    move.l #50, %d1
    
    add.b %d1, %d0
    add.w %d1, %d0
    add.l %d1, %d0
    
    cmp.l #250, %d0
    bne TEST_FAIL
    
    /* ADDI - add immediate */
    addi.l #100, %d0
    cmp.l #350, %d0
    bne TEST_FAIL
    
    /* ADDQ - add quick */
    addq.l #8, %d0
    cmp.l #358, %d0
    bne TEST_FAIL
    
    /* ADDA - add to address */
    move.l #0x1000, %a0
    adda.l #0x100, %a0
    cmp.l #0x1100, %a0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* SUB variations */
    /* =================================================================== */
    move.l #500, %d0
    move.l #100, %d1
    
    sub.l %d1, %d0
    cmp.l #400, %d0
    bne TEST_FAIL
    
    /* SUBI - subtract immediate */
    subi.l #50, %d0
    cmp.l #350, %d0
    bne TEST_FAIL
    
    /* SUBQ - subtract quick */
    subq.l #5, %d0
    cmp.l #345, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* MUL/DIV */
    /* =================================================================== */
    move.l #12, %d0
    move.l #8, %d1
    mulu.w %d1, %d0
    
    cmp.l #96, %d0
    bne TEST_FAIL
    
    move.l #144, %d0
    move.l #12, %d1
    divu.w %d1, %d0
    
    and.l #0xFFFF, %d0
    cmp.l #12, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* NEG/NEGX */
    /* =================================================================== */
    move.l #42, %d0
    neg.l %d0
    cmp.l #-42, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* CLR */
    /* =================================================================== */
    move.l #0xFFFFFFFF, %d0
    clr.b %d0
    cmp.l #0xFFFFFF00, %d0
    bne TEST_FAIL
    
    clr.w %d0
    cmp.l #0xFFFF0000, %d0
    bne TEST_FAIL
    
    clr.l %d0
    tst.l %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* EXT - sign extend */
    /* =================================================================== */
    move.l #0xFFFFFF80, %d0  | -128 as byte
    ext.w %d0
    cmp.w #0xFF80, %d0
    bne TEST_FAIL
    
    ext.l %d0
    cmp.l #0xFFFFFF80, %d0
    bne TEST_FAIL
    
    rts

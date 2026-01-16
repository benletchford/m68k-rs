.include "entry.s"
/* Test: Disassembler Coverage - Branch and Control Instructions */
/* Simplified version avoiding DBF and CCR manipulation */

run_test:
    /* =================================================================== */
    /* BRA - Branch Always */
    /* =================================================================== */
    bra skip1
    bra TEST_FAIL           | Should not execute
skip1:
    
    /* =================================================================== */
    /* Bcc - Conditional branches (test multiple conditions) */
    /* =================================================================== */
    
    /* BEQ/BNE */
    move.l #0, %d0
    tst.l %d0
    bne TEST_FAIL
    beq skip2
    bra TEST_FAIL
skip2:

    /* BPL/BMI */
    move.l #-1, %d0
    tst.l %d0
    bpl TEST_FAIL
    bmi skip3
    bra TEST_FAIL
skip3:

    /* BGT/BLE (signed comparison) */
    move.l #10, %d0
    cmp.l #5, %d0
    ble TEST_FAIL
    bgt skip6
    bra TEST_FAIL
skip6:

    /* BGE/BLT */
    move.l #5, %d0
    cmp.l #5, %d0
    blt TEST_FAIL
    bge skip7
    bra TEST_FAIL
skip7:

    /* BHI/BLS (unsigned comparison) */
    move.l #10, %d0
    cmp.l #5, %d0
    bls TEST_FAIL
    bhi skip8
    bra TEST_FAIL
skip8:
    
    /* =================================================================== */
    /* JSR/RTS - Subroutine call */
    /* =================================================================== */
    jsr subroutine
    cmp.l #42, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* BSR - Branch to subroutine */
    /* =================================================================== */
    bsr subroutine
    cmp.l #42, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* JMP - Jump */
    /* =================================================================== */
    jmp skip9
    bra TEST_FAIL
skip9:
    
    rts

subroutine:
    move.l #42, %d0
    rts

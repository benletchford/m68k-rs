.include "entry.s"
/* Test: 68020+ Exception Frame Formats */
/* Verifies correct stack frame format for different exception types */

.set BERR_VEC, 0x08            | Bus Error vector
.set AERR_VEC, 0x0C            | Address Error vector

run_test:
    /* =================================================================== */
    /* Test 1: Verify Format 0 frame for TRAP */
    /* =================================================================== */
    /* Install custom TRAP #0 handler */
    lea trap_handler_0, %a0
    mov.l %a0, 0x80             | TRAP #0 vector
    
    clr.l %d6                   | Counter
    trap #0
    
    /* D6 should be set by handler */
    cmp.l #1, %d6
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: Verify frame format field */
    /* =================================================================== */
    /* D7 contains the frame format from the handler */
    /* Format 0 = 0, Format 2 = 2, etc. */
    and.l #0xF000, %d7          | Mask to format field
    /* 68040 TRAP should use Format 0 (0x0xxx) */
    cmp.l #0x0000, %d7
    bne TEST_FAIL
    
    rts

/* TRAP #0 handler - captures frame format */
trap_handler_0:
    addq.l #1, %d6
    /* Read the format/vector word from stack */
    /* On 68020+: SP+0 = SR, SP+2 = PC.hi, SP+4 = PC.lo, SP+6 = Format/Vector */
    move.w 6(%sp), %d7          | Get format/vector word
    rte

.include "entry.s"
/* Test: Division edge cases and divide-by-zero */

run_test:
    clr.l %d0
    
    /* Install div-by-zero handler */
    lea div_zero_handler, %a0
    move.l %a0, 0x14        | Division by zero vector
    
    /* DIVU normal case */
    move.l #1000, %d1
    divu.w #10, %d1
    cmp.l #100, %d1         | Quotient in low word
    bne TEST_FAIL
    
    /* DIVU with remainder */
    move.l #105, %d2
    divu.w #10, %d2
    move.l %d2, %d3
    swap %d3
    cmp.w #5, %d3           | Remainder in high word
    bne TEST_FAIL
    
    /* DIVU overflow (result > 65535) */
    move.l #0x10000, %d4
    divu.w #1, %d4
    bvs 1f                  | Overflow should be set
    bra TEST_FAIL
1:
    
    /* DIVU by zero - should trap and go to handler which sets D0=1 */
    clr.l %d7               | Use D7 as success marker (set by handler)
    clr.l %d5
    move.l #100, %d6
    divu.w %d5, %d6         | Should jump to handler
    
    /* Check that handler was called (D7 should be 1) */
    cmp.l #1, %d7
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

div_zero_handler:
    move.l #1, %d7          | Mark that handler was called
    | Skip over the next instruction (bra TEST_FAIL should not be there now)
    | For 68010+, stacked PC points past DIVU, so we return to the check
    rte

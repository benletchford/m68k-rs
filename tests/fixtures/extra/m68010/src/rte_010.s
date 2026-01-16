.include "entry.s"
/* Test: 68010 RTE Stack Frame Format */
/* Verifies 68010 exception frame has format word at SP+6 with format=$8 for Address Error */

run_test:
    /* Install custom exception handler */
    lea exception_handler, %a0
    move.l %a0, 0x0C        | Address Error vector (vector 3)
    
    /* Trigger Address Error */
    clr.l %d0               | Mark as not visited (D0=0)
    move.l #1, %a1          | Odd address 
    move.w (%a1), %d1       | Trigger Address Error
    
    /* We should never get here - exception handler sets pass/fail directly */
    bra TEST_FAIL           
    
exception_handler:
    /* 68010 stack frame:
     * SP+0: Status Register (word)
     * SP+2: Program Counter (long)
     * SP+6: Format/Vector (word) - 68010 specific
     *
     * Format nibble is in upper 4 bits of format/vector word.
     * For 68010 Address Error, format should be $8.
     */
    
    /* Verify format field (upper nibble of format/vector word) */
    move.w 6(%sp), %d1      | Get format/vector word
    lsr.w #8, %d1           | Shift to get upper byte  
    lsr.w #4, %d1           | Shift to get format nibble
    and.w #0xF, %d1
    
    /* 68010 uses format $8 for bus/address errors */
    cmp.w #8, %d1
    bne format_wrong
    
    /* Format is correct - signal PASS and halt */
    mov.l #1, TEST_PASS_REG
    stop #0x2700
    
format_wrong:
    /* Format was not $8 - signal FAIL */
    bra TEST_FAIL

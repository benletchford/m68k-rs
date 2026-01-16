.include "entry.s"
/* Test: LC040 MMU Positive - MMU ops should work on LC040 */

run_test:
    clr.l %d0               | D0 = 0 (success)
    
    /* PTEST should work on LC040 (has MMU, no FPU) */
    /* Use a simple PTEST operation */
    lea test_addr, %a0
    .word 0xF010            | PTEST encoding
    .word 0x8200
    
    /* If we get here, PTEST executed without trapping */
    rts

.data
test_addr:
    .long 0x12345678

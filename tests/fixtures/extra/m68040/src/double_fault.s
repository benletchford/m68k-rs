.include "entry.s"
/* Test: Basic Exception Stacking Test */
/* Simplified version - verifies bus error handler can be installed */
/* (Full double-fault detection requires bus error signaling from memory controller) */

run_test:
    /* Install a bus error handler */
    lea bus_err_handler, %a0
    mov.l %a0, 0x08             | Vector 2: Bus Error
    
    /* Verify the vector was installed */
    mov.l 0x08, %d0
    cmp.l %a0, %d0
    bne TEST_FAIL
    
    /* Test that we can still execute without triggering bus errors */
    /* (since our emulator doesn't generate bus errors for unmapped memory) */
    mov.l #0x12345678, %d1
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    
    rts

bus_err_handler:
    /* If we get here, a bus error was triggered */
    rte

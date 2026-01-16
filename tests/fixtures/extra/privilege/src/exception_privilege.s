.include "entry.s"
/* Test: Verify SR.S bit set on exception */
/* Note: 68040 handles misaligned access transparently, so we use TRAP */
/* Note: After RTE returns to user mode, must use TRAP to return to supervisor for RTS */

run_test:
    clr.l %d0
    
    /* Initialize USP for user mode */
    lea 0x200, %a0
    move.l %a0, %usp
    
    /* Install TRAP #1 handler (the test exception) */
    lea trap_handler, %a0
    move.l %a0, 0x84            | TRAP #1 vector
    
    /* Install TRAP #2 handler (for returning to supervisor) */
    lea trap_return, %a0
    move.l %a0, 0x88            | TRAP #2 vector
    
    /* Switch to user mode */
    andi.w #0xDFFF, %sr
    
    /* Trigger TRAP #1 - this will cause an exception */
    trap #1
    
    /* After RTE: back in user mode, return to supervisor via TRAP #2 */
    trap #2
    bra TEST_FAIL               | Shouldn't reach here

trap_handler:
    /* Verify we're in supervisor mode */
    move.w %sr, %d1
    btst #13, %d1
    beq TEST_FAIL
    
    move.l #1, %d0
    rte

trap_return:
    /* Return to caller in supervisor mode */
    addq.l #8, %sp              | Pop exception frame (format 0: 8 bytes)
    rts

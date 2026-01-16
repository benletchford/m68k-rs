.include "entry.s"
/* Test: USP/SSP switching during exceptions */
/* Note: After RTE returns to user mode, must use TRAP to get back to supervisor for RTS */

run_test:
    clr.l %d0
    
    /* Set up test USP value */
    move.l #0x1000, %a0
    move.l %a0, %usp
    
    /* Install TRAP handlers */
    lea trap_handler, %a1
    move.l %a1, 0x80        | TRAP #0 vector
    lea trap_return, %a1
    move.l %a1, 0x84        | TRAP #1 vector for returning
    
    /* Switch to user mode with a known A7 value */
    /* We set USP already, so clear S bit to switch to USP */
    andi.w #0xDFFF, %sr     | Clear S bit - now SP = USP = $1000
    
    /* Trigger TRAP - should switch to SSP */
    trap #0
    
    /* After RTE, we're back in user mode */
    /* In user mode, we CAN'T access USP directly - just check that SP is sane */
    /* The TRAP handler already verified SSP/USP separation */
    
    cmp.l #1, %d0           | D0 was set by trap handler if SSP != USP
    bne TEST_FAIL
    
    /* Return via TRAP #1 (can't RTS from user mode - return addr on SSP) */
    trap #1
    bra TEST_FAIL           | Shouldn't reach here

trap_handler:
    /* In supervisor mode now, check SSP is not the USP value */
    move.l %sp, %d1
    move.l %usp, %a3        | This is OK in supervisor mode
    move.l %a3, %d2
    
    /* D1 = current SSP, D2 = USP */
    /* SSP should be some value, USP should be $1000 */
    /* They must be different for this test to pass */
    cmp.l %d1, %d2
    beq TEST_FAIL           | SSP == USP would be wrong
    
    move.l #1, %d0          | Mark success
    rte

trap_return:
    /* Return to caller in supervisor mode */
    addq.l #8, %sp          | Pop exception frame (format 0: 8 bytes)
    rts

.include "entry.s"
/* Test: Interrupt Handling */

.set INT_HANDLER_LOC, 0x64     | Level 1 autovector (vector 25)
.set INT_LVL2_LOC, 0x68        | Level 2 autovector (vector 26)
.set INT_LVL7_LOC, 0x7C        | Level 7 autovector (vector 31)

run_test:
    /* Save original vectors */
    mov.l INT_HANDLER_LOC, %d7
    mov.l INT_LVL7_LOC, %a6
    
    /* =================================================================== */
    /* Test 1: Level 1 Interrupt Handling */
    /* =================================================================== */
    lea int_handler_1, %a0
    mov.l %a0, INT_HANDLER_LOC
    clr.l %d6                   | Counter for verifying handler ran
    mov.w #0x2000, %sr          | Set IPL=0 to allow level 1 interrupts
    
    /* Trigger interrupt via test device */
    mov.l #1, INTERRUPT_REG     | Request level 1 interrupt
    nop                         | Allow interrupt to be processed
    nop
    
    cmp.l #1, %d6               | Handler should have incremented D6
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: IPL Masking - Level below mask should not trigger */
    /* =================================================================== */
    clr.l %d6
    mov.w #0x2300, %sr          | IPL = 3, supervisor mode
    
    mov.l #1, INTERRUPT_REG     | Request level 1 (masked by IPL 3)
    nop
    nop
    
    cmp.l #0, %d6               | Handler should NOT have run
    bne TEST_FAIL
    
    mov.w #0x2700, %sr          | Restore IPL = 7
    
    /* =================================================================== */
    /* Test 3: Level 7 Interrupt (NMI - cannot be masked) */
    /* =================================================================== */
    lea int_handler_7, %a0
    mov.l %a0, INT_LVL7_LOC
    clr.l %d6
    
    mov.l #7, INTERRUPT_REG     | Request level 7 interrupt
    nop
    nop
    
    cmp.l #7, %d6               | Handler should have set D6 to 7
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: Verify SR saved and restored correctly */
    /* =================================================================== */
    lea int_handler_sr, %a0
    mov.l %a0, INT_LVL7_LOC     | Use level 7 (NMI) which can't be masked
    
    mov.w #0x2700, %sr          | Set known SR value (IPL=7)
    clr.l %d5                   | Will hold captured SR
    
    mov.l #7, INTERRUPT_REG     | Use level 7 (NMI) since level 1 is masked
    nop
    nop
    
    /* D5 now contains SR from exception frame */
    and.w #0x0700, %d5          | Mask to just IPL bits
    cmp.w #0x0700, %d5          | IPL should have been 7
    bne TEST_FAIL
    
    /* Restore original vectors */
    mov.l %d7, INT_HANDLER_LOC
    mov.l %a6, INT_LVL7_LOC
    
    rts

/* Level 1 interrupt handler */
int_handler_1:
    addq.l #1, %d6
    rte

/* Level 7 interrupt handler */
int_handler_7:
    mov.l #7, %d6
    rte

/* SR capture handler */
int_handler_sr:
    mov.w (%sp), %d5            | Read SR from exception frame
    addq.l #1, %d6
    rte

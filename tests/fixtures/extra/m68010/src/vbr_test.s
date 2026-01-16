.include "entry.s"
/* Test: Vector Base Register (VBR) */
/* Verifies VBR relocation on 68010+ */

.set VBR_TEST_ADDR, 0x300000

run_test:
    /* Save original VBR */
    movec %vbr, %a0
    move.l %a0, %d7         | Save in D7
    
    /* Test 1: Write VBR */
    move.l #VBR_TEST_ADDR, %a1
    movec %a1, %vbr
    
    /* Test 2: Read back VBR */
    movec %vbr, %a2
    cmp.l %a1, %a2
    bne TEST_FAIL
    
    /* Test 3: Verify exception uses new VBR */
    /* Setup a CHK exception handler at new VBR location */
    lea chk_handler, %a3
    move.l %a3, 0x18(%a1)   | Vector 6 (CHK) at offset 0x18
    
    /* Trigger CHK exception */
    clr.l %d0               | Mark as not visited
    move.w #10, %d1
    chk.w #5, %d1           | Should trap (10 > 5)
    
    /* If we get here without handler being called, fail */
    cmp.l #1, %d0
    bne TEST_FAIL
    
    /* Restore original VBR */
    movec %d7, %vbr
    
    rts

chk_handler:
    move.l #1, %d0          | Mark handler visited
    | For 68010+, stacked PC already points past CHK instruction
    rte


.include "entry.s"
/* Test: MOVE from CCR */
/* Verifies MOVE CCR, <ea> works on 68010+ (illegal on 68000) */

.set TEST_MEM, 0x300000         | Use RAM area for test

run_test:
    /* Test 1: MOVE CCR captures actual current CCR flags */
    /* Use ADDX to set X flag and verify MOVE CCR captures it */
    move.w #0x2710, %sr     | Set X flag only, keep supervisor+int mask  
    move.w %ccr, %d0        | Capture CCR to D0
    
    and.w #0x1F, %d0        | Mask to CCR bits
    cmp.w #0x10, %d0        | Should have X flag only
    bne TEST_FAIL
    
    /* Test 2: MOVE CCR to memory works */
    move.w #0x271F, %sr     | Set all CCR flags
    lea TEST_MEM, %a0
    move.w %ccr, (%a0)      | Write CCR to memory (before any flag-changing ops)
    
    move.w (%a0), %d1
    and.w #0x1F, %d1
    cmp.w #0x1F, %d1        | All flags should be set
    bne TEST_FAIL
    
    /* Test 3: MOVE CCR with zero CCR */
    move.w #0x2700, %sr     | Clear all CCR flags
    move.w %ccr, %d2        | Capture CCR (all flags clear)
    
    and.w #0x1F, %d2
    bne TEST_FAIL           | Should be zero
    
    rts

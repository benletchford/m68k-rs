.include "entry.s"
/* Test: Address Mode Stress Test */
/* Complex addressing mode interactions and edge cases */

.set DATA_LOC, STACK2_BASE

run_test:
    /* =================================================================== */
    /* Test 1: Register used as Source and Destination Address */
    /* MOVE.L -(A0), (A0)+  vs  MOVE.L (A0)+, -(A0) */
    /* =================================================================== */
    /* Setup data */
    lea DATA_LOC, %a0
    mov.l #0x11111111, (%a0)
    mov.l #0x22222222, 4(%a0)
    
    /* Case A: MOVE.L (A0)+, -(A0) */
    /* A0 points to DATA_LOC */
    /* Source (A0)+ reads 0x11111111, A0 becomes DATA_LOC+4 */
    /* Dest -(A0) decrements A0 to DATA_LOC, writes 0x11111111 there */
    /* Net result: value stays same, A0 points to DATA_LOC */
    
    lea DATA_LOC, %a0
    move.l (%a0)+, -(%a0)
    
    cmp.l #DATA_LOC, %a0        | A0 should be back at start
    bne TEST_FAIL
    move.l (%a0), %d0
    cmp.l #0x11111111, %d0      | Value should be preserved
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: Register interactions - MOVE.L A0, -(A0) */
    /* Only Valid if Decrement (Target Calc) happens BEFORE Source Read? */
    /* Or Source Read happens before Decrement? */
    /* 68040 UM: Source EA calculated, Data fetched. Dest EA calculated. Data written. */
    /* So Source A0 is read (unmodified). Dest -(A0) decrements A0. */
    /* Value written is ORIGINAL A0. */
    /* =================================================================== */
    lea DATA_LOC, %a0
    move.l %a0, %d0             | Save original value
    move.l %a0, -(%a0)          | Pushes A0 value to [A0-4]
    
    /* If stored value is ORIGINAL A0, and A0 is now A0-4 */
    /* Then (%a0) == stored_value == ORIGINAL A0 */
    /* So (%a0) != %a0 */
    
    move.l (%a0), %d1
    cmp.l %d0, %d1              | Stored value should be ORIGINAL A0
    bne TEST_FAIL
    
    move.l %a0, %d1
    sub.l #4, %d0
    cmp.l %d0, %d1              | A0 should be decremented by 4
    bne TEST_FAIL

    /* =================================================================== */

    /* Test 3: Stack Pointer Manipulation in Subroutine */
    /* =================================================================== */
    mov.l %sp, %d7              | Save SP
    bsr mangle_sp
    cmp.l %sp, %d7              | SP should be restored
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: Nested Exception Stack Alignment */
    /* 68040 requires stack frame alignment to word boundary */
    /* =================================================================== */
    mov.l %sp, %d0
    btst #0, %d0                | Check odd bit
    bne TEST_FAIL               | SP must be even
    
    /* =================================================================== */
    /* Test 5: Index scaling edge cases */
    /* =================================================================== */
    lea DATA_LOC, %a0
    mov.l #0, %d0
    lea (0, %a0, %d0.l*8), %a1  | Scale *8
    cmp.l %a0, %a1
    bne TEST_FAIL
    
    rts

mangle_sp:
    /* Manually push data to stack, then check if RTS works */
    /* This tests if RTS relies on compiled stack usage or actual memory */
    move.l #0x12345678, -(%sp)
    move.l #0x87654321, -(%sp)
    
    /* Pop manually */
    move.l (%sp)+, %d0
    cmp.l #0x87654321, %d0
    bne TEST_FAIL
    
    move.l (%sp)+, %d0
    cmp.l #0x12345678, %d0
    bne TEST_FAIL
    
    rts

.include "entry.s"
/* Test: Cycle Timing - Exception Overhead */
/* Tests timing of exception entry/exit sequences */

.set TRAP_VEC_0, 0x80

run_test:
    /* =================================================================== */
    /* Test 1: TRAP instruction timing */
    /* TRAP takes 34 cycles on 68000 (varies by CPU type) */
    /* =================================================================== */
    lea trap_handler, %a0
    move.l %a0, TRAP_VEC_0
    
    clr.l %d6               | Counter
    
    trap #0                 | Execute TRAP, handler increments D6
    
    cmp.l #1, %d6
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: Multiple TRAP timing */
    /* =================================================================== */
    trap #0
    trap #0
    trap #0
    
    cmp.l #4, %d6           | 1 + 3 = 4
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: RTE timing */
    /* RTE takes 20 cycles on 68000 */
    /* The trap handler already uses RTE */
    /* =================================================================== */
    move.l #10, %d6         | Reset counter
    
    trap #0
    
    cmp.l #11, %d6
    bne TEST_FAIL
    
    rts

trap_handler:
    addq.l #1, %d6
    rte

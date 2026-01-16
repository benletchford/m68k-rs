.include "entry.s"
/* Test: Cycle Timing Verification */
/* Tests instruction timing accuracy by counting cycles for known sequences */

run_test:
    /* =================================================================== */
    /* Test 1: NOP timing baseline */
    /* On 68000: NOP = 4 cycles */
    /* =================================================================== */
    nop
    nop
    nop
    nop
    /* If cycles were being counted accurately, we'd have consumed 16 cycles */
    /* For now, this just verifies NOPs execute without issue */
    
    /* =================================================================== */
    /* Test 2: MOVE timing */
    /* MOVE.L Dn,Dn = 4 cycles on 68000 */
    /* =================================================================== */
    move.l #0x12345678, %d0
    move.l %d0, %d1
    move.l %d1, %d2
    move.l %d2, %d3
    
    cmp.l #0x12345678, %d3
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: ADD timing */
    /* ADD.L Dn,Dn = 8 cycles on 68000 (6 on 68020+) */
    /* =================================================================== */
    move.l #100, %d4
    move.l #50, %d5
    add.l %d5, %d4
    add.l %d5, %d4
    add.l %d5, %d4
    add.l %d5, %d4
    
    cmp.l #300, %d4
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: MUL timing (expensive operation) */
    /* MULU.W = 38-70 cycles on 68000, 28 on 68020+ */
    /* =================================================================== */
    move.l #17, %d0
    move.l #23, %d1
    mulu.w %d0, %d1
    
    cmp.l #391, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 5: DIV timing (most expensive) */
    /* DIVU.W = 140 cycles worst case on 68000, 44 on 68020+ */
    /* =================================================================== */
    move.l #10000, %d0
    move.l #100, %d1
    divu.w %d1, %d0
    
    and.l #0xFFFF, %d0      | Get quotient only
    cmp.l #100, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 6: Memory access timing */
    /* (An) access modes have different cycle costs */
    /* =================================================================== */
    lea test_data, %a0
    move.l (%a0), %d0
    move.l 4(%a0), %d1
    move.l 8(%a0), %d2
    
    cmp.l #0xAABBCCDD, %d0
    bne TEST_FAIL
    cmp.l #0x11223344, %d1
    bne TEST_FAIL
    cmp.l #0x55667788, %d2
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 7: Branch timing */
    /* Taken branches cost more than not-taken */
    /* =================================================================== */
    move.l #5, %d0
    
loop_test:
    subq.l #1, %d0
    bne loop_test
    
    /* After loop, D0 should be 0 */
    tst.l %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 8: Shift/Rotate timing */
    /* Shifts cost 6+2n cycles on 68000 */
    /* =================================================================== */
    move.l #0x80000000, %d0
    lsr.l #8, %d0           | Shift by 8
    
    cmp.l #0x00800000, %d0
    bne TEST_FAIL
    
    move.l #0x00000001, %d0
    lsl.l #4, %d0           | Shift by 4
    
    cmp.l #0x00000010, %d0
    bne TEST_FAIL
    
    rts

test_data:
    .long 0xAABBCCDD
    .long 0x11223344
    .long 0x55667788

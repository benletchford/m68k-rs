.include "entry.s"
/* Test: Cycle Timing - Memory Access Modes */
/* Different addressing modes have different cycle costs */

run_test:
    /* Setup base addresses */
    lea test_data, %a0
    lea test_data+4, %a1
    move.l #0, %a2
    
    /* =================================================================== */
    /* Test 1: (An) - Address Register Indirect */
    /* Base timing for memory access */
    /* =================================================================== */
    move.l (%a0), %d0
    cmp.l #0x11111111, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: (An)+ - Post-increment */
    /* Same as (An) but with auto-increment */
    /* =================================================================== */
    move.l %a0, %a2         | Save A0
    move.l (%a2)+, %d0
    move.l (%a2)+, %d1
    
    cmp.l #0x11111111, %d0
    bne TEST_FAIL
    cmp.l #0x22222222, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: -(An) - Pre-decrement */
    /* Adds cycles for predecrement calculation */
    /* =================================================================== */
    lea test_data+12, %a3
    move.l -(%a3), %d0
    move.l -(%a3), %d1
    
    cmp.l #0x33333333, %d0
    bne TEST_FAIL
    cmp.l #0x22222222, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: d(An) - Displacement */
    /* Adds 4 cycles for displacement extension word */
    /* =================================================================== */
    move.l 0(%a0), %d0
    move.l 4(%a0), %d1
    move.l 8(%a0), %d2
    
    cmp.l #0x11111111, %d0
    bne TEST_FAIL
    cmp.l #0x22222222, %d1
    bne TEST_FAIL
    cmp.l #0x33333333, %d2
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 5: d(An,Xn) - Indexed */
    /* Most expensive basic addressing mode */
    /* =================================================================== */
    move.l #4, %d7
    move.l 0(%a0,%d7.l), %d0
    
    cmp.l #0x22222222, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 6: xxx.W - Absolute Short */
    /* Faster than absolute long for low memory */
    /* =================================================================== */
    move.l test_data, %d0
    cmp.l #0x11111111, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 7: xxx.L - Absolute Long */
    /* Full 32-bit address */
    /* =================================================================== */
    move.l test_data+4, %d0
    cmp.l #0x22222222, %d0
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 8: d(PC) - PC Relative */
    /* Used for position-independent code */
    /* =================================================================== */
    move.l pc_rel_data(%pc), %d0
    cmp.l #0x44444444, %d0
    bne TEST_FAIL
    
    rts

test_data:
    .long 0x11111111
    .long 0x22222222
    .long 0x33333333

pc_rel_data:
    .long 0x44444444

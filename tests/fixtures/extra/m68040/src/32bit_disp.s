.include "entry.s"
/* Test: 32-bit Displacement Addressing (68020+) */

.set DATA_LOC, STACK2_BASE - 0x100

run_test:
    /* Setup test data */
    lea DATA_LOC, %a0
    move.l #0x12345678, (%a0)+
    move.l #0xABCDEF01, (%a0)+
    move.l #0xFEDCBA98, (%a0)+
    move.l #0x87654321, (%a0)+
    
    /* Test 1: Large positive displacement (bd.l,An,Xn) */
    lea DATA_LOC-0x10000, %a0
    move.l #0, %d0
    move.l (0x10000,%a0,%d0.l), %d1
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    
    /* Test 2: Negative displacement */
    lea DATA_LOC+0x10000, %a0
    move.l (-0x10000,%a0,%d0.l), %d1
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    
    /* Test 3: Displacement with index */
    lea DATA_LOC-0x8000, %a0
    move.l #4, %d0
    move.l (0x8000,%a0,%d0.l), %d1
    cmp.l #0xABCDEF01, %d1
    bne TEST_FAIL
    
    /* Test 4: At 16-bit boundary */
    lea DATA_LOC-0x8000, %a0
    move.l #0, %d0
    move.l (0x8000,%a0,%d0.l), %d1
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    
    /* Test 5: LEA with 32-bit displacement */
    lea DATA_LOC-0x10000, %a0
    lea (0x10004,%a0), %a1
    move.l (%a1), %d1
    cmp.l #0xABCDEF01, %d1
    bne TEST_FAIL
    
    rts

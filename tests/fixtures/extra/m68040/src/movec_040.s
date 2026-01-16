.include "entry.s"
/* Test: 68040 MOVEC Additional Registers */
/* Tests MMUSR, URP, SRP, TC, DACR0/1, IACR0/1 */

run_test:
    /* =================================================================== */
    /* Test 1: MMUSR - MMU Status Register */
    /* =================================================================== */
    movec %mmusr, %d0           | Read MMUSR
    /* Just verify we can read it without exception */
    
    /* =================================================================== */
    /* Test 2: URP - User Root Pointer */
    /* =================================================================== */
    move.l #0x12340000, %d0
    movec %d0, %urp             | Write URP
    movec %urp, %d1             | Read back
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 3: SRP - Supervisor Root Pointer */
    /* =================================================================== */
    move.l #0x56780000, %d0
    movec %d0, %srp             | Write SRP
    movec %srp, %d1             | Read back
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: TC - Translation Control */
    /* =================================================================== */
    move.l #0x00008000, %d0     | Enable bit set
    movec %d0, %tc              | Write TC
    movec %tc, %d1              | Read back
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Disable for safety */
    move.l #0, %d0
    movec %d0, %tc
    
    /* =================================================================== */
    /* Test 5: DACR0 - Data Access Control 0 */
    /* =================================================================== */
    move.l #0x00004000, %d0
    movec %d0, %dacr0           | Write DACR0
    movec %dacr0, %d1           | Read back
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 6: IACR0 - Instruction Access Control 0 */
    /* =================================================================== */
    move.l #0x00004000, %d0
    movec %d0, %iacr0           | Write IACR0
    movec %iacr0, %d1           | Read back
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Clear registers */
    move.l #0, %d0
    movec %d0, %dacr0
    movec %d0, %dacr1
    movec %d0, %iacr0
    movec %d0, %iacr1
    
    rts

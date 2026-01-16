.include "entry.s"
/* Test: MMU ATC (Address Translation Cache) Operations */
/* Tests that ATC-related registers and operations work */

run_test:
    /* =================================================================== */
    /* Test 1: MMUSR (MMU Status Register) read */
    /* Verify MMUSR is readable */
    /* =================================================================== */
    
    movec %mmusr, %d0
    /* Just verify no exception - value is implementation-defined */
    
    /* =================================================================== */
    /* Test 2: PFLUSH - Flush ATC entries */
    /* 68040 uses PFLUSHA (flush all) or PFLUSH with FC/EA */
    /* =================================================================== */
    
    /* PFLUSHA - flush all ATC entries */
    /* Opcode: F518 (PFLUSHA) */
    .word 0xF518
    
    /* Should execute without exception */
    
    /* =================================================================== */
    /* Test 3: DACR0/DACR1 - Data Access Control */
    /* Transparent translation regions for data */
    /* =================================================================== */
    
    /* Set DACR0 to cover low memory */
    move.l #0x0000C000, %d0 | Base=0, E=1, W=1
    movec %d0, %dacr0
    
    movec %dacr0, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Set DACR1 for high memory region */
    move.l #0xFF00C000, %d0 | Base=0xFF, E=1, W=1
    movec %d0, %dacr1
    
    movec %dacr1, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 4: IACR0/IACR1 - Instruction Access Control */
    /* Transparent translation regions for instructions */
    /* =================================================================== */
    
    /* Set IACR0 */
    move.l #0x0000C000, %d0
    movec %d0, %iacr0
    
    movec %iacr0, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* Set IACR1 */
    move.l #0x8000C000, %d0
    movec %d0, %iacr1
    
    movec %iacr1, %d1
    cmp.l %d0, %d1
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 5: Clear all transparent translation */
    /* =================================================================== */
    
    move.l #0, %d0
    movec %d0, %dacr0
    movec %d0, %dacr1
    movec %d0, %iacr0
    movec %d0, %iacr1
    
    /* Verify all cleared */
    movec %dacr0, %d1
    tst.l %d1
    bne TEST_FAIL
    
    movec %dacr1, %d1
    tst.l %d1
    bne TEST_FAIL
    
    movec %iacr0, %d1
    tst.l %d1
    bne TEST_FAIL
    
    movec %iacr1, %d1
    tst.l %d1
    bne TEST_FAIL
    
    rts

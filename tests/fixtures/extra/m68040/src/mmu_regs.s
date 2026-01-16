.include "entry.s"
/* Test: MMU Control Registers */

run_test:
    /* Test 1: Read TC (Translation Control) via MOVEC */
    movec %tc, %d0
    /* Just verify no crash */
    
    /* Test 2: Read URP (User Root Pointer) */
    movec %urp, %d0
    
    /* Test 3: Read SRP (Supervisor Root Pointer) */
    movec %srp, %d0
    
    /* Test 4: Read DTT0 (Data Transparent Translation 0) */
    movec %dtt0, %d0
    
    /* Test 5: Read DTT1 */
    movec %dtt1, %d0
    
    /* Test 6: Read ITT0 (Instruction Transparent Translation 0) */
    movec %itt0, %d0
    
    /* Test 7: Read ITT1 */
    movec %itt1, %d0
    
    /* Test 8: Read MMUSR (only 68040) */
    movec %mmusr, %d0
    
    rts

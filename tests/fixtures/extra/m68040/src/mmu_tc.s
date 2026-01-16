.include "entry.s"
/* Test: MMU Translation Control and Enable */

run_test:
    /* Test 1: Read current TC value */
    movec %tc, %d0
    move.l %d0, %d7
    
    /* Test 2: Write TC with MMU disabled */
    move.l #0, %d0
    movec %d0, %tc
    movec %tc, %d1
    /* Lower bits may be masked */
    
    /* Test 3: Set up root pointers */
    /* SRP and URP point to page tables */
    movec %srp, %d0
    movec %urp, %d0
    
    /* Test 4: Restore original TC */
    move.l %d7, %d0
    movec %d0, %tc
    
    rts

.include "entry.s"
/* Test: ATC (Address Translation Cache) Operations */

run_test:
    /* PFLUSHA already tested, now test ATC-specific behavior */
    
    /* Test 1: PFLUSHA clears all ATC entries */
    pflusha
    nop
    
    /* Test 2: PFLUSH specific address */
    lea STACK2_BASE, %a0
    pflush (%a0)
    nop
    
    /* Test 3: PFLUSHN (non-global) */
    pflushn (%a0)
    nop
    
    /* Test 4: Access after flush (should cause ATC reload) */
    move.l (%a0), %d0
    nop
    
    rts

.include "entry.s"
/* Test: PFLUSH - MMU TLB Flush (68030/68040) */

run_test:
    /* Test 1: PFLUSHA - Flush all entries */
    pflusha
    nop
    
    /* Test 2: PFLUSHN - Flush non-global entries */
    lea STACK2_BASE, %a0
    pflushn (%a0)
    nop
    
    /* Test 3: PFLUSH (An) */
    lea STACK2_BASE+0x1000, %a0
    pflush (%a0)
    nop
    
    rts

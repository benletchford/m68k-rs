.include "entry.s"
/* Test: CINV/CPUSH - Cache Invalidate/Push (68040 only) */

run_test:
    /* Test 1: CINVA DC - Invalidate all data cache */
    cinva %dc
    nop
    
    /* Test 2: CINVA IC - Invalidate all instruction cache */
    cinva %ic
    nop
    
    /* Test 3: CINVA BC - Invalidate both caches */
    cinva %bc
    nop
    
    /* Test 4: CPUSHA DC */
    cpusha %dc
    nop
    
    /* Test 5: CPUSHA IC */
    cpusha %ic
    nop
    
    /* Test 6: CPUSHA BC */
    cpusha %bc
    nop
    
    /* Test 7: CINVL - Invalidate line */
    lea STACK2_BASE, %a0
    cinvl %dc, (%a0)
    
    /* Test 8: CPUSHL - Push line */
    cpushl %dc, (%a0)
    
    /* Test 9: CINVP - Invalidate page */
    cinvp %dc, (%a0)
    
    /* Test 10: CPUSHP - Push page */
    cpushp %dc, (%a0)
    
    rts

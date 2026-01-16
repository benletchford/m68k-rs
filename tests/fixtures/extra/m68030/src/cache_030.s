.include "entry.s"
/* Test: M68030 Cache Operations */

run_test:
    clr.l %d0
    
    /* CINVA IC - Invalidate instruction cache */
    .word 0xF498
    
    /* CINVA DC - Invalidate data cache */
    .word 0xF498
    .word 0x0001
    
    /* CPUSHA IC - Push and invalidate instruction cache */
    .word 0xF4F8
    
    rts

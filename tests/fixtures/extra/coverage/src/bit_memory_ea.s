.include "entry.s"
/* Test: Bit ops with complex effective addresses */

run_test:
    clr.l %d0
    
    /* Test with (An)+ */
    move.l #0x2000, %a0
    move.b #0, (%a0)
    move.b #0, 1(%a0)
    
    move.l #0x2000, %a1
    bset #3, (%a1)+
    bset #5, (%a1)+
    
    move.b 0x2000, %d1
    cmp.b #8, %d1
    bne TEST_FAIL
    
    move.b 0x2001, %d1
    cmp.b #32, %d1
    bne TEST_FAIL
    
    /* Test with -(An) */
    move.l #0x2002, %a2
    clr.b -(%a2)
    bset #7, (%a2)
    
    move.b 0x2001, %d2
    cmp.b #0x80, %d2
    bne TEST_FAIL
    
    /* Test with displacement */
    move.l #0x2010, %a3
    clr.b 4(%a3)
    bset #2, 4(%a3)
    
    move.b 0x2014, %d3
    cmp.b #4, %d3
    bne TEST_FAIL
    
    /* Test with index */
    move.l #0x2020, %a4
    move.l #6, %d4
    clr.b 0(%a4,%d4.l)
    bset #1, 0(%a4,%d4.l)
    
    move.b 0x2026, %d5
    cmp.b #2, %d5
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

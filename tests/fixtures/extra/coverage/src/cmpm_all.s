.include "entry.s"
/* Test: CMPM all sizes */

run_test:
    clr.l %d0
    
    /* Setup test data */
    move.l #0x2000, %a0
    move.l #0x2010, %a1
    
    /* CMPM.b - equal */
    move.b #0x55, (%a0)
    move.b #0x55, (%a1)
    
    move.l #0x2000, %a2
    move.l #0x2010, %a3
    cmpm.b (%a2)+, (%a3)+
    bne TEST_FAIL
    
    /* Verify post-increment */
    cmp.l #0x2001, %a2
    bne TEST_FAIL
    
    /* CMPM.w - not equal */
    move.w #0x1234, 0x2002
    move.w #0x1235, 0x2012
    
    move.l #0x2002, %a4
    move.l #0x2012, %a5
    cmpm.w (%a4)+, (%a5)+
    beq TEST_FAIL
    
    /* CMPM.l - signed comparison */
    move.l #0x80000000, 0x2004  | -2147483648 (signed)
    move.l #0x70000000, 0x2014  | +1879048192 (signed)
    
    move.l #0x2004, %a6
    move.l #0x2014, %a0
    cmpm.l (%a6)+, (%a0)+       | dest - source = 0x70000000 - 0x80000000
    blt TEST_FAIL               | Signed: dest(+) > source(-), so should NOT be less
    
    move.l #1, %d0
    rts

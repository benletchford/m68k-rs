.include "entry.s"
/* Test: BCHG and BTST on data registers and memory */

run_test:
    clr.l %d0
    
    /* Test BTST - should not modify value */
    move.l #0x80, %d1
    btst #7, %d1
    bne 1f                  | Should be set
    bra TEST_FAIL
1:  cmp.l #0x80, %d1        | Value unchanged
    bne TEST_FAIL
    
    btst #0, %d1
    beq 2f                  | Should be clear
    bra TEST_FAIL
2:
    
    /* Test BCHG on data register */
    move.l #0x55, %d2
    bchg #0, %d2
    cmp.l #0x54, %d2
    bne TEST_FAIL
    
    bchg #0, %d2
    cmp.l #0x55, %d2
    bne TEST_FAIL
    
    /* Test BTST on memory */
    move.l #0x2000, %a0
    move.b #0xAA, (%a0)
    btst #7, (%a0)
    bne 3f
    bra TEST_FAIL
3:  btst #0, (%a0)
    beq 4f
    bra TEST_FAIL
4:
    
    /* Test BCHG on memory */
    bchg #3, (%a0)
    move.b (%a0), %d3
    cmp.b #0xA2, %d3
    bne TEST_FAIL
    
    /* Test dynamic bit number */
    move.l #6, %d4
    move.l #0x40, %d5
    btst %d4, %d5
    bne 5f
    bra TEST_FAIL
5:
    
    move.l #1, %d0
    rts

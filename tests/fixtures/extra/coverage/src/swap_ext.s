.include "entry.s"
/* Test: SWAP and EXT operations */

run_test:
    clr.l %d0
    
    /* SWAP */
    move.l #0x12345678, %d1
    swap %d1
    cmp.l #0x56781234, %d1
    bne TEST_FAIL
    
    /* SWAP twice = original */
    swap %d1
    cmp.l #0x12345678, %d1
    bne TEST_FAIL
    
    /* EXT.w - sign extend byte to word */
    move.l #0x87, %d2
    ext.w %d2
    cmp.w #0xFF87, %d2
    bne TEST_FAIL
    
    move.l #0x7F, %d3
    ext.w %d3
    cmp.w #0x007F, %d3
    bne TEST_FAIL
    
    /* EXT.l - sign extend word to long */
    move.l #0x8000, %d4
    ext.l %d4
    cmp.l #0xFFFF8000, %d4
    bne TEST_FAIL
    
    move.l #0x7FFF, %d5
    ext.l %d5
    cmp.l #0x00007FFF, %d5
    bne TEST_FAIL
    
    /* EXTB.l - sign extend byte to long (68020+) */
    #ifdef M68020
    move.l #0x80, %d6
    extb.l %d6
    cmp.l #0xFFFFFF80, %d6
    bne TEST_FAIL
    #endif
    
    move.l #1, %d0
    rts

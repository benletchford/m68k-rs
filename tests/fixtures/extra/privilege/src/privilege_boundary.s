.include "entry.s"
/* Test: Privilege boundary - verify SR modification works in supervisor mode */
/* Note: Tests that ANDI/ORI to SR work correctly in supervisor mode */

run_test:
    clr.l %d0
    
    /* Save original SR */
    move.w %sr, %d1
    
    /* Test: ANDI to SR (should work in supervisor) */
    andi.w #0xFFF0, %sr
    
    /* Verify flags cleared */
    move.w %sr, %d2
    andi.w #0x000F, %d2
    bne TEST_FAIL           | Lower nibble should be 0
    
    /* Test: ORI to SR (should work in supervisor) */
    ori.w #0x0700, %sr      | Set interrupt mask
    
    /* Verify interrupt mask set */
    move.w %sr, %d2
    andi.w #0x0700, %d2
    cmp.w #0x0700, %d2
    bne TEST_FAIL
    
    /* Restore S bit to supervisor */
    ori.w #0x2000, %sr
    
    move.l #1, %d0
    rts

.include "entry.s"
/* Test: PACK - Pack BCD (68020+) */

run_test:
    /* PACK Dx,Dy,#adj: Dest = (src[11:8] << 4) | src[3:0] + adj */
    
    /* Test 1: Pack 0x0302 with adj=0 -> 0x32 */
    move.w #0x0302, %d0
    pack %d0, %d1, #0
    cmp.b #0x32, %d1
    bne TEST_FAIL
    
    /* Test 2: Pack 0x0908 with adj=0 -> 0x98 */
    move.w #0x0908, %d0
    pack %d0, %d1, #0
    cmp.b #0x98, %d1
    bne TEST_FAIL
    
    /* Test 3: With adjustment */
    move.w #0x0302, %d0
    pack %d0, %d1, #0x30
    cmp.b #0x62, %d1
    bne TEST_FAIL
    
    /* Test 4: ASCII digits to BCD */
    /* '39' (0x3339) + adj=0xFFCC -> packed + adj = 0x39 + 0xFFCC = 0x05 */
    move.w #0x3339, %d0
    pack %d0, %d1, #0xFFCC
    cmp.b #0x05, %d1
    bne TEST_FAIL
    
    /* Test 5: Pack 0x0000 -> 0x00 */
    move.w #0x0000, %d0
    pack %d0, %d1, #0
    cmp.b #0x00, %d1
    bne TEST_FAIL
    
    rts

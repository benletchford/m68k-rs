.include "entry.s"
/* Test: UNPK - Unpack BCD (68020+) */

run_test:
    /* UNPK Dx,Dy,#adj: Dest[11:8]=src[7:4], Dest[3:0]=src[3:0] + adj */
    
    /* Test 1: Unpack 0x32 with adj=0 -> 0x0302 */
    move.b #0x32, %d0
    unpk %d0, %d1, #0
    cmp.w #0x0302, %d1
    bne TEST_FAIL
    
    /* Test 2: Unpack 0x98 with adj=0 -> 0x0908 */
    move.b #0x98, %d0
    unpk %d0, %d1, #0
    cmp.w #0x0908, %d1
    bne TEST_FAIL
    
    /* Test 3: With adjustment for ASCII */
    /* 0x39 + adj=0x3030 -> 0x3339 ('39') */
    move.b #0x39, %d0
    unpk %d0, %d1, #0x3030
    cmp.w #0x3339, %d1
    bne TEST_FAIL
    
    /* Test 4: Unpack 0x00 -> 0x0000 */
    move.b #0x00, %d0
    unpk %d0, %d1, #0
    cmp.w #0x0000, %d1
    bne TEST_FAIL
    
    /* Test 5: 0xFF -> 0x0F0F */
    move.b #0xFF, %d0
    unpk %d0, %d1, #0
    cmp.w #0x0F0F, %d1
    bne TEST_FAIL
    
    rts

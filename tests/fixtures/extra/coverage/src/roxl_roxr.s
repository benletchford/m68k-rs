.include "entry.s"
/* Test: Rotate through extend ROXL/ROXR */

run_test:
    clr.l %d0
    
    /* ROXL.b with X=0 */
    andi.w #0xFFEF, %sr     | Clear X flag
    move.l #0x81, %d1
    roxl.b #1, %d1
    cmp.b #0x02, %d1        | Rotated, X=0 shifted in
    bne TEST_FAIL
    
    /* ROXL.b with X=1 */
    ori.w #0x0010, %sr      | Set X flag
    move.l #0x40, %d2
    roxl.b #1, %d2
    cmp.b #0x81, %d2        | X=1 shifted in
    bne TEST_FAIL
    
    /* ROXL.w */
    andi.w #0xFFEF, %sr
    move.l #0x8000, %d3
    roxl.w #1, %d3
    cmp.w #0x0000, %d3
    bne TEST_FAIL
    
    /* ROXR.b with X=1 */
    ori.w #0x0010, %sr
    move.l #0x01, %d4
    roxr.b #1, %d4
    cmp.b #0x80, %d4        | X shifted in from left
    bne TEST_FAIL
    
    /* ROXL.l */
    andi.w #0xFFEF, %sr
    move.l #0x80000000, %d5
    roxl.l #1, %d5
    bcc TEST_FAIL           | C should be set from MSB shifted out (check BEFORE cmp)
    cmp.l #0x00000000, %d5  | Now verify result
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

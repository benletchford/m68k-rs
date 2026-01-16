.include "entry.s"
/* Test: Shift count > 32 (mod 64 behavior) */

run_test:
    clr.l %d0
    
    /* Shift counts are modulo 64 for register-based shifts */
    /* Count 65 mod 64 = 1 */
    move.l #65, %d1
    move.l #0x80000000, %d2
    lsl.l %d1, %d2
    cmp.l #0x00000000, %d2  | 0x80000000 << 1 = 0
    bne TEST_FAIL
    
    /* Count 33 mod 64 = 33, which is >= 32 bits, so result is 0 */
    move.l #33, %d3
    move.l #0xFFFFFFFF, %d4
    lsr.l %d3, %d4
    cmp.l #0x00000000, %d4  | All bits shifted out
    bne TEST_FAIL
    
    /* Count 64 mod 64 = 0 (no shift) */
    move.l #64, %d5
    move.l #0x12345678, %d6
    asl.l %d5, %d6
    cmp.l #0x12345678, %d6  | No change
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

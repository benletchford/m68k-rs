.include "entry.s"
/* Test: Memory shift/rotate operations */

run_test:
    clr.l %d0
    
    /* ASL memory (always 1 bit) */
    move.l #0x2000, %a0
    move.w #0x4000, (%a0)
    asl.w (%a0)
    move.w (%a0), %d1
    cmp.w #0x8000, %d1
    bne TEST_FAIL
    
    /* ASR memory */
    move.w #0x8000, (%a0)
    asr.w (%a0)
    move.w (%a0), %d1
    cmp.w #0xC000, %d1      | Sign-extended
    bne TEST_FAIL
    
    /* LSL memory */
    move.w #0x1234, (%a0)
    lsl.w (%a0)
    move.w (%a0), %d1
    cmp.w #0x2468, %d1
    bne TEST_FAIL
    
    /* LSR memory */
    move.w #0x8000, (%a0)
    lsr.w (%a0)
    move.w (%a0), %d1
    cmp.w #0x4000, %d1      | Zero-filled
    bne TEST_FAIL
    
    /* ROL memory */
    move.w #0x8001, (%a0)
    rol.w (%a0)
    move.w (%a0), %d1
    cmp.w #0x0003, %d1
    bne TEST_FAIL
    
    /* ROR memory */
    move.w #0x8001, (%a0)
    ror.w (%a0)
    move.w (%a0), %d1
    cmp.w #0xC000, %d1
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

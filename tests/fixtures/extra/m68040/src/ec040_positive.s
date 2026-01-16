.include "entry.s"
/* Test: EC040 Positive - Verify 040 features work (no MMU/FPU) */

run_test:
    clr.l %d0
    
    /* Test MOVE16 - should work on EC040 */
    lea src_data, %a0
    lea dst_data, %a1
    
    /* Align addresses to 16-byte boundary */
    move.l %a0, %d1
    andi.l #0xFFFFFFF0, %d1
    move.l %d1, %a0
    
    move.l %a1, %d1
    andi.l #0xFFFFFFF0, %d1
    move.l %d1, %a1
    
    /* MOVE16 (A0)+, (A1)+ */
    .word 0xF620
    .word 0x8000
    
    rts

.data
.align 4
src_data:
    .space 32, 0xAA
dst_data:
    .space 32, 0x00

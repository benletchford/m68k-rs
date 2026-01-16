.include "entry.s"
/* Test: M68030 MOVE16 - Burst mode with different alignment */

run_test:
    clr.l %d0
    
    /* 030 MOVE16 has different alignment rules */
    lea src_030, %a0
    lea dst_030, %a1
    
    /* Align to 16-byte boundary */
    move.l %a0, %d1
    andi.l #0xFFFFFFF0, %d1
    move.l %d1, %a0
    
    move.l %a1, %d1
    andi.l #0xFFFFFFF0, %d1
    move.l %d1, %a1
    
    /* MOVE16 on 030 */
    .word 0xF620
    .word 0x8000
    
    rts

.data
.align 4
src_030:
    .space 32, 0x55
dst_030:
    .space 32, 0x00

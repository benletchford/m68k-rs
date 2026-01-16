.include "entry.s"
/* Test: MMU Instructions on EC040 (should trap) */

run_test:
    /* Install Line-F handler */
    lea line_f_handler, %a0
    move.l %a0, 0x2C
    
    clr.l %d0               | D0 = 0
    
    /* PTEST on EC040 should trap */
    .word 0xF010            | PTEST encoding
    .word 0x8200
    
    rts

line_f_handler:
    move.l #1, %d0
    addq.l #4, 2(%sp)
    rte

.include "entry.s"
/* Test: Architecture Variance - FPU Trap */
/* LC040/EC040: Should trigger Line-F Exception */
/* 68040/020+FPU: Should execute */

.set FLINE_VEC, 0x2C           | Vector 11: Line-F Emulator

run_test:
    /* Status register (D0): 0 = Success (No Fault), 1 = Fault occurred */
    clr.l %d0
    
    /* Install Line-F Handler */
    mov.l FLINE_VEC, %d7        | Save original
    lea fline_handler, %a0
    mov.l %a0, FLINE_VEC
    
    /* Attempt FPU Instruction */
    /* FMOVE.L %D0, %FP0 - valid on 68881/2 and 68040 */
    fmove.l %d0, %fp0
    
    /* Restore vector */
    mov.l %d7, FLINE_VEC
    
    rts

fline_handler:
    mov.l #1, %d0               | Mark that fault occurred
    
    /* Line-F stack frame is Standard 4-word (Format 0) usually? */
    /* PC (4), SR (2), Format/Vector (2) = 8 bytes */
    /* We can skip instruction and RTE? */
    /* FMOVE is 4 bytes? */
    
    addq.l #4, 2(%sp)           | Skip instruction (adjust PC on stack)
    rte

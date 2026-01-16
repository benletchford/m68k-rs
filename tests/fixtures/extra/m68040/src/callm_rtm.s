.include "entry.s"
/* Test: CALLM/RTM Module Instructions */
/* These are 68020-only instructions, emulated as illegal on 68030/040 */
/* Expected: F-line exception (illegal instruction) on 68040 */

.set FLINE_VEC, 0x2C           | Vector 11: Line-F Emulator

run_test:
    /* Save original Line-F vector */
    mov.l FLINE_VEC, %a5
    
    /* Install our handler */
    lea fline_handler, %a0
    mov.l %a0, FLINE_VEC
    
    clr.l %d6                   | Counter: how many F-line exceptions?
    
    /* =================================================================== */
    /* Test 1: CALLM should trigger F-line exception on 68040 */
    /* =================================================================== */
    /* CALLM is encoded as 0x06C0-0x06FF range */
    /* On 68040, this is an unimplemented instruction */
    
    .short 0x06C0               | CALLM #0, (A0) encoding
    .short 0x0000               | Module descriptor
    
    cmp.l #1, %d6               | Should have caught one exception
    bne TEST_FAIL
    
    /* =================================================================== */
    /* Test 2: RTM should trigger F-line exception on 68040 */
    /* =================================================================== */
    /* RTM is encoded as 0x06C0 with register field */
    
    .short 0x06C8               | RTM A0 encoding
    
    cmp.l #2, %d6               | Should have caught second exception
    bne TEST_FAIL
    
    /* Restore original vector */
    mov.l %a5, FLINE_VEC
    
    rts

fline_handler:
    addq.l #1, %d6
    | Check if CALLM (0x06C0-0x06C7) or RTM (0x06C8-0x06CF)
    | CALLM has extension word (4 bytes), RTM is 2 bytes
    move.l 2(%sp), %a0
    move.w (%a0), %d0
    and.w #0xFFF8, %d0
    cmp.w #0x06C8, %d0
    beq skip_2
    addq.l #4, 2(%sp)           | CALLM: Skip 4 bytes
    bra done_skip
skip_2:
    addq.l #2, 2(%sp)           | RTM: Skip 2 bytes
done_skip:
    rte

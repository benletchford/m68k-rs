.include "entry.s"
/* Test: Self-Modifying Code (SMC) Test */
/* Verifies dynamic code generation in RAM works correctly */
/* (Test 1 removed - patching ROM is not possible) */

.set CODE_BUFFER, STACK2_BASE + 0x200

run_test:
    /* =================================================================== */
    /* Test: Generate code in RAM and execute it */
    /* =================================================================== */
    lea CODE_BUFFER, %a0
    
    /* Write: MOVE.L #0xDEADBEEF, %D1; RTS */
    mov.w #0x223C, (%a0)+       | MOVE.L #imm, D1
    mov.l #0xDEADBEEF, (%a0)+   | Immediate
    mov.w #0x4E75, (%a0)+       | RTS
    
    jsr CODE_BUFFER
    
    cmp.l #0xDEADBEEF, %d1
    bne TEST_FAIL
    
    /* Test 2: Generate a return value calculation */
    lea CODE_BUFFER, %a0
    
    /* Write: MOVE.L #42, %D0; RTS */
    mov.w #0x203C, (%a0)+       | MOVE.L #imm, D0
    mov.l #42, (%a0)+           | Immediate value
    mov.w #0x4E75, (%a0)+       | RTS
    
    jsr CODE_BUFFER
    
    cmp.l #42, %d0
    bne TEST_FAIL
    
    rts

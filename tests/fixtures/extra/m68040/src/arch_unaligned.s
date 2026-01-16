.include "entry.s"
/* Test: Architecture Variance - Unaligned Access */
/* 68000/010: Should trigger Address Error */
/* 68020+: Should succeed */

.set ADDR_ERR_VEC, 0x0C        | Vector 3: Address Error (or 0x0C for vector 3)
.set UNALIGNED_LOC, STACK2_BASE + 1

run_test:
    /* Status register (D0): 0 = Success (No Fault), 1 = Fault occurred */
    clr.l %d0
    
    /* Install Address Error Handler */
    mov.l ADDR_ERR_VEC, %d7     | Save original
    lea addr_err_handler, %a0
    mov.l %a0, ADDR_ERR_VEC
    
    /* Attempt Unaligned Word Write */
    lea UNALIGNED_LOC, %a0
    mov.w #0x1234, (%a0)
    
    /* If we get here without exception, D0 is still 0 */
    
    /* Restore vector */
    mov.l %d7, ADDR_ERR_VEC
    
    rts

addr_err_handler:
    mov.l #1, %d0               | Mark that fault occurred
    
    /* Stack cleanup for Address Error (Group 1/2 Exception) */
    /* Stack Frame varies by CPU! This is tricky for a generic handler */
    /* 68000: PC(4), SR(2), IR(2), AccessAddr(4), FnCode(2) = 14 bytes */
    /* 68010: Format(2), ... = 26 bytes short frame? */
    /* 68020/030: Bus Error frame? Wait, unaligned on 020 doesn't fault */
    /* So we mostly care about 68000 cleanup */
    
    /* For 68000: Skip the instruction that caused it? */
    /* The saved PC is usually pointing to instruction or next? */
    /* 68000 Address Error PC points to "next instruction" usually? */
    /* Or "current instruction"? */
    /* Actually, we just want to EXIT the test or skip. */
    
    /* Since we can't easily skip without decoding, let's just RETURN from the test */
    /* But we are in exception context. We can't just RTS */
    /* We need to clean up stack and then act as if run_test returned */
    
    /* Simplified strategy: Just STOP. The Rust harness can inspect D0 */
    /* But we want cleaner exit. */
    /* Let's try to restore stack pointer to initial state and RTS from run_test? */
    /* run_test was called via JSR. Stack has Return Address */
    /* But now we have Exception stack on top. */
    
    /* If 68000: 14 bytes frame or similar. */
    /* If we assume this test is ONLY run on CPUs that fault or don't... */
    /* If it faults, we are likely on 68000/010. */
    
    /* Let's trigger a STOP 0x2700, and rely on Rust harness inspecting registers from Stopped state? */
    /* m68k harness loops until `cpu.stopped != 0`. */
    
    stop #0x2700
    
    rte

.include "entry.s"
/* Test: RESET Instruction (Supervisor only) */

run_test:
    reset
    nop
    rts

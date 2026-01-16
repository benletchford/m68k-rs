.include "entry.s"
/* Test: MMU PTEST Instruction (68040) */
/* 68040 PTEST format differs from 68030 */

run_test:
    /* 68040 PTEST syntax: ptestr/ptestw (An) */
    /* No function code parameter in 68040 */
    
    /* Test 1: PTESTR - Read test */
    lea STACK2_BASE, %a0
    ptestr (%a0)
    nop
    
    /* Read MMUSR after PTEST */
    movec %mmusr, %d0
    
    /* Test 2: PTESTW - Write test */
    lea STACK2_BASE+0x1000, %a0
    ptestw (%a0)
    nop
    
    movec %mmusr, %d0
    
    rts

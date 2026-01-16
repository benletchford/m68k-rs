.include "entry.s"
/* Test: RTE from different exception types */

run_test:
    clr.l %d0
    
    /* Test 1: RTE from TRAP - standard 6-byte frame on 68000 */
    lea trap_handler, %a0
    move.l %a0, 0x80        | Install at vector 32 (TRAP #0)
    trap #0
    
    cmp.l #1, %d0
    bne TEST_FAIL
    
    /* Test 2: RTE from Address Error - 14-byte frame on 68000 */
    lea addr_handler, %a1
    move.l %a1, 0x0C        | Install at vector 3 (Address Error)
    move.w 0x1001, %d1      | Trigger Address Error
    
    cmp.l #2, %d0
    bne TEST_FAIL
    
    rts

trap_handler:
    move.l #1, %d0
    rte

addr_handler:
    move.l #2, %d0
    
    /* 68000 Address Error frame (14 bytes) - pop and return manually */
    move.l (%sp), %a0       | Get return PC
    addq.l #4, %a0          | Skip faulting instruction
    move.w 4(%sp), %d7      | Get SR
    lea 14(%sp), %sp        | Pop 14-byte frame
    move.w %d7, %sr         | Restore SR
    jmp (%a0)               | Return

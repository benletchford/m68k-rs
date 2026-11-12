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
    lea after_addr_error, %a6
    move.w 0x1001, %d1      | Trigger Address Error
after_addr_error:

    cmp.l #2, %d0
    bne TEST_FAIL
    
    rts

trap_handler:
    move.l #1, %d0
    rte

addr_handler:
    move.l #2, %d0

    /* 68000 group 0 frame (14 bytes): status word (SP+0), access address
       (SP+2), instruction register (SP+6), SR (SP+8), PC (SP+10). The
       stacked PC is only loosely defined (2 to 10 bytes past the opcode),
       so resume at the continuation address the test left in A6: replace
       the frame with a three-word one and RTE. */
    cmp.l #0x1001, 2(%sp)   | Access address is the odd operand address
    bne TEST_FAIL
    move.w 8(%sp), %d7      | Stacked SR
    lea 14(%sp), %sp        | Pop the group 0 frame
    move.l %a6, -(%sp)      | Continuation PC
    move.w %d7, -(%sp)      | SR
    rte

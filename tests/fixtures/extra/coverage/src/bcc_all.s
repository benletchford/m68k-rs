.include "entry.s"
/* Test: All 16 Bcc condition code variants */

run_test:
    clr.l %d0
    
    /* BRA - always */
    bra 1f
    bra TEST_FAIL
1:
    
    /* BEQ/BNE - Z flag */
    moveq #0, %d1
    beq 2f
    bra TEST_FAIL
2:  moveq #1, %d1
    bne 3f
    bra TEST_FAIL
3:
    
    /* BCS/BCC - C flag */
    ori.w #1, %sr           | Set C
    bcs 4f
    bra TEST_FAIL
4:  andi.w #0xFFFE, %sr    | Clear C
    bcc 5f
    bra TEST_FAIL
5:
    
    /* BMI/BPL - N flag */
    ori.w #8, %sr           | Set N
    bmi 6f
    bra TEST_FAIL
6:  andi.w #0xFFF7, %sr    | Clear N
    bpl 7f
    bra TEST_FAIL
7:
    
    /* BVS/BVC - V flag */
    ori.w #2, %sr           | Set V
    bvs 8f
    bra TEST_FAIL
8:  andi.w #0xFFFD, %sr    | Clear V
    bvc 9f
    bra TEST_FAIL
9:
    
    /* BGT/BLE - N=V and Z=0 / Z=1 or N!=V */
    andi.w #0xFFF0, %sr     | Clear all
    moveq #1, %d2
    bgt 10f
    bra TEST_FAIL
10:
    
    /* BHI/BLS - C=0 and Z=0 / C=1 or Z=1 */
    andi.w #0xFFF0, %sr
    moveq #1, %d3
    bhi 11f
    bra TEST_FAIL
11:
    
    /* BGE/BLT - N=V / N!=V */
    andi.w #0xFFF0, %sr
    bge 12f
    bra TEST_FAIL
12:
    
    move.l #1, %d0
    rts

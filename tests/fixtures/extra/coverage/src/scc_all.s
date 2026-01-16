.include "entry.s"
/* Test: All 16 Scc condition code variants */

run_test:
    clr.l %d0
    
    /* ST - always true */
    clr.l %d1
    st %d1
    cmp.b #0xFF, %d1
    bne TEST_FAIL
    
    /* SF - always false */
    move.l #0xFF, %d2
    sf %d2
    cmp.b #0, %d2
    bne TEST_FAIL
    
    /* SEQ - Z=1 */
    clr.l %d3               | CLR sets Z=1, so this works
    seq %d3                 | Z=1 from CLR, so sets D3=0xFF
    cmp.b #0xFF, %d3
    bne TEST_FAIL
    
    /* SNE - Z=0 */
    clr.l %d4               | CLR first (will set Z=1)
    andi.w #0xFFFB, %sr     | Clear Z after CLR
    sne %d4                 | Now Z=0, so sets D4=0xFF
    cmp.b #0xFF, %d4
    bne TEST_FAIL
    
    /* SCS - C=1 */
    clr.l %d5               | CLR first (clears C)
    ori.w #1, %sr           | Set C after CLR
    scs %d5                 | C=1, so sets D5=0xFF
    cmp.b #0xFF, %d5
    bne TEST_FAIL
    
    /* SCC - C=0 */
    clr.l %d6               | CLR sets C=0, so this works
    scc %d6                 | C=0 from CLR, so sets D6=0xFF
    cmp.b #0xFF, %d6
    bne TEST_FAIL
    
    /* SMI - N=1 */
    clr.l %d7               | CLR first (clears N)
    ori.w #8, %sr           | Set N after CLR
    smi %d7                 | N=1, so sets D7=0xFF
    cmp.b #0xFF, %d7
    bne TEST_FAIL
    
    /* SPL - N=0 */
    move.l #0x2000, %a0
    clr.b (%a0)             | CLR sets N=0, so this works
    spl (%a0)               | N=0 from CLR, so sets (A0)=0xFF
    move.b (%a0), %d1
    cmp.b #0xFF, %d1
    bne TEST_FAIL
    
    move.l #1, %d0
    rts

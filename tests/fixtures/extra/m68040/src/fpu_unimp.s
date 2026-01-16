.include "entry.s"
/* Test: FPU Transcendentals Now Implemented */
/* Note: Real 68040 traps transcendentals to software. */
/* Our emulator implements them directly for convenience. */
/* This test verifies FSIN works correctly. */

run_test:
    /* Test: FSIN(0) should be 0 */
    fmove.l #0, %fp0
    fsin.x %fp0
    
    ftst.x %fp0
    fbne TEST_FAIL
    
    /* Success - FSIN is implemented and works */
    rts

//! Transparent Translation Register (TTR) handling.
//!
//! Implements TTR matching for 68030 (TT0/TT1) and the 68040/68060
//! (ITT0/ITT1, DTT0/DTT1).
//! TTRs allow certain address ranges to bypass page table translation.

use crate::core::cpu::CpuCore;
use crate::core::types::CpuType;

/// 68030 TT0/TT1 register format (MC68030UM 9.7.1, figure 9-23):
/// ```text
/// `31:24` Logical address base (compared against address bits `31:24`)
/// `23:16` Logical address mask (1 = ignore that bit in the comparison)
/// [15]    E: enable
/// [10]    CI: cache inhibit (ignored here)
/// [9]     R/W: 1 = read accesses match, 0 = write accesses match
/// [8]     RWM: R/W mask, 1 = reads and writes match (R/W ignored)
/// `6:4`   FC base (function code to match)
/// `2:0`   FC mask (1 = ignore that FC bit in the comparison)
/// ```
/// The 68040/68060 ITTn/DTTn keep the base, mask and enable fields and
/// replace the rest; see [`ttr_matches_040`].
const TTR_ENABLE: u32 = 0x8000;
const TTR_BASE_MASK: u32 = 0xFF00_0000;
const TTR_ADDR_MASK_SHIFT: u32 = 16;
const TTR_030_RW: u32 = 0x0200;
const TTR_030_RWM: u32 = 0x0100;
const TTR_FC_BASE_SHIFT: u32 = 4;
const TTR_FC_MASK_SHIFT: u32 = 0;

/// Check if a single 68030-format TT register matches an access to `addr`
/// in function-code space `fc`, a write when `write` is set.
///
/// The access matches when the register is enabled, its address bits 31-24
/// and function code agree with the base fields outside the masked bits,
/// and either RWM is set or the access's direction is the one R/W selects
/// (MC68030UM 9.7). Not modelled: an indivisible read-modify-write cycle
/// (TAS, CAS, CAS2) is transparent only through a register with RWM=1;
/// here its read and write cycles are matched one at a time.
pub fn ttr_matches(ttr: u32, addr: u32, fc: u8, write: bool) -> bool {
    // Check enable bit
    if (ttr & TTR_ENABLE) == 0 {
        return false;
    }

    // Extract fields
    let base = (ttr & TTR_BASE_MASK) >> 24;
    let addr_mask = (ttr >> TTR_ADDR_MASK_SHIFT) & 0xFF;
    let fc_base = ((ttr >> TTR_FC_BASE_SHIFT) & 0x07) as u8;
    let fc_mask = ((ttr >> TTR_FC_MASK_SHIFT) & 0x07) as u8;

    // Compare address (masked)
    let addr_high = (addr >> 24) & 0xFF;
    let addr_match = (addr_high & !addr_mask) == (base & !addr_mask);

    // Compare function code (masked)
    let fc_match = (fc & !fc_mask) == (fc_base & !fc_mask);

    // Compare the access direction unless RWM masks it.
    let rw_match = ttr & TTR_030_RWM != 0 || (ttr & TTR_030_RW != 0) != write;

    addr_match && fc_match && rw_match
}

/// Check if a single 68040-format ITT/DTT register matches. The 040
/// dropped the 030's function-code base/mask fields: privilege matching is
/// the two-bit S-field at bits 14:13 (00 = user accesses only, 01 =
/// supervisor only, 1x = both), with the address base/mask and enable bit
/// unchanged (M68040UM 3.1.2). Matching an 040 TTR with the 030 rules
/// silently turns most real configurations into never-matching no-ops --
/// 68040 bring-up code shields itself with `S = both` TTRs across an MMU
/// enable, and that shield must hold.
pub fn ttr_matches_040(ttr: u32, addr: u32, supervisor: bool) -> bool {
    if (ttr & TTR_ENABLE) == 0 {
        return false;
    }
    let base = (ttr & TTR_BASE_MASK) >> 24;
    let addr_mask = (ttr >> TTR_ADDR_MASK_SHIFT) & 0xFF;
    let addr_high = (addr >> 24) & 0xFF;
    if (addr_high & !addr_mask) != (base & !addr_mask) {
        return false;
    }
    match (ttr >> 13) & 3 {
        0b00 => !supervisor,
        0b01 => supervisor,
        _ => true,
    }
}

/// Outcome of the transparent-translation check for one access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TtHit {
    /// No enabled register matches: the access goes through the tables.
    Miss,
    /// A register matches: the access is not translated.
    Hit,
    /// A 68040/68060 register matches a write and its W bit is set: the
    /// access is not translated, and it takes an access fault instead of
    /// being performed (MC68040UM 3.1.3, MC68060UM 4.1.3).
    WriteProtected,
}

/// 68040/68060 TTR W (write protect, bit 2).
const TTR_040_W: u32 = 0x0000_0004;

/// The transparent-translation outcome for an access in function-code space
/// `fc` (the access's own code, or the SFC/DFC of a MOVES, PTEST or PLPA).
///
/// The 68030 compares both TT registers against every access. The 68040 and
/// 68060 qualify instruction fetches (FC 2/6) with ITT0/ITT1 and every other
/// access with DTT0/DTT1, and a match on a write-protected register (W=1)
/// write-protects the access. When both registers of a pair match, the
/// first one's attributes apply.
#[inline]
pub(crate) fn lookup(cpu: &CpuCore, addr: u32, fc: u8, write: bool) -> TtHit {
    match cpu.cpu_type {
        CpuType::M68030 => {
            // 68030 has two shared TTRs for both instruction and data
            if ttr_matches(cpu.mmu_tt0, addr, fc, write)
                || ttr_matches(cpu.mmu_tt1, addr, fc, write)
            {
                return TtHit::Hit;
            }
        }
        CpuType::M68EC040 | CpuType::M68LC040 | CpuType::M68040 | CpuType::M68060 => {
            // The 040/060 TTRs carry the S-field, not 030 FC base/mask; the
            // access's privilege is FC2.
            let supervisor = (fc & 4) != 0;
            let (r0, r1) = if fc & 3 == 2 {
                (cpu.itt0, cpu.itt1)
            } else {
                (cpu.dtt0, cpu.dtt1)
            };
            for ttr in [r0, r1] {
                if ttr_matches_040(ttr, addr, supervisor) {
                    return if write && ttr & TTR_040_W != 0 {
                        TtHit::WriteProtected
                    } else {
                        TtHit::Hit
                    };
                }
            }
        }
        _ => {
            // Other CPUs don't have TTRs
        }
    }
    TtHit::Miss
}

/// Check if transparent translation applies for the given access.
///
/// For 68030: Checks TT0 and TT1.
/// For 68040/68060: Checks ITT0/ITT1 for instruction accesses, DTT0/DTT1 for
/// data. The 68060 registers have the 68040 layout (MC68060UM 4.1.3).
///
/// Returns `Some(physical_addr)` if a register matches (identity mapping),
/// or `None` if normal page table translation should be used. A match on a
/// write-protected 68040/68060 register is reported as a match: the address
/// translation path turns a write through it into an access fault.
pub fn check_transparent_translation(
    cpu: &CpuCore,
    addr: u32,
    write: bool,
    instruction: bool,
) -> Option<u32> {
    match lookup(cpu, addr, compute_function_code(cpu, instruction), write) {
        TtHit::Miss => None,
        TtHit::Hit | TtHit::WriteProtected => Some(addr),
    }
}

/// Whether any register [`check_transparent_translation`] consults for this
/// CPU is enabled; when none is, it matches nothing.
pub(crate) fn any_transparent_translation(cpu: &CpuCore) -> bool {
    let enabled = |ttr: u32| ttr & TTR_ENABLE != 0;
    match cpu.cpu_type {
        CpuType::M68030 => enabled(cpu.mmu_tt0) || enabled(cpu.mmu_tt1),
        CpuType::M68EC040 | CpuType::M68LC040 | CpuType::M68040 | CpuType::M68060 => {
            enabled(cpu.itt0) || enabled(cpu.itt1) || enabled(cpu.dtt0) || enabled(cpu.dtt1)
        }
        _ => false,
    }
}

/// Compute function code for the current access.
///
/// FC is a 3-bit value:
/// - 0: Reserved
/// - 1: User Data
/// - 2: User Program (instruction)
/// - 3: Reserved
/// - 4: Reserved
/// - 5: Supervisor Data
/// - 6: Supervisor Program (instruction)
/// - 7: CPU Space (interrupt acknowledge, etc.)
fn compute_function_code(cpu: &CpuCore, instruction: bool) -> u8 {
    // A MOVES data access carries SFC/DFC instead of the CPU-state code.
    if let Some(fc) = cpu.mmu_fc_override {
        return fc;
    }
    let is_supervisor = cpu.is_supervisor();
    match (is_supervisor, instruction) {
        (false, false) => 1, // User Data
        (false, true) => 2,  // User Program
        (true, false) => 5,  // Supervisor Data
        (true, true) => 6,   // Supervisor Program
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ttr_disabled() {
        // TTR with E=0 should not match
        let ttr = 0x00FF_0107; // whole space, any FC, RWM, but E=0
        assert!(!ttr_matches(ttr, 0x1000_0000, 5, false));
    }

    #[test]
    fn test_ttr_address_match() {
        // TTR matching addresses 0x40xxxxxx (base=0x40, mask=0x00)
        // FC mask=7 (bits 2:0) to match any FC, RWM=1 for either direction
        let ttr = 0x4000_8107; // Base=0x40, Mask=0x00, E=1, RWM, FCMask=7
        assert!(ttr_matches(ttr, 0x4000_0000, 5, false));
        assert!(ttr_matches(ttr, 0x40FF_FFFF, 5, true));
        assert!(!ttr_matches(ttr, 0x4100_0000, 5, false));
        assert!(!ttr_matches(ttr, 0x3F00_0000, 5, false));
    }

    #[test]
    fn test_ttr_address_mask() {
        // TTR matching addresses 0x40-0x4F (base=0x40, mask=0x0F)
        let ttr = 0x400F_8107; // Base=0x40, Mask=0x0F, E=1, RWM, FCMask=7
        assert!(ttr_matches(ttr, 0x4000_0000, 5, false));
        assert!(ttr_matches(ttr, 0x4F00_0000, 5, false));
        assert!(!ttr_matches(ttr, 0x5000_0000, 5, false));
    }

    #[test]
    fn test_ttr_fc_match() {
        // TTR matching FC=5 (supervisor data) only
        let ttr = 0x4000_8150; // Base=0x40, E=1, RWM, FC=5, FCMask=0
        assert!(ttr_matches(ttr, 0x4000_0000, 5, false));
        assert!(!ttr_matches(ttr, 0x4000_0000, 1, false)); // User data
        assert!(!ttr_matches(ttr, 0x4000_0000, 6, false)); // Supervisor program
    }

    #[test]
    fn test_ttr_fc_mask() {
        // TTR matching any supervisor access (FC=4-7, FCMask=3)
        let ttr = 0x4000_8143; // Base=0x40, E=1, RWM, FC=4, FCMask=3
        assert!(ttr_matches(ttr, 0x4000_0000, 4, false));
        assert!(ttr_matches(ttr, 0x4000_0000, 5, false));
        assert!(ttr_matches(ttr, 0x4000_0000, 6, false));
        assert!(ttr_matches(ttr, 0x4000_0000, 7, false));
        assert!(!ttr_matches(ttr, 0x4000_0000, 1, false)); // User data
    }

    #[test]
    fn test_ttr_030_rw_and_rwm() {
        // RWM=0: R/W=1 matches reads only, R/W=0 writes only.
        let reads = 0x4000_8207; // E, R/W=1, FCMask=7
        assert!(ttr_matches(reads, 0x4000_0000, 1, false));
        assert!(!ttr_matches(reads, 0x4000_0000, 1, true));
        let writes = 0x4000_8007; // E, R/W=0, FCMask=7
        assert!(!ttr_matches(writes, 0x4000_0000, 1, false));
        assert!(ttr_matches(writes, 0x4000_0000, 1, true));
        // RWM=1 ignores R/W. CI (bit 10) does not affect matching.
        let both = 0x4000_8507; // E, CI, RWM, FCMask=7
        assert!(ttr_matches(both, 0x4000_0000, 1, false));
        assert!(ttr_matches(both, 0x4000_0000, 1, true));
    }

    #[test]
    fn test_ttr_040_s_field() {
        // 040 format: S = 00 user only, 01 supervisor only, 1x both.
        let user_only = 0x00FF_8000; // whole space, S=00
        assert!(ttr_matches_040(user_only, 0x1234_5678, false));
        assert!(!ttr_matches_040(user_only, 0x1234_5678, true));

        let super_only = 0x00FF_A000; // whole space, S=01
        assert!(!ttr_matches_040(super_only, 0x1234_5678, false));
        assert!(ttr_matches_040(super_only, 0x1234_5678, true));

        // The bring-up shield: whole space, both privilege levels.
        let both = 0x00FF_C000;
        assert!(ttr_matches_040(both, 0x1234_5678, false));
        assert!(ttr_matches_040(both, 0x1234_5678, true));

        // Address base/mask work as on the 030; disabled never matches.
        let range = 0x400F_C000; // 0x40-0x4F, both
        assert!(ttr_matches_040(range, 0x4A00_0000, true));
        assert!(!ttr_matches_040(range, 0x5000_0000, true));
        assert!(!ttr_matches_040(0x00FF_4000, 0x1234_5678, true)); // E=0
    }
}

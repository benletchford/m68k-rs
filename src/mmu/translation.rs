//! Address translation (PMMU table walk)

use crate::core::cpu::CpuCore;
use crate::core::memory::{AddressBus, BusFaultKind};

use super::{MmuFault, MmuFaultKind, MmuResult};

fn buserr(address: u32) -> MmuFault {
    MmuFault {
        kind: MmuFaultKind::BusError,
        address,
    }
}

fn access_fault(address: u32) -> MmuFault {
    MmuFault {
        kind: MmuFaultKind::AccessLevelViolation,
        address,
    }
}

fn config_fault(address: u32) -> MmuFault {
    MmuFault {
        kind: MmuFaultKind::ConfigurationError,
        address,
    }
}

fn read_u32_phys<B: AddressBus>(bus: &mut B, addr: u32) -> MmuResult<u32> {
    bus.try_read_long(addr).map_err(|f| {
        if matches!(f.kind, BusFaultKind::BusError) {
            buserr(f.address)
        } else {
            buserr(addr)
        }
    })
}

/// Perform 68030/68040 PMMU translation.
///
/// This implementation follows the structure of Musashi's `pmmu_translate_addr()` algorithm.
/// It currently supports:
/// - CRP/SRP selection via TC bit 25 (0x0200_0000)
/// - Root/table modes 2 (4-byte descriptors) and 3 (8-byte descriptors)
/// - Early-termination descriptors (mode 1) at table A/B/C
/// - Transparent Translation Registers (TTRs) for 68030/68040
///
/// TODO:
/// - Access permission checks and precise MMUSR (`mmu_sr`) bits
/// - Page descriptor root mode (root_limit & 3 == 1)
pub fn translate<B: AddressBus>(
    cpu: &mut CpuCore,
    bus: &mut B,
    logical: u32,
    write: bool,
    supervisor: bool,
    instruction: bool,
) -> MmuResult<u32> {
    // If MMU not enabled, identity-map.
    if !cpu.pmmu_enabled || !cpu.has_pmmu {
        return Ok(logical);
    }

    // During exception processing, bypass translation to prevent recursive faults.
    // Real hardware uses transparent translation or physical addressing for exception frames.
    if cpu.exception_processing {
        return Ok(logical);
    }

    // Check Transparent Translation Registers first - they bypass page table walk.
    if let Some(phys) = super::ttr::check_transparent_translation(cpu, logical, write, instruction)
    {
        return Ok(phys);
    }

    // Root pointer selection: if SRP enabled and supervisor, use SRP; else CRP.
    let use_srp = (cpu.mmu_tc & 0x0200_0000) != 0 && supervisor;
    let (root_aptr, root_limit) = if use_srp {
        (cpu.mmu_srp_aptr, cpu.mmu_srp_limit)
    } else {
        (cpu.mmu_crp_aptr, cpu.mmu_crp_limit)
    };

    // Initial shift / table bits (Musashi):
    // is = tc[19:16], abits=tc[15:12], bbits=tc[11:8], cbits=tc[7:4]
    let is = (cpu.mmu_tc >> 16) & 0xF;
    let abits = (cpu.mmu_tc >> 12) & 0xF;
    let bbits = (cpu.mmu_tc >> 8) & 0xF;
    let cbits = (cpu.mmu_tc >> 4) & 0xF;

    let addr_in = logical;

    #[inline]
    fn top_index(addr: u32, left_shift: u32, bits: u32) -> u32 {
        if bits == 0 {
            return 0;
        }
        // bits is 1..=32. When bits==32, shift right by 0.
        let rshift = 32u32.saturating_sub(bits);
        addr.wrapping_shl(left_shift) >> rshift
    }

    #[inline]
    fn low_bits(addr: u32, shift: u32) -> u32 {
        if shift >= 32 {
            0
        } else {
            addr.wrapping_shl(shift) >> shift
        }
    }

    // Table A offset.
    let mut tofs = top_index(addr_in, is, abits);

    let mut tbl_entry: u32;
    let tamode: u32;

    match root_limit & 3 {
        0 => return Err(config_fault(logical)),
        1 => return Err(config_fault(logical)), // page descriptor root mode not implemented yet
        2 => {
            // 4-byte descriptors
            tofs = tofs.wrapping_mul(4);
            let e = read_u32_phys(bus, tofs.wrapping_add(root_aptr & 0xFFFF_FFFC))?;
            tbl_entry = e;
            tamode = e & 3;
        }
        3 => {
            // 8-byte descriptors: mode in high long, pointer/base in low long
            tofs = tofs.wrapping_mul(8);
            let hi = read_u32_phys(bus, tofs.wrapping_add(root_aptr & 0xFFFF_FFFC))?;
            let lo = read_u32_phys(
                bus,
                tofs.wrapping_add(root_aptr & 0xFFFF_FFFC).wrapping_add(4),
            )?;
            tamode = hi & 3;
            tbl_entry = lo;
        }
        _ => unreachable!(),
    }

    // Table B offset and pointer from A entry.
    tofs = top_index(addr_in, is + abits, bbits);
    let mut tptr = tbl_entry & 0xFFFF_FFF0;
    let tbmode: u32;

    match tamode {
        0 => return Err(access_fault(logical)),
        1 => {
            // Early termination descriptor (Musashi uses &0xffffff00).
            let base = tbl_entry & 0xFFFF_FF00;
            let shift = is + abits;
            let addr_out = low_bits(addr_in, shift).wrapping_add(base);
            return Ok(addr_out);
        }
        2 => {
            tofs = tofs.wrapping_mul(4);
            tbl_entry = read_u32_phys(bus, tofs.wrapping_add(tptr))?;
            tbmode = tbl_entry & 3;
        }
        3 => {
            tofs = tofs.wrapping_mul(8);
            let hi = read_u32_phys(bus, tofs.wrapping_add(tptr))?;
            let lo = read_u32_phys(bus, tofs.wrapping_add(tptr).wrapping_add(4))?;
            tbmode = hi & 3;
            tbl_entry = lo;
        }
        _ => return Err(access_fault(logical)),
    }

    // Table C
    tofs = top_index(addr_in, is + abits + bbits, cbits);
    tptr = tbl_entry & 0xFFFF_FFF0;
    let tcmode: u32;

    match tbmode {
        0 => return Err(access_fault(logical)),
        1 => {
            let base = tbl_entry & 0xFFFF_FF00;
            let shift = is + abits + bbits;
            let addr_out = low_bits(addr_in, shift).wrapping_add(base);
            return Ok(addr_out);
        }
        2 => {
            tofs = tofs.wrapping_mul(4);
            tbl_entry = read_u32_phys(bus, tofs.wrapping_add(tptr))?;
            tcmode = tbl_entry & 3;
        }
        3 => {
            tofs = tofs.wrapping_mul(8);
            let hi = read_u32_phys(bus, tofs.wrapping_add(tptr))?;
            let lo = read_u32_phys(bus, tofs.wrapping_add(tptr).wrapping_add(4))?;
            tcmode = hi & 3;
            tbl_entry = lo;
        }
        _ => return Err(access_fault(logical)),
    }

    // Final termination at table C.
    match tcmode {
        1 => {
            let base = tbl_entry & 0xFFFF_FF00;
            let shift = is + abits + bbits + cbits;
            Ok(low_bits(addr_in, shift).wrapping_add(base))
        }
        _ => Err(access_fault(logical)),
    }
}

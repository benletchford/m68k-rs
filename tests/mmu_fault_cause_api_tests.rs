//! `MmuFaultCause` is a public, exhaustive enum: code outside the crate
//! matches every variant without a wildcard arm. The match below is that
//! code. It stops compiling if a variant is added, which is the point: a new
//! cause is a breaking change and has to be made as one.

use m68k::mmu::MmuFaultCause;

fn describe(cause: MmuFaultCause) -> &'static str {
    match cause {
        MmuFaultCause::PointerA => "pointer A",
        MmuFaultCause::PointerB => "pointer B",
        MmuFaultCause::Indirect => "indirect",
        MmuFaultCause::PageFault => "page fault",
        MmuFaultCause::WriteProtect => "write protect",
        MmuFaultCause::SupervisorProtect => "supervisor protect",
        MmuFaultCause::TableWalkBusError => "table walk bus error",
        MmuFaultCause::AccessBusError => "access bus error",
    }
}

#[test]
fn every_fault_cause_is_matched_without_a_wildcard() {
    assert_eq!(describe(MmuFaultCause::WriteProtect), "write protect");
}

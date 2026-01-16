//! Status register and CCR flag constants.
//!
//! Flag manipulation methods are now in cpu.rs.

/// Status Register bit positions.
pub const SR_CARRY: u16 = 0x0001;
pub const SR_OVERFLOW: u16 = 0x0002;
pub const SR_ZERO: u16 = 0x0004;
pub const SR_NEGATIVE: u16 = 0x0008;
pub const SR_EXTEND: u16 = 0x0010;
pub const SR_INT_MASK: u16 = 0x0700;
pub const SR_SUPERVISOR: u16 = 0x2000;
pub const SR_TRACE: u16 = 0x8000;

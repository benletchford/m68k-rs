//! Floating-point emulation for the 68881/68882 and the on-chip
//! 68040/68060 FPUs.
//!
//! The public register value is [`FloatX80`]. Instruction execution uses a
//! pure-Rust 80-bit extended-precision engine, including FPCR rounding modes,
//! exception accumulation, packed-decimal conversion, and transcendental
//! operations. CPU-specific instruction availability and exception behavior
//! are enforced by [`CpuCore`](crate::CpuCore).

mod dd;
mod exceptions_040;
mod operations;
mod packed;
mod softfloat;
mod state_040;
mod transcendental;
mod two_over_pi;
mod types;

pub use state_040::{Fpu040Frame, Fpu040State};
pub use types::*;

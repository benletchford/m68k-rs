//! # Core
//!
//! Core M68000 family CPU emulation engine.

pub mod addressing;
pub mod cpu;
pub mod decode;
pub mod ea;
pub mod exceptions;
pub mod execute;
pub mod instructions;
pub mod interrupts;
pub(crate) mod mem_ops;
pub mod memory;
pub mod op_cache;
pub mod registers;
pub mod status;
pub mod timing;
pub mod trace_jit;
pub mod types;

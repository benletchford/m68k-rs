//! Benchmark workloads: hand-encoded 68000 machine code.
//!
//! Every workload is a repeating unit ("iteration") of a fixed number of
//! instructions, so instruction throughput can be computed exactly. Where
//! possible an iteration increments a data register, giving an exact
//! per-emulator retired-instruction count even if the two cores disagree on
//! cycle accounting.
//!
//! Run the harness with `--disasm` to see each workload disassembled by the
//! m68k-rs disassembler (a sanity check on the hand-encoded opcodes).

/// Guest memory size. Power of two so both cores wrap addresses the same way
/// (the 68000's 24-bit address bus).
pub const MEM_SIZE: usize = 0x0100_0000;
/// Where workload code is placed / where execution starts.
pub const CODE_BASE: u32 = 0x1000;
/// Initial supervisor stack pointer (only the call/return workload uses it).
pub const SSP: u32 = 0x0080_0000;

pub enum Code {
    /// Fill all of guest memory with one opcode; the PC runs linearly and
    /// wraps around the 24-bit address space.
    Fill(u16),
    /// Place these words at [`CODE_BASE`]; the rest of memory is zero.
    Words(&'static [u16]),
}

/// A data register that increments exactly once per iteration.
pub struct CountReg {
    pub reg: usize,
}

pub struct Workload {
    pub name: &'static str,
    pub code: Code,
    /// Instructions per iteration of the repeating unit.
    pub instrs_per_iter: u32,
    /// How many iterations a timed run executes.
    pub target_iters: u64,
    /// Initial data register values, applied before every run.
    pub d_init: &'static [(usize, u32)],
    /// Register-based exact iteration counter, when the workload has one.
    pub count_reg: Option<CountReg>,
}

impl Workload {
    pub fn target_instrs(&self) -> u64 {
        self.target_iters * self.instrs_per_iter as u64
    }
}

pub fn all() -> Vec<Workload> {
    vec![
        Workload {
            name: "linear NOP",
            code: Code::Fill(0x4E71), // NOP
            instrs_per_iter: 1,
            target_iters: 40_000_000,
            d_init: &[],
            count_reg: None,
        },
        Workload {
            name: "linear MOVEQ",
            code: Code::Fill(0x7001), // MOVEQ #1,D0
            instrs_per_iter: 1,
            target_iters: 40_000_000,
            d_init: &[],
            count_reg: None,
        },
        Workload {
            name: "linear ADDQ.L",
            code: Code::Fill(0x5280), // ADDQ.L #1,D0
            instrs_per_iter: 1,
            target_iters: 25_000_000,
            d_init: &[(0, 0)],
            count_reg: Some(CountReg { reg: 0 }),
        },
        Workload {
            name: "loop ADDQ/BRA",
            // ADDQ.L #1,D0 ; BRA.S -4
            code: Code::Words(&[0x5280, 0x60FC]),
            instrs_per_iter: 2,
            target_iters: 15_000_000,
            d_init: &[(0, 0)],
            count_reg: Some(CountReg { reg: 0 }),
        },
        Workload {
            name: "loop TST/BNE",
            // TST.L D0 ; BNE.S -4   (D0 = 3, so always taken)
            code: Code::Words(&[0x4A80, 0x66FC]),
            instrs_per_iter: 2,
            target_iters: 15_000_000,
            d_init: &[(0, 3)],
            count_reg: None,
        },
        Workload {
            name: "loop reg mix",
            // MOVE.L D0,D2 ; ADD.L D1,D2 ; ADDQ.L #1,D2 ; EOR.L D0,D2 ;
            // TST.L D2 ; BRA.S -12
            code: Code::Words(&[0x2400, 0xD481, 0x5282, 0xB182, 0x4A82, 0x60F4]),
            instrs_per_iter: 6,
            target_iters: 5_000_000,
            d_init: &[(0, 3), (1, 2)],
            count_reg: None,
        },
        Workload {
            name: "memcpy 4KB",
            // Copies 1024 longwords from $8000 to $10000, counting passes in D1:
            //   $1000: LEA $00008000.L,A0
            //   $1006: LEA $00010000.L,A1
            //   $100C: MOVE.W #1023,D0
            //   $1010: MOVE.L (A0)+,(A1)+
            //   $1012: DBRA D0,$1010
            //   $1016: ADDQ.W #1,D1
            //   $1018: BRA.S $1000
            // Per iteration: 3 setup + 1024 moves + 1024 dbra + addq + bra = 2053.
            code: Code::Words(&[
                0x41F9, 0x0000, 0x8000, // LEA $8000.L,A0
                0x43F9, 0x0001, 0x0000, // LEA $10000.L,A1
                0x303C, 0x03FF, // MOVE.W #$03FF,D0
                0x22D8, // MOVE.L (A0)+,(A1)+
                0x51C8, 0xFFFC, // DBRA D0,-4
                0x5241, // ADDQ.W #1,D1
                0x60E6, // BRA.S -26
            ]),
            instrs_per_iter: 2053,
            target_iters: 6_000,
            d_init: &[(1, 0)],
            count_reg: Some(CountReg { reg: 1 }),
        },
        Workload {
            name: "call/return",
            //   $1000: ADDQ.L #1,D0
            //   $1002: BSR.S $1008
            //   $1004: BRA.S $1000
            //   $1006: NOP        (padding, never executed)
            //   $1008: RTS
            code: Code::Words(&[0x5280, 0x6104, 0x60FA, 0x4E71, 0x4E75]),
            instrs_per_iter: 4,
            target_iters: 5_000_000,
            d_init: &[(0, 0)],
            count_reg: Some(CountReg { reg: 0 }),
        },
    ]
}

/// Build the initial guest memory image for a workload, including the reset
/// vectors (SSP at 0, initial PC at 4).
pub fn build_image(w: &Workload) -> Vec<u8> {
    let mut mem = vec![0u8; MEM_SIZE];
    match w.code {
        Code::Fill(op) => {
            let bytes = op.to_be_bytes();
            for chunk in mem.chunks_exact_mut(2) {
                chunk[0] = bytes[0];
                chunk[1] = bytes[1];
            }
        }
        Code::Words(words) => {
            for (i, word) in words.iter().enumerate() {
                let a = CODE_BASE as usize + i * 2;
                mem[a..a + 2].copy_from_slice(&word.to_be_bytes());
            }
        }
    }
    mem[0..4].copy_from_slice(&SSP.to_be_bytes());
    mem[4..8].copy_from_slice(&CODE_BASE.to_be_bytes());
    mem
}

/// For `Code::Fill` workloads: what the first 8 bytes (overwritten by the
/// reset vectors) must be restored to after reset, so wrapped execution runs
/// through clean opcodes.
pub fn post_reset_patch(w: &Workload) -> Option<[u8; 8]> {
    match w.code {
        Code::Fill(op) => {
            let b = op.to_be_bytes();
            Some([b[0], b[1], b[0], b[1], b[0], b[1], b[0], b[1]])
        }
        Code::Words(_) => None,
    }
}

pub fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x1_0000_01b3);
    }
    h
}

//! m68k-rs engine adapters: the cycle-accurate interpreter (`execute`) and
//! the instruction-budgeted fast path (`run_batch`, decoded-op cache +
//! trace JIT over a fastmem window).

use crate::engine::{Engine, EngineState};
use crate::workloads::{self, CODE_BASE, MEM_SIZE, SSP, Workload};
use m68k::{AddressBus, BatchExit, CpuCore, CpuType, LinearMemoryBus};

/// Flat mask-and-index bus with no fastmem window: the interpreter engine
/// uses this so `execute` pays one (inlined) bus call per access, matching
/// the shape of Musashi's memory interface.
struct FlatBus {
    memory: Vec<u8>,
}

const MEM_MASK: usize = MEM_SIZE - 1;

impl AddressBus for FlatBus {
    #[inline]
    fn read_byte(&mut self, address: u32) -> u8 {
        self.memory[address as usize & MEM_MASK]
    }

    #[inline]
    fn read_word(&mut self, address: u32) -> u16 {
        let a = address as usize & MEM_MASK;
        ((self.memory[a] as u16) << 8) | self.memory[(a + 1) & MEM_MASK] as u16
    }

    #[inline]
    fn read_long(&mut self, address: u32) -> u32 {
        let a = address as usize & MEM_MASK;
        ((self.memory[a] as u32) << 24)
            | ((self.memory[(a + 1) & MEM_MASK] as u32) << 16)
            | ((self.memory[(a + 2) & MEM_MASK] as u32) << 8)
            | self.memory[(a + 3) & MEM_MASK] as u32
    }

    #[inline]
    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[address as usize & MEM_MASK] = value;
    }

    #[inline]
    fn write_word(&mut self, address: u32, value: u16) {
        let a = address as usize & MEM_MASK;
        self.memory[a] = (value >> 8) as u8;
        self.memory[(a + 1) & MEM_MASK] = value as u8;
    }

    #[inline]
    fn write_long(&mut self, address: u32, value: u32) {
        let a = address as usize & MEM_MASK;
        self.memory[a] = (value >> 24) as u8;
        self.memory[(a + 1) & MEM_MASK] = (value >> 16) as u8;
        self.memory[(a + 2) & MEM_MASK] = (value >> 8) as u8;
        self.memory[(a + 3) & MEM_MASK] = value as u8;
    }
}

fn fresh_cpu(w: &Workload) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.set_sr(0x2700);
    cpu.pc = CODE_BASE;
    cpu.set_a(7, SSP);
    for &(reg, val) in w.d_init {
        cpu.set_d(reg, val);
    }
    cpu
}

/// m68k-rs cycle-accurate interpreter on a plain bus.
pub struct NativeInterp {
    cpu: CpuCore,
    bus: FlatBus,
}

impl NativeInterp {
    pub fn new() -> Self {
        Self {
            cpu: CpuCore::new(),
            bus: FlatBus {
                memory: vec![0; MEM_SIZE],
            },
        }
    }
}

impl Engine for NativeInterp {
    fn name(&self) -> &'static str {
        "m68k-rs execute"
    }

    fn load_workload(&mut self, w: &Workload) {
        self.bus.memory = workloads::build_image(w);
        if let Some(patch) = workloads::post_reset_patch(w) {
            self.bus.memory[0..8].copy_from_slice(&patch);
        }
        self.cpu = fresh_cpu(w);
    }

    fn reset_run(&mut self, w: &Workload) {
        self.cpu = fresh_cpu(w);
    }

    fn run_cycles(&mut self, budget: i64) -> i64 {
        self.cpu.execute(
            &mut self.bus,
            i32::try_from(budget).expect("cycle budget exceeds i32"),
        ) as i64
    }

    fn run_instructions(&mut self, _budget: u64) -> u64 {
        unreachable!("interpreter engine is cycle-budgeted")
    }

    fn is_cycle_budgeted(&self) -> bool {
        true
    }

    fn state(&mut self) -> EngineState {
        EngineState {
            d: std::array::from_fn(|i| self.cpu.d(i)),
            mem_hash: workloads::fnv1a(&self.bus.memory),
        }
    }
}

/// m68k-rs `run_batch` fast path on a `LinearMemoryBus` (exposes a fastmem
/// window, enabling the decoded-op cache and trace JIT).
pub struct NativeBatch {
    cpu: CpuCore,
    bus: LinearMemoryBus,
}

impl NativeBatch {
    pub fn new() -> Self {
        Self {
            cpu: CpuCore::new(),
            bus: LinearMemoryBus::new(MEM_SIZE),
        }
    }
}

impl Engine for NativeBatch {
    fn name(&self) -> &'static str {
        "m68k-rs run_batch"
    }

    fn load_workload(&mut self, w: &Workload) {
        let mut image = workloads::build_image(w);
        if let Some(patch) = workloads::post_reset_patch(w) {
            image[0..8].copy_from_slice(&patch);
        }
        self.bus = LinearMemoryBus::from_vec(image);
        self.cpu = fresh_cpu(w);
    }

    fn reset_run(&mut self, w: &Workload) {
        self.cpu = fresh_cpu(w);
    }

    fn run_cycles(&mut self, _budget: i64) -> i64 {
        unreachable!("batch engine is instruction-budgeted")
    }

    fn run_instructions(&mut self, budget: u64) -> u64 {
        let budget = u32::try_from(budget).expect("instruction budget exceeds u32");
        let result = self.cpu.run_batch(&mut self.bus, budget, &[]);
        assert_eq!(
            result.exit,
            BatchExit::BudgetExhausted,
            "batch run exited early"
        );
        result.instructions as u64
    }

    fn is_cycle_budgeted(&self) -> bool {
        false
    }

    fn state(&mut self) -> EngineState {
        EngineState {
            d: std::array::from_fn(|i| self.cpu.d(i)),
            mem_hash: workloads::fnv1a(self.bus.as_slice()),
        }
    }
}

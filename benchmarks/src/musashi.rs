//! FFI bindings and engine wrapper for the vendored Musashi core.

use crate::engine::{Engine, EngineState};
use crate::workloads::{self, CODE_BASE, MEM_SIZE, SSP, Workload};
use std::ffi::{c_int, c_uint, c_void};

const M68K_CPU_TYPE_68000: c_uint = 1;

// m68k_register_t values (see vendor/musashi/m68k.h).
const REG_D0: c_uint = 0;
const REG_PC: c_uint = 16;
const REG_SR: c_uint = 17;
const REG_SP: c_uint = 18;

unsafe extern "C" {
    fn m68k_init();
    fn m68k_set_cpu_type(cpu_type: c_uint);
    fn m68k_pulse_reset();
    fn m68k_execute(num_cycles: c_int) -> c_int;
    fn m68k_set_reg(reg: c_uint, value: c_uint);
    fn m68k_get_reg(context: *mut c_void, reg: c_uint) -> c_uint;
    fn musashi_mem_ptr() -> *mut u8;
}

fn mem() -> &'static mut [u8] {
    // Safety: single-threaded harness; the shim owns a static 16MB buffer.
    unsafe { std::slice::from_raw_parts_mut(musashi_mem_ptr(), MEM_SIZE) }
}

/// Musashi has global (single-instance) CPU state, so this engine is a
/// zero-sized wrapper around it. The harness runs engines sequentially.
pub struct Musashi;

impl Musashi {
    pub fn new() -> Self {
        unsafe {
            m68k_init();
            m68k_set_cpu_type(M68K_CPU_TYPE_68000);
        }
        Musashi
    }

    fn reset_regs(&mut self, w: &Workload) {
        unsafe {
            m68k_set_reg(REG_SR, 0x2700);
            m68k_set_reg(REG_SP, SSP);
            m68k_set_reg(REG_PC, CODE_BASE);
            for i in 0..8 {
                m68k_set_reg(REG_D0 + i, 0);
            }
            for &(reg, val) in w.d_init {
                m68k_set_reg(REG_D0 + reg as c_uint, val);
            }
        }
    }
}

impl Engine for Musashi {
    fn name(&self) -> &'static str {
        "Musashi"
    }

    fn load_workload(&mut self, w: &Workload) {
        let image = workloads::build_image(w);
        mem().copy_from_slice(&image);
        unsafe {
            // Reset once with valid vectors in place; this also charges
            // Musashi's pending reset cycles, which the execute(1) call below
            // absorbs (it retires no instructions).
            m68k_pulse_reset();
            let _ = m68k_execute(1);
        }
        if let Some(patch) = workloads::post_reset_patch(w) {
            mem()[0..8].copy_from_slice(&patch);
        }
        self.reset_regs(w);
    }

    fn reset_run(&mut self, w: &Workload) {
        self.reset_regs(w);
    }

    fn run_cycles(&mut self, budget: i64) -> i64 {
        unsafe { m68k_execute(c_int::try_from(budget).expect("cycle budget exceeds i32")) as i64 }
    }

    fn run_instructions(&mut self, _budget: u64) -> u64 {
        unreachable!("Musashi is cycle-budgeted")
    }

    fn is_cycle_budgeted(&self) -> bool {
        true
    }

    fn state(&mut self) -> EngineState {
        let mut d = [0u32; 8];
        for (i, slot) in d.iter_mut().enumerate() {
            *slot = unsafe { m68k_get_reg(std::ptr::null_mut(), REG_D0 + i as c_uint) };
        }
        EngineState {
            d,
            mem_hash: workloads::fnv1a(mem()),
        }
    }
}

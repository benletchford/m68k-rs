//! Executable storage for bounded batches of foreground compilations.
//!
//! Cranelift's JITModule deliberately does not free machine code on Drop.
//! Keep an explicit lease beside every callable entry, then free a retired
//! module when the last entry disappears. IR-only worker snapshots do not
//! contain executable ownership and can outlive the foreground module.
use super::*;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;

// Keep each compilation independently reclaimable. A trace body and its
// checked entry remain together; unrelated traces cannot pin their storage.
// Count attempts, not successes: failed compilation also leaves declarations.
const COMPILES_PER_MODULE: usize = 1;

#[derive(Default)]
pub(super) struct CodeLease {
    retired: RefCell<Option<JITModule>>,
}

impl Drop for CodeLease {
    fn drop(&mut self) {
        if let Some(module) = self.retired.get_mut().take() {
            // SAFETY: all foreground callable entries own this lease. Execution
            // borrows its trace/region, so the final lease cannot drop during a
            // call. Worker snapshots contain only IR; their compiled regions
            // use independent modules and reject unresolved foreground calls.
            unsafe { module.free_memory() };
        }
    }
}

pub(super) struct CodeModule {
    module: Option<JITModule>,
    lease: Rc<CodeLease>,
    attempts: usize,
}

impl CodeModule {
    pub(super) fn new() -> Option<Self> {
        Some(Self {
            module: Some(JITModule::new(
                JITBuilder::new(default_libcall_names()).ok()?,
            )),
            lease: Rc::new(CodeLease::default()),
            attempts: 0,
        })
    }

    pub(super) fn lease(&self) -> Rc<CodeLease> {
        Rc::clone(&self.lease)
    }
}

impl Deref for CodeModule {
    type Target = JITModule;
    fn deref(&self) -> &JITModule {
        self.module.as_ref().unwrap()
    }
}

impl DerefMut for CodeModule {
    fn deref_mut(&mut self) -> &mut JITModule {
        self.module.as_mut().unwrap()
    }
}

impl Drop for CodeModule {
    fn drop(&mut self) {
        // Retire the mutable owner before dropping its lease. Outstanding
        // traces may keep this allocation alive, but cannot mutate the module.
        *self.lease.retired.borrow_mut() = self.module.take();
    }
}

impl TraceJit {
    pub(super) fn prepare_code_module(&mut self) -> Option<()> {
        if self.module.as_ref()?.attempts >= COMPILES_PER_MODULE {
            // If construction fails, preserve existing callable entries while
            // declining further compilation until a later attempt can retry.
            self.module = Some(CodeModule::new()?);
        }
        self.module.as_mut()?.attempts += 1;
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::memory::LinearMemoryBus;

    fn compile(jit: &mut TraceJit, value: u16) -> CompiledTrace {
        let mut bus = LinearMemoryBus::new(0x1000);
        for (pc, word) in [(0x100, 0x7000 | value), (0x102, 0x5281), (0x104, 0x60fa)] {
            bus.write_word_at(pc, word);
        }
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        let ops = [0x100, 0x102, 0x104]
            .into_iter()
            .map(|pc| decode_trace_op(&cpu, &mut bus, pc, CpuType::M68040).unwrap())
            .collect();
        jit.compile_decoded_ops(&cpu, 0x100, CpuType::M68040, ops, Some(0x100))
            .unwrap()
    }

    #[test]
    fn retired_batch_stays_callable_until_its_last_trace_drops() {
        let mut jit = TraceJit::new_base();
        let old = compile(&mut jit, 7);
        let old_lease = Rc::downgrade(&old._code_lease);
        for _ in 0..COMPILES_PER_MODULE {
            drop(compile(&mut jit, 9));
        }
        assert!(old._code_lease.retired.borrow().is_some());
        assert!(!Rc::ptr_eq(
            &old._code_lease,
            &jit.module.as_ref().unwrap().lease
        ));
        // Even dropping the JIT must not invalidate a separately owned trace.
        drop(jit);
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(CpuType::M68040);
        unsafe { old.call_native_raw(&mut cpu, 3) };
        assert_eq!(cpu.d(0), 7);
        assert_eq!(cpu.d(1), 3);
        drop(old);
        assert!(old_lease.upgrade().is_none());
    }

    #[test]
    fn failed_attempts_also_retire_declarations() {
        let mut jit = TraceJit::new_base();
        let old_lease = Rc::downgrade(&jit.module.as_ref().unwrap().lease);
        for index in 0..COMPILES_PER_MODULE {
            jit.prepare_code_module().unwrap();
            let module = jit.module.as_mut().unwrap();
            let signature = module.make_signature();
            // Model a backend refusal after declaration, before publication.
            module
                .declare_function(&format!("refused_{index}"), Linkage::Local, &signature)
                .unwrap();
        }
        jit.prepare_code_module().unwrap();
        assert!(old_lease.upgrade().is_none());
        assert_eq!(
            jit.module
                .as_ref()
                .unwrap()
                .declarations()
                .get_functions()
                .count(),
            0
        );
    }

    #[test]
    fn unwinding_frees_unpublished_module() {
        let mut weak = None;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let module = CodeModule::new().unwrap();
            weak = Some(Rc::downgrade(&module.lease));
            panic!("simulated compiler unwind");
        }));
        assert!(result.is_err());
        assert!(weak.unwrap().upgrade().is_none());
    }
}

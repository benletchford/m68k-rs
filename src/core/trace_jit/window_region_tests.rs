//! Multiple physical direct-memory regions, exercised through public batching.
use crate::core::cpu::CpuCore;
use crate::core::memory::{AddressBus, FastMem, FastMemRegion, MAX_FAST_MEM_REGIONS};
use crate::core::types::CpuType;

#[derive(Clone)]
struct Bus {
    primary: Vec<u8>,
    ram: Vec<u8>,
    rom: Vec<u8>,
    direct: bool,
    primary_enabled: bool,
    extra_enabled: bool,
    reads: usize,
    writes: usize,
}
const PRIMARY: u32 = 0x100000;
const RAM: u32 = 0x200000;
const ROM: u32 = 0x800000;
impl Bus {
    fn new(words: &[u16]) -> Self {
        let mut b = Self {
            primary: vec![0; 1024],
            ram: vec![0; 1024],
            rom: vec![0; 1024],
            direct: true,
            primary_enabled: true,
            extra_enabled: true,
            reads: 0,
            writes: 0,
        };
        for (chunk, word) in b.rom.chunks_exact_mut(2).zip(words) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        b
    }
}
impl AddressBus for Bus {
    fn read_byte(&mut self, a: u32) -> u8 {
        self.reads += 1;
        for (base, data) in [(PRIMARY, &self.primary), (RAM, &self.ram), (ROM, &self.rom)] {
            if let Some(v) = data.get(a.wrapping_sub(base) as usize) {
                return *v;
            }
        }
        0xa5
    }
    fn read_word(&mut self, a: u32) -> u16 {
        u16::from_be_bytes([self.read_byte(a), self.read_byte(a.wrapping_add(1))])
    }
    fn read_long(&mut self, a: u32) -> u32 {
        (u32::from(self.read_word(a)) << 16) | u32::from(self.read_word(a.wrapping_add(2)))
    }
    fn write_byte(&mut self, a: u32, v: u8) {
        self.writes += 1;
        for (base, data) in [(PRIMARY, &mut self.primary), (RAM, &mut self.ram)] {
            if let Some(byte) = data.get_mut(a.wrapping_sub(base) as usize) {
                *byte = v;
                return;
            }
        }
    }
    fn write_word(&mut self, a: u32, v: u16) {
        for (i, b) in v.to_be_bytes().into_iter().enumerate() {
            self.write_byte(a.wrapping_add(i as u32), b);
        }
    }
    fn write_long(&mut self, a: u32, v: u32) {
        for (i, b) in v.to_be_bytes().into_iter().enumerate() {
            self.write_byte(a.wrapping_add(i as u32), b);
        }
    }
    fn fast_mem(&mut self) -> Option<FastMem> {
        (self.direct && self.primary_enabled).then_some(FastMem {
            ptr: self.primary.as_mut_ptr(),
            base: PRIMARY,
            len: self.primary.len() as u32,
        })
    }
    fn fast_mem_regions(&mut self) -> [Option<FastMemRegion>; MAX_FAST_MEM_REGIONS] {
        let mut out = [None; MAX_FAST_MEM_REGIONS];
        if self.direct && self.extra_enabled {
            out[0] = Some(FastMemRegion {
                ptr: self.rom.as_mut_ptr(),
                base: ROM,
                len: self.rom.len() as u32,
                writable: false,
            });
            out[1] = Some(FastMemRegion {
                ptr: self.ram.as_mut_ptr(),
                base: RAM,
                len: self.ram.len() as u32,
                writable: true,
            });
        }
        out
    }
}
fn cpu() -> CpuCore {
    let mut c = CpuCore::new();
    c.set_cpu_type(CpuType::M68040);
    c.set_sr(0x2700);
    c.pc = ROM;
    c
}
fn compare(c: &CpuCore, b: &Bus, expected: &CpuCore, memory: &Bus) {
    assert_eq!(c.pc, expected.pc);
    assert_eq!(c.dar, expected.dar);
    assert_eq!(c.get_sr(), expected.get_sr());
    assert_eq!(b.ram, memory.ram);
    assert_eq!(b.rom, memory.rom);
    assert_eq!(b.primary, memory.primary);
}
#[test]
fn rom_code_and_separate_data_banks_match_dispatch() {
    // move.l (a0),d0; addq.l #1,d0; move.l d0,(a1); bra.s loop
    for primary_enabled in [false, true] {
        let mut b = Bus::new(&[0x2010, 0x5280, 0x2280, 0x60f8]);
        b.primary_enabled = primary_enabled;
        let mut c = cpu();
        c.set_a(0, RAM);
        c.set_a(1, RAM);
        let mut reference = b.clone();
        reference.direct = false;
        let mut expected = cpu();
        expected.dar = c.dar;
        for _ in 0..16384 {
            expected.step(&mut reference);
        }
        assert_eq!(c.run_batch(&mut b, 16384, &[]).instructions, 16384);
        compare(&c, &b, &expected, &reference);
        assert!(
            b.reads < reference.reads / 4,
            "ROM and second-bank reads bypass bus"
        );
        assert_eq!(b.writes, 0, "second-bank writes use their own allocation");
    }
}
#[test]
fn readonly_stores_and_rmw_fall_back_including_promoted_rom() {
    for primary_enabled in [false, true] {
        for op in [0x2080, 0x5290] {
            // move.l d0,(a0); addq.l #1,(a0)
            let mut b = Bus::new(&[op, 0x60fc]);
            b.primary_enabled = primary_enabled;
            let mut c = cpu();
            c.set_a(0, ROM + 128);
            c.set_d(0, 0x12345678);
            let mut reference = b.clone();
            reference.direct = false;
            let mut expected = cpu();
            expected.dar = c.dar;
            for _ in 0..8192 {
                expected.step(&mut reference);
            }
            c.run_batch(&mut b, 8192, &[]);
            compare(&c, &b, &expected, &reference);
            assert_eq!(b.writes, reference.writes);
            assert!(b.writes > 0);
        }
    }
}
#[test]
fn crossing_region_end_uses_bus_without_partial_postincrement() {
    let mut b = Bus::new(&[0x22d8]); // move.l (a0)+,(a1)+
    let mut c = cpu();
    c.set_a(0, RAM + 1022);
    c.set_a(1, PRIMARY + 128);
    let mut reference = b.clone();
    reference.direct = false;
    let mut expected = cpu();
    expected.dar = c.dar;
    expected.step(&mut reference);
    c.run_batch(&mut b, 1, &[]);
    compare(&c, &b, &expected, &reference);
    assert_eq!(c.a(0), RAM + 1026);
}
#[test]
fn changed_code_and_replaced_backing_are_seen_on_next_batch() {
    let mut b = Bus::new(&[0x2010, 0x5280, 0x2280, 0x60f8]);
    let mut c = cpu();
    c.set_a(0, RAM);
    c.set_a(1, RAM);
    c.run_batch(&mut b, 16384, &[]);
    b.rom[2..4].copy_from_slice(&0x5480u16.to_be_bytes()); // addq.l #2,d0
    b.ram = vec![0; 1024];
    c.run_batch(&mut b, 4096, &[]);
    assert_eq!(u32::from_be_bytes(b.ram[..4].try_into().unwrap()), 2048);
    b.direct = false; // a watchpoint/debugger can withhold every region
    b.reads = 0;
    c.run_batch(&mut b, 4096, &[]);
    assert_eq!(u32::from_be_bytes(b.ram[..4].try_into().unwrap()), 4096);
    assert!(b.reads >= 4096);
}

#[test]
fn resident_mmu_resolves_extra_regions_and_enforces_both_permissions() {
    let mut b = Bus::new(&[0x4e71]);
    let mut c = cpu();
    for region in b.fast_mem_regions().into_iter().flatten() {
        c.capture_extra_window(region);
    }
    c.pmmu_enabled = true;
    c.mmu_tc = 0x8000;
    c.atc.insert(1, true, RAM, false, false);
    c.atc.insert(2, true, ROM, false, false);
    c.atc.insert(3, true, RAM, true, false);
    c.refresh_window_mmu();
    assert!(c.window_access_off(0x1000, 4, true, false).is_some());
    assert!(c.window_access_off(0x2000, 4, false, true).is_some());
    assert!(
        c.window_access_off(0x2000, 4, true, false).is_none(),
        "ROM permission survives ATC"
    );
    assert!(
        c.window_access_off(0x3000, 4, true, false).is_none(),
        "ATC write protection survives RAM mapping"
    );
    assert!(
        c.window_access_off(0x3ffe, 4, false, false).is_none(),
        "page crossing falls back"
    );
}

fn install_windows(c: &mut CpuCore, b: &mut Bus) {
    c.fm_region_count = 0;
    c.fm_readonly = false;
    c.fm_len = 0;
    if let Some(fm) = b.fast_mem() {
        c.fm_ptr = fm.ptr as usize;
        c.fm_base = fm.base;
        c.fm_len = fm.len;
    }
    for region in b.fast_mem_regions().into_iter().flatten() {
        c.capture_extra_window(region);
    }
    c.refresh_window_mmu();
}

/// Invoke the warmed native body itself so fallback cannot make a broken
/// generated guard look correct. Portable builds exercise the same contract.
fn compiled_once(c: &mut CpuCore, b: &mut Bus, pc: u32) -> u32 {
    install_windows(c, b);
    let retired = super::with_trace_jit(|jit| {
        let super::TraceSlot::Compiled(trace) = &jit.slots[super::trace_cache_index(pc)] else {
            panic!("the warm-up must compile the loop");
        };
        assert_eq!(trace.pc, pc);
        assert_eq!(trace.translated, c.fm_mmu.enabled());
        assert_eq!(trace.multi_window, c.fm_readonly || c.fm_region_count != 0);
        assert_eq!(trace.primary_readonly, c.fm_readonly);
        #[cfg(all(feature = "jit", not(target_family = "wasm")))]
        // SAFETY: live, disjoint windows remain installed for the call; the
        // unchanged code was compiled with this CPU and window mode.
        let packed = unsafe { trace.call_native_raw(c, 1) };
        #[cfg(any(not(feature = "jit"), target_family = "wasm"))]
        let packed = super::execute_portable_trace(
            c,
            &trace.ops,
            super::CodeSpans::caller(trace.code_start, trace.code_end),
        );
        super::trace_return_retired(packed)
    });
    c.fm_ptr = 0;
    c.fm_base = 0;
    c.fm_len = 0;
    c.fm_region_count = 0;
    c.fm_readonly = false;
    c.fm_mmu = Default::default();
    retired
}

#[test]
fn compiled_postincrement_guards_both_operands_before_commit() {
    // MOVE.L (A0)+,(A1)+; SUBQ.L #4,A0; SUBQ.L #4,A1; BRA.S loop
    for primary_enabled in [false, true] {
        let mut b = Bus::new(&[0x22d8, 0x5988, 0x5989, 0x60f8]);
        b.primary_enabled = primary_enabled;
        b.ram[..4].copy_from_slice(&0x12345678u32.to_be_bytes());
        let mut c = cpu();
        c.set_a(0, RAM);
        c.set_a(1, RAM + 128);
        c.run_batch(&mut b, 16384, &[]);
        assert_eq!(compiled_once(&mut c, &mut b, ROM), 4);
        // A source crossing the region, a destination crossing it, and a
        // read-only destination must each bail before either increment/store.
        for (source, destination) in [(RAM + 1022, RAM + 128), (RAM, RAM + 1022), (RAM, ROM + 128)]
        {
            c.pc = ROM;
            c.set_a(0, source);
            c.set_a(1, destination);
            let registers = c.dar;
            let status = c.get_sr();
            let memory = b.clone();
            assert_eq!(compiled_once(&mut c, &mut b, ROM), 0);
            assert_eq!(c.pc, ROM);
            assert_eq!(c.dar, registers);
            assert_eq!(c.get_sr(), status);
            assert_eq!(b.ram, memory.ram);
            assert_eq!(b.rom, memory.rom);
            assert_eq!(b.writes, memory.writes);
            // The ordinary public batch path must then replay exactly once.
            let mut reference = b.clone();
            reference.direct = false;
            let mut expected = cpu();
            expected.dar = c.dar;
            expected.set_sr(c.get_sr());
            for _ in 0..4 {
                expected.step(&mut reference);
            }
            assert_eq!(c.run_batch(&mut b, 4, &[]).instructions, 4);
            compare(&c, &b, &expected, &reference);
        }
    }
}

#[test]
fn compiled_mmu_guards_translation_and_region_permissions() {
    let mut b = Bus::new(&[0x22d8, 0x5988, 0x5989, 0x60f8]);
    b.ram[..4].copy_from_slice(&0x12345678u32.to_be_bytes());
    let mut c = cpu();
    c.pc = 0x8000;
    c.pmmu_enabled = true;
    c.mmu_tc = 0x8000; // 68040, 4 KB pages
    c.atc.insert(8, true, ROM, false, false);
    c.atc.insert(4, true, RAM, false, false);
    c.atc.insert(5, true, PRIMARY, false, false);
    c.set_a(0, 0x4000);
    c.set_a(1, 0x5000);
    c.run_batch(&mut b, 16384, &[]);
    assert_eq!(compiled_once(&mut c, &mut b, 0x8000), 4);
    assert_eq!(&b.primary[..4], &0x12345678u32.to_be_bytes());
    for mode in 0..4 {
        c.pc = 0x8000;
        c.set_a(0, if mode == 0 { 0x4ffe } else { 0x4000 });
        c.atc.insert(
            5,
            true,
            if mode == 1 { ROM } else { PRIMARY },
            mode == 2,
            false,
        );
        if mode == 3 {
            c.atc.flush_page(5);
        }
        let registers = c.dar;
        let memory = b.clone();
        assert_eq!(
            compiled_once(&mut c, &mut b, 0x8000),
            0,
            "page crossing, ROM write, ATC protection, or ATC miss: {mode}"
        );
        assert_eq!(c.dar, registers);
        assert_eq!(b.primary, memory.primary);
        assert_eq!(b.rom, memory.rom);
        assert_eq!(b.writes, memory.writes);
    }
}

#[test]
fn switching_between_single_and_multiple_windows_rerecords() {
    let words = [0x2010u16, 0x5280, 0x2080, 0x60f8];
    let mut b = Bus::new(&words);
    b.primary[..8].copy_from_slice(&b.rom[..8]);
    let mut c = cpu();
    for extra in [false, true, false] {
        b.extra_enabled = extra;
        c.pc = PRIMARY;
        c.set_a(0, PRIMARY + 128);
        c.run_batch(&mut b, 16384, &[]);
        assert_eq!(compiled_once(&mut c, &mut b, PRIMARY), 4);
    }
}

#[test]
fn switching_primary_write_permission_rerecords() {
    // move.l (a0),d0; addq.l #1,d0; move.l d0,(a1); bra.s loop, from ROM.
    // Without a writable FastMem the ROM region is promoted to a read-only
    // primary; stores to RAM must still resolve through the region table.
    let mut b = Bus::new(&[0x2010, 0x5280, 0x2280, 0x60f8]);
    let mut c = cpu();
    c.set_a(0, RAM);
    c.set_a(1, RAM);
    for primary_enabled in [false, true, false] {
        b.primary_enabled = primary_enabled;
        c.pc = ROM;
        c.run_batch(&mut b, 16384, &[]);
        let before = u32::from_be_bytes(b.ram[..4].try_into().unwrap());
        assert_eq!(compiled_once(&mut c, &mut b, ROM), 4);
        let after = u32::from_be_bytes(b.ram[..4].try_into().unwrap());
        assert_eq!(
            after,
            before.wrapping_add(1),
            "store reached the RAM region"
        );
    }
}

/// x86-64 traces validate their code in a generated entry. Code that lives
/// in an extra region must validate there too, not return to the host on
/// every entry, and a changed byte must still be caught before execution.
#[cfg(all(feature = "jit", target_arch = "x86_64"))]
#[test]
fn generated_validation_reads_code_from_extra_regions() {
    let mut b = Bus::new(&[0x2010, 0x5280, 0x2280, 0x60f8]);
    let mut c = cpu();
    c.set_a(0, RAM);
    c.set_a(1, RAM + 128);
    c.run_batch(&mut b, 16384, &[]);
    for changed in [false, true] {
        if changed {
            b.rom[2] ^= 0x01; // addq.l #1,d0 -> addq.l #1,d1
        }
        install_windows(&mut c, &mut b);
        c.pc = ROM;
        let registers = c.dar;
        let packed = super::with_trace_jit(|jit| {
            let super::TraceSlot::Compiled(trace) = &jit.slots[super::trace_cache_index(ROM)]
            else {
                panic!("the warm-up must compile the ROM loop");
            };
            assert!(trace.multi_window);
            // SAFETY: live, disjoint windows remain installed for the call.
            unsafe { trace.call_checked(&mut c, 1) }
        });
        if changed {
            assert!(super::trace_return_validation_needed(packed));
            assert_eq!(c.dar, registers, "validation precedes execution");
        } else {
            assert!(!super::trace_return_validation_needed(packed));
            assert_eq!(super::trace_return_retired(packed), 4);
        }
        c.fm_ptr = 0;
        c.fm_base = 0;
        c.fm_len = 0;
        c.fm_region_count = 0;
        c.fm_readonly = false;
    }
}

#[test]
fn invalid_and_aliased_regions_are_not_installed() {
    let mut b = Bus::new(&[0x4e71]);
    let mut c = cpu();
    install_windows(&mut c, &mut b);
    let count = c.fm_region_count;
    for region in [
        FastMemRegion {
            ptr: b.ram.as_mut_ptr(),
            base: 0x900000,
            len: 1024,
            writable: true,
        },
        FastMemRegion {
            ptr: b.primary.as_mut_ptr(),
            base: 0x900000,
            len: 1024,
            writable: true,
        },
        FastMemRegion {
            ptr: &mut c as *mut CpuCore as *mut u8,
            base: 0x900000,
            len: 4,
            writable: true,
        },
        FastMemRegion {
            ptr: b.ram.as_mut_ptr(),
            base: u32::MAX - 1,
            len: 4,
            writable: true,
        },
        FastMemRegion {
            ptr: std::ptr::null_mut(),
            base: 0x900000,
            len: 4,
            writable: true,
        },
    ] {
        c.capture_extra_window(region);
        assert_eq!(c.fm_region_count, count);
    }
}

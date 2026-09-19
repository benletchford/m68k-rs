#![cfg(all(feature = "jit", feature = "trace-profile"))]
use m68k::{AddressBus, CpuCore, CpuType, FastMem, TrackedMem};

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Begin(u32, u32),
    Write(u32, u32, u32),
    End(Option<u32>),
}
#[derive(Clone)]
struct Bus {
    ram: Vec<u8>,
    tracked: bool,
    fast: bool,
    events: Vec<Event>,
    protect: bool,
}
impl Bus {
    fn store(&mut self, a: u32, v: u32, n: u32) {
        if !self.fast {
            self.events.push(Event::Write(a, v, n));
        }
        if !self.protect {
            for i in 0..n {
                self.ram[(a + i) as usize] = (v >> ((n - 1 - i) * 8)) as u8;
            }
        }
    }
}
impl AddressBus for Bus {
    fn read_byte(&mut self, a: u32) -> u8 {
        self.ram[a as usize]
    }
    fn read_word(&mut self, a: u32) -> u16 {
        u16::from_be_bytes(self.ram[a as usize..a as usize + 2].try_into().unwrap())
    }
    fn read_long(&mut self, a: u32) -> u32 {
        u32::from_be_bytes(self.ram[a as usize..a as usize + 4].try_into().unwrap())
    }
    fn write_byte(&mut self, a: u32, v: u8) {
        self.store(a, v.into(), 1)
    }
    fn write_word(&mut self, a: u32, v: u16) {
        self.store(a, v.into(), 2)
    }
    fn write_long(&mut self, a: u32, v: u32) {
        self.store(a, v, 4)
    }
    fn begin_memory_copy(&mut self, a: u32, n: u32) -> bool {
        if self.fast {
            return false;
        }
        self.events.push(Event::Begin(a, n));
        true
    }
    fn end_memory_copy(&mut self, a: Option<u32>) {
        self.events.push(Event::End(a));
    }
    fn fast_mem(&mut self) -> Option<FastMem> {
        self.fast.then_some(FastMem {
            ptr: self.ram.as_mut_ptr(),
            base: 0,
            len: self.ram.len() as u32,
        })
    }
    fn tracked_mem(&mut self) -> Option<TrackedMem> {
        self.tracked.then_some(TrackedMem {
            ptr: self.ram.as_ptr(),
            base: 0,
            len: self.ram.len() as u32,
        })
    }
}
fn run_pair(opcode: u16, overlap: bool, protect: bool) {
    run_pair_with_batches(opcode, overlap, protect, &[(2048, true)]);
}

fn run_pair_with_batches(opcode: u16, overlap: bool, protect: bool, batches: &[(u32, bool)]) {
    run_pair_with_cpu(opcode, overlap, protect, batches, CpuType::M68040);
}

fn run_pair_with_cpu(
    opcode: u16,
    overlap: bool,
    protect: bool,
    batches: &[(u32, bool)],
    cpu_type: CpuType,
) {
    let mut bus = Bus {
        ram: vec![0; 0x10000],
        tracked: false,
        fast: false,
        events: Vec::new(),
        protect,
    };
    for i in 0..0x2000 {
        bus.ram[0x1000 + i] = (i * 19 + 3) as u8;
    }
    for (i, word) in [opcode, 0x51c8, 0xfffc].into_iter().enumerate() {
        bus.ram[0x100 + i * 2..0x102 + i * 2].copy_from_slice(&word.to_be_bytes());
    }
    let make_cpu = || {
        let mut cpu = CpuCore::new();
        cpu.set_cpu_type(cpu_type);
        cpu.pc = 0x100;
        cpu.set_d(0, 1023);
        cpu.set_d(1, 0x89abcdef);
        cpu.set_a(0, 0x1000);
        cpu.set_a(1, if overlap { 0x1001 } else { 0x5000 });
        cpu.set_ccr(0x10);
        cpu
    };
    let mut expected = make_cpu();
    let mut actual = make_cpu();
    let mut actual_bus = bus.clone();
    actual_bus.tracked = true;
    m68k::core::trace_profile::reset();
    let mut retired = 0;
    for &(budget, tracked) in batches {
        actual_bus.tracked = tracked;
        retired += actual
            .run_batch(&mut actual_bus, budget, &[0x106])
            .instructions;
    }
    let profile = m68k::core::trace_profile::snapshot();
    for _ in 0..2048 {
        expected.step(&mut bus);
    }
    assert_eq!(expected.pc, 0x106);
    assert_eq!(retired, 2048);
    assert_eq!(actual.pc, expected.pc);
    assert_eq!(actual.get_ccr(), expected.get_ccr());
    for r in 0..8 {
        assert_eq!(actual.d(r), expected.d(r));
        assert_eq!(actual.a(r), expected.a(r));
    }
    assert_eq!(
        actual_bus.ram, bus.ram,
        "tracked native writes differ from step"
    );
    assert_eq!(
        actual_bus.events, bus.events,
        "write/copy notifications differ from step"
    );
    if matches!(cpu_type, CpuType::M68000 | CpuType::M68010) {
        assert_eq!(profile.rows.iter().map(|r| r.jit_retired).sum::<u64>(), 0);
        return;
    }
    assert!(
        profile.rows.iter().map(|r| r.jit_retired).sum::<u64>() > 1000,
        "test must execute native instructions: {:?}",
        profile.rows
    );
}
#[test]
fn tracked_byte_copy_matches_interpreter() {
    run_pair(0x12d8, false, false);
}
#[test]
fn tracked_word_copy_matches_interpreter() {
    run_pair(0x32d8, false, false);
}
#[test]
fn tracked_long_copy_matches_interpreter() {
    run_pair(0x22d8, false, false);
}
#[test]
fn tracked_overlapping_copy_preserves_notifications() {
    run_pair(0x22d8, true, false);
}
#[test]
fn tracked_write_protection_is_preserved() {
    run_pair(0x22d8, false, true);
}
#[test]
fn tracked_register_store_has_no_copy_notification() {
    run_pair(0x22c1, false, false);
}
#[test]
fn tracked_rmw_has_no_copy_notification() {
    run_pair(0x5291, false, false);
}

#[test]
fn tracked_copy_respects_short_budgets_and_window_changes() {
    // Warm a native trace, stop inside its loop using short budgets, decline
    // the window, and then reuse the trace with a tracked window again.
    run_pair_with_batches(
        0x22d8,
        false,
        false,
        &[
            (1400, true),
            (1, true),
            (3, true),
            (7, true),
            (43, false),
            (594, true),
        ],
    );
}

#[test]
fn pre_020_retains_interpreter_store_sequencing() {
    for cpu_type in [CpuType::M68000, CpuType::M68010] {
        run_pair_with_cpu(0x22d8, false, false, &[(2048, true)], cpu_type);
    }
}

#[test]
fn compiled_stores_follow_changes_between_raw_and_tracked_windows() {
    let mut bus = Bus {
        ram: vec![0; 0x10000],
        tracked: false,
        fast: true,
        events: Vec::new(),
        protect: false,
    };
    for (i, word) in [0x2290u16, 0x51c8, 0xfffc].into_iter().enumerate() {
        bus.ram[0x100 + i * 2..0x102 + i * 2].copy_from_slice(&word.to_be_bytes());
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68040);
    cpu.set_a(0, 0x1000);
    cpu.set_a(1, 0x5000);
    for (round, tracked) in [false, true, false, true].into_iter().enumerate() {
        bus.fast = !tracked;
        bus.tracked = tracked;
        bus.events.clear();
        let value = 0x12345678u32 + round as u32;
        bus.ram[0x1000..0x1004].copy_from_slice(&value.to_be_bytes());
        cpu.pc = 0x100;
        cpu.set_d(0, 1023);
        m68k::core::trace_profile::reset();
        assert_eq!(cpu.run_batch(&mut bus, 2048, &[0x106]).instructions, 2048);
        assert_eq!(cpu.pc, 0x106);
        assert_eq!(bus.read_long(0x5000), value);
        let expected: Vec<_> = if tracked {
            (0..1024)
                .flat_map(|_| {
                    [
                        Event::Begin(0x1000, 4),
                        Event::Write(0x5000, value, 4),
                        Event::End(Some(0x5000)),
                    ]
                })
                .collect()
        } else {
            Vec::new()
        };
        assert_eq!(bus.events, expected, "wrong store mode in round {round}");
        assert!(
            m68k::core::trace_profile::snapshot()
                .rows
                .iter()
                .map(|r| r.jit_retired)
                .sum::<u64>()
                > 1000
        );
    }
}

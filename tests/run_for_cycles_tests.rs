use m68k::{AddressBus, CpuCore, CpuType, CycleBatchExit, LinearMemoryBus, StepResult};

fn cpu_at(cpu_type: CpuType, pc: u32) -> CpuCore {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.pc = pc;
    cpu.set_sr(0x2700);
    cpu.set_a(7, 0x8000);
    cpu
}

fn bus_with(words: &[(u32, u16)]) -> LinearMemoryBus {
    let mut bus = LinearMemoryBus::new(0x10000);
    for &(address, word) in words {
        bus.load(address, &word.to_be_bytes());
    }
    bus
}

fn assert_cpu_state_eq(left: &CpuCore, right: &CpuCore) {
    assert_eq!(left.pc, right.pc, "pc");
    assert_eq!(left.ppc, right.ppc, "ppc");
    assert_eq!(left.get_sr(), right.get_sr(), "sr");
    assert_eq!(left.stopped, right.stopped, "stopped");
    for register in 0..8 {
        assert_eq!(left.d(register), right.d(register), "D{register}");
        assert_eq!(left.a(register), right.a(register), "A{register}");
    }
}

#[test]
fn non_positive_budget_returns_without_fetching() {
    let mut bus = bus_with(&[(0x1000, 0x7001)]);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);

    for budget in [0, -1] {
        let result = cpu.run_for_cycles(&mut bus, budget);
        assert_eq!(result.cycles, 0);
        assert_eq!(result.instructions, 0);
        assert_eq!(result.exit, CycleBatchExit::BudgetExhausted);
        assert_eq!(cpu.pc, 0x1000);
    }
}

#[test]
fn budget_overshoot_stops_only_at_complete_instruction_boundaries() {
    let mut bus = bus_with(&[
        (0x1000, 0x4E71), // NOP: 4 cycles
        (0x1002, 0x4E71), // NOP: 4 cycles
        (0x1004, 0x4E71),
    ]);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);

    let result = cpu.run_for_cycles(&mut bus, 5);

    assert_eq!(result.cycles, 8);
    assert_eq!(result.instructions, 2);
    assert_eq!(result.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(cpu.pc, 0x1004);
}

#[test]
fn cycle_batch_matches_repeated_step_state_and_cycles() {
    let words = [
        (0x1000, 0x7003), // MOVEQ #3,D0
        (0x1002, 0x5280), // ADDQ.L #1,D0
        (0x1004, 0x4840), // SWAP D0
        (0x1006, 0x4E71), // NOP
        (0x1008, 0x4E71),
    ];
    let mut batch_bus = bus_with(&words);
    let mut step_bus = bus_with(&words);
    let mut batch_cpu = cpu_at(CpuType::M68020, 0x1000);
    let mut step_cpu = cpu_at(CpuType::M68020, 0x1000);

    let result = batch_cpu.run_for_cycles(&mut batch_bus, 9);

    let mut cycles = 0;
    let mut instructions = 0;
    while cycles < 9 {
        match step_cpu.step(&mut step_bus) {
            StepResult::Ok {
                cycles: step_cycles,
            } => {
                cycles += step_cycles;
                instructions += 1;
            }
            other => panic!("unexpected step result: {other:?}"),
        }
    }

    assert_eq!(result.cycles, cycles);
    assert_eq!(result.instructions, instructions);
    assert_eq!(result.exit, CycleBatchExit::BudgetExhausted);
    assert_cpu_state_eq(&batch_cpu, &step_cpu);
}

#[test]
fn surfaced_traps_match_step_state_and_are_not_counted() {
    let cases = [
        (
            CpuType::M68000,
            0xA123,
            CycleBatchExit::AlineTrap { opcode: 0xA123 },
        ),
        (
            CpuType::M68000,
            0xF123,
            CycleBatchExit::FlineTrap { opcode: 0xF123 },
        ),
        (
            CpuType::M68000,
            0x4E42,
            CycleBatchExit::TrapInstruction { trap_num: 2 },
        ),
        (
            CpuType::M68010,
            0x484B,
            CycleBatchExit::Breakpoint { bp_num: 3 },
        ),
        (
            CpuType::M68000,
            0x4AFC,
            CycleBatchExit::IllegalInstruction { opcode: 0x4AFC },
        ),
    ];

    for (cpu_type, opcode, expected_exit) in cases {
        let words = [(0x1000, 0x4E71), (0x1002, opcode)];
        let mut batch_bus = bus_with(&words);
        let mut step_bus = bus_with(&words);
        let mut batch_cpu = cpu_at(cpu_type, 0x1000);
        let mut step_cpu = cpu_at(cpu_type, 0x1000);

        assert!(matches!(
            step_cpu.step(&mut step_bus),
            StepResult::Ok { .. }
        ));
        let expected_step = step_cpu.step(&mut step_bus);
        let result = batch_cpu.run_for_cycles(&mut batch_bus, 1000);

        assert_eq!(result.cycles, 4, "{opcode:#06x}");
        assert_eq!(result.instructions, 1, "{opcode:#06x}");
        assert_eq!(result.exit, expected_exit, "{opcode:#06x}");
        assert_cpu_state_eq(&batch_cpu, &step_cpu);
        assert!(
            !matches!(expected_step, StepResult::Ok { .. } | StepResult::Stopped),
            "{opcode:#06x}: {expected_step:?}"
        );
    }
}

#[test]
fn stop_is_distinct_and_the_stop_instruction_is_counted() {
    let mut bus = bus_with(&[
        (0x1000, 0x4E72), // STOP #$2700
        (0x1002, 0x2700),
        (0x1004, 0x4E71),
    ]);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);

    let result = cpu.run_for_cycles(&mut bus, 100);

    assert_eq!(result.cycles, 4);
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::Stopped);
    assert_eq!(cpu.pc, 0x1004);
    assert!(cpu.is_stopped());

    let result = cpu.run_for_cycles(&mut bus, 100);
    assert_eq!(result.cycles, 0);
    assert_eq!(result.instructions, 0);
    assert_eq!(result.exit, CycleBatchExit::Stopped);
}

#[derive(Clone)]
struct EventBus {
    memory: Vec<u8>,
    overlay_memory: Vec<u8>,
    overlay_enabled: bool,
    resets: u32,
    boundary_write_address: Option<u32>,
    boundary_on_interrupt_acknowledge: bool,
    boundary_requested: bool,
}

impl EventBus {
    fn new() -> Self {
        Self {
            memory: vec![0; 0x10000],
            overlay_memory: vec![0; 0x10000],
            overlay_enabled: false,
            resets: 0,
            boundary_write_address: None,
            boundary_on_interrupt_acknowledge: false,
            boundary_requested: false,
        }
    }

    fn load_word(&mut self, address: u32, value: u16) {
        self.write_word(address, value);
    }

    fn load_long(&mut self, address: u32, value: u32) {
        self.write_long(address, value);
    }

    fn load_overlay_word(&mut self, address: u32, value: u16) {
        let [hi, lo] = value.to_be_bytes();
        self.overlay_memory[address as usize & 0xFFFF] = hi;
        self.overlay_memory[address.wrapping_add(1) as usize & 0xFFFF] = lo;
    }
}

impl AddressBus for EventBus {
    fn read_byte(&mut self, address: u32) -> u8 {
        let index = address as usize & 0xFFFF;
        if self.overlay_enabled && (0x1000..0x2000).contains(&address) {
            self.overlay_memory[index]
        } else {
            self.memory[index]
        }
    }

    fn read_word(&mut self, address: u32) -> u16 {
        u16::from_be_bytes([
            self.read_byte(address),
            self.read_byte(address.wrapping_add(1)),
        ])
    }

    fn read_long(&mut self, address: u32) -> u32 {
        u32::from_be_bytes([
            self.read_byte(address),
            self.read_byte(address.wrapping_add(1)),
            self.read_byte(address.wrapping_add(2)),
            self.read_byte(address.wrapping_add(3)),
        ])
    }

    fn write_byte(&mut self, address: u32, value: u8) {
        self.memory[address as usize & 0xFFFF] = value;
    }

    fn write_word(&mut self, address: u32, value: u16) {
        let [hi, lo] = value.to_be_bytes();
        self.write_byte(address, hi);
        self.write_byte(address.wrapping_add(1), lo);
        if self.boundary_write_address == Some(address) {
            self.boundary_requested = true;
        }
    }

    fn write_long(&mut self, address: u32, value: u32) {
        for (offset, byte) in value.to_be_bytes().into_iter().enumerate() {
            self.write_byte(address.wrapping_add(offset as u32), byte);
        }
    }

    fn interrupt_acknowledge(&mut self, _level: u8) -> u32 {
        if self.boundary_on_interrupt_acknowledge {
            self.boundary_requested = true;
        }
        u32::MAX
    }

    fn take_boundary_request(&mut self) -> bool {
        std::mem::take(&mut self.boundary_requested)
    }

    fn reset_devices(&mut self) {
        self.resets += 1;
    }
}

#[test]
fn bus_request_returns_after_the_current_instruction_and_before_the_next() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x3080); // MOVE.W D0,(A0)
    bus.load_word(0x1002, 0x7201); // Old mapping: MOVEQ #1,D1
    bus.load_word(0x1004, 0x4E71); // NOP
    bus.load_overlay_word(0x1002, 0x7202); // New mapping: MOVEQ #2,D1
    bus.load_overlay_word(0x1004, 0x4E71);
    bus.boundary_write_address = Some(0x4000);

    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_a(0, 0x4000);
    cpu.set_d(0, 0xABCD);

    // The boundary reason wins even though this instruction crosses the
    // one-cycle budget.
    let result = cpu.run_for_cycles(&mut bus, 1);

    assert_eq!(result.cycles, 8);
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(bus.read_word(0x4000), 0xABCD);
    assert_eq!(cpu.d(1), 0);
    assert_eq!(cpu.pc, 0x1002);

    // The 68000 prefetched the old opcode as part of the requesting
    // instruction. Apply the mapping change and discard that stale word
    // before resuming.
    bus.overlay_enabled = true;
    cpu.invalidate_prefetch();

    // The request was consumed by the boundary check, so resuming starts
    // with the instruction that follows the requesting write, fetched from
    // the new mapping.
    let resumed = cpu.run_for_cycles(&mut bus, 1);
    assert_eq!(resumed.cycles, 4);
    assert_eq!(resumed.instructions, 1);
    assert_eq!(resumed.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(cpu.d(1), 2);
    assert_eq!(cpu.pc, 0x1004);
}

#[test]
fn reset_completes_synchronously_and_counts_as_an_instruction() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x4E70);
    bus.load_word(0x1002, 0x4E71);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);

    let result = cpu.run_for_cycles(&mut bus, 1);

    assert_eq!(result.cycles, 132);
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(bus.resets, 1);
}

#[test]
fn internally_taken_exception_completes_and_counts_the_instruction() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x82C0); // DIVU.W D0,D1
    bus.load_long(5 * 4, 0x2000); // divide-by-zero vector
    bus.load_word(0x2000, 0x4E71);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_d(0, 0);
    cpu.set_d(1, 0x1234);

    let result = cpu.run_for_cycles(&mut bus, 1);

    assert!(result.cycles > 1);
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(cpu.pc, 0x2000);
}

#[test]
fn entry_interrupt_is_charged_without_counting_an_instruction() {
    let mut bus = EventBus::new();
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x4E71);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_irq(3);

    let result = cpu.run_for_cycles(&mut bus, 1);

    assert_eq!(result.cycles, 44);
    assert_eq!(result.instructions, 0);
    assert_eq!(result.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(cpu.pc, 0x2000);
}

#[test]
fn entry_interrupt_boundary_returns_before_the_handler_instruction() {
    let mut bus = EventBus::new();
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x7001); // MOVEQ #1,D0
    bus.boundary_on_interrupt_acknowledge = true;
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_irq(3);

    let result = cpu.run_for_cycles(&mut bus, 100);

    assert_eq!(result.cycles, 44);
    assert_eq!(result.instructions, 0);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.pc, 0x2000);
    assert_eq!(cpu.d(0), 0);

    // The request was consumed, so resuming starts with the first handler
    // instruction rather than returning the same boundary again.
    let resumed = cpu.run_for_cycles(&mut bus, 1);
    assert_eq!(resumed.cycles, 4);
    assert_eq!(resumed.instructions, 1);
    assert_eq!(resumed.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(cpu.pc, 0x2002);
    assert_eq!(cpu.d(0), 1);
}

#[test]
fn entry_interrupt_boundary_precedes_budget_exhaustion() {
    let mut bus = EventBus::new();
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x4E71);
    bus.boundary_on_interrupt_acknowledge = true;
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_irq(3);

    let result = cpu.run_for_cycles(&mut bus, 1);

    assert_eq!(result.cycles, 44);
    assert_eq!(result.instructions, 0);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.pc, 0x2000);
}

#[test]
fn cycle_batch_matches_step_after_fast_path_warmup() {
    let words = [(0x1000, 0x5280), (0x1002, 0x60FC)];
    let mut batch_bus = bus_with(&words);
    let mut step_bus = bus_with(&words);
    let mut batch_cpu = cpu_at(CpuType::M68020, 0x1000);
    let mut step_cpu = cpu_at(CpuType::M68020, 0x1000);

    let warmup = batch_cpu.run_batch(&mut batch_bus, 10_000, &[]);
    assert_eq!(warmup.instructions, 10_000);
    for _ in 0..10_000 {
        assert!(matches!(
            step_cpu.step(&mut step_bus),
            StepResult::Ok { .. }
        ));
    }
    assert_cpu_state_eq(&batch_cpu, &step_cpu);

    let result = batch_cpu.run_for_cycles(&mut batch_bus, 25);
    let mut cycles = 0;
    let mut instructions = 0;
    while cycles < 25 {
        match step_cpu.step(&mut step_bus) {
            StepResult::Ok {
                cycles: step_cycles,
            } => {
                cycles += step_cycles;
                instructions += 1;
            }
            other => panic!("unexpected step result: {other:?}"),
        }
    }

    assert_eq!(result.cycles, cycles);
    assert_eq!(result.instructions, instructions);
    assert_eq!(result.exit, CycleBatchExit::BudgetExhausted);
    assert_cpu_state_eq(&batch_cpu, &step_cpu);
}

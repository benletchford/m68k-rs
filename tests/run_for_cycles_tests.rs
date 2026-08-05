use m68k::{
    AddressBus, CpuCore, CpuType, CycleBatchControl, CycleBatchExit, CycleBoundaryEvent,
    LinearMemoryBus, StepResult,
};

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

#[test]
fn hook_return_precedes_budget_and_resume_starts_at_next_instruction() {
    let mut bus = bus_with(&[
        (0x1000, 0x4E71), // NOP
        (0x1002, 0x4E71), // NOP
        (0x1004, 0x4E71), // NOP
    ]);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    let mut observations = Vec::new();

    let result = cpu.run_for_cycles_with_hook(&mut bus, 1, |cpu, _bus, cycles| {
        observations.push((cpu.ppc, cpu.pc, cycles));
        CycleBatchControl::Return
    });

    assert_eq!(observations, vec![(0x1000, 0x1002, 4)]);
    assert_eq!(result.cycles, 4);
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.pc, 0x1002);

    let resumed = cpu.run_for_cycles(&mut bus, 1);
    assert_eq!(resumed.cycles, 4);
    assert_eq!(resumed.instructions, 1);
    assert_eq!(resumed.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(cpu.pc, 0x1004);
}

#[test]
fn always_continue_hook_matches_the_original_runner() {
    let words = [
        (0x1000, 0x5280), // ADDQ.L #1,D0
        (0x1002, 0x60FC), // BRA.S $1000
    ];
    let mut plain_bus = bus_with(&words);
    let mut hooked_bus = bus_with(&words);
    let mut plain_cpu = cpu_at(CpuType::M68000, 0x1000);
    let mut hooked_cpu = cpu_at(CpuType::M68000, 0x1000);
    let mut hook_calls = 0;

    let plain = plain_cpu.run_for_cycles(&mut plain_bus, 50);
    let hooked = hooked_cpu.run_for_cycles_with_hook(&mut hooked_bus, 50, |_, _, cycles| {
        assert!(cycles > 0);
        hook_calls += 1;
        CycleBatchControl::Continue
    });

    assert_eq!(hooked, plain);
    assert_eq!(hook_calls, hooked.instructions);
    assert_cpu_state_eq(&hooked_cpu, &plain_cpu);
}

#[test]
fn hook_reports_data_dependent_instruction_cycles_individually() {
    let mut bus = bus_with(&[
        (0x1000, 0xC0C1), // MULU.W D1,D0
        (0x1002, 0xC0C1), // MULU.W D1,D0
        (0x1004, 0x4E71),
    ]);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_d(0, 2);
    cpu.set_d(1, 1);
    let mut observed_cycles = Vec::new();

    let result = cpu.run_for_cycles_with_hook(&mut bus, 1_000, |cpu, _, cycles| {
        observed_cycles.push(cycles);
        if observed_cycles.len() == 1 {
            // 68000 MULU timing depends on the number of set bits in the source.
            cpu.set_d(1, 0xFFFF);
            CycleBatchControl::Continue
        } else {
            CycleBatchControl::Return
        }
    });

    assert_eq!(observed_cycles.len(), 2);
    assert!(observed_cycles[1] > observed_cycles[0]);
    assert_eq!(result.cycles, observed_cycles.iter().sum());
    assert_eq!(result.instructions, 2);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
}

#[test]
fn stop_retains_its_exit_without_invoking_the_hook() {
    let mut bus = bus_with(&[(0x1000, 0x4E72), (0x1002, 0x2700)]);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    let mut hook_calls = 0;

    let result = cpu.run_for_cycles_with_hook(&mut bus, 100, |_, _, _| {
        hook_calls += 1;
        CycleBatchControl::Return
    });

    assert_eq!(hook_calls, 0);
    assert_eq!(result.cycles, 4);
    assert_eq!(result.instructions, 1);
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
    record_word_reads: bool,
    word_reads: Vec<u32>,
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
            record_word_reads: false,
            word_reads: Vec::new(),
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

    fn start_recording_word_reads(&mut self) {
        self.word_reads.clear();
        self.record_word_reads = true;
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
        if self.record_word_reads {
            self.word_reads.push(address);
        }
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
fn hook_updates_bus_state_before_the_next_instruction() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x4E71); // NOP
    bus.load_word(0x1002, 0x1210); // MOVE.B (A0),D1
    bus.load_word(0x1004, 0x4E71); // NOP
    bus.write_byte(0x4000, 0x11);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_a(0, 0x4000);
    let mut observations = Vec::new();

    let result = cpu.run_for_cycles_with_hook(&mut bus, 100, |cpu, bus, cycles| {
        observations.push((cpu.ppc, cycles));
        if cpu.ppc == 0x1000 {
            bus.write_byte(0x4000, 0x7B);
            CycleBatchControl::Continue
        } else {
            CycleBatchControl::Return
        }
    });

    assert_eq!(observations, vec![(0x1000, 4), (0x1002, 8)]);
    assert_eq!(result.cycles, 12);
    assert_eq!(result.instructions, 2);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.d(1) & 0xFF, 0x7B);
    assert_eq!(cpu.pc, 0x1004);
}

#[test]
fn irq_raised_by_hook_is_taken_before_the_next_ordinary_instruction() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x4E71); // NOP
    bus.load_word(0x1002, 0x7201); // MOVEQ #1,D1 (must not execute first)
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x7007); // handler: MOVEQ #7,D0
    bus.load_word(0x2002, 0x4E71);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    let mut observations = Vec::new();

    let result = cpu.run_for_cycles_with_hook(&mut bus, 100, |cpu, _bus, cycles| {
        observations.push((cpu.ppc, cycles));
        if cpu.ppc == 0x1000 {
            cpu.set_irq(3);
            CycleBatchControl::Continue
        } else {
            CycleBatchControl::Return
        }
    });

    assert_eq!(observations, vec![(0x1000, 4), (0x2000, 4)]);
    assert_eq!(result.cycles, 52); // NOP + level-3 entry + handler MOVEQ
    assert_eq!(result.instructions, 2);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.d(0), 7);
    assert_eq!(cpu.d(1), 0);
    assert_eq!(cpu.pc, 0x2002);
}

#[test]
fn bus_boundary_precedes_an_irq_newly_raised_by_the_hook() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x3080); // MOVE.W D0,(A0)
    bus.load_word(0x1002, 0x4E71);
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x4E71);
    bus.boundary_write_address = Some(0x4000);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_a(0, 0x4000);

    let result = cpu.run_for_cycles_with_hook(&mut bus, 100, |cpu, _, _| {
        cpu.set_irq(3);
        CycleBatchControl::Continue
    });

    assert_eq!(result.cycles, 8);
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.pc, 0x1002);

    let resumed = cpu.run_for_cycles(&mut bus, 1);
    assert_eq!(resumed.cycles, 44);
    assert_eq!(resumed.instructions, 0);
    assert_eq!(resumed.exit, CycleBatchExit::BudgetExhausted);
    assert_eq!(cpu.pc, 0x2000);
}

#[test]
fn entry_interrupt_boundary_does_not_invoke_instruction_hook() {
    let mut bus = EventBus::new();
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x4E71);
    bus.boundary_on_interrupt_acknowledge = true;
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_irq(3);
    let mut hook_calls = 0;

    let result = cpu.run_for_cycles_with_hook(&mut bus, 100, |_, _, _| {
        hook_calls += 1;
        CycleBatchControl::Continue
    });

    assert_eq!(hook_calls, 0);
    assert_eq!(result.cycles, 44);
    assert_eq!(result.instructions, 0);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.pc, 0x2000);
}

#[test]
fn boundary_hook_reports_entry_interrupt_and_resumes_at_handler() {
    let mut bus = EventBus::new();
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x7001); // handler: MOVEQ #1,D0
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_irq(3);
    let mut observations = Vec::new();

    let result = cpu.run_for_cycles_with_boundary_hook(&mut bus, 100, |cpu, _, event| {
        observations.push((event, cpu.pc, cpu.d(0)));
        CycleBatchControl::Return
    });

    assert_eq!(
        observations,
        vec![(CycleBoundaryEvent::InterruptEntry { cycles: 44 }, 0x2000, 0,)]
    );
    assert_eq!(result.cycles, 44);
    assert_eq!(result.instructions, 0);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);

    let resumed = cpu.run_for_cycles(&mut bus, 1);
    assert_eq!(resumed.cycles, 4);
    assert_eq!(resumed.instructions, 1);
    assert_eq!(cpu.d(0), 1);
    assert_eq!(cpu.pc, 0x2002);
}

#[test]
fn hook_created_irq_reports_entry_before_the_handler_instruction() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x4E71); // NOP
    bus.load_word(0x1002, 0x7201); // MOVEQ #1,D1 (must not execute)
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x1210); // handler: MOVE.B (A0),D1
    bus.load_word(0x2002, 0x4E71);
    bus.write_byte(0x4000, 0x11);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_a(0, 0x4000);
    let mut observations = Vec::new();

    let result = cpu.run_for_cycles_with_boundary_hook(&mut bus, 100, |cpu, bus, event| {
        observations.push((event, cpu.ppc, cpu.pc));
        match event {
            CycleBoundaryEvent::Instruction { .. } if cpu.ppc == 0x1000 => {
                cpu.set_irq(3);
                CycleBatchControl::Continue
            }
            CycleBoundaryEvent::InterruptEntry { .. } => {
                bus.write_byte(0x4000, 0x7B);
                CycleBatchControl::Continue
            }
            CycleBoundaryEvent::Instruction { .. } => CycleBatchControl::Return,
        }
    });

    assert_eq!(
        observations,
        vec![
            (
                CycleBoundaryEvent::Instruction { cycles: 4 },
                0x1000,
                0x1002,
            ),
            (
                CycleBoundaryEvent::InterruptEntry { cycles: 44 },
                0x1000,
                0x2000,
            ),
            (
                CycleBoundaryEvent::Instruction { cycles: 8 },
                0x2000,
                0x2002,
            ),
        ]
    );
    assert_eq!(result.cycles, 56);
    assert_eq!(result.instructions, 2);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.d(1) & 0xFF, 0x7B);
    assert_eq!(cpu.pc, 0x2002);
}

#[test]
fn instruction_boundary_precedes_an_irq_unmasked_by_that_instruction() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x46FC); // MOVE.W #$2000,SR
    bus.load_word(0x1002, 0x2000); // lower interrupt mask from 3 to 0
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x4E71);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2300); // level 3 remains masked on batch entry
    cpu.set_irq(3);
    let mut observations = Vec::new();

    let result = cpu.run_for_cycles_with_boundary_hook(&mut bus, 100, |cpu, _, event| {
        observations.push((event, cpu.pc));
        match event {
            CycleBoundaryEvent::Instruction { .. } => CycleBatchControl::Continue,
            CycleBoundaryEvent::InterruptEntry { .. } => CycleBatchControl::Return,
        }
    });

    assert_eq!(observations.len(), 2);
    assert!(matches!(
        observations[0],
        (CycleBoundaryEvent::Instruction { .. }, 0x1004)
    ));
    assert_eq!(
        observations[1],
        (CycleBoundaryEvent::InterruptEntry { cycles: 44 }, 0x2000)
    );
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.pc, 0x2000);
}

#[test]
fn stop_that_unmasks_an_irq_reports_only_the_interrupt_boundary() {
    let mut bus = EventBus::new();
    bus.load_word(0x1000, 0x4E72); // STOP #$2000
    bus.load_word(0x1002, 0x2000);
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x4E71);
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2300); // level 3 remains masked on batch entry
    cpu.set_irq(3);
    let mut events = Vec::new();

    let result = cpu.run_for_cycles_with_boundary_hook(&mut bus, 100, |_, _, event| {
        events.push(event);
        CycleBatchControl::Return
    });

    assert_eq!(
        events,
        vec![CycleBoundaryEvent::InterruptEntry { cycles: 44 }]
    );
    assert_eq!(result.instructions, 1);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert!(!cpu.is_stopped());
    assert_eq!(cpu.pc, 0x2000);
}

#[test]
fn entry_bus_request_still_returns_after_the_boundary_event() {
    let mut bus = EventBus::new();
    bus.load_long(0x6C, 0x2000); // level-3 autovector
    bus.load_word(0x2000, 0x4E71);
    bus.boundary_on_interrupt_acknowledge = true;
    let mut cpu = cpu_at(CpuType::M68000, 0x1000);
    cpu.set_sr(0x2000); // supervisor, interrupt mask 0
    cpu.set_irq(3);
    let mut events = Vec::new();

    let result = cpu.run_for_cycles_with_boundary_hook(&mut bus, 100, |_, _, event| {
        events.push(event);
        CycleBatchControl::Continue
    });

    assert_eq!(
        events,
        vec![CycleBoundaryEvent::InterruptEntry { cycles: 44 }]
    );
    assert_eq!(result.cycles, 44);
    assert_eq!(result.instructions, 0);
    assert_eq!(result.exit, CycleBatchExit::BoundaryRequested);
    assert_eq!(cpu.pc, 0x2000);
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

#[test]
fn boundary_hook_decoded_subset_matches_step_for_each_supported_cpu_model() {
    let words = [
        (0x1000, 0x4E71), // NOP
        (0x1002, 0x7001), // MOVEQ #1,D0
        (0x1004, 0x4E71), // NOP
        (0x1006, 0x76FE), // MOVEQ #-2,D3
    ];

    for cpu_type in [
        CpuType::M68000,
        CpuType::M68010,
        CpuType::M68020,
        CpuType::M68030,
        CpuType::M68040,
    ] {
        let mut precise_bus = bus_with(&words);
        let mut decoded_bus = bus_with(&words);
        let mut precise_cpu = cpu_at(cpu_type, 0x1000);
        let mut decoded_cpu = cpu_at(cpu_type, 0x1000);
        let mut precise_hooks = 0;
        let mut decoded_hooks = 0;

        let precise = precise_cpu.run_for_cycles_with_hook(&mut precise_bus, 16, |_, _, _| {
            precise_hooks += 1;
            CycleBatchControl::Continue
        });
        let decoded =
            decoded_cpu.run_for_cycles_with_boundary_hook(&mut decoded_bus, 16, |_, _, event| {
                if matches!(event, CycleBoundaryEvent::Instruction { .. }) {
                    decoded_hooks += 1;
                }
                CycleBatchControl::Continue
            });

        assert_eq!(decoded, precise, "{cpu_type:?}");
        assert_eq!(decoded_hooks, precise_hooks, "{cpu_type:?}");
        assert_cpu_state_eq(&decoded_cpu, &precise_cpu);
    }
}

#[test]
fn boundary_hook_decoded_subset_falls_back_without_changing_fetch_order() {
    let mut initial_bus = EventBus::new();
    initial_bus.load_word(0x1000, 0x7001); // MOVEQ #1,D0
    initial_bus.load_word(0x1002, 0x5280); // ADDQ.L #1,D0: unsupported
    initial_bus.load_word(0x1004, 0x4E71); // NOP
    initial_bus.load_word(0x1006, 0x76FE); // MOVEQ #-2,D3
    initial_bus.start_recording_word_reads();

    let mut precise_bus = initial_bus.clone();
    let mut decoded_bus = initial_bus;
    let mut precise_cpu = cpu_at(CpuType::M68000, 0x1000);
    let mut decoded_cpu = cpu_at(CpuType::M68000, 0x1000);
    let mut precise_hooks = 0;
    let mut decoded_hooks = 0;

    let precise = precise_cpu.run_for_cycles_with_hook(&mut precise_bus, 20, |_, _, _| {
        precise_hooks += 1;
        CycleBatchControl::Continue
    });
    let decoded =
        decoded_cpu.run_for_cycles_with_boundary_hook(&mut decoded_bus, 20, |_, _, event| {
            if matches!(event, CycleBoundaryEvent::Instruction { .. }) {
                decoded_hooks += 1;
            }
            CycleBatchControl::Continue
        });

    assert_eq!(decoded, precise);
    assert_eq!(decoded_hooks, precise_hooks);
    assert_cpu_state_eq(&decoded_cpu, &precise_cpu);
    assert_eq!(decoded_bus.word_reads, precise_bus.word_reads);
}

#[test]
fn boundary_hook_decoded_subset_observes_hook_cpu_state_changes() {
    let words = [
        (0x1000, 0x4E71), // NOP
        (0x1002, 0x7001), // skipped by the hook's PC update
        (0x1004, 0x7202), // MOVEQ #2,D1
        (0x1006, 0x4E71), // M68040 fallback
    ];
    let mut precise_bus = bus_with(&words);
    let mut decoded_bus = bus_with(&words);
    let mut precise_cpu = cpu_at(CpuType::M68020, 0x1000);
    let mut decoded_cpu = cpu_at(CpuType::M68020, 0x1000);

    let update_after_first_instruction = |cpu: &mut CpuCore| {
        if cpu.ppc == 0x1000 {
            cpu.set_cpu_type(CpuType::M68040);
            cpu.set_sr(cpu.get_sr() | 1);
            cpu.pc = 0x1004;
            cpu.invalidate_prefetch();
        }
    };
    let precise = precise_cpu.run_for_cycles_with_hook(&mut precise_bus, 12, |cpu, _, _| {
        update_after_first_instruction(cpu);
        CycleBatchControl::Continue
    });

    let decoded =
        decoded_cpu.run_for_cycles_with_boundary_hook(&mut decoded_bus, 12, |cpu, _, event| {
            if matches!(event, CycleBoundaryEvent::Instruction { .. }) {
                update_after_first_instruction(cpu);
            }
            CycleBatchControl::Continue
        });

    assert_eq!(decoded, precise);
    assert_cpu_state_eq(&decoded_cpu, &precise_cpu);
    assert_eq!(decoded_cpu.d(1), 2);
}

#[test]
fn boundary_hook_decoded_subset_matches_step_when_hook_raises_irq() {
    for cpu_type in [CpuType::M68000, CpuType::M68020, CpuType::M68040] {
        let mut initial_bus = EventBus::new();
        initial_bus.load_word(0x1000, 0x7001); // MOVEQ #1,D0: fast-path target
        initial_bus.load_word(0x1002, 0x7201); // MOVEQ #1,D1: must not execute
        initial_bus.load_long(0x6C, 0x2000); // level-3 autovector
        initial_bus.load_word(0x2000, 0x4E71); // handler entry: must not execute
        initial_bus.boundary_on_interrupt_acknowledge = true;

        let mut precise_bus = initial_bus.clone();
        let mut decoded_bus = initial_bus;
        let mut precise_cpu = cpu_at(cpu_type, 0x1000);
        let mut decoded_cpu = cpu_at(cpu_type, 0x1000);
        precise_cpu.set_sr(0x2000); // supervisor, interrupt mask 0
        decoded_cpu.set_sr(0x2000);

        let mut precise_instruction_events = Vec::new();
        let precise =
            precise_cpu.run_for_cycles_with_hook(&mut precise_bus, 100, |cpu, _, cycles| {
                precise_instruction_events.push(CycleBoundaryEvent::Instruction { cycles });
                if cpu.ppc == 0x1000 {
                    cpu.set_irq(3);
                }
                CycleBatchControl::Continue
            });
        assert_eq!(precise_instruction_events.len(), 1, "{cpu_type:?}");
        let mut precise_events = precise_instruction_events;
        let instruction_cycles = match precise_events[0] {
            CycleBoundaryEvent::Instruction { cycles } => cycles,
            CycleBoundaryEvent::InterruptEntry { .. } => unreachable!(),
        };
        precise_events.push(CycleBoundaryEvent::InterruptEntry {
            cycles: precise.cycles - instruction_cycles,
        });

        let mut decoded_events = Vec::new();
        let decoded = decoded_cpu.run_for_cycles_with_boundary_hook(
            &mut decoded_bus,
            100,
            |cpu, _, event| {
                decoded_events.push(event);
                if matches!(event, CycleBoundaryEvent::Instruction { .. }) && cpu.ppc == 0x1000 {
                    cpu.set_irq(3);
                }
                CycleBatchControl::Continue
            },
        );

        assert_eq!(decoded, precise, "{cpu_type:?}");
        assert_eq!(decoded_events, precise_events, "{cpu_type:?}");
        assert_eq!(
            decoded_events,
            vec![
                CycleBoundaryEvent::Instruction {
                    cycles: instruction_cycles,
                },
                CycleBoundaryEvent::InterruptEntry {
                    cycles: precise.cycles - instruction_cycles,
                },
            ],
            "{cpu_type:?}"
        );
        assert_eq!(decoded.instructions, 1, "{cpu_type:?}");
        assert_eq!(
            decoded.exit,
            CycleBatchExit::BoundaryRequested,
            "{cpu_type:?}"
        );
        assert_eq!(decoded_cpu.pc, 0x2000, "{cpu_type:?}");
        assert_eq!(decoded_cpu.d(1), 0, "{cpu_type:?}");
        assert_cpu_state_eq(&decoded_cpu, &precise_cpu);
        assert_eq!(decoded_bus.memory, precise_bus.memory, "{cpu_type:?}");
        assert_eq!(
            &decoded_bus.memory[0x7F00..0x8000],
            &precise_bus.memory[0x7F00..0x8000],
            "{cpu_type:?} stack frame"
        );
    }
}

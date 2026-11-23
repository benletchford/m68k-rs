//! Electrical lock edges surround only the operand transfers, including the
//! last write on an unsuccessful 040 compare (M68000PRM, CAS/CAS2).
use m68k::core::memory::{BusFault, BusFaultKind};
use m68k::{AddressBus, CpuCore, CpuType, LinearMemoryBus};

const DATA: u32 = 0x600;

struct Bus {
    ram: LinearMemoryBus,
    lock: bool,
    edges: Vec<bool>,
    operands: Vec<(u32, bool)>,
    fault: bool,
}

impl Bus {
    fn cycle(&mut self, addr: u32, write: bool) {
        if (DATA..DATA + 16).contains(&addr) {
            assert!(self.lock);
            self.operands.push((addr, write));
        } else {
            assert!(
                !self.lock,
                "locked instruction fetch or exception access at {addr:x}"
            );
        }
    }
}

impl AddressBus for Bus {
    fn set_rmw_lock(&mut self, asserted: bool) {
        assert_ne!(self.lock, asserted);
        self.lock = asserted;
        self.edges.push(asserted);
    }
    fn read_byte(&mut self, addr: u32) -> u8 {
        self.cycle(addr, false);
        self.ram.read_byte(addr)
    }
    fn try_read_byte(&mut self, addr: u32) -> Result<u8, BusFault> {
        let value = self.read_byte(addr);
        if self.fault && addr == DATA {
            Err(BusFault {
                address: addr,
                kind: BusFaultKind::BusError,
            })
        } else {
            Ok(value)
        }
    }
    fn read_word(&mut self, addr: u32) -> u16 {
        self.cycle(addr, false);
        self.ram.read_word(addr)
    }
    fn read_long(&mut self, addr: u32) -> u32 {
        self.cycle(addr, false);
        self.ram.read_long(addr)
    }
    fn write_byte(&mut self, addr: u32, value: u8) {
        self.cycle(addr, true);
        self.ram.write_byte(addr, value);
    }
    fn write_word(&mut self, addr: u32, value: u16) {
        self.cycle(addr, true);
        self.ram.write_word(addr, value);
    }
    fn write_long(&mut self, addr: u32, value: u32) {
        self.cycle(addr, true);
        self.ram.write_long(addr, value);
    }
}

fn run(model: CpuType, words: &[u16], mismatch: bool, fault: bool) -> Bus {
    let mut bus = Bus {
        ram: LinearMemoryBus::new(0x1000),
        edges: vec![],
        operands: vec![],
        lock: false,
        fault,
    };
    bus.ram.write_long(0, 0x0f00);
    bus.ram.write_long(4, 0x200);
    bus.ram.write_long(8, 0x400);
    for (i, &word) in words.iter().enumerate() {
        bus.ram.write_word(0x200 + 2 * i as u32, word);
    }
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(model);
    cpu.reset(&mut bus);
    cpu.set_a(0, DATA);
    cpu.set_a(1, DATA + 8);
    cpu.set_d(0, u32::from(mismatch));
    cpu.step(&mut bus);
    assert!(!bus.lock);
    bus
}

#[test]
fn tas_skips_registers_and_releases_before_exception_entry() {
    for model in [CpuType::M68000, CpuType::M68010, CpuType::M68040] {
        assert!(run(model, &[0x4ac0], false, false).edges.is_empty());
        for fault in [false, true] {
            let b = run(model, &[0x4ad0], false, fault);
            assert_eq!(b.edges, [true, false]);
            let mut ops = vec![(DATA, false)];
            if !fault {
                ops.push((DATA, true));
            }
            assert_eq!(b.operands, ops);
        }
    }
}

#[test]
fn cas_and_cas2_hold_one_grant_through_the_final_operand() {
    for model in [CpuType::M68020, CpuType::M68040] {
        for pair in [false, true] {
            for mismatch in [false, true] {
                let ins: &[u16] = if pair {
                    &[0x0cfc, 0x8080, 0x90c1]
                } else {
                    &[0x0cd0, 0x0040]
                };
                let b = run(model, ins, mismatch, false);
                assert_eq!(b.edges, [true, false]);
                let mut ops = vec![(DATA, false)];
                if pair {
                    ops.push((DATA + 8, false));
                }
                if !mismatch || model == CpuType::M68040 {
                    ops.push((DATA, true));
                }
                if pair && !mismatch {
                    ops.push((DATA + 8, true));
                }
                assert_eq!(b.operands, ops);
            }
        }
    }
}

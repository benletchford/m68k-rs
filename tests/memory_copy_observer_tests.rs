use m68k::core::memory::{BusFault, BusFaultKind};
use m68k::{AddressBus, CpuCore, CpuType, StepResult};

struct Bus {
    bytes: Vec<u8>,
    tags: Vec<u8>,
    saved: Vec<u8>,
    copies: Vec<(u32, Option<u32>)>,
    fail_write: Option<u32>,
}

impl Bus {
    fn new() -> Self {
        Self {
            bytes: vec![0; 0x10000],
            tags: vec![0; 0x10000],
            saved: vec![],
            copies: vec![],
            fail_write: None,
        }
    }
}

impl AddressBus for Bus {
    fn read_byte(&mut self, a: u32) -> u8 {
        self.bytes[a as usize]
    }
    fn read_word(&mut self, a: u32) -> u16 {
        u16::from_be_bytes([self.read_byte(a), self.read_byte(a + 1)])
    }
    fn read_long(&mut self, a: u32) -> u32 {
        (u32::from(self.read_word(a)) << 16) | u32::from(self.read_word(a + 2))
    }
    fn write_byte(&mut self, a: u32, v: u8) {
        self.bytes[a as usize] = v;
        self.tags[a as usize] = 0;
    }
    fn write_word(&mut self, a: u32, v: u16) {
        self.write_byte(a, (v >> 8) as u8);
        self.write_byte(a + 1, v as u8);
    }
    fn write_long(&mut self, a: u32, v: u32) {
        self.write_word(a, (v >> 16) as u16);
        self.write_word(a + 2, v as u16);
    }
    fn try_write_word(&mut self, a: u32, v: u16) -> Result<(), BusFault> {
        if self.fail_write == Some(a) {
            Err(BusFault {
                kind: BusFaultKind::BusError,
                address: a,
            })
        } else {
            self.write_word(a, v);
            Ok(())
        }
    }
    fn begin_memory_copy(&mut self, source: u32, bytes: u32) -> bool {
        self.saved = self.tags[source as usize..(source + bytes) as usize].to_vec();
        self.copies.push((source, None));
        true
    }
    fn end_memory_copy(&mut self, dest: Option<u32>) {
        self.copies.last_mut().unwrap().1 = dest;
        if let Some(dest) = dest {
            self.tags[dest as usize..dest as usize + self.saved.len()].copy_from_slice(&self.saved);
        }
        self.saved.clear();
    }
}

fn run(cpu_type: CpuType, opcode: u16, dest: u32, batch: bool) -> (CpuCore, Bus) {
    let mut bus = Bus::new();
    bus.write_word(0x100, opcode);
    bus.write_word(0x102, 0x4e71);
    bus.write_word(0x104, 0x4e71);
    bus.bytes[0x200..0x204].copy_from_slice(&[1, 2, 3, 4]);
    bus.tags[0x200..0x204].copy_from_slice(&[11, 22, 33, 44]);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(cpu_type);
    cpu.pc = 0x100;
    cpu.set_a(0, 0x200);
    cpu.set_a(1, dest);
    if batch {
        assert_eq!(cpu.run_batch(&mut bus, 1, &[]).instructions, 1);
    } else {
        assert!(matches!(cpu.step(&mut bus), StepResult::Ok { .. }));
    }
    (cpu, bus)
}

#[test]
fn memory_moves_preserve_metadata_in_step_and_batch_without_changing_bytes() {
    for cpu in [CpuType::M68000, CpuType::M68020, CpuType::M68040] {
        for batch in [false, true] {
            for (op, size) in [(0x12d8, 1), (0x32d8, 2), (0x22d8, 4)] {
                let (cpu, bus) = run(cpu, op, 0x300, batch);
                assert_eq!(bus.copies, [(0x200, Some(0x300))]);
                assert_eq!(
                    &bus.bytes[0x300..0x300 + size],
                    &bus.bytes[0x200..0x200 + size]
                );
                assert_eq!(
                    &bus.tags[0x300..0x300 + size],
                    &bus.tags[0x200..0x200 + size]
                );
                assert_eq!(cpu.a(0), 0x200 + size as u32);
                assert_eq!(cpu.a(1), 0x300 + size as u32);
            }
        }
    }
}

#[test]
fn overlapping_move_snapshots_before_the_destination_write() {
    for cpu in [CpuType::M68000, CpuType::M68020] {
        let (_, bus) = run(cpu, 0x22d8, 0x202, true);
        assert_eq!(&bus.tags[0x202..0x206], &[11, 22, 33, 44]);
        assert_eq!(&bus.bytes[0x202..0x206], &[1, 2, 3, 4]);
    }
}

#[test]
fn register_stores_and_clear_do_not_restore_metadata() {
    for opcode in [0x2280, 0x4291] {
        // MOVE.L D0,(A1); CLR.L (A1)
        let (_, bus) = run(CpuType::M68020, opcode, 0x200, true);
        assert!(bus.copies.is_empty());
        assert_eq!(&bus.tags[0x200..0x204], &[0; 4]);
    }
}

#[test]
fn failed_destination_discards_the_snapshot() {
    let mut bus = Bus::new();
    bus.write_word(0x100, 0x3290); // MOVE.W (A0),(A1)
    bus.write_long(8, 0x400); // bus error handler
    bus.fail_write = Some(0x300);
    bus.tags[0x200] = 42;
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);
    cpu.pc = 0x100;
    cpu.set_a(7, 0x8000);
    cpu.set_a(0, 0x200);
    cpu.set_a(1, 0x300);
    let _ = cpu.step(&mut bus);
    assert_eq!(bus.copies, [(0x200, None)]);
    assert!(bus.saved.is_empty());
    assert_eq!(bus.tags[0x300], 0);
}

#[test]
fn destination_addressing_modes_report_the_written_range() {
    for cpu_type in [CpuType::M68000, CpuType::M68020] {
        for (opcode, extension, dest) in [
            (0x3290, vec![], 0x300),               // (A1)
            (0x3310, vec![], 0x2fe),               // -(A1)
            (0x3350, vec![0x0006], 0x306),         // d16(A1)
            (0x31d0, vec![0x0300], 0x300),         // absolute word
            (0x33d0, vec![0x0000, 0x0300], 0x300), // absolute long
        ] {
            let mut bus = Bus::new();
            bus.write_word(0x100, opcode);
            for (i, word) in extension.iter().enumerate() {
                bus.write_word(0x102 + i as u32 * 2, *word);
            }
            bus.write_word(0x102 + extension.len() as u32 * 2, 0x4e71);
            bus.tags[0x200] = 42;
            let mut cpu = CpuCore::new();
            cpu.set_cpu_type(cpu_type);
            cpu.pc = 0x100;
            cpu.set_a(0, 0x200);
            cpu.set_a(1, 0x300);
            assert!(matches!(cpu.step(&mut bus), StepResult::Ok { .. }));
            assert_eq!(
                bus.copies,
                [(0x200, Some(dest))],
                "{cpu_type:?} {opcode:04x}"
            );
            assert_eq!(bus.tags[dest as usize], 42);
        }
    }
}

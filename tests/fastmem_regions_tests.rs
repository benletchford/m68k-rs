//! Multiple fastmem regions and read-only ROM through `run_batch`, checked
//! against single-stepping the same program over the bus alone. Without the
//! `jit` feature this exercises the interpreter's window paths directly.
use m68k::{AddressBus, CpuCore, CpuType, FastMem, FastMemRegion, MAX_FAST_MEM_REGIONS};

const PRIMARY: u32 = 0x10_0000;
const RAM: u32 = 0x20_0000;
const ROM: u32 = 0x80_0000;
const LEN: usize = 0x1000;

/// Which windows the bus offers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    /// A primary RAM window plus read-only ROM and a second RAM bank.
    PrimaryRomRam,
    /// The RAM bank as the primary window, ROM as the only extra region.
    RamRom,
    /// No primary window: the read-only ROM region becomes the primary
    /// window and the RAM bank an extra region.
    RomRam,
}
const LAYOUTS: [Layout; 3] = [Layout::PrimaryRomRam, Layout::RamRom, Layout::RomRam];

#[derive(Clone)]
struct Bus {
    primary: Vec<u8>,
    ram: Vec<u8>,
    rom: Vec<u8>,
    /// Offer the windows at all (the reference run does not).
    direct: bool,
    layout: Layout,
    reads: usize,
    writes: usize,
}

impl Bus {
    fn new(code: &[u16]) -> Self {
        let mut rom = vec![0; LEN];
        for (chunk, word) in rom.chunks_exact_mut(2).zip(code) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        Self {
            primary: vec![0; LEN],
            ram: (0..LEN).map(|i| (i * 7 + 3) as u8).collect(),
            rom,
            direct: true,
            layout: Layout::PrimaryRomRam,
            reads: 0,
            writes: 0,
        }
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
        // ROM ignores writes.
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
        let (ptr, base) = match self.layout {
            _ if !self.direct => return None,
            Layout::PrimaryRomRam => (self.primary.as_mut_ptr(), PRIMARY),
            Layout::RamRom => (self.ram.as_mut_ptr(), RAM),
            Layout::RomRam => return None,
        };
        Some(FastMem {
            ptr,
            base,
            len: LEN as u32,
        })
    }
    fn fast_mem_regions(&mut self) -> [Option<FastMemRegion>; MAX_FAST_MEM_REGIONS] {
        let mut out = [None; MAX_FAST_MEM_REGIONS];
        if self.direct {
            out[0] = Some(FastMemRegion {
                ptr: self.rom.as_mut_ptr(),
                base: ROM,
                len: LEN as u32,
                writable: false,
            });
            if self.layout != Layout::RamRom {
                out[1] = Some(FastMemRegion {
                    ptr: self.ram.as_mut_ptr(),
                    base: RAM,
                    len: LEN as u32,
                    writable: true,
                });
            }
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

/// Run `steps` instructions of `code` (in ROM) with A0..A2 set to `a`,
/// through the windows and through the bus alone, and require the same
/// registers and memory. Returns both buses.
fn compare(code: &[u16], a: [u32; 3], layout: Layout, steps: u32) -> (Bus, Bus) {
    let mut bus = Bus::new(code);
    bus.layout = layout;
    let mut reference = bus.clone();
    reference.direct = false;
    let mut c = cpu();
    let mut expected = cpu();
    for (i, &value) in a.iter().enumerate() {
        c.set_a(i, value);
        expected.set_a(i, value);
    }
    for _ in 0..steps {
        expected.step(&mut reference);
    }
    let mut done = 0;
    while done < steps {
        done += c.run_batch(&mut bus, steps - done, &[]).instructions;
    }
    assert_eq!(c.pc, expected.pc);
    for i in 0..8 {
        assert_eq!(c.d(i), expected.d(i), "d{i}");
        assert_eq!(c.a(i), expected.a(i), "a{i}");
    }
    assert_eq!(c.get_sr(), expected.get_sr());
    assert!(bus.primary == reference.primary, "primary window differs");
    assert!(bus.ram == reference.ram, "RAM bank differs");
    assert!(bus.rom == reference.rom, "ROM changed");
    (bus, reference)
}

#[test]
fn rom_code_with_data_in_primary_and_second_bank_bypasses_the_bus() {
    // loop: move.l (a0)+,d0 ; add.l d0,(a1)+ ; move.w (a2),d1 ;
    //       addq.w #1,d1 ; move.w d1,(a2) ; dbra d3,loop ; bra.s start
    let code = [
        0x2018, 0xd199, 0x3212, 0x5241, 0x3481, 0x51cb, 0xfff4, 0x60f0,
    ];
    for layout in LAYOUTS {
        let a = if layout == Layout::PrimaryRomRam {
            [RAM, PRIMARY + 0x800, PRIMARY + 0xffe]
        } else {
            [RAM, RAM + 0x800, RAM + 0xffe]
        };
        let (bus, reference) = compare(&code, a, layout, 6 * 256);
        assert!(
            bus.reads * 8 < reference.reads,
            "{} of {} reads used the bus",
            bus.reads,
            reference.reads
        );
        assert_eq!(bus.writes, 0, "RAM writes go straight to their windows");
    }
}

#[test]
fn stores_and_read_modify_writes_to_rom_take_the_bus() {
    // move.l d0,(a0) ; addq.l #1,(a1) ; not.b (a2) ; bra.s start
    let code = [0x2080, 0x5291, 0x4612, 0x60f8];
    for layout in LAYOUTS {
        let (bus, reference) = compare(&code, [ROM + 0x100, ROM + 0x200, ROM + 0x301], layout, 400);
        assert_eq!(
            bus.writes, reference.writes,
            "every ROM store reaches the bus"
        );
        assert!(bus.writes > 0);
    }
}

#[test]
fn accesses_straddling_a_window_edge_take_the_bus() {
    // move.l (a0),d0 ; move.l d0,(a1) ; move.w (a2),d2 ; bra.s start
    let code = [0x2010, 0x2280, 0x3412, 0x60f8];
    let (bus, reference) = compare(
        &code,
        [RAM + LEN as u32 - 2, PRIMARY + LEN as u32 - 1, PRIMARY - 1],
        Layout::PrimaryRomRam,
        400,
    );
    assert_eq!(bus.writes, reference.writes);
}

# m68k-rs

A safe, pure Rust implementation of the Motorola 68000 family CPU emulator.

Strong for both low-level hardware-accurate emulation and high-level emulation (HLE).

[![Rust CI](https://github.com/benletchford/m68k-rs/actions/workflows/rust.yml/badge.svg)](https://github.com/benletchford/m68k-rs/actions/workflows/rust.yml)
[![Crates.io](https://img.shields.io/crates/v/m68k.svg)](https://crates.io/crates/m68k)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Features

- **Complete CPU family support**: M68000, M68010, M68020, M68030, M68040, and variants (EC/LC)
- **Zero dependencies**: Pure Rust with no external runtime dependencies
- **Safe Rust**: No unsafe code blocks
- **FPU emulation**: Full 68881/68882/68040 floating-point unit support
- **MMU emulation**: 68030/68040 PMMU with table walks and transparent translation
- **HLE-ready**: Built-in trap interception for High-Level Emulation
- **Extensively tested**: Validated against multiple industry-standard test suites

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
m68k = "0.1"
```

### Basic Usage

```rust
use m68k::{CpuCore, CpuType, AddressBus, StepResult};

// Implement your memory bus
struct MyBus { memory: Vec<u8> }

impl AddressBus for MyBus {
    fn read_byte(&mut self, addr: u32) -> u8 {
        self.memory.get(addr as usize).copied().unwrap_or(0)
    }
    fn write_byte(&mut self, addr: u32, val: u8) {
        if let Some(m) = self.memory.get_mut(addr as usize) { *m = val; }
    }
    fn read_word(&mut self, addr: u32) -> u16 {
        let hi = self.read_byte(addr) as u16;
        let lo = self.read_byte(addr + 1) as u16;
        (hi << 8) | lo
    }
    fn write_word(&mut self, addr: u32, val: u16) {
        self.write_byte(addr, (val >> 8) as u8);
        self.write_byte(addr + 1, val as u8);
    }
    fn read_long(&mut self, addr: u32) -> u32 {
        ((self.read_word(addr) as u32) << 16) | self.read_word(addr + 2) as u32
    }
    fn write_long(&mut self, addr: u32, val: u32) {
        self.write_word(addr, (val >> 16) as u16);
        self.write_word(addr + 2, val as u16);
    }
}

fn main() {
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68000);

    let mut bus = MyBus { memory: vec![0; 0x10000] };

    // Set up vectors: SSP at 0x1000, PC at 0x400
    bus.write_long(0, 0x1000);
    bus.write_long(4, 0x400);

    // Write a NOP instruction at 0x400
    bus.write_word(0x400, 0x4E71);

    cpu.reset(&mut bus);

    loop {
        match cpu.step(&mut bus) {
            StepResult::Ok { cycles } => println!("Executed: {} cycles", cycles),
            StepResult::Stopped => break,
        }
    }
}
```

### High-Level Emulation (HLE)

Intercept traps for OS emulation or debugger integration with CPU/bus access:

```rust
use m68k::{AddressBus, CpuCore, HleHandler};

struct MacToolbox;

impl HleHandler for MacToolbox {
    fn handle_aline(
        &mut self,
        cpu: &mut CpuCore,
        bus: &mut dyn AddressBus,
        opcode: u16,
    ) -> bool {
        println!("A-line trap: {:04X} at PC=0x{:08X}", opcode, cpu.pc);
        // ... implement HLE by reading/writing through `bus` ...
        true // Handled - don't take exception
    }
}

fn emulate(cpu: &mut CpuCore, bus: &mut impl AddressBus) {
    let mut hle = MacToolbox;
    let result = cpu.step_with_hle_handler(bus, &mut hle);
}
```

Other `HleHandler` callbacks you can implement (all are optional):

- `handle_fline(cpu, bus, opcode)`: intercepts 0xFxxx (Line-F) instructions (e.g., FPU probes).
- `handle_trap(cpu, bus, trap_num)`: intercepts `TRAP #n`.
- `handle_breakpoint(cpu, bus, bp_num)`: intercepts `BKPT #n`.
- `handle_illegal(cpu, bus, opcode)`: intercepts illegal instructions.

Return `true` to indicate the HLE handled the trap, or `false` to fall back to the real hardware exception.

## Supported CPU Types

| CPU        | Description                            |
| ---------- | -------------------------------------- |
| `M68000`   | Original 68000 (24-bit address bus)    |
| `M68010`   | 68010 with virtual memory support      |
| `M68EC020` | 68020 embedded controller (no MMU)     |
| `M68020`   | Full 68020 with 32-bit address bus     |
| `M68EC030` | 68030 embedded controller (no MMU)     |
| `M68030`   | Full 68030 with on-chip MMU            |
| `M68EC040` | 68040 embedded controller (no FPU/MMU) |
| `M68LC040` | 68040 lite (no FPU)                    |
| `M68040`   | Full 68040 with FPU and MMU            |
| `SCC68070` | Philips SCC68070 variant               |

## Validation & Testing

This emulator has been rigorously validated against multiple industry-standard test suites to ensure correctness:

### SingleStepTests (m68000)

The [SingleStepTests](https://github.com/SingleStepTests/m68000) project provides exhaustive per-instruction test vectors derived from real hardware and cycle-accurate emulators. Our test suite runs **all 101 instruction categories** with thousands of test cases each, covering:

- All addressing modes and operand sizes
- Edge cases for condition codes (CCR/SR)
- BCD arithmetic (ABCD, SBCD, NBCD)
- Multiply/divide overflow handling
- Exception frame generation

### Musashi Reference Implementation

We validate against [Musashi](https://github.com/kstenerud/Musashi), the gold-standard M68000 emulator used in MAME and countless other projects. Our integration tests:

- Execute complete Musashi test binaries
- Verify register state, memory contents, and exception handling
- Cover 68000 through 68040 instruction sets

### Cross-CPU Verification

Additional test suites verify behavior across CPU generations:

- **68040 FPU tests**: Floating-point transcendental functions, rounding modes
- **MMU translation tests**: Table walks, TTR matching, fault handling
- **Privilege tests**: User/supervisor mode transitions, TRAP behavior
- **Exception tests**: Double-fault detection, address error frames

### Test Coverage

```
tests/
├── singlestep_m68000_v1_tests.rs   # 101 instruction test files
├── musashi_tests.rs                 # Musashi integration suite
├── cross_cpu_tests.rs               # Multi-generation verification
├── m68040_tests.rs                  # 68040-specific features
├── mmu_fault_tests.rs               # MMU and exception handling
├── hle_interception_tests.rs        # Trap handler API tests
└── fixtures/
    ├── m68000/                      # SingleStepTests submodule
    └── Musashi/                     # Musashi reference submodule
```

## Architecture

```
m68k/
├── core/           # CPU core, registers, execution loop
├── dasm/           # Disassembler
├── fpu/            # 68881/68882/68040 FPU emulation
└── mmu/            # 68030/68040 PMMU emulation
```

### Key Types

| Type         | Description                        |
| ------------ | ---------------------------------- |
| `CpuCore`    | Main CPU state and execution       |
| `CpuType`    | CPU model selection enum           |
| `AddressBus` | Trait for memory/IO implementation |
| `HleHandler` | Trait for HLE interception         |
| `StepResult` | Instruction execution result       |

## Performance

The emulator is designed for correctness first, with performance as a secondary goal. Typical use cases (classic computer emulation, game console emulation) run at many multiples of original hardware speed on modern CPUs.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please ensure:

1. All tests pass: `cargo test`
2. No clippy warnings: `cargo clippy -- -D warnings`
3. Code is formatted: `cargo fmt`

## Acknowledgments

- [Musashi](https://github.com/kstenerud/Musashi) - Reference implementation and test fixtures
- [SingleStepTests](https://github.com/SingleStepTests/m68000) - Exhaustive instruction test vectors
- The M68000 Programmer's Reference Manual

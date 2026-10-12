//! Bounded diagnostic for replacing native traces at one guest address.
use m68k::{CpuCore, CpuType, core::memory::LinearMemoryBus};
use std::time::Instant;

fn main() {
    let rounds: u32 = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "10000".into())
        .parse()
        .unwrap();
    let hot_traces: u32 = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "0".into())
        .parse()
        .unwrap();
    if std::env::args().nth(3).as_deref() == Some("dense") {
        dense_cache(rounds, hot_traces);
        return;
    }
    assert!(hot_traces <= 128);
    let mut bus = LinearMemoryBus::new(0x10000);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68040);
    cpu.set_sr(0x2700);
    // MOVEQ #value,D0; ADDQ.L #1,D1; BRA.S back to MOVEQ.
    bus.write_word_at(0x0102, 0x5281);
    bus.write_word_at(0x0104, 0x60FA);
    // Long-lived loops with different sizes, each warmed in a different batch.
    // One live trace can pin the otherwise dead compilations in that batch.
    for index in 0..hot_traces {
        let pc = 0x200 + index * 128;
        let ops = 3 + index % 30;
        bus.write_word_at(pc, 0x707f);
        for op in 1..ops - 1 {
            bus.write_word_at(pc + op * 2, 0x5281);
        }
        bus.write_word_at(
            pc + (ops - 1) * 2,
            0x6000 | ((-(ops as i16 * 2)) as u8 as u16),
        );
    }
    let started = Instant::now();
    for round in 0..rounds {
        let value = round & 0x7f;
        bus.write_word_at(0x0100, 0x7000 | value as u16);
        cpu.pc = 0x0100;
        cpu.run_batch(&mut bus, 3000, &[]);
        assert_eq!(cpu.d(0), value);
        if hot_traces != 0 && round % 64 == 0 {
            let index = (round / 64) % hot_traces;
            let ops = 3 + index % 30;
            cpu.pc = 0x200 + index * 128;
            let before = cpu.d(1);
            cpu.run_batch(&mut bus, ops * 1000, &[]);
            assert_eq!(cpu.d(0), 127);
            assert_eq!(cpu.d(1) - before, (ops - 2) * 1000);
        }
        if round % 1000 == 0 {
            println!(
                "round={round} elapsed={:.3} d1={}",
                started.elapsed().as_secs_f64(),
                cpu.d(1)
            );
        }
    }
    println!(
        "done rounds={rounds} elapsed={:.3} pc={} d0={} d1={} ccr={}",
        started.elapsed().as_secs_f64(),
        cpu.pc,
        cpu.d(0),
        cpu.d(1),
        cpu.get_ccr()
    );
    // Give external process-memory tools a bounded opportunity to sample the final state.
    std::thread::sleep(std::time::Duration::from_secs(3));
}

fn dense_cache(rounds: u32, traces: u32) {
    assert!((1..=4096).contains(&traces));
    let mut bus = LinearMemoryBus::new(0x10000);
    let mut cpu = CpuCore::new();
    cpu.set_cpu_type(CpuType::M68040);
    cpu.set_sr(0x2700);
    for index in 0..traces {
        let pc = 0x200 + index * 6;
        bus.write_word_at(pc, 0x7000 | (index & 127) as u16);
        bus.write_word_at(pc + 2, 0x5281);
        bus.write_word_at(pc + 4, 0x60fa);
    }
    let started = Instant::now();
    for round in 0..rounds {
        let index = round % traces;
        cpu.pc = 0x200 + index * 6;
        cpu.run_batch(&mut bus, 3000, &[]);
        assert_eq!(cpu.d(0), index & 127);
        if round % 1000 == 0 {
            println!(
                "round={round} elapsed={:.3} d1={}",
                started.elapsed().as_secs_f64(),
                cpu.d(1)
            );
        }
    }
    assert_eq!(cpu.d(1), rounds * 1000);
    println!(
        "done rounds={rounds} elapsed={:.3} pc={} d0={} d1={} ccr={}",
        started.elapsed().as_secs_f64(),
        cpu.pc,
        cpu.d(0),
        cpu.d(1),
        cpu.get_ccr()
    );
    std::thread::sleep(std::time::Duration::from_secs(3));
}

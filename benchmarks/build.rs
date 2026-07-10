//! Builds the vendored Musashi core.
//!
//! Musashi generates the bulk of its opcode handlers at build time: `m68kmake`
//! (a host tool) reads `m68k_in.c` and emits `m68kops.c`/`m68kops.h`. We
//! replicate its Makefile here: compile `m68kmake`, run it into OUT_DIR, then
//! compile the core plus our flat-memory shim into a static library.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let vendor = manifest.join("vendor/musashi");

    println!("cargo:rerun-if-changed=vendor/musashi");
    println!("cargo:rerun-if-changed=shim/bench_shim.c");

    // 1. Build the m68kmake generator as a host tool.
    let compiler = cc::Build::new().get_compiler();
    let m68kmake = out.join("m68kmake");
    let status = Command::new(compiler.path())
        .arg("-O2")
        .arg("-o")
        .arg(&m68kmake)
        .arg(vendor.join("m68kmake.c"))
        .status()
        .expect("failed to run C compiler for m68kmake");
    assert!(status.success(), "compiling m68kmake failed");

    // 2. Generate m68kops.c / m68kops.h into OUT_DIR.
    let status = Command::new(&m68kmake)
        .arg(&out)
        .arg(vendor.join("m68k_in.c"))
        .status()
        .expect("failed to run m68kmake");
    assert!(status.success(), "m68kmake generation failed");

    // 3. Compile the core. m68kfpu.c and m68kmmu.h are #included by m68kcpu.c.
    cc::Build::new()
        .file(vendor.join("m68kcpu.c"))
        .file(out.join("m68kops.c"))
        .file(vendor.join("softfloat/softfloat.c"))
        .file(manifest.join("shim/bench_shim.c"))
        .include(&vendor)
        .include(&out)
        .warnings(false)
        .compile("musashi");
}

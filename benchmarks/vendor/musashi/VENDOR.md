# Vendored: Musashi

- Upstream: https://github.com/kstenerud/Musashi (Karl Stenerud)
- Version: 4.10
- Obtained from the GitLab mirror https://gitlab.com/0xTJ/Musashi at commit
  `9bb5f45d521926dff03e9065d43aa16a7189ec87` (2024-08-24).
- License: MIT (see the header of `m68k.h` / `readme.txt`).
- Local changes: none to source files. The `example/` directory was dropped;
  `m68kops.c`/`m68kops.h` are generated at build time by `m68kmake` (driven
  from `../../build.rs`), exactly as upstream's Makefile does.

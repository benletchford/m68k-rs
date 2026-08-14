# Changelog

## [0.10.14](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.13...m68k-v0.10.14) (2026-08-14)


### Bug Fixes

* **m68000/m68010:** issue MOVEM mem-to-reg discarded read after long transfers ([#126](https://github.com/benletchford/m68k-rs/issues/126)) ([9d1fbe0](https://github.com/benletchford/m68k-rs/commit/9d1fbe0984a7f24fbb46ddf62da1034bd6925e24))

## [0.10.13](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.12...m68k-v0.10.13) (2026-08-12)


### Performance Improvements

* admit absolute-addressed TST into the memory-TST family ([#117](https://github.com/benletchford/m68k-rs/issues/117)) ([d5af2c4](https://github.com/benletchford/m68k-rs/commit/d5af2c490533abe8ab611b5ede7616e55dd9cbf3))

## [0.10.12](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.11...m68k-v0.10.12) (2026-08-12)


### Performance Improvements

* record through constant-target JSRs and make call permission durable ([0d8cdc7](https://github.com/benletchford/m68k-rs/commit/0d8cdc7c537d2669faf8024d3838dd3176d3e3e2))

## [0.10.11](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.10...m68k-v0.10.11) (2026-08-12)


### Performance Improvements

* record through BSR calls on a retry, with per-segment SMC intervals ([8cf482f](https://github.com/benletchford/m68k-rs/commit/8cf482f03b03e684f2f666d9ee1cc94094772c8f))

## [0.10.10](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.9...m68k-v0.10.10) (2026-08-11)


### Performance Improvements

* admit AND and OR from memory into the register-ALU family ([14145e6](https://github.com/benletchford/m68k-rs/commit/14145e6184d3a1b2d287e810199337e88a804db1))

## [0.10.9](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.8...m68k-v0.10.9) (2026-08-11)


### Performance Improvements

* admit absolute-addressed CLR into the memory-CLR family ([75baded](https://github.com/benletchford/m68k-rs/commit/75baded6c8c661a965d32fd0ed7a7d750bc1da76))

## [0.10.8](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.7...m68k-v0.10.8) (2026-08-11)


### Performance Improvements

* seed trace candidacy from guard exits and chain compiled continuations ([04f41a0](https://github.com/benletchford/m68k-rs/commit/04f41a0d4957e0420383175c967b9025bcd07320))

## [0.10.7](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.6...m68k-v0.10.7) (2026-08-11)


### Performance Improvements

* salvage a blocked recording's prefix through its last branch ([0c64208](https://github.com/benletchford/m68k-rs/commit/0c64208069ed0fe4b2ccbab7841a76c4253119da))

## [0.10.6](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.5...m68k-v0.10.6) (2026-08-10)


### Performance Improvements

* admit LINK/UNLK frame ops into traces ([865ebd9](https://github.com/benletchford/m68k-rs/commit/865ebd9e5b78e1c06638a6010a9304b17fbf4973))

## [0.10.5](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.4...m68k-v0.10.5) (2026-08-09)


### Performance Improvements

* JIT immediate MOVE stores to memory destinations ([40a8cac](https://github.com/benletchford/m68k-rs/commit/40a8cace840b35f4c32f72aa044849104062db35))

## [0.10.4](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.3...m68k-v0.10.4) (2026-08-09)


### Performance Improvements

* JIT CLR to predecrement destinations ([e0368a0](https://github.com/benletchford/m68k-rs/commit/e0368a0e81a668ba367479d66f459b150cdfc9eb))

## [0.10.3](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.2...m68k-v0.10.3) (2026-08-09)


### Performance Improvements

* jit stores to brief-indexed destinations (MOVE, CLR) ([6c7e8c5](https://github.com/benletchford/m68k-rs/commit/6c7e8c5743a59a7511477c46f1fbecaa0c0e5634))

## [0.10.2](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.1...m68k-v0.10.2) (2026-08-09)


### Performance Improvements

* jit register-to-memory subtracts ([8796494](https://github.com/benletchford/m68k-rs/commit/879649408f1417a5050ff8f122e59939e1f1b1be))

## [0.10.1](https://github.com/benletchford/m68k-rs/compare/m68k-v0.10.0...m68k-v0.10.1) (2026-08-09)


### Performance Improvements

* JIT register-count shifts ([662c6a1](https://github.com/benletchford/m68k-rs/commit/662c6a1df0197aa658d1afb1dbfb3f9df7d67957))

## [0.10.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.9.1...m68k-v0.10.0) (2026-08-09)


### Features

* classify pure poll loops at trace compile time ([5ecdc72](https://github.com/benletchford/m68k-rs/commit/5ecdc72a54431f7d1cf6c98d928c098b51a721fc))

## [0.9.1](https://github.com/benletchford/m68k-rs/compare/m68k-v0.9.0...m68k-v0.9.1) (2026-08-08)


### Performance Improvements

* jit immediate word multiplies ([e1eb2ae](https://github.com/benletchford/m68k-rs/commit/e1eb2aeb3ed3c30312a36ddf7251f814de1259ea))

## [0.9.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.8.3...m68k-v0.9.0) (2026-08-08)


### Features

* **trace-profile:** report silent-rejection opcodes ([ff6455f](https://github.com/benletchford/m68k-rs/commit/ff6455f95f1c14c7a8341034c169d778921545cd))


### Performance Improvements

* **execute:** decode register CLR for boundary hooks ([b1d4891](https://github.com/benletchford/m68k-rs/commit/b1d4891f586a64d56b6ca31fbabe7fc0e84d83a1))

## [0.8.3](https://github.com/benletchford/m68k-rs/compare/m68k-v0.8.2...m68k-v0.8.3) (2026-08-06)


### Performance Improvements

* **execute:** add precise decoded dispatch for boundary-hook runs ([3a2c591](https://github.com/benletchford/m68k-rs/commit/3a2c5918ba4b9eecc112eb2a50ae56c1fedbb920))

## [0.8.2](https://github.com/benletchford/m68k-rs/compare/m68k-v0.8.1...m68k-v0.8.2) (2026-08-06)


### Bug Fixes

* **trace-profile:** attribute recordings that end with no unsupported opcode ([153436b](https://github.com/benletchford/m68k-rs/commit/153436b8bb62b5281345489a904eae135de6265b))


### Performance Improvements

* JIT brief-indexed LEA ([934f5ca](https://github.com/benletchford/m68k-rs/commit/934f5ca60438c09adfb5a90bd11b78be000dc8e5))

## [0.8.1](https://github.com/benletchford/m68k-rs/compare/m68k-v0.8.0...m68k-v0.8.1) (2026-08-06)


### Performance Improvements

* JIT displacement PEA pushes ([6313f2f](https://github.com/benletchford/m68k-rs/commit/6313f2fc49187fff335ef8e47ac89da89f28a9ac))

## [0.8.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.7.2...m68k-v0.8.0) (2026-08-05)


### Features

* report exact trace recording paths ([0edf7df](https://github.com/benletchford/m68k-rs/commit/0edf7df5bb70803671fe468ddbc85fe2f3a5c921))

## [0.7.2](https://github.com/benletchford/m68k-rs/compare/m68k-v0.7.1...m68k-v0.7.2) (2026-08-05)


### Performance Improvements

* JIT indexed immediate word compares ([bfe4a2d](https://github.com/benletchford/m68k-rs/commit/bfe4a2d9839cc3dd18942552829c541cdbd0f09d))

## [0.7.1](https://github.com/benletchford/m68k-rs/compare/m68k-v0.7.0...m68k-v0.7.1) (2026-08-05)


### Performance Improvements

* JIT brief-indexed memory-source ALU operations ([0bc6b89](https://github.com/benletchford/m68k-rs/commit/0bc6b891db8b1f99b0edf806dfa12a8e64cd8e85))
* JIT fixed-point state update loops ([303f5e3](https://github.com/benletchford/m68k-rs/commit/303f5e3309fbde2dd603c5d99b3c18d60de85046))

## [0.7.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.6.0...m68k-v0.7.0) (2026-08-04)


### Features

* add exact-width memory bit-field bus accesses ([d4ad107](https://github.com/benletchford/m68k-rs/commit/d4ad10793271fecc6f0c381d7c89cc3e38449c45))
* expose interrupt-entry cycle boundaries ([04a3bb0](https://github.com/benletchford/m68k-rs/commit/04a3bb07a907d7792eb7d445fce48066c6e5a6cf))


### Performance Improvements

* continue trace recording through full dispatch ([aba25ec](https://github.com/benletchford/m68k-rs/commit/aba25ec5dd0d41705ed0cb6b46e9690a46928ac8))
* JIT packed lookup generator loops ([95c25ac](https://github.com/benletchford/m68k-rs/commit/95c25acdda9d2691769157787c5834822f29d57b))

## [0.6.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.5.3...m68k-v0.6.0) (2026-08-04)


### Features

* add instruction-boundary cycle hooks ([#62](https://github.com/benletchford/m68k-rs/issues/62)) ([af37111](https://github.com/benletchford/m68k-rs/commit/af3711149ad2b4c1cd49511d09a6e98bd48bc758))

## [0.5.3](https://github.com/benletchford/m68k-rs/compare/m68k-v0.5.2...m68k-v0.5.3) (2026-08-04)


### Performance Improvements

* JIT guarded indexed scans ([72f69e2](https://github.com/benletchford/m68k-rs/commit/72f69e2857346c5536f394d4b447c0ac8775fac3))

## [0.5.2](https://github.com/benletchford/m68k-rs/compare/m68k-v0.5.1...m68k-v0.5.2) (2026-08-02)


### Bug Fixes

* honor bus boundaries after interrupt entry ([1e7124b](https://github.com/benletchford/m68k-rs/commit/1e7124ba3c9dfa7b6ad272e4d8dd03664f77f1ad))

## [0.5.1](https://github.com/benletchford/m68k-rs/compare/m68k-v0.5.0...m68k-v0.5.1) (2026-08-02)


### Bug Fixes

* remove 020 result forwarding and make the DBcc refill alignment dependent ([ebe891f](https://github.com/benletchford/m68k-rs/commit/ebe891f0e808c693f18ed97b75edb9105f7a32f2))

## [0.5.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.4.0...m68k-v0.5.0) (2026-08-02)


### Features

* calibrate 020 taken-branch refill and model result forwarding from real hardware ([0b9117a](https://github.com/benletchford/m68k-rs/commit/0b9117a83217a61fc3096bbf4860d59bd0fdf8d5))

## [0.4.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.3.2...m68k-v0.4.0) (2026-08-01)


### Features

* add bus-requested cycle boundaries ([#48](https://github.com/benletchford/m68k-rs/issues/48)) ([70ead97](https://github.com/benletchford/m68k-rs/commit/70ead97e05c22fcfe5101c7567e9fe1ebb88b412))

## [0.3.2](https://github.com/benletchford/m68k-rs/compare/m68k-v0.3.1...m68k-v0.3.2) (2026-07-31)


### Bug Fixes

* correct public API documentation ([e0d0b57](https://github.com/benletchford/m68k-rs/commit/e0d0b57fe5f0007a87d0e4027f29039bd5105cc5))

## [0.3.1](https://github.com/benletchford/m68k-rs/compare/m68k-v0.3.0...m68k-v0.3.1) (2026-07-31)


### Bug Fixes

* make native JIT dependencies opt in ([39e767a](https://github.com/benletchford/m68k-rs/commit/39e767a00b6365880a71e44f0805c97aae7f9a02))

## [0.3.0](https://github.com/benletchford/m68k-rs/compare/m68k-v0.2.5...m68k-v0.3.0) (2026-07-31)


### Features

* unify cycle-exact and high-performance emulation ([8d11edf](https://github.com/benletchford/m68k-rs/commit/8d11edf75feb62dd3211bfda7a3cb9a781f6b3e6))

## [0.2.5](https://github.com/benletchford/m68k-rs/compare/m68k-v0.2.4...m68k-v0.2.5) (2026-07-25)


### Performance Improvements

* batch profitable self-loops in native JIT ([36922d8](https://github.com/benletchford/m68k-rs/commit/36922d8a0f892353be9572c3f7c12173c68a854a))
* JIT brief-indexed MOVE and postincrement ADD loops ([db116c2](https://github.com/benletchford/m68k-rs/commit/db116c2fbba94c06a6bdf0ab98742d63e914bef0))
* JIT data-only MOVEM.W postincrement loops ([00f635f](https://github.com/benletchford/m68k-rs/commit/00f635f2c93cafec0567956423bc7bc0a90548f1))
* JIT immediate ASR and LSL shifts ([0897f04](https://github.com/benletchford/m68k-rs/commit/0897f045c38a77ddb30bcbd1309bc773fa499c84))
* JIT memory-source ADD and SUB traces ([99c0f19](https://github.com/benletchford/m68k-rs/commit/99c0f19383506b994db67d779214bfcac16b226e))
* JIT profitable indirect JSR boundaries ([abad647](https://github.com/benletchford/m68k-rs/commit/abad647e23814c07f22824ec556797dd97b621a2))

## [0.2.4](https://github.com/benletchford/m68k-rs/compare/m68k-v0.2.3...m68k-v0.2.4) (2026-07-24)


### Bug Fixes

* count successful calls in trace adaptation ([a2dcfbd](https://github.com/benletchford/m68k-rs/commit/a2dcfbd5c107a0bf09ffbf6c24d289d19e624f88))


### Performance Improvements

* adapt JIT traces to dominant branch paths ([4b95efb](https://github.com/benletchford/m68k-rs/commit/4b95efb1317ba2cccc763426ac64690a67d1489a))

## [0.2.3](https://github.com/benletchford/m68k-rs/compare/m68k-v0.2.2...m68k-v0.2.3) (2026-07-24)


### Performance Improvements

* JIT hot memory-source CMP loops ([3571e27](https://github.com/benletchford/m68k-rs/commit/3571e2754399bc5ef229f97687fac585a911df16))
* JIT hot memory-source CMP loops ([a497005](https://github.com/benletchford/m68k-rs/commit/a49700513c4c2e789c103eb8642e6368e2c83a93))
* JIT two-instruction self-loops ([6433e09](https://github.com/benletchford/m68k-rs/commit/6433e094a82fc8bf7db23c214e60ab152200105d))
* JIT two-instruction self-loops ([cdc342a](https://github.com/benletchford/m68k-rs/commit/cdc342a3f82eab52308a31407678c508af6477db))

## [0.2.2](https://github.com/benletchford/m68k-rs/compare/m68k-v0.2.1...m68k-v0.2.2) (2026-07-23)


### Performance Improvements

* extend traces for Classic Mac memory loops ([a714fc0](https://github.com/benletchford/m68k-rs/commit/a714fc02f71d5747cf84b974e334476be643deb4))
* extend traces for Classic Mac memory loops ([b30fe3b](https://github.com/benletchford/m68k-rs/commit/b30fe3b4b95ff1281cc4ebd67634303b8f01b92a))
* fast-path classic Mac displacement memory ops ([6e8e018](https://github.com/benletchford/m68k-rs/commit/6e8e018e7cfd485a04566d1927bb289cf73a01ae))
* fast-path classic Mac displacement memory ops ([287cd02](https://github.com/benletchford/m68k-rs/commit/287cd024b05ab0a50ae96e7c42fec11b48b4e8df))
* inline generic memory helpers ([146c350](https://github.com/benletchford/m68k-rs/commit/146c350b62c7a7d33d9332625a6bb837f55585d3))
* inline generic memory helpers ([128a6a2](https://github.com/benletchford/m68k-rs/commit/128a6a218a9880fc6f3ed03ad860ea5804063505))
* inline memory operation internals ([e782cff](https://github.com/benletchford/m68k-rs/commit/e782cffeb365b3ca9bb5a7a806429a36735bbac6))
* inline memory operation internals ([a90d60a](https://github.com/benletchford/m68k-rs/commit/a90d60a82998b7a17e434356234301b26418f325))
* inline the decoded simple-op executor ([9a808c5](https://github.com/benletchford/m68k-rs/commit/9a808c5cdac218ea1008cd0017dc226ddc28fccc))
* inline the decoded simple-op executor ([af33e5b](https://github.com/benletchford/m68k-rs/commit/af33e5b18e268f69f3449e4a2415014b0564bf3c))
* record multi-block JIT regions ([6dfea1e](https://github.com/benletchford/m68k-rs/commit/6dfea1e86385d1e1e0cc594f688809bb9139b7a6))
* record multi-block JIT regions ([7fa51ad](https://github.com/benletchford/m68k-rs/commit/7fa51adfd44f69d5118a751d2656b379b1b268a1))

## [0.2.1](https://github.com/benletchford/m68k-rs/compare/m68k-v0.2.0...m68k-v0.2.1) (2026-07-16)


### Bug Fixes

* release-please ([609e8d3](https://github.com/benletchford/m68k-rs/commit/609e8d385b7dc5a69e7a244ff225029da511b491))
* unused mut ([7f28225](https://github.com/benletchford/m68k-rs/commit/7f282250ed0379d2b67a20229fa6867d92c16508))

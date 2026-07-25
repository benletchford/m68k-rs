# Changelog

## [0.2.5](https://github.com/benletchford/m68k-rs/compare/m68k-v0.2.4...m68k-v0.2.5) (2026-07-25)


### Performance Improvements

* batch profitable self-loops in native JIT ([36922d8](https://github.com/benletchford/m68k-rs/commit/36922d8a0f892353be9572c3f7c12173c68a854a))
* JIT brief-indexed MOVE and postincrement ADD loops ([db116c2](https://github.com/benletchford/m68k-rs/commit/db116c2fbba94c06a6bdf0ab98742d63e914bef0))
* JIT data-only MOVEM.W postincrement loops ([00f635f](https://github.com/benletchford/m68k-rs/commit/00f635f2c93cafec0567956423bc7bc0a90548f1))
* JIT immediate ASR and LSL shifts ([0897f04](https://github.com/benletchford/m68k-rs/commit/0897f045c38a77ddb30bcbd1309bc773fa499c84))
* JIT immediate ASR and LSL shifts ([d9e89a1](https://github.com/benletchford/m68k-rs/commit/d9e89a1d20bfd59ee26c026bb7d6ce1b7d500630))
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

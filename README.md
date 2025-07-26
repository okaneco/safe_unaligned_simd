# `safe_unaligned_simd`

[![Build Status](https://github.com/okaneco/safe_unaligned_simd/workflows/Rust%20CI/badge.svg)](https://github.com/okaneco/safe_unaligned_simd)
[![Crates.io](https://img.shields.io/crates/v/safe_unaligned_simd.svg)](https://crates.io/crates/safe_unaligned_simd)
[![Docs.rs](https://docs.rs/safe_unaligned_simd/badge.svg)](https://docs.rs/safe_unaligned_simd)

Safe wrappers for unaligned SIMD load and store operations.

The goal of this crate is to remove the need for "unnecessary `unsafe`" code when using memory vector intrinsics, with no alignment requirements.

Platform-intrinsics that take raw pointers have been wrapped in functions that receive Rust reference types as arguments.

**MSRV**: 1.87

## Implemented Intrinsics

### `x86`, `x86_64`
- `sse`, `sse2`, `avx`

Some example function signatures:
```rust
#[target_feature(enable = "sse")]
fn _mm_storeu_ps(mem_addr: &mut [f32; 4], a: __m128);
#[target_feature(enable = "sse2")]
fn _mm_store_sd(mem_addr: &mut f64, a: __m128d);
#[target_feature(enable = "avx")]
fn _mm256_loadu2_m128(hiaddr: &[f32; 4], loaddr: &[f32; 4]) -> __m256;
```

Currently, there is no plan to implement gather/scatter or masked load/store intrinsics for this platform.

### `aarch64` / `arm64ec`
- `neon`

Some example function signatures:
```rust
#[target_feature(enable = "neon")]
fn vld2_dup_s8(from: &[i8; 2]) -> int8x8x2_t;
#[target_feature(enable = "neon")]
fn vst1q_f64(into: &mut [f64; 2], val: float64x2_t);
```

### Other platforms

Not yet supported.

## License
This crate is licensed under either
- the [MIT License](LICENSE-MIT), or
- the [Apache License (Version 2.0)](LICENSE-APACHE)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

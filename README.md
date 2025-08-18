# `safe_unaligned_simd`

[![Build Status](https://github.com/okaneco/safe_unaligned_simd/actions/workflows/rust-ci.yml/badge.svg?branch=master)](https://github.com/okaneco/safe_unaligned_simd)
[![Crates.io](https://img.shields.io/crates/v/safe_unaligned_simd.svg)](https://crates.io/crates/safe_unaligned_simd)
[![Docs.rs](https://docs.rs/safe_unaligned_simd/badge.svg)](https://docs.rs/safe_unaligned_simd)

Safe wrappers for unaligned SIMD load and store operations.

The goal of this crate is to remove the need for "unnecessary `unsafe`" code when using vector intrinsics to access memory, with no alignment requirements.

Platform-intrinsics that take raw pointers have been wrapped in functions that receive Rust reference types as arguments.

**MSRV**: 1.88

## Supported target architectures

### `x86` / `x86_64`
- `sse`, `sse2`, `avx`

Some functions have variants that are generic over `Cell` array types, which allow for mutation of shared references.
See the [`cell`](./src/x86/cell.rs) module for an example.

Example function signatures:
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

Example function signatures:
```rust
#[target_feature(enable = "neon")]
fn vld2_dup_s8(from: &[i8; 2]) -> int8x8x2_t;
#[target_feature(enable = "neon")]
fn vst1q_f64(into: &mut [f64; 2], val: float64x2_t);
```

### `wasm32`
- `simd128`

Example function signatures:
```rust
#[target_feature(enable = "simd128")]
pub fn v128_load8_splat<T: Is1ByteUnaligned>(t: &T) -> v128;
#[target_feature(enable = "simd128")]
pub fn v128_store<T: Is16BytesUnaligned>(t: &mut T, v: v128);
```

## License
This crate is licensed under either
- the [MIT License](LICENSE-MIT), or
- the [Apache License (Version 2.0)](LICENSE-APACHE)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

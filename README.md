# `safe_unaligned_simd`

[![Build Status](https://github.com/okaneco/safe_unaligned_simd/actions/workflows/rust-ci.yml/badge.svg?branch=master)](https://github.com/okaneco/safe_unaligned_simd)
[![Crates.io](https://img.shields.io/crates/v/safe_unaligned_simd.svg)](https://crates.io/crates/safe_unaligned_simd)
[![Docs.rs](https://docs.rs/safe_unaligned_simd/badge.svg)](https://docs.rs/safe_unaligned_simd)

Safe wrappers for unaligned SIMD load and store operations.

The goal of this crate is to remove the need for "unnecessary `unsafe`" code when using vector intrinsics to access memory, with no alignment requirements.

Platform-intrinsics that take raw pointers have been wrapped in functions that receive Rust reference types as arguments.

**MSRV**: `1.88`

## Why use this crate?

This crate is compatible with runtime feature detection.

Unlike some other safe architecture intrinsic wrappers, this crate does not lock the user into `#[cfg()]`-gating SIMD code behind compile-time CPU target feature declaration.

## Supported target architectures

### `x86` / `x86_64`
- `sse`, `sse2`, `avx`, `avx512f`, `avx512vl`, `avx512bw`, `avx512vbmi2`

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

Currently, there is no plan to implement gather/scatter or `avx2` masked load/store intrinsics for this platform.

`avx512` - AVX-512 intrinsics require `rustc 1.89` or later.

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

## A note on creating mutable array references from slices

Beware of accidentally creating mutable references to temporary arrays.

Rust will implicitly clone an array from a slice and return a mutable reference to that clone if not wrapped properly in parentheses.

```rust
// Valid mutable array reference creation
let out_data: &mut [u8; 4] = (&mut chunk[..4]).try_into().unwrap();
let out_data = TryInto::<&mut [u8; 4]>::try_into(&mut chunk[..4]).unwrap();

// Incorrect creation of a mutable reference: this clones the chunk and returns
// a mutable reference to the copy. If we modify `out_data` after this point,
// the changes will not reflect back in our original `chunk` slice.
// 🚫🈲⛔❌ - Do not use the following line
let out_data = &mut chunk[..4].try_into().unwrap();
```

A better solution is to use [`as_mut_array`][as_mut_array] to sidestep this entirely.<br>
As of the time of this writing (`rustc 1.91`), `as_mut_array` is unstable but in the process of being stabilized.

[as_mut_array]: https://doc.rust-lang.org/stable/std/primitive.slice.html#method.as_mut_array

## License
This crate is licensed under either
- the [MIT License](LICENSE-MIT), or
- the [Apache License (Version 2.0)](LICENSE-APACHE)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

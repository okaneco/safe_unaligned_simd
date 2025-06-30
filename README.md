# `safe_unaligned_simd`

Safe wrappers for unaligned SIMD load and store operations.

The goal of this crate is to remove the need for "unnecessary `unsafe`" code when using vector intrinsics with no alignment requirements.

Platform-intrinsics that take raw pointers have been wrapped in functions that receive Rust reference types as arguments.

**MSRV**: 1.87

## Implemented Intrinsics

### `x86_64`
- `sse`, `sse2`, `avx`

Currently, there is no plan to implement gather/scatter or masked load/store intrinsics for this platform.

### Other platforms

To be determined.

## License
This crate is licensed under either
- the [MIT License](LICENSE-MIT), or
- the [Apache License (Version 2.0)](LICENSE-APACHE)

at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.

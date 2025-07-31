# `safe_unaligned_simd` changelog

## Version 0.2.0 - 2025-07

Version bump for changing the signature of some `sse2` functions.

### Major changes

- Added `neon` support for load and store on `aarch64`/`arm64ec` architectures
- Adjusted `sse2` signatures of `_mm_loadu/storeu_si[16|32|64]` to use traits generic over 16, 32, and 64-bit width types
- Added `Is[16|32|64|128|256]CellUnaligned` traits for `Cell` types in `x86`
- MSRV increased to `1.88`

#### aarch64 / arm64ec

See the `aarch64` module for all implemented intrinsics.
Mainly `VLD1` and `VST1` intrinsics were added.

#### x86 / x86_64

All of the remaining `std::arch` `x86`/`x86_64` intrinsics that take `u8` pointers are now wrapped by functions that use marker traits for the argument size.

For example, a function which takes an input argument bounded by `<T: Is16BitsUnaligned>` can be called with `[u8; 2]`, `[i8; 2]`, `[u16; 1]`, `[i16; 1]`, `u16`, or `i16`.
This affects the `_mm_loadu/storeu_si[16|32|64]` functions and matches the behavior of the `_si[128|256]` functions.

Additionally, `Is__BitsUnaligned` now has a corresponding `Is__CellUnaligned` trait which enables using intrinsics with `&Cell<[T; N]>` and `&[Cell<T>; N]` types.
These traits allow for operating on overlapping slices similar to how one can use `std::arch` intrinsics with raw pointers.
The functions are located within an architecture's `cell` module.

### Notable PRs

[`#15`][15] - Implement generic load/store for Cell types of 16/32/64 bits for x86  
[`#8`][8] - Abstractions around `aarch64` neon loads and stores  
[`#6`][6] - Add traits for 16, 32, and 64 bit unaligned operations in x86  
[`#4`][4] - Add cell variants for generic load/store (x86)

## Version 0.1.1 - 2025-07

[`#3`][3] - Added `x86` support

## Version 0.1.0 - 2025-06

Initial release

[15]: https://github.com/okaneco/safe_unaligned_simd/pull/15
[8]: https://github.com/okaneco/safe_unaligned_simd/pull/8
[6]: https://github.com/okaneco/safe_unaligned_simd/pull/6
[4]: https://github.com/okaneco/safe_unaligned_simd/pull/4
[3]: https://github.com/okaneco/safe_unaligned_simd/pull/3

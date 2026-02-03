//! Safe wrappers for unaligned SIMD load and store operations.
//!
//! ## Overview
//!
//! The goal of this crate is to remove the need for "unnecessary `unsafe`" code
//! when using vector intrinsics to access memory, with no alignment
//! requirements.
//!
//! Platform-intrinsics that take raw pointers have been wrapped in functions
//! that receive Rust reference types as arguments.
//!
//! ## Safely using platform intrinsics
//!
//! Users are responsible for ensuring that the CPU supports the intended target feature.
//!
//! Feature detection can be done at compile-time by using `#[cfg]` attributes on functions or at runtime using an `is_[arch]_feature_detected!` macro from `std::arch`.
//!
//! `unsafe` is needed to call into functions annotated with `#[target_feature]`, but [it's safe to call other functions with the same target features][rustc-1.86].
//!
//! See [the `std::arch` module documentation][stdarch] for a full explanation and [the `rustc` 1.87 release notes][rustc-1.87] for a simple example of runtime feature detection with fallback.
//!
//! [rustc-1.86]: https://blog.rust-lang.org/2025/04/03/Rust-1.86.0/#allow-safe-functions-to-be-marked-with-the-target-feature-attribute
//! [rustc-1.87]: https://blog.rust-lang.org/2025/05/15/Rust-1.87.0/#safe-architecture-intrinsics
//! [stdarch]: https://doc.rust-lang.org/stable/std/arch/index.html#overview
//!
//! ## Supported target architectures
//!
//! ### `x86` / `x86_64`
//! - `sse`, `sse2`, `avx`, `avx512f`, `avx512vl`, `avx512bw`, `avx512vbmi2`
//!
//! Some functions have variants that are generic over `Cell` array types,
//! which allow for mutation of shared references.
//!
//! Currently, there is no plan to implement gather/scatter or `avx2` masked
//! load/store intrinsics for this platform.
//!
//! ### `aarch64`, `arm64ec`
//! - `neon`
//!
//! Intrinsics that load / store individual lanes are not designed yet.
//!
//! ### `wasm32`
//! - `simd128`
//!
//! ## A note on creating mutable array references from slices
//!
//! **_tl;dr:_ Use [`as_mut_array`][as_mut_array] to avoid this bug, stable since `1.93`.**
//!
//! Beware of accidentally creating mutable references to temporary arrays.
//!
//! Rust will implicitly clone an array from a slice and return a mutable reference to that clone if not wrapped properly in parentheses.
//!
//! ```rust,ignore
//! # let mut chunk = [0u8; 8];
//! // Valid mutable array reference creation
//! let out_data: &mut [u8; 4] = chunk[..4].as_mut_array().unwrap(); // since 1.93
//! let out_data: &mut [u8; 4] = (&mut chunk[..4]).try_into().unwrap();
//! let out_data = TryInto::<&mut [u8; 4]>::try_into(&mut chunk[..4]).unwrap();
//!
//! // Incorrect creation of a mutable reference: this clones the chunk and returns
//! // a mutable reference to the copy. If we modify `out_data` after this point,
//! // the changes will not reflect back in our original `chunk` slice.
//! // 🚫🈲⛔❌ - Do not use the following line
//! let out_data = &mut chunk[..4].try_into().unwrap();
//! # *out_data = [1u8; 4];
//! ```
//!
//! The now-stable [`as_mut_array`][as_mut_array] sidesteps this issue entirely.<br>
//!
//! [as_mut_array]: https://doc.rust-lang.org/1.93.0/std/primitive.slice.html#method.as_mut_array
#![forbid(missing_docs, non_ascii_idents)]
#![cfg_attr(not(test), no_std)]

#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch = "wasm32",))]
mod common_traits;

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
pub mod aarch64;

#[cfg(target_arch = "wasm32")]
pub mod wasm32;

#[cfg(target_arch = "x86")]
pub mod x86;

#[cfg(target_arch = "x86_64")]
mod x86;
#[cfg(target_arch = "x86_64")]
pub mod x86_64 {
    //! Platform-specific intrinsics for `x86_64`.

    #[cfg(target_arch = "x86_64")]
    pub use crate::x86::*;
}

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
//! ## Implemented Intrinsics
//!
//! ### `x86`, `x86_64`
//! - `sse`, `sse2`, `avx`
//!
//! Some functions have variants that are generic over `Cell` array types,
//! which allow for mutation of shared references.
//!
//! Currently, there is no plan to implement gather/scatter or masked load/store
//! intrinsics for this platform.
//!
//! ### `aarch64`, `arm64ec`
//! - `neon`
//!
//! Intrinsics that load / store individual lanes are not designed yet.
//!
//! ### Other platforms
//!
//! Not yet supported.
#![forbid(missing_docs, non_ascii_idents)]
#![cfg_attr(not(test), no_std)]

#[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
pub mod aarch64;

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

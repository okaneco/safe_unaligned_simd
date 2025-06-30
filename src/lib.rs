//! Safe wrappers for unaligned SIMD load and store operations.
//!
//! ## Overview
//!
//! The goal of this crate is to remove the need for "unnecessary `unsafe`" code
//! when using vector intrinsics with no alignment requirements.
//!
//! Platform-intrinsics that take raw pointers have been wrapped in functions
//! that receive Rust reference types as arguments.
//!
//! ## Implemented Intrinsics
//!
//! ### `x86_64`
//! - `sse`, `sse2`, `avx`
//!
//! Currently, there is no plan to implement gather/scatter or masked load/store
//! intrinsics for this platform.
//!
//! ### Other platforms
//!
//! To be determined.
#![forbid(missing_docs, non_ascii_idents)]
#![cfg_attr(not(test), no_std)]

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

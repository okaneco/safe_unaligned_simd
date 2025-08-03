//! Platform-specific intrinsics for `x86`.
#![allow(
    // Boilerplate that would repeat each function description for little benefit
    clippy::missing_safety_doc
)]

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128i, __m256i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128i, __m256i};

mod sse;
pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

mod avx;
pub use self::avx::*;

pub mod cell;

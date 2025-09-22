//! Platform-specific intrinsics for `x86`.
#![allow(
    // Boilerplate that would repeat each function description for little benefit
    clippy::missing_safety_doc
)]

mod sse;
pub use self::sse::*;

mod sse2;
pub use self::sse2::*;

mod avx;
pub use self::avx::*;

#[cfg(feature = "avx512")]
mod avx512f;
#[cfg(feature = "avx512")]
pub use self::avx512f::*;

#[cfg(feature = "avx512")]
mod avx512bw;
#[cfg(feature = "avx512")]
pub use self::avx512bw::*;

#[cfg(feature = "avx512")]
mod avx512vbmi2;
#[cfg(feature = "avx512")]
pub use self::avx512vbmi2::*;

pub mod cell;

pub use crate::common_traits::{
    Is16BitsUnaligned, Is16CellUnaligned, Is32BitsUnaligned, Is32CellUnaligned, Is64BitsUnaligned,
    Is64CellUnaligned, Is128BitsUnaligned, Is128CellUnaligned, Is256BitsUnaligned,
    Is256CellUnaligned, Is512BitsUnaligned,
};

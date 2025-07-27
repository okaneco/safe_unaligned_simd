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

// Internal module for sealing SIMD traits.
mod private {
    pub trait Sealed {}
}

/// A trait that marks a type as valid for unaligned operations as an `i16`.
pub trait Is16BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as an `i32`.
pub trait Is32BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as an `i64`.
pub trait Is64BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as [`__m128i`],
/// an x86-specific 128-bit integer vector type.
pub trait Is128BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as [`__m256i`],
/// an x86-specific 256-bit integer vector type.
pub trait Is256BitsUnaligned: private::Sealed {}

///////////////////////////////////////////
// Start of `Cell` trait implementations //
///////////////////////////////////////////

impl<T, const N: usize> private::Sealed for [core::cell::Cell<T>; N] where [T; N]: private::Sealed {}
impl<T, const N: usize> private::Sealed for core::cell::Cell<[T; N]> where [T; N]: private::Sealed {}

/// Marks a cell-like type as valid for unaligned operations as [`__m128i`], an
/// x86-specific 128-bit integer vector type, on shared references.
pub trait Is128CellUnaligned: private::Sealed {}

impl<T, const N: usize> Is128CellUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is128BitsUnaligned
{
}
impl<T, const N: usize> Is128CellUnaligned for core::cell::Cell<[T; N]> where
    [T; N]: Is128BitsUnaligned
{
}

/// Marks a cell-like type as valid for unaligned operations as [`__m256i`], an
/// x86-specific 256-bit integer vector type, on shared references.
pub trait Is256CellUnaligned: private::Sealed {}

impl<T, const N: usize> Is256CellUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is256BitsUnaligned
{
}
impl<T, const N: usize> Is256CellUnaligned for core::cell::Cell<[T; N]> where
    [T; N]: Is256BitsUnaligned
{
}

macro_rules! impl_N_bits_traits {
    (
        impl $trait:path [$target:ty] for {
            $($source:ty,)*
        }
    ) => {
        $(
            const _: () =
                const { assert!(size_of::<$source>() == size_of::<$target>()) };

            impl private::Sealed for $source {}
            impl $trait for $source {}
        )*
    };
}

impl_N_bits_traits! {
    impl Is16BitsUnaligned [i16] for {
        [u8; 2],
        [i8; 2],
        [u16; 1],
        [i16; 1],
        u16,
        i16,
    }
}

impl_N_bits_traits! {
    impl Is32BitsUnaligned [i32] for {
        [u8; 4],
        [i8; 4],
        [u16; 2],
        [i16; 2],
        [u32; 1],
        [i32; 1],
        u32,
        i32,
    }
}

impl_N_bits_traits! {
    impl Is64BitsUnaligned [i64] for {
        [u8; 8],
        [i8; 8],
        [u16; 4],
        [i16; 4],
        [u32; 2],
        [i32; 2],
        [u64; 1],
        [i64; 1],
        u64,
        i64,
    }
}

impl_N_bits_traits! {
    impl Is128BitsUnaligned [__m128i] for {
        [u8; 16],
        [i8; 16],
        [u16; 8],
        [i16; 8],
        [u32; 4],
        [i32; 4],
        [u64; 2],
        [i64; 2],
    }
}

impl_N_bits_traits! {
    impl Is256BitsUnaligned [__m256i] for {
        [u8; 32],
        [i8; 32],
        [u16; 16],
        [i16; 16],
        [u32; 8],
        [i32; 8],
        [u64; 4],
        [i64; 4],
    }
}

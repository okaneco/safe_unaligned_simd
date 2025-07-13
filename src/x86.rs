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

// Internal module for sealing SIMD traits.
mod private {
    pub trait Sealed {}
}

/// A trait that marks a type as valid for unaligned operations as [`__m128i`],
/// an x86-specific 128-bit integer vector type.
pub trait Is128BitsUnaligned: private::Sealed {}

macro_rules! impl_128_bits_traits {
    ([$array_type:ty; $array_len:literal] => $vector:ty ) => {
        const _: () =
            const { assert!((size_of::<$array_type>() * $array_len) == size_of::<$vector>()) };

        impl private::Sealed for [$array_type; $array_len] {}
        impl Is128BitsUnaligned for [$array_type; $array_len] {}
    };
}

impl_128_bits_traits!([u8; 16] => __m128i);
impl_128_bits_traits!([i8; 16] => __m128i);
impl_128_bits_traits!([u16; 8] => __m128i);
impl_128_bits_traits!([i16; 8] => __m128i);
impl_128_bits_traits!([u32; 4] => __m128i);
impl_128_bits_traits!([i32; 4] => __m128i);
impl_128_bits_traits!([u64; 2] => __m128i);
impl_128_bits_traits!([i64; 2] => __m128i);

/// A trait that marks a type as valid for unaligned operations as [`__m256i`],
/// an x86-specific 256-bit integer vector type.
pub trait Is256BitsUnaligned: private::Sealed {}

macro_rules! impl_256_bits_traits {
    ([$array_type:ty; $array_len:literal] => $vector:ty ) => {
        const _: () =
            const { assert!((size_of::<$array_type>() * $array_len) == size_of::<$vector>()) };

        impl private::Sealed for [$array_type; $array_len] {}
        impl Is256BitsUnaligned for [$array_type; $array_len] {}
    };
}

impl_256_bits_traits!([u8; 32] => __m256i);
impl_256_bits_traits!([i8; 32] => __m256i);
impl_256_bits_traits!([u16; 16] => __m256i);
impl_256_bits_traits!([i16; 16] => __m256i);
impl_256_bits_traits!([u32; 8] => __m256i);
impl_256_bits_traits!([i32; 8] => __m256i);
impl_256_bits_traits!([u64; 4] => __m256i);
impl_256_bits_traits!([i64; 4] => __m256i);

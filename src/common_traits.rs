//! Module for traits common to SIMD intrinsic implementations.
//!
//! Some platforms, such as x86 and wasm32, have "bag of bits" vector register
//! types which can internally represent various types of packed integer arrays.
//!
//! These traits provide abstractions over the bit-width that these vector
//! types' load and store intrinsics operate on.

// Internal module for sealing SIMD traits.
mod private {
    pub trait Sealed {}
}

/// A trait that marks a type as valid for unaligned operations as an [`i16`].
pub trait Is16BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as an [`i32`].
pub trait Is32BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as an [`i64`].
pub trait Is64BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as a 128-bit
/// integer vector type such as [`__m128i`][x86] or [`v128`][wasm32].
///
/// [x86]: https://doc.rust-lang.org/stable/core/arch/x86/struct.__m128i.html
/// [wasm32]: https://doc.rust-lang.org/stable/core/arch/wasm32/struct.v128.html
pub trait Is128BitsUnaligned: private::Sealed {}

/// A trait that marks a type as valid for unaligned operations as a 256-bit
/// integer vector type such as [`__m256i`][x86].
///
/// [x86]: https://doc.rust-lang.org/stable/core/arch/x86/struct.__m256i.html
pub trait Is256BitsUnaligned: private::Sealed {}

////////////////////////////
// Start of `Cell` traits //
////////////////////////////

impl<T, const N: usize> private::Sealed for [core::cell::Cell<T>; N] where [T; N]: private::Sealed {}
impl<T, const N: usize> private::Sealed for core::cell::Cell<[T; N]> where [T; N]: private::Sealed {}

/// A trait that marks a cell-like type as valid for unaligned operations as an
/// [`i16`].
pub trait Is16CellUnaligned: private::Sealed {}

impl<T, const N: usize> Is16CellUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is16BitsUnaligned
{
}
impl<T, const N: usize> Is16CellUnaligned for core::cell::Cell<[T; N]> where
    [T; N]: Is16BitsUnaligned
{
}

/// A trait that marks a cell-like type as valid for unaligned operations as an
/// [`i32`].
pub trait Is32CellUnaligned: private::Sealed {}

impl<T, const N: usize> Is32CellUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is32BitsUnaligned
{
}
impl<T, const N: usize> Is32CellUnaligned for core::cell::Cell<[T; N]> where
    [T; N]: Is32BitsUnaligned
{
}

/// A trait that marks a cell-like type as valid for unaligned operations as an
/// [`i64`].
pub trait Is64CellUnaligned: private::Sealed {}

impl<T, const N: usize> Is64CellUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is64BitsUnaligned
{
}
impl<T, const N: usize> Is64CellUnaligned for core::cell::Cell<[T; N]> where
    [T; N]: Is64BitsUnaligned
{
}

/// A trait that marks a cell-like type as valid for unaligned operations as a
/// 128-bit integer vector type such as [`__m128i`][x86] or [`v128`][wasm32].
///
/// [x86]: https://doc.rust-lang.org/stable/core/arch/x86/struct.__m128i.html
/// [wasm32]: https://doc.rust-lang.org/stable/core/arch/wasm32/struct.v128.html
pub trait Is128CellUnaligned: private::Sealed {}

impl<T, const N: usize> Is128CellUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is128BitsUnaligned
{
}
impl<T, const N: usize> Is128CellUnaligned for core::cell::Cell<[T; N]> where
    [T; N]: Is128BitsUnaligned
{
}

/// A trait that marks a cell-like type as valid for unaligned operations as a
/// 256-bit integer vector type such as [`__m256i`][x86].
///
/// [x86]: https://doc.rust-lang.org/stable/core/arch/x86/struct.__m256i.html
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
    impl Is16CellUnaligned [i16] for {
        core::cell::Cell<u16>,
        core::cell::Cell<i16>,
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
    impl Is32CellUnaligned [i32] for {
        core::cell::Cell<u32>,
        core::cell::Cell<i32>,
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
    impl Is64CellUnaligned [i64] for {
        core::cell::Cell<u64>,
        core::cell::Cell<i64>,
    }
}

impl_N_bits_traits! {
    impl Is128BitsUnaligned [i128] for {
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
    impl Is256BitsUnaligned [[i128; 2]] for {
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

#[cfg(target_arch = "x86")]
use core::arch::x86::{__m128i, __m256i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{__m128i, __m256i};

// Sanity check:
// We define the 128/256-bit unaligned trait types in terms of `i128`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const _: () = assert!(size_of::<i128>() == size_of::<__m128i>());
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const _: () = assert!(size_of::<[i128; 2]>() == size_of::<__m256i>());

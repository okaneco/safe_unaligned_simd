//! Platform-specific intrinsics for `wasm32`.
use core::arch::wasm32::{self as arch, v128};
use core::ptr;

/// A marker trait of types valid for unaligned operations that operate on a single byte.
pub trait Is1ByteUnaligned: private::Sealed {}

/// A marker trait of types valid for unaligned operations that operate on two bytes.
pub trait Is2BytesUnaligned: private::Sealed {}

/// A marker trait of types valid for unaligned operations that operate on four bytes.
pub trait Is4BytesUnaligned: private::Sealed {}

/// A marker trait of types valid for unaligned operations that operate on eight bytes.
pub trait Is8BytesUnaligned: private::Sealed {}

/// A marker trait of types valid for unaligned operations that operate on sixteen bytes.
pub trait Is16BytesUnaligned: private::Sealed {}

// Internal module for sealing SIMD traits.
mod private {
    pub trait Sealed {}
}

/// Safe wrapper around [`arch::i16x8_load_extend_i8x8`].
#[target_feature(enable = "simd128")]
pub fn i16x8_load_extend_i8x8<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::i16x8_load_extend_i8x8(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::i16x8_load_extend_u8x8`].
#[target_feature(enable = "simd128")]
pub fn i16x8_load_extend_u8x8<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::i16x8_load_extend_u8x8(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::i32x4_load_extend_i16x4`].
#[target_feature(enable = "simd128")]
pub fn i32x4_load_extend_i16x4<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::i32x4_load_extend_i16x4(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::i32x4_load_extend_u16x4`].
#[target_feature(enable = "simd128")]
pub fn i32x4_load_extend_u16x4<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::i32x4_load_extend_u16x4(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::i64x2_load_extend_i32x2`].
#[target_feature(enable = "simd128")]
pub fn i64x2_load_extend_i32x2<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::i64x2_load_extend_i32x2(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::i64x2_load_extend_u32x2`].
#[target_feature(enable = "simd128")]
pub fn i64x2_load_extend_u32x2<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::i64x2_load_extend_u32x2(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::u16x8_load_extend_u8x8`].
#[target_feature(enable = "simd128")]
pub fn u16x8_load_extend_u8x8<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::u16x8_load_extend_u8x8(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::u32x4_load_extend_u16x4`].
#[target_feature(enable = "simd128")]
pub fn u32x4_load_extend_u16x4<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::u32x4_load_extend_u16x4(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::u64x2_load_extend_u32x2`].
#[target_feature(enable = "simd128")]
pub fn u64x2_load_extend_u32x2<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::u64x2_load_extend_u32x2(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_load`].
#[target_feature(enable = "simd128")]
pub fn v128_load<T: Is16BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::v128_load(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_load8_splat`].
#[target_feature(enable = "simd128")]
pub fn v128_load8_splat<T: Is1ByteUnaligned>(t: &T) -> v128 {
    unsafe { arch::v128_load8_splat(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_load16_splat`].
#[target_feature(enable = "simd128")]
pub fn v128_load16_splat<T: Is2BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::v128_load16_splat(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_load32_splat`].
#[target_feature(enable = "simd128")]
pub fn v128_load32_splat<T: Is4BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::v128_load32_splat(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_load32_zero`].
#[target_feature(enable = "simd128")]
pub fn v128_load32_zero<T: Is4BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::v128_load32_zero(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_load64_splat`].
#[target_feature(enable = "simd128")]
pub fn v128_load64_splat<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::v128_load64_splat(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_load64_zero`].
#[target_feature(enable = "simd128")]
pub fn v128_load64_zero<T: Is8BytesUnaligned>(t: &T) -> v128 {
    unsafe { arch::v128_load64_zero(ptr::from_ref(t).cast()) }
}

/// Safe wrapper around [`arch::v128_store`].
#[target_feature(enable = "simd128")]
pub fn v128_store<T: Is16BytesUnaligned>(t: &mut T, v: v128) {
    unsafe { arch::v128_store(ptr::from_mut(t).cast(), v) }
}

macro_rules! impl_N_bits_traits {
    (
        impl $trait:path [$N:literal] for {
            $($ty:ty,)*
        }
    ) => {
        $(
            const _: () =
                const { assert!(size_of::<$ty>() == $N) };

            impl private::Sealed for $ty {}
            impl $trait for $ty {}
        )*
    };
}

impl_N_bits_traits! {
    impl Is1ByteUnaligned [1] for {
        [u8; 1],
        [i8; 1],
        u8,
        i8,
    }
}

impl_N_bits_traits! {
    impl Is2BytesUnaligned [2] for {
        [u8; 2],
        [i8; 2],
        [u16; 1],
        [i16; 1],
        u16,
        i16,
    }
}

impl_N_bits_traits! {
    impl Is4BytesUnaligned [4] for {
        [u8; 4],
        [i8; 4],
        [u16; 2],
        [i16; 2],
        [u32; 1],
        [i32; 1],
        [f32; 1],
        u32,
        i32,
        f32,
    }
}

impl_N_bits_traits! {
    impl Is8BytesUnaligned [8] for {
        [u8; 8],
        [i8; 8],
        [u16; 4],
        [i16; 4],
        [u32; 2],
        [i32; 2],
        [f32; 2],
        [u64; 1],
        [i64; 1],
        [f64; 1],
        f64,
        u64,
        i64,
    }
}

impl<T, const N: usize> private::Sealed for [core::cell::Cell<T>; N] where [T; N]: private::Sealed {}
impl<T> private::Sealed for core::cell::Cell<T> where T: private::Sealed {}

// Mark cell types as valid for their respective sizes. The wasm32 platform has many more load
// operations than stores, so in contrast to other platforms, for brevity we implement the size
// traits for cells instead of bounding the inner / array types in a separate trait. This makes
// Cells redundant for stores (due to Cell::get_mut) but makes the list of exposed loads more
// terse.
//
// As we mark the cell type, and arrays of such cells, the specific underlying types must be chosen
// to avoid overlapping impls. This is a dance around the type checker as the many recursive impls
// have potential of infinitely recursing until the next generation. Hence we make sure to always
// reduce the matched type: both [Cell<T>; N] and Cell<T> take out a cell without introducing more
// types to the obligations. E.g. Cell<[T; N]> for [T: N]  would not work.

impl<T, const N: usize> Is1ByteUnaligned for [core::cell::Cell<T>; N] where [T; N]: Is1ByteUnaligned {}

impl<T> Is1ByteUnaligned for core::cell::Cell<T> where T: Is1ByteUnaligned {}

impl<T, const N: usize> Is2BytesUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is2BytesUnaligned
{
}

impl<T> Is2BytesUnaligned for core::cell::Cell<T> where T: Is2BytesUnaligned {}

impl<T, const N: usize> Is4BytesUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is4BytesUnaligned
{
}

impl<T> Is4BytesUnaligned for core::cell::Cell<T> where T: Is4BytesUnaligned {}

impl<T, const N: usize> Is8BytesUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is8BytesUnaligned
{
}

impl<T> Is8BytesUnaligned for core::cell::Cell<T> where T: Is8BytesUnaligned {}

impl<T, const N: usize> Is16BytesUnaligned for [core::cell::Cell<T>; N] where
    [T; N]: Is16BytesUnaligned
{
}

impl<T> Is16BytesUnaligned for core::cell::Cell<T> where T: Is16BytesUnaligned {}

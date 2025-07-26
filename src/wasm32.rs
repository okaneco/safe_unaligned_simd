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

impl_N_bits_traits! {
    impl Is16BytesUnaligned [16] for {
        [u8; 16],
        [i8; 16],
        [u16; 8],
        [i16; 8],
        [u32; 4],
        [i32; 4],
        [f32; 4],
        [u64; 2],
        [i64; 2],
        [f64; 2],
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

#[cfg(test)]
mod tests {
    use core::arch::wasm32::{self as arch, v128};

    fn assert_v128_bytes<const N: usize>(val: v128, data: &[[u8; N]]) {
        assert_eq!(
            unsafe { core::mem::transmute::<_, [u8; 16]>(val) },
            unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * N) }
        );

        // assert_eq!(arch::to_bytes(val), <[u8; N]>::flat(data))
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_i16x8_load_extend_i8x8() {
        #[target_feature(enable = "simd128")]
        fn test(a: [i8; 8]) {
            let v = super::i16x8_load_extend_i8x8(&a);
            assert_v128_bytes(v, &a.map(|i| (i as i16).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| i as i8);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    pub fn test_i16x8_load_extend_u8x8() {
        #[target_feature(enable = "simd128")]
        fn test(a: [u8; 8]) {
            let v = super::u16x8_load_extend_u8x8(&a);
            assert_v128_bytes(v, &a.map(|i| (i as i16).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| u8::MAX - i as u8);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_i32x4_load_extend_i16x4() {
        #[target_feature(enable = "simd128")]
        fn test(a: [i16; 4]) {
            let v = super::i32x4_load_extend_i16x4(&a);
            assert_v128_bytes(v, &a.map(|i| (i as i32).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| i as i16);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    pub fn test_i32x4_load_extend_u16x4() {
        #[target_feature(enable = "simd128")]
        fn test(a: [u16; 4]) {
            let v = super::u32x4_load_extend_u16x4(&a);
            assert_v128_bytes(v, &a.map(|i| (i as i32).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| u16::MAX - i as u16);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_i64x2_load_extend_i32x2() {
        #[target_feature(enable = "simd128")]
        fn test(a: [i32; 2]) {
            let v = super::i64x2_load_extend_i32x2(&a);
            assert_v128_bytes(v, &a.map(|i| (i as i64).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| i as i32);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    pub fn test_i64x2_load_extend_u32x2() {
        #[target_feature(enable = "simd128")]
        fn test(a: [u32; 2]) {
            let v = super::u64x2_load_extend_u32x2(&a);
            assert_v128_bytes(v, &a.map(|i| (i as i64).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| u32::MAX - i as u32);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    pub fn test_u16x8_load_extend_u8x8() {
        #[target_feature(enable = "simd128")]
        fn test(a: [u8; 8]) {
            let v = super::u16x8_load_extend_u8x8(&a);
            assert_v128_bytes(v, &a.map(|i| (i as u16).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| i as u8);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    pub fn test_u32x4_load_extend_u16x4() {
        #[target_feature(enable = "simd128")]
        fn test(a: [u16; 4]) {
            let v = super::u32x4_load_extend_u16x4(&a);
            assert_v128_bytes(v, &a.map(|i| (i as u32).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| i as u16);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    pub fn test_u64x2_load_extend_u32x2() {
        #[target_feature(enable = "simd128")]
        fn test(a: [u32; 2]) {
            let v = super::u64x2_load_extend_u32x2(&a);
            assert_v128_bytes(v, &a.map(|i| (i as u64).to_ne_bytes()));
        }

        let a = core::array::from_fn(|i| i as u32);
        test(a)
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_load() {
        #[target_feature(enable = "simd128")]
        fn test(a: &[i8; 16]) {
            let v = super::v128_load(a);
            let expected: [u8; 16] = core::array::from_fn(|i| 1 + i as u8);
            assert_v128_bytes(v, &[expected]);
        }

        let a: [i8; 17] = core::array::from_fn(|i| i as i8);
        // Try to unalign, even if our stack happens to be aligned.
        test(a[1..].try_into().unwrap())
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_load8_splat() {
        #[target_feature(enable = "simd128")]
        fn test(a: &i8) {
            let v = super::v128_load8_splat(a);
            let expected = [[42u8]; 16];
            assert_v128_bytes(v, &expected);
        }

        let a: [i8; 3] = [0, 42, 0];
        test(&a[1]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_load16_splat() {
        #[target_feature(enable = "simd128")]
        fn test(a: &u16) {
            let v = super::v128_load16_splat(a);
            let expected = [42u16.to_ne_bytes(); 8];
            assert_v128_bytes(v, &expected);
        }

        let a: [u16; 3] = [0, 42, 0];
        test(&a[1]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_load32_splat() {
        #[target_feature(enable = "simd128")]
        fn test(a: &u32) {
            let v = super::v128_load32_splat(a);
            let expected = [42u32.to_ne_bytes(); 4];
            assert_v128_bytes(v, &expected);
        }

        let a: [u32; 3] = [0, 42, 0];
        test(&a[1]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_load32_zero() {
        #[target_feature(enable = "simd128")]
        fn test(a: &u32) {
            let v = super::v128_load32_zero(a);

            let expected = [
                42u32.to_ne_bytes(),
                0u32.to_ne_bytes(),
                0u32.to_ne_bytes(),
                0u32.to_ne_bytes(),
            ];

            assert_v128_bytes(v, &expected);
        }

        let a: [u32; 3] = [0, 42, 0];
        test(&a[1]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_load64_splat() {
        #[target_feature(enable = "simd128")]
        fn test(a: &[u8; 8]) {
            let v = super::v128_load64_splat(a);
            let expected = [42u64.to_ne_bytes(); 2];
            assert_v128_bytes(v, &expected);
        }

        let a: [_; 3] = [[0; 8], 42u64.to_ne_bytes(), [0; 8]];
        test(&a[1]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_load64_zero() {
        #[target_feature(enable = "simd128")]
        fn test(a: &u64) {
            let v = super::v128_load64_zero(a);
            let expected = [42u64.to_ne_bytes(), 0u64.to_ne_bytes()];
            assert_v128_bytes(v, &expected);
        }

        let a: [u64; 3] = [0, 42, 0];
        test(&a[1]);
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_store_i8() {
        #[target_feature(enable = "simd128")]
        fn test() {
            let mut into = [42i8; 16];
            let v = arch::u8x16_splat(1);
            super::v128_store(&mut into, v);
            assert_eq!(into, [1i8; 16]);
        }

        test()
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_store_u16() {
        #[target_feature(enable = "simd128")]
        fn test() {
            let mut into = [42u16; 8];
            let v = arch::u16x8_splat(1);
            super::v128_store(&mut into, v);
            assert_eq!(into, [1u16; 8]);
        }

        test()
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_store_i32() {
        #[target_feature(enable = "simd128")]
        fn test() {
            let mut into = [42i32; 4];
            let v = arch::i32x4_splat(1);
            super::v128_store(&mut into, v);
            assert_eq!(into, [1i32; 4]);
        }

        test()
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_store_u64() {
        #[target_feature(enable = "simd128")]
        fn test() {
            let mut into = [42u64; 2];
            let v = arch::u64x2_splat(1);
            super::v128_store(&mut into, v);
            assert_eq!(into, [1u64; 2]);
        }

        test()
    }

    #[test]
    #[cfg_attr(not(target_feature = "simd128"), ignore)]
    fn test_v128_store_f64() {
        #[target_feature(enable = "simd128")]
        fn test() {
            let mut into = [42f64; 2];
            let v = arch::f64x2_splat(1.0);
            super::v128_store(&mut into, v);
            assert_eq!(into, [1.0f64; 2]);
        }

        test()
    }
}

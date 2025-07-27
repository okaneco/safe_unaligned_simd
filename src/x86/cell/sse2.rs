#[cfg(target_arch = "x86")]
use core::arch::x86::{self as arch, __m128i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{self as arch, __m128i};
use core::ptr;

#[cfg(target_arch = "x86")]
use crate::x86::{Is16CellUnaligned, Is32CellUnaligned, Is64CellUnaligned, Is128CellUnaligned};
#[cfg(target_arch = "x86_64")]
use crate::x86_64::{Is16CellUnaligned, Is32CellUnaligned, Is64CellUnaligned, Is128CellUnaligned};

/// Loads a 64-bit integer from memory into first element of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_epi64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadl_epi64<T: Is128CellUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadl_epi64(ptr::from_ref(mem_addr).cast()) }
}

/// Loads 128-bits of integer data from memory into a new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si128)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si128<T: Is128CellUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_si128(ptr::from_ref(mem_addr).cast_mut().cast()) }
}

/// Loads unaligned 16-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si16)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si16<T: Is16CellUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_si16(ptr::from_ref(mem_addr).cast()) }
}

/// Loads unaligned 32-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si32)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si32<T: Is32CellUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_si32(ptr::from_ref(mem_addr).cast()) }
}

/// Loads unaligned 64-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si64<T: Is64CellUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_si64(ptr::from_ref(mem_addr).cast()) }
}

/// Stores the lower 64-bit integer `a` to a memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storel_epi64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storel_epi64<T: Is128CellUnaligned>(mem_addr: &T, a: __m128i) {
    unsafe { arch::_mm_storel_epi64(ptr::from_ref(mem_addr).cast_mut().cast(), a) }
}

/// Stores 128-bits of integer data from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si128)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si128<T: Is128CellUnaligned>(mem_addr: &T, a: __m128i) {
    unsafe { arch::_mm_storeu_si128(ptr::from_ref(mem_addr).cast_mut().cast(), a) }
}

/// Store 16-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si16)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si16<T: Is16CellUnaligned>(mem_addr: &T, a: __m128i) {
    unsafe { arch::_mm_storeu_si16(ptr::from_ref(mem_addr).cast_mut().cast(), a) }
}

/// Store 32-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si32)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si32<T: Is32CellUnaligned>(mem_addr: &T, a: __m128i) {
    unsafe { arch::_mm_storeu_si32(ptr::from_ref(mem_addr).cast_mut().cast(), a) }
}

/// Store 64-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si64<T: Is64CellUnaligned>(mem_addr: &T, a: __m128i) {
    unsafe { arch::_mm_storeu_si64(ptr::from_ref(mem_addr).cast_mut().cast(), a) }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128i};

    use core::{array, cell::Cell};

    // SAFETY: The `x86_64` target baseline includes `sse` and `sse2`.

    fn assert_eq_m128i(a: __m128i, b: __m128i) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    #[test]
    fn test_mm_loadl_epi64() {
        let mut a = [20, 25];
        unsafe { test(&mut a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &mut [i64; 2]) {
            let val = Cell::from_mut(a);
            let r = super::_mm_loadl_epi64(val);
            let target = arch::_mm_set_epi64x(0, 20);

            assert_eq_m128i(r, target)
        }
    }

    #[test]
    fn test_mm_storel_epi64() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_set_epi64x(i64::MIN, i64::MAX);
            let mut x = [0; 2];
            let val = Cell::from_mut(&mut x);
            super::_mm_storel_epi64(val, a);

            assert_eq!(x[0], i64::MAX);
        }
    }

    macro_rules! test_loadu_storeu_siXYZ {
        ($testname:ident, $loadu:ident, $storeu:ident, [$target:ty] for $($source:ty,)*) => {
            #[test]
            fn $testname() {
                $({
                    let mut a = array::from_fn(|i| i as $source);
                    unsafe { test(&mut a) }

                    const LOAD_INDEX: usize = 1;
                    const STORE_INDEX: usize = 3;
                    const ARRAY_SIZE: usize = size_of::<$target>() + STORE_INDEX;

                    #[target_feature(enable = "sse2")]
                    fn test(a: &mut [$source; ARRAY_SIZE]) {
                        // Number of sources that fit within $target
                        const N: usize = size_of::<$target>() / size_of::<$source>();

                        // The equivalent scalar operation of this SIMD load and store
                        let mut result = a.clone();
                        result.copy_within(LOAD_INDEX..LOAD_INDEX + N, STORE_INDEX);

                        let val = Cell::from_mut(&mut a[..]).as_slice_of_cells();

                        let load: &[_; N] = val[LOAD_INDEX..][..N].try_into().unwrap();
                        let store: &[_; N] = val[STORE_INDEX..][..N].try_into().unwrap();

                        let r = super::$loadu(load);
                        super::$storeu(store, r);

                        assert_eq!(*a, result);
                    }
                })*
            }
        };
    }

    macro_rules! test_loadu_storeu_siXYZ_scalar {
        ($testname:ident, $loadu:ident, $storeu:ident, [$target:ty] for $($source:ty,)*) => {
            #[test]
            fn $testname() {
                $({
                    unsafe { test_scalar() }

                    #[target_feature(enable = "sse2")]
                    fn test_scalar() {
                        let mut a: $source = 1;

                        const NUM: $source = (<$source>::MAX).wrapping_add(100);

                        let load = Cell::from(NUM);
                        let store = Cell::from_mut(&mut a);

                        let r = super::$loadu(&load);
                        super::$storeu(store, r);

                        assert_eq!(a, NUM);
                    }
                })*
            }
        };
    }

    // loadu_si16 and storeu_si16 variants

    test_loadu_storeu_siXYZ!(
        test_mm_loadu_si16,
        _mm_loadu_si16,
        _mm_storeu_si16,
        [i16] for
        u8,
        i8,
        u16,
        i16,
    );

    test_loadu_storeu_siXYZ_scalar!(
        test_mm_loadu_si16_scalar,
        _mm_loadu_si16,
        _mm_storeu_si16,
        [i16] for
        u16,
        i16,
    );

    // loadu_si32 and storeu_si32 variants

    test_loadu_storeu_siXYZ!(
        test_mm_loadu_si32,
        _mm_loadu_si32,
        _mm_storeu_si32,
        [i32] for
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
    );

    test_loadu_storeu_siXYZ_scalar!(
        test_mm_loadu_si32_scalar,
        _mm_loadu_si32,
        _mm_storeu_si32,
        [i32] for
        u32,
        i32,
    );

    // loadu_si64 and storeu_si64 variants

    test_loadu_storeu_siXYZ!(
        test_mm_loadu_si64,
        _mm_loadu_si64,
        _mm_storeu_si64,
        [i64] for
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
    );

    test_loadu_storeu_siXYZ_scalar!(
        test_mm_loadu_si64_scalar,
        _mm_loadu_si64,
        _mm_storeu_si64,
        [i64] for
        u64,
        i64,
    );

    // loadu_si128 and storeu_si128 variants

    test_loadu_storeu_siXYZ!(
        test_mm_loadu_si128,
        _mm_loadu_si128,
        _mm_storeu_si128,
        [__m128i] for
        u8,
        i8,
        u16,
        i16,
        u32,
        i32,
        u64,
        i64,
    );
}

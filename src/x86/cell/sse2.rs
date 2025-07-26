#[cfg(target_arch = "x86")]
use core::arch::x86::{self as arch, __m128i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{self as arch, __m128i};
use core::ptr;

#[cfg(target_arch = "x86")]
use crate::x86::Is128CellUnaligned;
#[cfg(target_arch = "x86_64")]
use crate::x86_64::Is128CellUnaligned;

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

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128i};

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
            let val = core::cell::Cell::from_mut(a);
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
            let val = core::cell::Cell::from_mut(&mut x);
            super::_mm_storel_epi64(val, a);

            assert_eq!(x[0], i64::MAX);
        }
    }

    #[test]
    fn test_mm_storeu_si128_epi32() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let mut a = [0u32, 1, 2, 3, 4];
            let val = core::cell::Cell::from_mut(&mut a[..]).as_slice_of_cells();

            let load: &[_; 4] = val[..4].try_into().unwrap();
            let store: &[_; 4] = val[1..].try_into().unwrap();

            let r = super::_mm_loadu_si128(load);
            super::_mm_storeu_si128(store, r);

            assert_eq!(a, [0, 0, 1, 2, 3]);
        }
    }

    #[test]
    fn test_mm_loadu_si128_i64() {
        let mut a = [-1, -2, 0];
        unsafe { test(&mut a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &mut [i64; 3]) {
            let val = core::cell::Cell::from_mut(&mut a[..]).as_slice_of_cells();

            let load: &[_; 2] = val[..2].try_into().unwrap();
            let store: &[_; 2] = val[1..].try_into().unwrap();

            let r = super::_mm_loadu_si128(load);
            super::_mm_storeu_si128(store, r);

            assert_eq!(*a, [-1, -1, -2]);
        }
    }
}

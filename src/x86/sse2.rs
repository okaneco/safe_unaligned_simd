#[cfg(target_arch = "x86")]
use core::arch::x86::{self as arch, __m128d, __m128i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{self as arch, __m128d, __m128i};
use core::{cell, ptr};

#[cfg(target_arch = "x86")]
use crate::x86::Is128BitsUnaligned;
#[cfg(target_arch = "x86_64")]
use crate::x86_64::Is128BitsUnaligned;

/// Loads a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_pd1)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_load_pd1(mem_addr: &f64) -> __m128d {
    _mm_load1_pd(mem_addr)
}

/// Loads a 64-bit double-precision value to the low element of a
/// 128-bit integer vector and clears the upper element.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_sd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_load_sd(mem_addr: &f64) -> __m128d {
    unsafe { arch::_mm_load_sd(mem_addr) }
}

/// Loads a double-precision (64-bit) floating-point element from memory
/// into both elements of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load1_pd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_load1_pd(mem_addr: &f64) -> __m128d {
    unsafe { arch::_mm_load1_pd(mem_addr) }
}

/// Loads a double-precision value into the high-order bits of a 128-bit
/// vector of `[2 x double]`. The low-order bits are copied from the low-order
/// bits of the first operand.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadh_pd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadh_pd(a: __m128d, mem_addr: &f64) -> __m128d {
    unsafe { arch::_mm_loadh_pd(a, mem_addr) }
}

/// Loads a 64-bit integer from memory into first element of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_epi64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadl_epi64<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadl_epi64(ptr::from_ref(mem_addr).cast()) }
}

/// Loads a 64-bit integer from memory into first element of returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_epi64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadl_epi64_cell<T: Is128BitsUnaligned>(mem_addr: &cell::Cell<T>) -> __m128i {
    unsafe { arch::_mm_loadl_epi64(ptr::from_ref(mem_addr).cast()) }
}

/// Loads a double-precision value into the low-order bits of a 128-bit
/// vector of `[2 x double]`. The high-order bits are copied from the
/// high-order bits of the first operand.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadl_pd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadl_pd(a: __m128d, mem_addr: &f64) -> __m128d {
    unsafe { arch::_mm_loadl_pd(a, mem_addr) }
}

/// Loads 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory into the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_pd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_pd(mem_addr: &[f64; 2]) -> __m128d {
    unsafe { arch::_mm_loadu_pd(mem_addr.as_ptr()) }
}

/// Loads 128-bits of integer data from memory into a new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si128)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si128<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_si128(ptr::from_ref(mem_addr).cast()) }
}

/// Loads 128-bits of integer data from memory into a new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si128)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si128_cell<T: Is128BitsUnaligned>(mem_addr: &cell::Cell<T>) -> __m128i {
    unsafe { arch::_mm_loadu_si128(ptr::from_ref(mem_addr).cast()) }
}

/// Loads unaligned 16-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si16)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si16(mem_addr: &[u8; 2]) -> __m128i {
    unsafe { arch::_mm_loadu_si16(mem_addr.as_ptr()) }
}

/// Loads unaligned 32-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si32)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si32(mem_addr: &[u8; 4]) -> __m128i {
    unsafe { arch::_mm_loadu_si32(mem_addr.as_ptr()) }
}

/// Loads unaligned 64-bits of integer data from memory into new vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_si64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_loadu_si64(mem_addr: &[u8; 8]) -> __m128i {
    unsafe { arch::_mm_loadu_si64(mem_addr.as_ptr()) }
}

/// Stores the lower 64 bits of a 128-bit vector of `[2 x double]` to a
/// memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store_sd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_store_sd(mem_addr: &mut f64, a: __m128d) {
    unsafe { arch::_mm_store_sd(ptr::from_mut(mem_addr), a) }
}

/// Stores the upper 64 bits of a 128-bit vector of `[2 x double]` to a
/// memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeh_pd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeh_pd(mem_addr: &mut f64, a: __m128d) {
    unsafe { arch::_mm_storeh_pd(ptr::from_mut(mem_addr), a) }
}

/// Stores the lower 64-bit integer `a` to a memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storel_epi64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storel_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
    unsafe { arch::_mm_storel_epi64(ptr::from_mut(mem_addr).cast(), a) }
}

/// Stores the lower 64-bit integer `a` to a memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storel_epi64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storel_epi64_cell<T: Is128BitsUnaligned>(mem_addr: &cell::Cell<T>, a: __m128i) {
    unsafe { arch::_mm_storel_epi64(mem_addr.as_ptr().cast(), a) }
}

/// Stores the lower 64 bits of a 128-bit vector of `[2 x double]` to a
/// memory location.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storel_pd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storel_pd(mem_addr: &mut f64, a: __m128d) {
    unsafe { arch::_mm_storel_pd(ptr::from_mut(mem_addr), a) }
}

/// Stores 128-bits (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_pd)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_pd(mem_addr: &mut [f64; 2], a: __m128d) {
    unsafe { arch::_mm_storeu_pd(mem_addr.as_mut_ptr().cast(), a) }
}

/// Stores 128-bits of integer data from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si128)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si128<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
    unsafe { arch::_mm_storeu_si128(ptr::from_mut(mem_addr).cast(), a) }
}

/// Stores 128-bits of integer data from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si128)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si128_cell<T: Is128BitsUnaligned>(mem_addr: &cell::Cell<T>, a: __m128i) {
    unsafe { arch::_mm_storeu_si128(mem_addr.as_ptr().cast(), a) }
}

/// Store 16-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si16)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si16(mem_addr: &mut [u8; 2], a: __m128i) {
    unsafe { arch::_mm_storeu_si16(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store 32-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si32)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si32(mem_addr: &mut [u8; 4], a: __m128i) {
    unsafe { arch::_mm_storeu_si32(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store 64-bit integer from the first element of `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_si64)
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_storeu_si64(mem_addr: &mut [u8; 8], a: __m128i) {
    unsafe { arch::_mm_storeu_si64(ptr::from_mut(mem_addr).cast(), a) }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128d, __m128i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128d, __m128i};

    // SAFETY: The `x86_64` target baseline includes `sse` and `sse2`.

    fn assert_eq_m128d(a: __m128d, b: __m128d) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m128i(a: __m128i, b: __m128i) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    #[test]
    fn test_mm_load1_pd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = 10.0;
            let r = super::_mm_load1_pd(&a);
            let target = arch::_mm_setr_pd(a, a);

            assert_eq_m128d(r, target);
        }
    }

    #[test]
    fn test_mm_load_sd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = 10.0;
            let r = super::_mm_load_sd(&a);
            let target = arch::_mm_setr_pd(a, 0.0);

            assert_eq_m128d(r, target)
        }
    }

    #[test]
    fn test_mm_loadh_pd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = 10.0;
            let b = arch::_mm_setr_pd(1.0, 2.0);
            let r = super::_mm_loadh_pd(b, &a);
            let target = arch::_mm_setr_pd(1.0, 10.0);

            assert_eq_m128d(r, target)
        }
    }

    #[test]
    fn test_mm_loadl_epi64() {
        let a = [20, 25];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[i64; 2]) {
            let r = super::_mm_loadl_epi64(a);
            let target = arch::_mm_set_epi64x(0, 20);

            assert_eq_m128i(r, target)
        }
    }

    #[test]
    fn test_mm_loadl_pd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = 10.0;
            let b = arch::_mm_setr_pd(1.0, 2.0);
            let r = super::_mm_loadl_pd(b, &a);
            let target = arch::_mm_setr_pd(10.0, 2.0);

            assert_eq_m128d(r, target)
        }
    }

    #[test]
    fn test_mm_loadu_pd() {
        let a = [1.0, 2.0];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[f64; 2]) {
            let r = super::_mm_loadu_pd(a);
            let target = arch::_mm_setr_pd(1.0, 2.0);

            assert_eq_m128d(r, target)
        }
    }

    #[test]
    fn test_mm_loadu_si16() {
        let a = [1, 2];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[u8; 2]) {
            let r = super::_mm_loadu_si16(a);
            let target = arch::_mm_setr_epi16(i16::from_le_bytes(*a), 0, 0, 0, 0, 0, 0, 0);

            assert_eq_m128i(r, target)
        }
    }

    #[test]
    fn test_mm_loadu_si32() {
        let a = [1, 2, 3, 4];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[u8; 4]) {
            let r = super::_mm_loadu_si32(a);
            let target = arch::_mm_setr_epi32(i32::from_le_bytes(*a), 0, 0, 0);

            assert_eq_m128i(r, target)
        }
    }

    #[test]
    fn test_mm_loadu_si64() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[u8; 8]) {
            let r = super::_mm_loadu_si64(a);
            let target = arch::_mm_set_epi64x(0, i64::from_le_bytes(*a));

            assert_eq_m128i(r, target)
        }
    }

    #[test]
    fn test_mm_store_sd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_setr_pd(1.0, 2.0);
            let mut x = 0.0;
            super::_mm_store_sd(&mut x, a);

            assert_eq!(x, 1.0);
        }
    }

    #[test]
    fn test_mm_storeh_pd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_setr_pd(1.0, 2.0);
            let mut x = 0.0;
            super::_mm_storeh_pd(&mut x, a);

            assert_eq!(x, 2.0);
        }
    }

    #[test]
    fn test_mm_storel_epi64() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_set_epi64x(i64::MIN, i64::MAX);
            let mut x = [0; 2];
            super::_mm_storel_epi64(&mut x, a);

            assert_eq!(x[0], i64::MAX);
        }
    }

    #[test]
    fn test_mm_storel_pd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_setr_pd(1.0, 2.0);
            let mut x = 0.0;
            super::_mm_storel_pd(&mut x, a);

            assert_eq!(x, 1.0);
        }
    }

    #[test]
    fn test_mm_storeu_pd() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_setr_pd(1.0, 2.0);
            let mut x = [0.0; 2];
            super::_mm_storeu_pd(&mut x, a);

            assert_eq!(x, [1.0, 2.0]);
        }
    }

    #[test]
    fn test_mm_storeu_si16() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
            let mut x = [0; 2];
            super::_mm_storeu_si16(&mut x, a);

            assert_eq!(u16::from_le_bytes(x), 1);
        }
    }

    #[test]
    fn test_mm_storeu_si32() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_setr_epi32(1, 2, 3, 4);
            let mut x = [0; 4];
            super::_mm_storeu_si32(&mut x, a);

            assert_eq!(u32::from_le_bytes(x), 1);
        }
    }

    #[test]
    fn test_mm_storeu_si64() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = arch::_mm_set_epi64x(2, 1);
            let mut x = [0; 8];
            super::_mm_storeu_si64(&mut x, a);

            assert_eq!(u64::from_le_bytes(x), 1);
        }
    }

    // `_mm_loadu_si128` family
    //
    // Test all 8 implementations of `Is128BitsUnaligned`.
    #[test]
    fn test_mm_loadu_si128_u8() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[u8; 16]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

            assert_eq_m128i(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_si128_i8() {
        let a = [
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        ];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[i8; 16]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_setr_epi8(
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
            );

            assert_eq_m128i(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_si128_u16() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[u16; 8]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);

            assert_eq_m128i(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_si128_i16() {
        let a = [-1, -2, -3, -4, -5, -6, -7, -8];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[i16; 8]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_setr_epi16(-1, -2, -3, -4, -5, -6, -7, -8);

            assert_eq_m128i(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_si128_u32() {
        let a = [1, 2, 3, 4];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[u32; 4]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_setr_epi32(1, 2, 3, 4);

            assert_eq_m128i(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_si128_i32() {
        let a = [-1, -2, -3, -4];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[i32; 4]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_setr_epi32(-1, -2, -3, -4);

            assert_eq_m128i(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_si128_u64() {
        let a = [1, 2];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[u64; 2]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_set_epi64x(2, 1);

            assert_eq_m128i(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_si128_i64() {
        let a = [-1, -2];
        unsafe { test(&a) }

        #[target_feature(enable = "sse2")]
        fn test(a: &[i64; 2]) {
            let r = super::_mm_loadu_si128(a);
            let target = arch::_mm_set_epi64x(-2, -1);

            assert_eq_m128i(r, target);
        }
    }

    // `_mm_storeu_si128` family
    //
    // Test all 8 implementations of `Is128BitsUnaligned`.
    #[test]
    fn test_mm_storeu_si128_u8() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let r = arch::_mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

            let mut x: [u8; 16] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            assert_eq!(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        }
    }

    #[test]
    fn test_mm_storeu_si128_i8() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let r = arch::_mm_setr_epi8(
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
            );

            let mut x: [i8; 16] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            let a = [
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
            ];
            assert_eq!(x, a);
        }
    }

    #[test]
    fn test_mm_storeu_si128_u16() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let r = arch::_mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);

            let mut x: [u16; 8] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            assert_eq!(x, [1, 2, 3, 4, 5, 6, 7, 8]);
        }
    }

    #[test]
    fn test_mm_storeu_si128_i16() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let r = arch::_mm_setr_epi16(-1, -2, -3, -4, -5, -6, -7, -8);

            let mut x: [i16; 8] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            assert_eq!(x, [-1, -2, -3, -4, -5, -6, -7, -8]);
        }
    }

    #[test]
    fn test_mm_storeu_si128_u32() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = [1, 2, 3, 4];
            let r = arch::_mm_setr_epi32(1, 2, 3, 4);

            let mut x: [u32; 4] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            assert_eq!(x, a);
        }
    }

    #[test]
    fn test_mm_storeu_si128_i32() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let r = arch::_mm_setr_epi32(-1, -2, -3, -4);

            let mut x: [i32; 4] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            assert_eq!(x, [-1, -2, -3, -4]);
        }
    }

    #[test]
    fn test_mm_storeu_si128_u64() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = [1, 2];
            let r = arch::_mm_set_epi64x(2, 1);

            let mut x: [u64; 2] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            assert_eq!(x, a);
        }
    }

    #[test]
    fn test_mm_storeu_si128_i64() {
        unsafe { test() }

        #[target_feature(enable = "sse2")]
        fn test() {
            let a = [-1, -2];
            let r = arch::_mm_set_epi64x(-2, -1);

            let mut x: [i64; 2] = core::array::from_fn(|_| !0);
            super::_mm_storeu_si128(&mut x, r);

            assert_eq!(x, a);
        }
    }
}

#[cfg(target_arch = "x86")]
use core::arch::x86::{self as arch, __m128, __m128d, __m256, __m256d, __m256i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{self as arch, __m128, __m128d, __m256, __m256d, __m256i};
use core::ptr;

#[cfg(target_arch = "x86")]
use crate::x86::{Is128BitsUnaligned, Is256BitsUnaligned};
#[cfg(target_arch = "x86_64")]
use crate::x86_64::{Is128BitsUnaligned, Is256BitsUnaligned};

/// Broadcasts 128 bits from memory (composed of 2 packed double-precision
/// (64-bit) floating-point elements) to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_pd)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_broadcast_pd(mem_addr: &__m128d) -> __m256d {
    // FIXME: Remove unsafe blocks when MSRV includes the safe version
    #[allow(unused_unsafe)]
    unsafe {
        arch::_mm256_broadcast_pd(mem_addr)
    }
}

/// Broadcasts 128 bits from memory (composed of 4 packed single-precision
/// (32-bit) floating-point elements) to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_ps)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_broadcast_ps(mem_addr: &__m128) -> __m256 {
    // FIXME: Remove unsafe blocks when MSRV includes the safe version
    #[allow(unused_unsafe)]
    unsafe {
        arch::_mm256_broadcast_ps(mem_addr)
    }
}

/// Broadcasts a double-precision (64-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_sd)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_broadcast_sd(mem_addr: &f64) -> __m256d {
    // FIXME: Remove unsafe blocks when MSRV includes the safe version
    #[allow(unused_unsafe)]
    unsafe {
        arch::_mm256_broadcast_sd(mem_addr)
    }
}

/// Broadcasts a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_broadcast_ss)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm_broadcast_ss(mem_addr: &f32) -> __m128 {
    // FIXME: Remove unsafe blocks when MSRV includes the safe version
    #[allow(unused_unsafe)]
    unsafe {
        arch::_mm_broadcast_ss(mem_addr)
    }
}

/// Broadcasts a single-precision (32-bit) floating-point element from memory
/// to all elements of the returned vector.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_broadcast_ss)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_broadcast_ss(mem_addr: &f32) -> __m256 {
    // FIXME: Remove unsafe blocks when MSRV includes the safe version
    #[allow(unused_unsafe)]
    unsafe {
        arch::_mm256_broadcast_ss(mem_addr)
    }
}

/// Loads 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_pd)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu_pd(mem_addr: &[f64; 4]) -> __m256d {
    unsafe { arch::_mm256_loadu_pd(mem_addr.as_ptr().cast()) }
}

/// Loads 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_ps)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu_ps(mem_addr: &[f32; 8]) -> __m256 {
    unsafe { arch::_mm256_loadu_ps(mem_addr.as_ptr().cast()) }
}

/// Loads 256-bits of integer data from memory into result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_si256)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu_si256<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu_si256(ptr::from_ref(mem_addr).cast()) }
}

/// Loads two 128-bit values (composed of 4 packed single-precision (32-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu2_m128)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu2_m128(hiaddr: &[f32; 4], loaddr: &[f32; 4]) -> __m256 {
    unsafe { arch::_mm256_loadu2_m128(hiaddr.as_ptr().cast(), loaddr.as_ptr().cast()) }
}

/// Loads two 128-bit values (composed of 2 packed double-precision (64-bit)
/// floating-point elements) from memory, and combine them into a 256-bit
/// value.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu2_m128d)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu2_m128d(hiaddr: &[f64; 2], loaddr: &[f64; 2]) -> __m256d {
    unsafe { arch::_mm256_loadu2_m128d(hiaddr.as_ptr().cast(), loaddr.as_ptr().cast()) }
}

/// Loads two 128-bit values (composed of integer data) from memory, and combine
/// them into a 256-bit value.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu2_m128i)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu2_m128i<T: Is128BitsUnaligned>(hiaddr: &T, loaddr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu2_m128i(ptr::from_ref(hiaddr).cast(), ptr::from_ref(loaddr).cast()) }
}

/// Stores 256-bits (composed of 4 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_pd)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu_pd(mem_addr: &mut [f64; 4], a: __m256d) {
    unsafe { arch::_mm256_storeu_pd(mem_addr.as_mut_ptr().cast(), a) }
}

/// Stores 256-bits (composed of 8 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_ps)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu_ps(mem_addr: &mut [f32; 8], a: __m256) {
    unsafe { arch::_mm256_storeu_ps(mem_addr.as_mut_ptr().cast(), a) }
}

/// Stores 256-bits of integer data from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_si256)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu_si256<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
    unsafe { arch::_mm256_storeu_si256(ptr::from_mut(mem_addr).cast(), a) }
}

/// Stores the high and low 128-bit halves (each composed of 4 packed
/// single-precision (32-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu2_m128)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu2_m128(hiaddr: &mut [f32; 4], loaddr: &mut [f32; 4], a: __m256) {
    unsafe { arch::_mm256_storeu2_m128(hiaddr.as_mut_ptr().cast(), loaddr.as_mut_ptr().cast(), a) }
}

/// Stores the high and low 128-bit halves (each composed of 2 packed
/// double-precision (64-bit) floating-point elements) from `a` into memory two
/// different 128-bit locations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu2_m128d)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu2_m128d(hiaddr: &mut [f64; 2], loaddr: &mut [f64; 2], a: __m256d) {
    unsafe { arch::_mm256_storeu2_m128d(hiaddr.as_mut_ptr().cast(), loaddr.as_mut_ptr().cast(), a) }
}

/// Stores the high and low 128-bit halves (each composed of integer data) from
/// `a` into memory two different 128-bit locations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu2_m128i)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu2_m128i<T: Is128BitsUnaligned>(hiaddr: &mut T, loaddr: &mut T, a: __m256i) {
    unsafe {
        arch::_mm256_storeu2_m128i(
            ptr::from_mut(hiaddr).cast(),
            ptr::from_mut(loaddr).cast(),
            a,
        )
    }
}

#[cfg(feature = "_avx_test")]
#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128, __m256, __m256d, __m256i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128, __m256, __m256d, __m256i};

    // Fail-safe for tests being run on a CPU that doesn't support `avx`
    static CPU_HAS_AVX: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| is_x86_feature_detected!("avx"));

    fn assert_eq_m256(a: __m256, b: __m256) {
        let a: [u8; 32] = unsafe { core::mem::transmute(a) };
        let b: [u8; 32] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m256d(a: __m256d, b: __m256d) {
        let a: [u8; 32] = unsafe { core::mem::transmute(a) };
        let b: [u8; 32] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m256i(a: __m256i, b: __m256i) {
        let a: [u8; 32] = unsafe { core::mem::transmute(a) };
        let b: [u8; 32] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m128(a: __m128, b: __m128) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    #[test]
    fn test_mm256_broadcast_pd() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let a = arch::_mm_setr_pd(4.0, 3.0);
            let r = super::_mm256_broadcast_pd(&a);
            let target = arch::_mm256_setr_pd(4.0, 3.0, 4.0, 3.0);

            assert_eq_m256d(r, target);
        }
    }

    #[test]
    fn test_mm256_broadcast_ps() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let a = arch::_mm_setr_ps(4.0, 3.0, 2.0, 5.0);
            let r = super::_mm256_broadcast_ps(&a);
            let target = arch::_mm256_setr_ps(4.0, 3.0, 2.0, 5.0, 4.0, 3.0, 2.0, 5.0);

            assert_eq_m256(r, target);
        }
    }

    #[test]
    fn test_mm256_broadcast_sd() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = super::_mm256_broadcast_sd(&3.0);
            let target = arch::_mm256_set1_pd(3.0);

            assert_eq_m256d(r, target);
        }
    }

    #[test]
    fn test_mm_broadcast_ss() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = super::_mm_broadcast_ss(&3.0);
            let target = arch::_mm_set1_ps(3.0);

            assert_eq_m128(r, target);
        }
    }

    #[test]
    fn test_mm256_broadcast_ss() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = super::_mm256_broadcast_ss(&3.0);
            let target = arch::_mm256_set1_ps(3.0);

            assert_eq_m256(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_pd() {
        assert!(*CPU_HAS_AVX);

        let a = [1.0, 2.0, 3.0, 4.0];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[f64; 4]) {
            let r = super::_mm256_loadu_pd(a);
            let target = arch::_mm256_setr_pd(1.0, 2.0, 3.0, 4.0);

            assert_eq_m256d(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_ps() {
        assert!(*CPU_HAS_AVX);

        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[f32; 8]) {
            let r = super::_mm256_loadu_ps(a);
            let target = arch::_mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

            assert_eq_m256(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu2_m128() {
        assert!(*CPU_HAS_AVX);

        let lo = [1.0, 2.0, 3.0, 4.0];
        let hi = [5.0, 6.0, 7.0, 8.0];
        unsafe { test(&hi, &lo) }

        #[target_feature(enable = "avx")]
        fn test(hi: &[f32; 4], lo: &[f32; 4]) {
            let r = super::_mm256_loadu2_m128(hi, lo);
            let target = arch::_mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

            assert_eq_m256(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu2_m128d() {
        assert!(*CPU_HAS_AVX);

        let lo = [1.0, 2.0];
        let hi = [3.0, 4.0];
        unsafe { test(&hi, &lo) }

        #[target_feature(enable = "avx")]
        fn test(hi: &[f64; 2], lo: &[f64; 2]) {
            let r = super::_mm256_loadu2_m128d(hi, lo);
            let target = arch::_mm256_setr_pd(1.0, 2.0, 3.0, 4.0);

            assert_eq_m256d(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu2_m128i() {
        assert!(*CPU_HAS_AVX);

        let lo = [10, 20];
        let hi = [30, 40];
        unsafe { test(&hi, &lo) }

        #[target_feature(enable = "avx")]
        fn test(hi: &[u64; 2], lo: &[u64; 2]) {
            let r = super::_mm256_loadu2_m128i(hi, lo);
            let target = arch::_mm256_setr_epi64x(10, 20, 30, 40);

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_storeu_pd() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let a = arch::_mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
            let mut x = [0.0; 4];
            super::_mm256_storeu_pd(&mut x, a);

            assert_eq!(x, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_mm256_storeu_ps() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let a = arch::_mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let mut x = [0.0; 8];
            super::_mm256_storeu_ps(&mut x, a);

            assert_eq!(x, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        }
    }

    #[test]
    fn test_mm256_storeu2_m128() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let mut hi = [0.0; 4];
            let mut lo = [0.0; 4];
            let a = arch::_mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            super::_mm256_storeu2_m128(&mut hi, &mut lo, a);

            assert_eq!(hi, [5.0, 6.0, 7.0, 8.0]);
            assert_eq!(lo, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_mm256_storeu2_m128d() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let mut hi = [0.0; 2];
            let mut lo = [0.0; 2];
            let a = arch::_mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
            super::_mm256_storeu2_m128d(&mut hi, &mut lo, a);

            assert_eq!(hi, [3.0, 4.0]);
            assert_eq!(lo, [1.0, 2.0]);
        }
    }

    #[test]
    fn test_mm256_storeu2_m128i() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let mut hi: [i64; 2] = [0; 2];
            let mut lo: [i64; 2] = [0; 2];
            let a = arch::_mm256_setr_epi64x(10, 20, 30, 40);
            super::_mm256_storeu2_m128i(&mut hi, &mut lo, a);

            assert_eq!(hi, [30, 40]);
            assert_eq!(lo, [10, 20]);
        }
    }

    // `_mm_loadu_si256` family
    //
    // Test all 8 implementations of `Is256BitsUnaligned`.
    #[test]
    fn test_mm256_loadu_si256_u8() {
        assert!(*CPU_HAS_AVX);

        let a = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[u8; 32]) {
            let r = super::_mm256_loadu_si256(a);
            let target = arch::_mm256_setr_epi8(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            );

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_si256_i8() {
        assert!(*CPU_HAS_AVX);

        let a = [
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19,
            -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32,
        ];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[i8; 32]) {
            let r = super::_mm256_loadu_si256(a);
            let target = arch::_mm256_setr_epi8(
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18,
                -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32,
            );

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_si256_u16() {
        assert!(*CPU_HAS_AVX);

        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[u16; 16]) {
            let r = super::_mm256_loadu_si256(a);
            let target =
                arch::_mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_si256_i16() {
        assert!(*CPU_HAS_AVX);

        let a = [
            -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
        ];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[i16; 16]) {
            let r = super::_mm256_loadu_si256(a);
            let target = arch::_mm256_setr_epi16(
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
            );

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_si256_u32() {
        assert!(*CPU_HAS_AVX);

        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[u32; 8]) {
            let r = super::_mm256_loadu_si256(a);
            let target = arch::_mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_si256_i32() {
        assert!(*CPU_HAS_AVX);

        let a = [-1, -2, -3, -4, -5, -6, -7, -8];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[i32; 8]) {
            let r = super::_mm256_loadu_si256(a);
            let target = arch::_mm256_setr_epi32(-1, -2, -3, -4, -5, -6, -7, -8);

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_si256_u64() {
        assert!(*CPU_HAS_AVX);

        let a = [1, 2, 3, 4];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[u64; 4]) {
            let r = super::_mm256_loadu_si256(a);
            let target = arch::_mm256_setr_epi64x(1, 2, 3, 4);

            assert_eq_m256i(r, target);
        }
    }

    #[test]
    fn test_mm256_loadu_si256_i64() {
        assert!(*CPU_HAS_AVX);

        let a = [-1, -2, -3, -4];
        unsafe { test(&a) }

        #[target_feature(enable = "avx")]
        fn test(a: &[i64; 4]) {
            let r = super::_mm256_loadu_si256(a);
            let target = arch::_mm256_setr_epi64x(-1, -2, -3, -4);

            assert_eq_m256i(r, target);
        }
    }

    // `_mm_storeu_si256` family
    //
    // Test all 8 implementations of `Is256BitsUnaligned`.
    #[test]
    fn test_mm256_storeu_si256_u8() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi8(
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            );

            let mut x: [u8; 32] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(
                x,
                [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                ]
            );
        }
    }

    #[test]
    fn test_mm256_storeu_si256_i8() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi8(
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18,
                -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32,
            );

            let mut x: [i8; 32] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(
                x,
                [
                    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17,
                    -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32,
                ]
            );
        }
    }

    #[test]
    fn test_mm256_storeu_si256_u16() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

            let mut x: [u16; 16] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(x, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        }
    }

    #[test]
    fn test_mm256_storeu_si256_i16() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi16(
                -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16,
            );

            let mut x: [i16; 16] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(
                x,
                [
                    -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16
                ]
            );
        }
    }

    #[test]
    fn test_mm256_storeu_si256_u32() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);

            let mut x: [u32; 8] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(x, [1, 2, 3, 4, 5, 6, 7, 8]);
        }
    }

    #[test]
    fn test_mm256_storeu_si256_i32() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi32(-1, -2, -3, -4, -5, -6, -7, -8);

            let mut x: [i32; 8] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(x, [-1, -2, -3, -4, -5, -6, -7, -8]);
        }
    }

    #[test]
    fn test_mm256_storeu_si256_u64() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi64x(1, 2, 3, 4);

            let mut x: [u64; 4] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(x, [1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_mm256_storeu_si256_i64() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let r = arch::_mm256_setr_epi64x(-1, -2, -3, -4);

            let mut x: [i64; 4] = core::array::from_fn(|_| !0);
            super::_mm256_storeu_si256(&mut x, r);

            assert_eq!(x, [-1, -2, -3, -4]);
        }
    }
}

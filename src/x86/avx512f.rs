#[cfg(target_arch = "x86")]
use core::arch::x86::{
    self as arch, __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
    __mmask8, __mmask16,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    self as arch, __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
    __mmask8, __mmask16,
};
use core::ptr;

#[cfg(target_arch = "x86")]
use crate::x86::{Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned};
#[cfg(target_arch = "x86_64")]
use crate::x86_64::{Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned};

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_expandloadu_epi32<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_expandloadu_epi32(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_expandloadu_epi32<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
    _mm_mask_expandloadu_epi32(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_expandloadu_epi32<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_expandloadu_epi32(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_expandloadu_epi32<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
    _mm256_mask_expandloadu_epi32(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_expandloadu_epi32<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask16,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_expandloadu_epi32(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 32-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_expandloadu_epi32<T: Is512BitsUnaligned>(
    k: __mmask16,
    mem_addr: &T,
) -> __m512i {
    _mm512_mask_expandloadu_epi32(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_expandloadu_epi64<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_expandloadu_epi64(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_expandloadu_epi64<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
    _mm_mask_expandloadu_epi64(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_expandloadu_epi64<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_expandloadu_epi64(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_expandloadu_epi64<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
    _mm256_mask_expandloadu_epi64(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_expandloadu_epi64<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask8,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_expandloadu_epi64(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 64-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_expandloadu_epi64<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m512i {
    _mm512_mask_expandloadu_epi64(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_expandloadu_pd(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
    unsafe { arch::_mm_mask_expandloadu_pd(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
    _mm_mask_expandloadu_pd(arch::_mm_setzero_pd(), k, mem_addr)
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_expandloadu_pd(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
    unsafe { arch::_mm256_mask_expandloadu_pd(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
    _mm256_mask_expandloadu_pd(arch::_mm256_setzero_pd(), k, mem_addr)
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_expandloadu_pd(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    unsafe { arch::_mm512_mask_expandloadu_pd(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active double-precision (64-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_expandloadu_pd(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    _mm512_mask_expandloadu_pd(arch::_mm512_setzero_pd(), k, mem_addr)
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_expandloadu_ps(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
    unsafe { arch::_mm_mask_expandloadu_ps(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_expandloadu_ps(k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
    _mm_mask_expandloadu_ps(arch::_mm_setzero_ps(), k, mem_addr)
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_expandloadu_ps(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
    unsafe { arch::_mm256_mask_expandloadu_ps(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_expandloadu_ps(k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
    _mm256_mask_expandloadu_ps(arch::_mm256_setzero_ps(), k, mem_addr)
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_expandloadu_ps(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    unsafe { arch::_mm512_mask_expandloadu_ps(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active single-precision (32-bit) floating-point elements from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_expandloadu_ps(k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    _mm512_mask_expandloadu_ps(arch::_mm512_setzero_ps(), k, mem_addr)
}

#[cfg(feature = "avx512f")]
#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{
        self as arch, __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
    };
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{
        self as arch, __m128, __m128d, __m128i, __m256, __m256d, __m256i, __m512, __m512d, __m512i,
    };

    use core::hint::black_box;

    // Fail-safe for tests being run on a CPU that doesn't support the instruction set
    static CPU_HAS_AVX512VL: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| is_x86_feature_detected!("avx512vl"));

    fn assert_eq_m128(a: __m128, b: __m128) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

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

    fn assert_eq_m512(a: __m512, b: __m512) {
        let a: [u8; 64] = unsafe { core::mem::transmute(a) };
        let b: [u8; 64] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m512d(a: __m512d, b: __m512d) {
        let a: [u8; 64] = unsafe { core::mem::transmute(a) };
        let b: [u8; 64] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m512i(a: __m512i, b: __m512i) {
        let a: [u8; 64] = unsafe { core::mem::transmute(a) };
        let b: [u8; 64] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_expandloadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi32(42);
            let a = &[1_i32, 2, 3, 4];
            let m = 0b11111000;
            let r = super::_mm_mask_expandloadu_epi32(src, m, black_box(a));
            let e = arch::_mm_set_epi32(1, 42, 42, 42);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_expandloadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i32, 2, 3, 4];
            let m = 0b11111000;
            let r = super::_mm_maskz_expandloadu_epi32(m, black_box(a));
            let e = arch::_mm_set_epi32(1, 0, 0, 0);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_expandloadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi32(42);
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11101000;
            let r = super::_mm256_mask_expandloadu_epi32(src, m, black_box(a));
            let e = arch::_mm256_set_epi32(4, 3, 2, 42, 1, 42, 42, 42);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_expandloadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11101000;
            let r = super::_mm256_maskz_expandloadu_epi32(m, black_box(a));
            let e = arch::_mm256_set_epi32(4, 3, 2, 0, 1, 0, 0, 0);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_expandloadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_epi32(42);
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm512_mask_expandloadu_epi32(src, m, black_box(a));
            let e = arch::_mm512_set_epi32(8, 7, 6, 42, 5, 42, 42, 42, 4, 3, 42, 42, 2, 42, 1, 42);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_expandloadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm512_maskz_expandloadu_epi32(m, black_box(a));
            let e = arch::_mm512_set_epi32(8, 7, 6, 0, 5, 0, 0, 0, 4, 3, 0, 0, 2, 0, 1, 0);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_expandloadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi64x(42);
            let a = &[1_i64, 2];
            let m = 0b11101000;
            let r = super::_mm_mask_expandloadu_epi64(src, m, black_box(a));
            let e = arch::_mm_set_epi64x(42, 42);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_expandloadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i64, 2];
            let m = 0b11101000;
            let r = super::_mm_maskz_expandloadu_epi64(m, black_box(a));
            let e = arch::_mm_set_epi64x(0, 0);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_expandloadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi64x(42);
            let a = &[1_i64, 2, 3, 4];
            let m = 0b11101000;
            let r = super::_mm256_mask_expandloadu_epi64(src, m, black_box(a));
            let e = arch::_mm256_set_epi64x(1, 42, 42, 42);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_expandloadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i64, 2, 3, 4];
            let m = 0b11101000;
            let r = super::_mm256_maskz_expandloadu_epi64(m, black_box(a));
            let e = arch::_mm256_set_epi64x(1, 0, 0, 0);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_expandloadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_epi64(42);
            let a = &[1_i64, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11101000;
            let r = super::_mm512_mask_expandloadu_epi64(src, m, black_box(a));
            let e = arch::_mm512_set_epi64(4, 3, 2, 42, 1, 42, 42, 42);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_expandloadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[1_i64, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11101000;
            let r = super::_mm512_maskz_expandloadu_epi64(m, black_box(a));
            let e = arch::_mm512_set_epi64(4, 3, 2, 0, 1, 0, 0, 0);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_expandloadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_pd(42.);
            let a = &[1.0f64, 2.];
            let m = 0b11101000;
            let r = super::_mm_mask_expandloadu_pd(src, m, black_box(a));
            let e = arch::_mm_set_pd(42., 42.);
            assert_eq_m128d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_expandloadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0f64, 2.];
            let m = 0b11101000;
            let r = super::_mm_maskz_expandloadu_pd(m, black_box(a));
            let e = arch::_mm_set_pd(0., 0.);
            assert_eq_m128d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_expandloadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_pd(42.);
            let a = &[1.0f64, 2., 3., 4.];
            let m = 0b11101000;
            let r = super::_mm256_mask_expandloadu_pd(src, m, black_box(a));
            let e = arch::_mm256_set_pd(1., 42., 42., 42.);
            assert_eq_m256d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_expandloadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0f64, 2., 3., 4.];
            let m = 0b11101000;
            let r = super::_mm256_maskz_expandloadu_pd(m, black_box(a));
            let e = arch::_mm256_set_pd(1., 0., 0., 0.);
            assert_eq_m256d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_expandloadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_pd(42.);
            let a = &[1.0f64, 2., 3., 4., 5., 6., 7., 8.];
            let m = 0b11101000;
            let r = super::_mm512_mask_expandloadu_pd(src, m, black_box(a));
            let e = arch::_mm512_set_pd(4., 3., 2., 42., 1., 42., 42., 42.);
            assert_eq_m512d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_expandloadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[1.0f64, 2., 3., 4., 5., 6., 7., 8.];
            let m = 0b11101000;
            let r = super::_mm512_maskz_expandloadu_pd(m, black_box(a));
            let e = arch::_mm512_set_pd(4., 3., 2., 0., 1., 0., 0., 0.);
            assert_eq_m512d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_expandloadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_ps(42.);
            let a = &[1.0f32, 2., 3., 4.];
            let m = 0b11101000;
            let r = super::_mm_mask_expandloadu_ps(src, m, black_box(a));
            let e = arch::_mm_set_ps(1., 42., 42., 42.);
            assert_eq_m128(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_expandloadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0f32, 2., 3., 4.];
            let m = 0b11101000;
            let r = super::_mm_maskz_expandloadu_ps(m, black_box(a));
            let e = arch::_mm_set_ps(1., 0., 0., 0.);
            assert_eq_m128(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_expandloadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_ps(42.);
            let a = &[1.0f32, 2., 3., 4., 5., 6., 7., 8.];
            let m = 0b11101000;
            let r = super::_mm256_mask_expandloadu_ps(src, m, black_box(a));
            let e = arch::_mm256_set_ps(4., 3., 2., 42., 1., 42., 42., 42.);
            assert_eq_m256(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_expandloadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0f32, 2., 3., 4., 5., 6., 7., 8.];
            let m = 0b11101000;
            let r = super::_mm256_maskz_expandloadu_ps(m, black_box(a));
            let e = arch::_mm256_set_ps(4., 3., 2., 0., 1., 0., 0., 0.);
            assert_eq_m256(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_expandloadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_ps(42.);
            let a = &[
                1.0f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ];
            let m = 0b11101000_11001010;
            let r = super::_mm512_mask_expandloadu_ps(src, m, black_box(a));
            let e = arch::_mm512_set_ps(
                8., 7., 6., 42., 5., 42., 42., 42., 4., 3., 42., 42., 2., 42., 1., 42.,
            );
            assert_eq_m512(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_expandloadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[
                1.0f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            ];
            let m = 0b11101000_11001010;
            let r = super::_mm512_maskz_expandloadu_ps(m, black_box(a));
            let e = arch::_mm512_set_ps(
                8., 7., 6., 0., 5., 0., 0., 0., 4., 3., 0., 0., 2., 0., 1., 0.,
            );
            assert_eq_m512(r, e);
        }
    }
}

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    self as arch, __m128i, __m256i, __m512i, __mmask8, __mmask16, __mmask32, __mmask64,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    self as arch, __m128i, __m256i, __m512i, __mmask8, __mmask16, __mmask32, __mmask64,
};
use core::ptr;

#[cfg(target_arch = "x86")]
use crate::x86::{Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned};
#[cfg(target_arch = "x86_64")]
use crate::x86_64::{Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned};

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm_mask_expandloadu_epi16<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_expandloadu_epi16(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm_maskz_expandloadu_epi16<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
    _mm_mask_expandloadu_epi16(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm256_mask_expandloadu_epi16<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask16,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_expandloadu_epi16(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm256_maskz_expandloadu_epi16<T: Is256BitsUnaligned>(
    k: __mmask16,
    mem_addr: &T,
) -> __m256i {
    _mm256_mask_expandloadu_epi16(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
pub fn _mm512_mask_expandloadu_epi16<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask32,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_expandloadu_epi16(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 16-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
pub fn _mm512_maskz_expandloadu_epi16<T: Is512BitsUnaligned>(
    k: __mmask32,
    mem_addr: &T,
) -> __m512i {
    _mm512_mask_expandloadu_epi16(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm_mask_expandloadu_epi8<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask16,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_expandloadu_epi8(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm_maskz_expandloadu_epi8<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i {
    _mm_mask_expandloadu_epi8(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm256_mask_expandloadu_epi8<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask32,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_expandloadu_epi8(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm256_maskz_expandloadu_epi8<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i {
    _mm256_mask_expandloadu_epi8(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
pub fn _mm512_mask_expandloadu_epi8<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask64,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_expandloadu_epi8(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load contiguous active 8-bit integers from unaligned memory at mem_addr (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_expandloadu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
pub fn _mm512_maskz_expandloadu_epi8<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: &T) -> __m512i {
    _mm512_mask_expandloadu_epi8(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm_mask_compressstoreu_epi16<T: Is128BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    unsafe { arch::_mm_mask_compressstoreu_epi16(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm256_mask_compressstoreu_epi16<T: Is256BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    unsafe { arch::_mm256_mask_compressstoreu_epi16(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi16)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
pub fn _mm512_mask_compressstoreu_epi16<T: Is512BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask32,
    a: __m512i,
) {
    unsafe { arch::_mm512_mask_compressstoreu_epi16(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm_mask_compressstoreu_epi8<T: Is128BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask16,
    a: __m128i,
) {
    unsafe { arch::_mm_mask_compressstoreu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
pub fn _mm256_mask_compressstoreu_epi8<T: Is256BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask32,
    a: __m256i,
) {
    unsafe { arch::_mm256_mask_compressstoreu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi8)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
pub fn _mm512_mask_compressstoreu_epi8<T: Is512BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask64,
    a: __m512i,
) {
    unsafe { arch::_mm512_mask_compressstoreu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128i, __m256i, __m512i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128i, __m256i, __m512i};

    use core::hint::black_box;

    // Fail-safe for tests being run on a CPU that doesn't support the instruction set
    static CPU_HAS_AVX512VBMI2: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| is_x86_feature_detected!("avx512vbmi2"));

    fn assert_eq_m128i(a: __m128i, b: __m128i) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m256i(a: __m256i, b: __m256i) {
        let a: [u8; 32] = unsafe { core::mem::transmute(a) };
        let b: [u8; 32] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    fn assert_eq_m512i(a: __m512i, b: __m512i) {
        let a: [u8; 64] = unsafe { core::mem::transmute(a) };
        let b: [u8; 64] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_expandloadu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi16(42);
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11101000;
            let r = super::_mm_mask_expandloadu_epi16(src, m, black_box(a));
            let e = arch::_mm_set_epi16(4, 3, 2, 42, 1, 42, 42, 42);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_expandloadu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11101000;
            let r = super::_mm_maskz_expandloadu_epi16(m, black_box(a));
            let e = arch::_mm_set_epi16(4, 3, 2, 0, 1, 0, 0, 0);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_expandloadu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi16(42);
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm256_mask_expandloadu_epi16(src, m, black_box(a));
            let e = arch::_mm256_set_epi16(8, 7, 6, 42, 5, 42, 42, 42, 4, 3, 42, 42, 2, 42, 1, 42);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_expandloadu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm256_maskz_expandloadu_epi16(m, black_box(a));
            let e = arch::_mm256_set_epi16(8, 7, 6, 0, 5, 0, 0, 0, 4, 3, 0, 0, 2, 0, 1, 0);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_expandloadu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2")]
        fn test() {
            let src = arch::_mm512_set1_epi16(42);
            let a = &[
                1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b11101000_11001010_11110000_00001111;
            let r = super::_mm512_mask_expandloadu_epi16(src, m, black_box(a));
            let e = arch::_mm512_set_epi16(
                16, 15, 14, 42, 13, 42, 42, 42, 12, 11, 42, 42, 10, 42, 9, 42, 8, 7, 6, 5, 42, 42,
                42, 42, 42, 42, 42, 42, 4, 3, 2, 1,
            );
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_expandloadu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2")]
        fn test() {
            let a = &[
                1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b11101000_11001010_11110000_00001111;
            let r = super::_mm512_maskz_expandloadu_epi16(m, black_box(a));
            let e = arch::_mm512_set_epi16(
                16, 15, 14, 0, 13, 0, 0, 0, 12, 11, 0, 0, 10, 0, 9, 0, 8, 7, 6, 5, 0, 0, 0, 0, 0,
                0, 0, 0, 4, 3, 2, 1,
            );
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_expandloadu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi8(42);
            let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm_mask_expandloadu_epi8(src, m, black_box(a));
            let e = arch::_mm_set_epi8(8, 7, 6, 42, 5, 42, 42, 42, 4, 3, 42, 42, 2, 42, 1, 42);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_expandloadu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm_maskz_expandloadu_epi8(m, black_box(a));
            let e = arch::_mm_set_epi8(8, 7, 6, 0, 5, 0, 0, 0, 4, 3, 0, 0, 2, 0, 1, 0);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_expandloadu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi8(42);
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b11101000_11001010_11110000_00001111;
            let r = super::_mm256_mask_expandloadu_epi8(src, m, black_box(a));
            let e = arch::_mm256_set_epi8(
                16, 15, 14, 42, 13, 42, 42, 42, 12, 11, 42, 42, 10, 42, 9, 42, 8, 7, 6, 5, 42, 42,
                42, 42, 42, 42, 42, 42, 4, 3, 2, 1,
            );
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_expandloadu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b11101000_11001010_11110000_00001111;
            let r = super::_mm256_maskz_expandloadu_epi8(m, black_box(a));
            let e = arch::_mm256_set_epi8(
                16, 15, 14, 0, 13, 0, 0, 0, 12, 11, 0, 0, 10, 0, 9, 0, 8, 7, 6, 5, 0, 0, 0, 0, 0,
                0, 0, 0, 4, 3, 2, 1,
            );
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_expandloadu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2")]
        fn test() {
            let src = arch::_mm512_set1_epi8(42);
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            ];
            let m = 0b11101000_11001010_11110000_00001111_11111111_00000000_10101010_01010101;
            let r = super::_mm512_mask_expandloadu_epi8(src, m, black_box(a));
            let e = arch::_mm512_set_epi8(
                32, 31, 30, 42, 29, 42, 42, 42, 28, 27, 42, 42, 26, 42, 25, 42, 24, 23, 22, 21, 42,
                42, 42, 42, 42, 42, 42, 42, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 42, 42,
                42, 42, 42, 42, 42, 42, 8, 42, 7, 42, 6, 42, 5, 42, 42, 4, 42, 3, 42, 2, 42, 1,
            );
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_expandloadu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2")]
        fn test() {
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            ];
            let m = 0b11101000_11001010_11110000_00001111_11111111_00000000_10101010_01010101;
            let r = super::_mm512_maskz_expandloadu_epi8(m, black_box(a));
            let e = arch::_mm512_set_epi8(
                32, 31, 30, 0, 29, 0, 0, 0, 28, 27, 0, 0, 26, 0, 25, 0, 24, 23, 22, 21, 0, 0, 0, 0,
                0, 0, 0, 0, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 0, 0, 0, 0, 0, 0, 0, 0,
                8, 0, 7, 0, 6, 0, 5, 0, 0, 4, 0, 3, 0, 2, 0, 1,
            );
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_compressstoreu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = arch::_mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
            let mut r = [0_i16; 8];
            super::_mm_mask_compressstoreu_epi16(&mut r, 0, a);
            assert_eq!(&r, &[0_i16; 8]);
            super::_mm_mask_compressstoreu_epi16(&mut r, 0b11110000, a);
            assert_eq!(&r, &[5, 6, 7, 8, 0, 0, 0, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_compressstoreu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = arch::_mm256_set_epi16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            let mut r = [0_i16; 16];
            super::_mm256_mask_compressstoreu_epi16(&mut r, 0, a);
            assert_eq!(&r, &[0_i16; 16]);
            super::_mm256_mask_compressstoreu_epi16(&mut r, 0b11110000_11001010, a);
            assert_eq!(&r, &[2, 4, 7, 8, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_compressstoreu_epi16() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2")]
        fn test() {
            let a = arch::_mm512_set_epi16(
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            );
            let mut r = [0_i16; 32];
            super::_mm512_mask_compressstoreu_epi16(&mut r, 0, a);
            assert_eq!(&r, &[0_i16; 32]);
            super::_mm512_mask_compressstoreu_epi16(
                &mut r,
                0b11110000_11001010_11111111_00000000,
                a,
            );
            assert_eq!(
                &r,
                &[
                    9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 24, 29, 30, 31, 32, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_compressstoreu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = arch::_mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            let mut r = [0_i8; 16];
            super::_mm_mask_compressstoreu_epi8(&mut r, 0, a);
            assert_eq!(&r, &[0_i8; 16]);
            super::_mm_mask_compressstoreu_epi8(&mut r, 0b11110000_11001010, a);
            assert_eq!(&r, &[2, 4, 7, 8, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_compressstoreu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2,avx512vl")]
        fn test() {
            let a = arch::_mm256_set_epi8(
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            );
            let mut r = [0_i8; 32];
            super::_mm256_mask_compressstoreu_epi8(&mut r, 0, a);
            assert_eq!(&r, &[0_i8; 32]);
            super::_mm256_mask_compressstoreu_epi8(
                &mut r,
                0b11110000_11001010_11111111_00000000,
                a,
            );
            assert_eq!(
                &r,
                &[
                    9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 23, 24, 29, 30, 31, 32, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_compressstoreu_epi8() {
        assert!(*CPU_HAS_AVX512VBMI2);
        unsafe { test() }

        #[target_feature(enable = "avx512vbmi2")]
        fn test() {
            let a = arch::_mm512_set_epi8(
                64, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44,
                43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23,
                22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            );
            let mut r = [0_i8; 64];
            super::_mm512_mask_compressstoreu_epi8(&mut r, 0, a);
            assert_eq!(&r, &[0_i8; 64]);
            super::_mm512_mask_compressstoreu_epi8(
                &mut r,
                0b11110000_11001010_11111111_00000000_10101010_01010101_11110000_00001111,
                a,
            );
            assert_eq!(
                &r,
                &[
                    1, 2, 3, 4, 13, 14, 15, 16, 17, 19, 21, 23, 26, 28, 30, 32, 41, 42, 43, 44, 45,
                    46, 47, 48, 50, 52, 55, 56, 61, 62, 63, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                ]
            );
        }
    }
}

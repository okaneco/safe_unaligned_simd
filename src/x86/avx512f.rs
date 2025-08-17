#[cfg(target_arch = "x86")]
use core::arch::x86::{self as arch, __m128i, __mmask8};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{self as arch, __m128i, __mmask8};
use core::ptr;

#[cfg(target_arch = "x86")]
use crate::x86::Is128BitsUnaligned;
#[cfg(target_arch = "x86_64")]
use crate::x86_64::Is128BitsUnaligned;

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

#[cfg(feature = "avx512f")]
#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128i};

    use core::hint::black_box;

    // Fail-safe for tests being run on a CPU that doesn't support the instructions
    static CPU_HAS_AVX512VL: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| is_x86_feature_detected!("avx512vl"));

    fn assert_eq_m128i(a: __m128i, b: __m128i) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
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
}

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

/// Load 128-bits (composed of 4 packed 32-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_loadu_epi32<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_epi32(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 32-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_loadu_epi32<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_loadu_epi32(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 32-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_loadu_epi32<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
    _mm_mask_loadu_epi32(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load 256-bits (composed of 8 packed 32-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_loadu_epi32<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu_epi32(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 32-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_loadu_epi32<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_loadu_epi32(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 32-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_loadu_epi32<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
    _mm256_mask_loadu_epi32(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load 512-bits (composed of 16 packed 32-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_loadu_epi32<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i {
    unsafe { arch::_mm512_loadu_epi32(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 32-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_loadu_epi32<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask16,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_loadu_epi32(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 32-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_loadu_epi32<T: Is512BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m512i {
    _mm512_mask_loadu_epi32(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Load 128-bits (composed of 2 packed 64-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_loadu_epi64<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_epi64(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 64-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_loadu_epi64<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_loadu_epi64(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 64-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_loadu_epi64<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
    _mm_mask_loadu_epi64(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load 256-bits (composed of 4 packed 64-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_loadu_epi64<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu_epi64(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 64-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_loadu_epi64<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask8,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_loadu_epi64(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 64-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_loadu_epi64<T: Is256BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m256i {
    _mm256_mask_loadu_epi64(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load 512-bits (composed of 8 packed 64-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_loadu_epi64<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i {
    unsafe { arch::_mm512_loadu_epi64(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 64-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_loadu_epi64<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask8,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_loadu_epi64(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 64-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_loadu_epi64<T: Is512BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m512i {
    _mm512_mask_loadu_epi64(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_loadu_pd(src: __m128d, k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
    unsafe { arch::_mm_mask_loadu_pd(src, k, mem_addr.as_ptr()) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 2]) -> __m128d {
    _mm_mask_loadu_pd(arch::_mm_setzero_pd(), k, mem_addr)
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_loadu_pd(src: __m256d, k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
    unsafe { arch::_mm256_mask_loadu_pd(src, k, mem_addr.as_ptr()) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 4]) -> __m256d {
    _mm256_mask_loadu_pd(arch::_mm256_setzero_pd(), k, mem_addr)
}

/// Loads 512-bits (composed of 8 packed double-precision (64-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_loadu_pd(mem_addr: &[f64; 8]) -> __m512d {
    unsafe { arch::_mm512_loadu_pd(mem_addr.as_ptr()) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_loadu_pd(src: __m512d, k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    unsafe { arch::_mm512_mask_loadu_pd(src, k, mem_addr.as_ptr()) }
}

/// Load packed double-precision (64-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_loadu_pd(k: __mmask8, mem_addr: &[f64; 8]) -> __m512d {
    _mm512_mask_loadu_pd(arch::_mm512_setzero_pd(), k, mem_addr)
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_loadu_ps(src: __m128, k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
    unsafe { arch::_mm_mask_loadu_ps(src, k, mem_addr.as_ptr()) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_maskz_loadu_ps(k: __mmask8, mem_addr: &[f32; 4]) -> __m128 {
    _mm_mask_loadu_ps(arch::_mm_setzero_ps(), k, mem_addr)
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_loadu_ps(src: __m256, k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
    unsafe { arch::_mm256_mask_loadu_ps(src, k, mem_addr.as_ptr()) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_maskz_loadu_ps(k: __mmask8, mem_addr: &[f32; 8]) -> __m256 {
    _mm256_mask_loadu_ps(arch::_mm256_setzero_ps(), k, mem_addr)
}

/// Loads 512-bits (composed of 16 packed single-precision (32-bit)
/// floating-point elements) from memory into result.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_loadu_ps(mem_addr: &[f32; 16]) -> __m512 {
    unsafe { arch::_mm512_loadu_ps(mem_addr.as_ptr()) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_loadu_ps(src: __m512, k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    unsafe { arch::_mm512_mask_loadu_ps(src, k, mem_addr.as_ptr()) }
}

/// Load packed single-precision (32-bit) floating-point elements from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_maskz_loadu_ps(k: __mmask16, mem_addr: &[f32; 16]) -> __m512 {
    _mm512_mask_loadu_ps(arch::_mm512_setzero_ps(), k, mem_addr)
}

/// Load 512-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_si512)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_loadu_si512<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i {
    unsafe { arch::_mm512_loadu_si512(ptr::from_ref(mem_addr).cast()) }
}

// Store intrinsics

/// Contiguously store the active 32-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_compressstoreu_epi32<T: Is128BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    unsafe { arch::_mm_mask_compressstoreu_epi32(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 32-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_compressstoreu_epi32<T: Is256BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    unsafe { arch::_mm256_mask_compressstoreu_epi32(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 32-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_compressstoreu_epi32<T: Is512BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask16,
    a: __m512i,
) {
    unsafe { arch::_mm512_mask_compressstoreu_epi32(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 64-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_compressstoreu_epi64<T: Is128BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    unsafe { arch::_mm_mask_compressstoreu_epi64(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 64-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_compressstoreu_epi64<T: Is256BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m256i,
) {
    unsafe { arch::_mm256_mask_compressstoreu_epi64(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active 64-bit integers in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_compressstoreu_epi64<T: Is512BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m512i,
) {
    unsafe { arch::_mm512_mask_compressstoreu_epi64(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Contiguously store the active double-precision (64-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_compressstoreu_pd(base_addr: &mut [f64; 2], k: __mmask8, a: __m128d) {
    unsafe { arch::_mm_mask_compressstoreu_pd(base_addr.as_mut_ptr().cast(), k, a) }
}

/// Contiguously store the active double-precision (64-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_compressstoreu_pd(base_addr: &mut [f64; 4], k: __mmask8, a: __m256d) {
    unsafe { arch::_mm256_mask_compressstoreu_pd(base_addr.as_mut_ptr().cast(), k, a) }
}

/// Contiguously store the active double-precision (64-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_compressstoreu_pd(base_addr: &mut [f64; 8], k: __mmask8, a: __m512d) {
    unsafe { arch::_mm512_mask_compressstoreu_pd(base_addr.as_mut_ptr().cast(), k, a) }
}

/// Contiguously store the active single-precision (32-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_compressstoreu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_compressstoreu_ps(base_addr: &mut [f32; 4], k: __mmask8, a: __m128) {
    unsafe { arch::_mm_mask_compressstoreu_ps(base_addr.as_mut_ptr().cast(), k, a) }
}

/// Contiguously store the active single-precision (32-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_compressstoreu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_compressstoreu_ps(base_addr: &mut [f32; 8], k: __mmask8, a: __m256) {
    unsafe { arch::_mm256_mask_compressstoreu_ps(base_addr.as_mut_ptr().cast(), k, a) }
}

/// Contiguously store the active single-precision (32-bit) floating-point elements in a (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_compressstoreu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_compressstoreu_ps(base_addr: &mut [f32; 16], k: __mmask16, a: __m512) {
    unsafe { arch::_mm512_mask_compressstoreu_ps(base_addr.as_mut_ptr().cast(), k, a) }
}

/// Store packed 32-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_storeu_epi32<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) {
    unsafe { arch::_mm_mask_storeu_epi32(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 128-bits (composed of 4 packed 32-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_storeu_epi32<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
    unsafe { arch::_mm_storeu_epi32(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 32-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_storeu_epi32<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) {
    unsafe { arch::_mm256_mask_storeu_epi32(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 256-bits (composed of 8 packed 32-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi32)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_storeu_epi32<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
    unsafe { arch::_mm256_storeu_epi32(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 32-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_storeu_epi32<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m512i) {
    unsafe { arch::_mm512_mask_storeu_epi32(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 512-bits (composed of 16 packed 32-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_storeu_epi32<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
    unsafe { arch::_mm512_storeu_epi32(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 64-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_storeu_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) {
    unsafe { arch::_mm_mask_storeu_epi64(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 128-bits (composed of 2 packed 64-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_storeu_epi64<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
    unsafe { arch::_mm_storeu_epi64(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 64-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_storeu_epi64<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m256i) {
    unsafe { arch::_mm256_mask_storeu_epi64(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 256-bits (composed of 4 packed 64-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi64)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_storeu_epi64<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
    unsafe { arch::_mm256_storeu_epi64(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 64-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_storeu_epi64<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m512i) {
    unsafe { arch::_mm512_mask_storeu_epi64(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 512-bits (composed of 8 packed 64-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_storeu_epi64<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
    unsafe { arch::_mm512_storeu_epi64(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed double-precision (64-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_storeu_pd(mem_addr: &mut [f64; 2], k: __mmask8, a: __m128d) {
    unsafe { arch::_mm_mask_storeu_pd(mem_addr.as_mut_ptr().cast(), k, a) }
}

/// Store packed double-precision (64-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_pd)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_storeu_pd(mem_addr: &mut [f64; 4], k: __mmask8, a: __m256d) {
    unsafe { arch::_mm256_mask_storeu_pd(mem_addr.as_mut_ptr().cast(), k, a) }
}

/// Store packed double-precision (64-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_storeu_pd(mem_addr: &mut [f64; 8], k: __mmask8, a: __m512d) {
    unsafe { arch::_mm512_mask_storeu_pd(mem_addr.as_mut_ptr().cast(), k, a) }
}

/// Stores 512-bits (composed of 8 packed double-precision (64-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_pd)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_storeu_pd(mem_addr: &mut [f64; 8], a: __m512d) {
    unsafe { arch::_mm512_storeu_pd(mem_addr.as_mut_ptr().cast(), a) }
}

/// Store packed single-precision (32-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm_mask_storeu_ps(mem_addr: &mut [f32; 4], k: __mmask8, a: __m128) {
    unsafe { arch::_mm_mask_storeu_ps(mem_addr.as_mut_ptr().cast(), k, a) }
}

/// Store packed single-precision (32-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_ps)
#[inline]
#[target_feature(enable = "avx512f,avx512vl")]
pub fn _mm256_mask_storeu_ps(mem_addr: &mut [f32; 8], k: __mmask8, a: __m256) {
    unsafe { arch::_mm256_mask_storeu_ps(mem_addr.as_mut_ptr().cast(), k, a) }
}

/// Store packed single-precision (32-bit) floating-point elements from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_mask_storeu_ps(mem_addr: &mut [f32; 16], k: __mmask16, a: __m512) {
    unsafe { arch::_mm512_mask_storeu_ps(mem_addr.as_mut_ptr().cast(), k, a) }
}

/// Stores 512-bits (composed of 16 packed single-precision (32-bit)
/// floating-point elements) from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_ps)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_storeu_ps(mem_addr: &mut [f32; 16], a: __m512) {
    unsafe { arch::_mm512_storeu_ps(mem_addr.as_mut_ptr().cast(), a) }
}

/// Store 512-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_si512)
#[inline]
#[target_feature(enable = "avx512f")]
pub fn _mm512_storeu_si512<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
    unsafe { arch::_mm512_storeu_si512(ptr::from_mut(mem_addr).cast(), a) }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use crate::x86::{_mm_loadu_pd, _mm_loadu_ps, _mm256_loadu_pd, _mm256_loadu_ps};
    #[cfg(target_arch = "x86_64")]
    use crate::x86_64::{_mm_loadu_pd, _mm_loadu_ps, _mm256_loadu_pd, _mm256_loadu_ps};

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

    #[test]
    fn test_mm_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[4, 3, 2, 5];
            let r = super::_mm_loadu_epi32(black_box(a));
            let e = arch::_mm_setr_epi32(4, 3, 2, 5);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi32(42);
            let a = &[1_i32, 2, 3, 4];
            let m = 0b1010;
            let r = super::_mm_mask_loadu_epi32(src, m, black_box(a));
            let e = arch::_mm_setr_epi32(42, 2, 42, 4);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i32, 2, 3, 4];
            let m = 0b1010;
            let r = super::_mm_maskz_loadu_epi32(m, black_box(a));
            let e = arch::_mm_setr_epi32(0, 2, 0, 4);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm256_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[4, 3, 2, 5, 8, 9, 64, 50];
            let r = super::_mm256_loadu_epi32(black_box(a));
            let e = arch::_mm256_setr_epi32(4, 3, 2, 5, 8, 9, 64, 50);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi32(42);
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11001010;
            let r = super::_mm256_mask_loadu_epi32(src, m, black_box(a));
            let e = arch::_mm256_setr_epi32(42, 2, 42, 4, 42, 42, 7, 8);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11001010;
            let r = super::_mm256_maskz_loadu_epi32(m, black_box(a));
            let e = arch::_mm256_setr_epi32(0, 2, 0, 4, 0, 0, 7, 8);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm512_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50];
            let r = super::_mm512_loadu_epi32(black_box(a));
            let e =
                arch::_mm512_setr_epi32(4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_epi32(42);
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm512_mask_loadu_epi32(src, m, black_box(a));
            let e =
                arch::_mm512_setr_epi32(42, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_loadu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[1_i32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm512_maskz_loadu_epi32(m, black_box(a));
            let e = arch::_mm512_setr_epi32(0, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    fn test_mm_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1i64, 2];
            let r = super::_mm_loadu_epi64(a);
            let e = arch::_mm_set_epi64x(2, 1);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi64x(42);
            let a = &[1_i64, 2];
            let m = 0b10;
            let r = super::_mm_mask_loadu_epi64(src, m, black_box(a));
            let e = arch::_mm_set_epi64x(2, 42);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i64, 2];
            let m = 0b10;
            let r = super::_mm_maskz_loadu_epi64(m, black_box(a));
            let e = arch::_mm_set_epi64x(2, 0);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm256_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1i64, 2, 3, 4];
            let r = super::_mm256_loadu_epi64(a);
            let e = arch::_mm256_set_epi64x(4, 3, 2, 1);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi64x(42);
            let a = &[1_i64, 2, 3, 4];
            let m = 0b1010;
            let r = super::_mm256_mask_loadu_epi64(src, m, black_box(a));
            let e = arch::_mm256_setr_epi64x(42, 2, 42, 4);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1_i64, 2, 3, 4];
            let m = 0b1010;
            let r = super::_mm256_maskz_loadu_epi64(m, black_box(a));
            let e = arch::_mm256_setr_epi64x(0, 2, 0, 4);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm512_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[1_i64, 2, 3, 4, 5, 6, 7, 8];
            let r = super::_mm512_loadu_epi64(a);
            let e = arch::_mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_epi64(42);
            let a = &[1_i64, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11001010;
            let r = super::_mm512_mask_loadu_epi64(src, m, black_box(a));
            let e = arch::_mm512_setr_epi64(42, 2, 42, 4, 42, 42, 7, 8);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_loadu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[1_i64, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11001010;
            let r = super::_mm512_maskz_loadu_epi64(m, black_box(a));
            let e = arch::_mm512_setr_epi64(0, 2, 0, 4, 0, 0, 7, 8);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_loadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_pd(42.0);
            let a = &[1.0_f64, 2.0];
            let m = 0b10;
            let r = super::_mm_mask_loadu_pd(src, m, black_box(a));
            let e = arch::_mm_setr_pd(42.0, 2.0);
            assert_eq_m128d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_loadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0_f64, 2.0];
            let m = 0b10;
            let r = super::_mm_maskz_loadu_pd(m, black_box(a));
            let e = arch::_mm_setr_pd(0.0, 2.0);
            assert_eq_m128d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_loadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_pd(42.0);
            let a = &[1.0_f64, 2.0, 3.0, 4.0];
            let m = 0b1010;
            let r = super::_mm256_mask_loadu_pd(src, m, black_box(a));
            let e = arch::_mm256_setr_pd(42.0, 2.0, 42.0, 4.0);
            assert_eq_m256d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_loadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0_f64, 2.0, 3.0, 4.0];
            let m = 0b1010;
            let r = super::_mm256_maskz_loadu_pd(m, black_box(a));
            let e = arch::_mm256_setr_pd(0.0, 2.0, 0.0, 4.0);
            assert_eq_m256d(r, e);
        }
    }

    #[test]
    fn test_mm512_loadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[4., 3., 2., 5., 8., 9., 64., 50.];
            let r = super::_mm512_loadu_pd(black_box(a));
            let e = arch::_mm512_setr_pd(4., 3., 2., 5., 8., 9., 64., 50.);
            assert_eq_m512d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_loadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_pd(42.0);
            let a = &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let m = 0b11001010;
            let r = super::_mm512_mask_loadu_pd(src, m, black_box(a));
            let e = arch::_mm512_setr_pd(42.0, 2.0, 42.0, 4.0, 42.0, 42.0, 7.0, 8.0);
            assert_eq_m512d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_loadu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let m = 0b11001010;
            let r = super::_mm512_maskz_loadu_pd(m, black_box(a));
            let e = arch::_mm512_setr_pd(0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0);
            assert_eq_m512d(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_loadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_ps(42.0);
            let a = &[1.0_f32, 2.0, 3.0, 4.0];
            let m = 0b1010;
            let r = super::_mm_mask_loadu_ps(src, m, black_box(a));
            let e = arch::_mm_setr_ps(42.0, 2.0, 42.0, 4.0);
            assert_eq_m128(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_maskz_loadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0_f32, 2.0, 3.0, 4.0];
            let m = 0b1010;
            let r = super::_mm_maskz_loadu_ps(m, black_box(a));
            let e = arch::_mm_setr_ps(0.0, 2.0, 0.0, 4.0);
            assert_eq_m128(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_loadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_ps(42.0);
            let a = &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let m = 0b11001010;
            let r = super::_mm256_mask_loadu_ps(src, m, black_box(a));
            let e = arch::_mm256_setr_ps(42.0, 2.0, 42.0, 4.0, 42.0, 42.0, 7.0, 8.0);
            assert_eq_m256(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_maskz_loadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let m = 0b11001010;
            let r = super::_mm256_maskz_loadu_ps(m, black_box(a));
            let e = arch::_mm256_setr_ps(0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0);
            assert_eq_m256(r, e);
        }
    }

    #[test]
    fn test_mm512_loadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[
                4., 3., 2., 5., 8., 9., 64., 50., -4., -3., -2., -5., -8., -9., -64., -50.,
            ];
            let r = super::_mm512_loadu_ps(black_box(a));
            let e = arch::_mm512_setr_ps(
                4., 3., 2., 5., 8., 9., 64., 50., -4., -3., -2., -5., -8., -9., -64., -50.,
            );
            assert_eq_m512(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_loadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let src = arch::_mm512_set1_ps(42.0);
            let a = &[
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ];
            let m = 0b11101000_11001010;
            let r = super::_mm512_mask_loadu_ps(src, m, black_box(a));
            let e = arch::_mm512_setr_ps(
                42.0, 2.0, 42.0, 4.0, 42.0, 42.0, 7.0, 8.0, 42.0, 42.0, 42.0, 12.0, 42.0, 14.0,
                15.0, 16.0,
            );
            assert_eq_m512(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_maskz_loadu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[
                1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ];
            let m = 0b11101000_11001010;
            let r = super::_mm512_maskz_loadu_ps(m, black_box(a));
            let e = arch::_mm512_setr_ps(
                0.0, 2.0, 0.0, 4.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 0.0, 12.0, 0.0, 14.0, 15.0, 16.0,
            );
            assert_eq_m512(r, e);
        }
    }

    #[test]
    fn test_mm512_loadu_si512() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = &[4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50];
            let r = super::_mm512_loadu_si512(black_box(a));
            let e =
                arch::_mm512_setr_epi32(4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_compressstoreu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm_setr_epi32(1, 2, 3, 4);
            let mut r = [0_i32; 4];
            super::_mm_mask_compressstoreu_epi32(&mut r, 0, a);
            assert_eq!(&r, &[0_i32; 4]);
            super::_mm_mask_compressstoreu_epi32(&mut r, 0b1011, a);
            assert_eq!(&r, &[1, 2, 4, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_compressstoreu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let mut r = [0_i32; 8];
            super::_mm256_mask_compressstoreu_epi32(&mut r, 0, a);
            assert_eq!(&r, &[0_i32; 8]);
            super::_mm256_mask_compressstoreu_epi32(&mut r, 0b11001010, a);
            assert_eq!(&r, &[2, 4, 7, 8, 0, 0, 0, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_compressstoreu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
            let mut r = [0_i32; 16];
            super::_mm512_mask_compressstoreu_epi32(&mut r, 0, a);
            assert_eq!(&r, &[0_i32; 16]);
            super::_mm512_mask_compressstoreu_epi32(&mut r, 0b1111000011001010, a);
            assert_eq!(&r, &[2, 4, 7, 8, 13, 14, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_compressstoreu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm_set_epi64x(2, 1);
            let mut r = [0_i64; 2];
            super::_mm_mask_compressstoreu_epi64(&mut r, 0, a);
            assert_eq!(&r, &[0_i64; 2]);
            super::_mm_mask_compressstoreu_epi64(&mut r, 0b10, a);
            assert_eq!(&r, &[2, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_compressstoreu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm256_setr_epi64x(1, 2, 3, 4);
            let mut r = [0_i64; 4];
            super::_mm256_mask_compressstoreu_epi64(&mut r, 0, a);
            assert_eq!(&r, &[0_i64; 4]);
            super::_mm256_mask_compressstoreu_epi64(&mut r, 0b1011, a);
            assert_eq!(&r, &[1, 2, 4, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_compressstoreu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
            let mut r = [0_i64; 8];
            super::_mm512_mask_compressstoreu_epi64(&mut r, 0, a);
            assert_eq!(&r, &[0_i64; 8]);
            super::_mm512_mask_compressstoreu_epi64(&mut r, 0b11001010, a);
            assert_eq!(&r, &[2, 4, 7, 8, 0, 0, 0, 0]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_compressstoreu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm_setr_pd(1., 2.);
            let mut r = [0.; 2];
            super::_mm_mask_compressstoreu_pd(&mut r, 0, a);
            assert_eq!(&r, &[0.; 2]);
            super::_mm_mask_compressstoreu_pd(&mut r, 0b10, a);
            assert_eq!(&r, &[2., 0.]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_compressstoreu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm256_setr_pd(1., 2., 3., 4.);
            let mut r = [0.; 4];
            super::_mm256_mask_compressstoreu_pd(&mut r, 0, a);
            assert_eq!(&r, &[0.; 4]);
            super::_mm256_mask_compressstoreu_pd(&mut r, 0b1011, a);
            assert_eq!(&r, &[1., 2., 4., 0.]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_compressstoreu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_setr_pd(1., 2., 3., 4., 5., 6., 7., 8.);
            let mut r = [0.; 8];
            super::_mm512_mask_compressstoreu_pd(&mut r, 0, a);
            assert_eq!(&r, &[0.; 8]);
            super::_mm512_mask_compressstoreu_pd(&mut r, 0b11001010, a);
            assert_eq!(&r, &[2., 4., 7., 8., 0., 0., 0., 0.]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_compressstoreu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm_setr_ps(1_f32, 2_f32, 3_f32, 4_f32);
            let mut r = [0.; 4];
            super::_mm_mask_compressstoreu_ps(&mut r, 0, a);
            assert_eq!(&r, &[0.; 4]);
            super::_mm_mask_compressstoreu_ps(&mut r, 0b1011, a);
            assert_eq!(&r, &[1_f32, 2_f32, 4_f32, 0_f32]);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_compressstoreu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm256_setr_ps(1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 6_f32, 7_f32, 8_f32);
            let mut r = [0_f32; 8];
            super::_mm256_mask_compressstoreu_ps(&mut r, 0, a);
            assert_eq!(&r, &[0_f32; 8]);
            super::_mm256_mask_compressstoreu_ps(&mut r, 0b11001010, a);
            assert_eq!(
                &r,
                &[2_f32, 4_f32, 7_f32, 8_f32, 0_f32, 0_f32, 0_f32, 0_f32]
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_compressstoreu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_setr_ps(
                1_f32, 2_f32, 3_f32, 4_f32, 5_f32, 6_f32, 7_f32, 8_f32, 9_f32, 10_f32, 11_f32,
                12_f32, 13_f32, 14_f32, 15_f32, 16_f32,
            );
            let mut r = [0_f32; 16];
            super::_mm512_mask_compressstoreu_ps(&mut r, 0, a);
            assert_eq!(&r, &[0_f32; 16]);
            super::_mm512_mask_compressstoreu_ps(&mut r, 0b1111000011001010, a);
            assert_eq!(
                &r,
                &[
                    2_f32, 4_f32, 7_f32, 8_f32, 13_f32, 14_f32, 15_f32, 16_f32, 0_f32, 0_f32,
                    0_f32, 0_f32, 0_f32, 0_f32, 0_f32, 0_f32
                ]
            );
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_storeu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_i32; 4];
            let a = arch::_mm_setr_epi32(1, 2, 3, 4);
            let m = 0b1010;
            super::_mm_mask_storeu_epi32(&mut r, m, a);
            let e = arch::_mm_setr_epi32(42, 2, 42, 4);
            assert_eq_m128i(super::_mm_loadu_epi32(&r), e);
        }
    }

    #[test]
    fn test_mm_storeu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm_set1_epi32(9);
            let mut r = arch::_mm_undefined_si128();
            super::_mm_storeu_epi32(&mut r, a);
            assert_eq_m128i(r, a);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_storeu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_i32; 8];
            let a = arch::_mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
            let m = 0b11001010;
            super::_mm256_mask_storeu_epi32(&mut r, m, a);
            let e = arch::_mm256_setr_epi32(42, 2, 42, 4, 42, 42, 7, 8);
            assert_eq_m256i(super::_mm256_loadu_epi32(&r), e);
        }
    }

    #[test]
    fn test_mm256_storeu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let a = arch::_mm256_set1_epi32(9);
            let mut r = arch::_mm256_undefined_si256();
            super::_mm256_storeu_epi32(&mut r, a);
            assert_eq_m256i(r, a);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_storeu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let mut r = [42_i32; 16];
            let a = arch::_mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
            let m = 0b11101000_11001010;
            super::_mm512_mask_storeu_epi32(&mut r, m, a);
            let e =
                arch::_mm512_setr_epi32(42, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16);
            assert_eq_m512i(super::_mm512_loadu_epi32(&r), e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_storeu_epi32() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_set1_epi32(9);
            let mut r = arch::_mm512_undefined_epi32();
            super::_mm512_storeu_epi32(&mut r, a);
            assert_eq_m512i(r, a);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_storeu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_i64; 2];
            let a = arch::_mm_set_epi64x(2, 1);
            let m = 0b10;
            super::_mm_mask_storeu_epi64(&mut r, m, a);
            let e = arch::_mm_set_epi64x(2, 42);
            assert_eq_m128i(super::_mm_loadu_epi64(&r), e);
        }
    }

    #[test]
    fn test_mm_storeu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_i64; 2];
            let a = arch::_mm_set_epi64x(2, 1);
            super::_mm_storeu_epi64(&mut r, a);
            assert_eq_m128i(super::_mm_loadu_epi64(&r), a);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_storeu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_i64; 4];
            let a = arch::_mm256_setr_epi64x(1, 2, 3, 4);
            let m = 0b1010;
            super::_mm256_mask_storeu_epi64(&mut r, m, a);
            let e = arch::_mm256_setr_epi64x(42, 2, 42, 4);
            assert_eq_m256i(super::_mm256_loadu_epi64(&r), e);
        }
    }

    #[test]
    fn test_mm256_storeu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_i64; 4];
            let a = arch::_mm256_setr_epi64x(1, 2, 3, 4);
            super::_mm256_storeu_epi64(&mut r, a);
            assert_eq_m256i(super::_mm256_loadu_epi64(&r), a);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_storeu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let mut r = [42_i64; 8];
            let a = arch::_mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
            let m = 0b11001010;
            super::_mm512_mask_storeu_epi64(&mut r, m, a);
            let e = arch::_mm512_setr_epi64(42, 2, 42, 4, 42, 42, 7, 8);
            assert_eq_m512i(super::_mm512_loadu_epi64(&r), e);
        }
    }

    #[test]
    fn test_mm512_storeu_epi64() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let mut r = [42_i64; 8];
            let a = arch::_mm512_setr_epi64(1, 2, 3, 4, 5, 6, 7, 8);
            super::_mm512_storeu_epi64(&mut r, a);
            assert_eq_m512i(super::_mm512_loadu_epi64(&r), a);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_storeu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_f64; 2];
            let a = arch::_mm_setr_pd(1.0, 2.0);
            let m = 0b10;
            super::_mm_mask_storeu_pd(&mut r, m, a);
            let e = arch::_mm_setr_pd(42.0, 2.0);
            assert_eq_m128d(_mm_loadu_pd(&r), e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_storeu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_f64; 4];
            let a = arch::_mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
            let m = 0b1010;
            super::_mm256_mask_storeu_pd(&mut r, m, a);
            let e = arch::_mm256_setr_pd(42.0, 2.0, 42.0, 4.0);
            assert_eq_m256d(_mm256_loadu_pd(&r), e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_storeu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let mut r = [42_f64; 8];
            let a = arch::_mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let m = 0b11001010;
            super::_mm512_mask_storeu_pd(&mut r, m, a);
            let e = arch::_mm512_setr_pd(42.0, 2.0, 42.0, 4.0, 42.0, 42.0, 7.0, 8.0);
            assert_eq_m512d(super::_mm512_loadu_pd(&r), e);
        }
    }

    #[test]
    fn test_mm512_storeu_pd() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_set1_pd(9.);
            let mut r = [42_f64; 8];
            super::_mm512_storeu_pd(&mut r, a);
            assert_eq_m512d(super::_mm512_loadu_pd(&r), a);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_storeu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_f32; 4];
            let a = arch::_mm_setr_ps(1.0, 2.0, 3.0, 4.0);
            let m = 0b1010;
            super::_mm_mask_storeu_ps(&mut r, m, a);
            let e = arch::_mm_setr_ps(42.0, 2.0, 42.0, 4.0);
            assert_eq_m128(_mm_loadu_ps(&r), e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_storeu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f,avx512vl")]
        fn test() {
            let mut r = [42_f32; 8];
            let a = arch::_mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
            let m = 0b11001010;
            super::_mm256_mask_storeu_ps(&mut r, m, a);
            let e = arch::_mm256_setr_ps(42.0, 2.0, 42.0, 4.0, 42.0, 42.0, 7.0, 8.0);
            assert_eq_m256(_mm256_loadu_ps(&r), e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_storeu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let mut r = [42_f32; 16];
            let a = arch::_mm512_setr_ps(
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            );
            let m = 0b11101000_11001010;
            super::_mm512_mask_storeu_ps(&mut r, m, a);
            let e = arch::_mm512_setr_ps(
                42.0, 2.0, 42.0, 4.0, 42.0, 42.0, 7.0, 8.0, 42.0, 42.0, 42.0, 12.0, 42.0, 14.0,
                15.0, 16.0,
            );
            assert_eq_m512(super::_mm512_loadu_ps(&r), e);
        }
    }

    #[test]
    fn test_mm512_storeu_ps() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_set1_ps(9.);
            let mut r = [42_f32; 16];
            super::_mm512_storeu_ps(&mut r, a);
            assert_eq_m512(super::_mm512_loadu_ps(&r), a);
        }
    }

    #[test]
    fn test_mm512_storeu_si512() {
        assert!(*CPU_HAS_AVX512VL);
        unsafe { test() }

        #[target_feature(enable = "avx512f")]
        fn test() {
            let a = arch::_mm512_set1_epi32(9);
            let mut r = arch::_mm512_undefined_epi32();
            super::_mm512_storeu_si512(&mut r, a);
            assert_eq_m512i(r, a);
        }
    }
}

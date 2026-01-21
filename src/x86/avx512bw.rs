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
use crate::x86::{Is64BitsUnaligned, Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned};
#[cfg(target_arch = "x86_64")]
use crate::x86_64::{
    Is64BitsUnaligned, Is128BitsUnaligned, Is256BitsUnaligned, Is512BitsUnaligned,
};

/// Load 128-bits (composed of 8 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_loadu_epi16<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_epi16(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_mask_loadu_epi16<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask8,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_loadu_epi16(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_maskz_loadu_epi16<T: Is128BitsUnaligned>(k: __mmask8, mem_addr: &T) -> __m128i {
    _mm_mask_loadu_epi16(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load 256-bits (composed of 16 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_loadu_epi16<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu_epi16(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_mask_loadu_epi16<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask16,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_loadu_epi16(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_maskz_loadu_epi16<T: Is256BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m256i {
    _mm256_mask_loadu_epi16(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load 512-bits (composed of 32 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_loadu_epi16<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i {
    unsafe { arch::_mm512_loadu_epi16(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 16-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_mask_loadu_epi16<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask32,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_loadu_epi16(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 16-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_maskz_loadu_epi16<T: Is512BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m512i {
    _mm512_mask_loadu_epi16(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Load 128-bits (composed of 16 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_loadu_epi8<T: Is128BitsUnaligned>(mem_addr: &T) -> __m128i {
    unsafe { arch::_mm_loadu_epi8(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_mask_loadu_epi8<T: Is128BitsUnaligned>(
    src: __m128i,
    k: __mmask16,
    mem_addr: &T,
) -> __m128i {
    unsafe { arch::_mm_mask_loadu_epi8(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_maskz_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_maskz_loadu_epi8<T: Is128BitsUnaligned>(k: __mmask16, mem_addr: &T) -> __m128i {
    _mm_mask_loadu_epi8(arch::_mm_setzero_si128(), k, mem_addr)
}

/// Load 256-bits (composed of 32 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_loadu_epi8<T: Is256BitsUnaligned>(mem_addr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu_epi8(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_mask_loadu_epi8<T: Is256BitsUnaligned>(
    src: __m256i,
    k: __mmask32,
    mem_addr: &T,
) -> __m256i {
    unsafe { arch::_mm256_mask_loadu_epi8(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_maskz_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_maskz_loadu_epi8<T: Is256BitsUnaligned>(k: __mmask32, mem_addr: &T) -> __m256i {
    _mm256_mask_loadu_epi8(arch::_mm256_setzero_si256(), k, mem_addr)
}

/// Load 512-bits (composed of 64 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_loadu_epi8<T: Is512BitsUnaligned>(mem_addr: &T) -> __m512i {
    unsafe { arch::_mm512_loadu_epi8(ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 8-bit integers from memory into dst using writemask k
/// (elements are copied from src when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_mask_loadu_epi8<T: Is512BitsUnaligned>(
    src: __m512i,
    k: __mmask64,
    mem_addr: &T,
) -> __m512i {
    unsafe { arch::_mm512_mask_loadu_epi8(src, k, ptr::from_ref(mem_addr).cast()) }
}

/// Load packed 8-bit integers from memory into dst using zeromask k
/// (elements are zeroed out when the corresponding mask bit is not set).
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_maskz_loadu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_maskz_loadu_epi8<T: Is512BitsUnaligned>(k: __mmask64, mem_addr: &T) -> __m512i {
    _mm512_mask_loadu_epi8(arch::_mm512_setzero_si512(), k, mem_addr)
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_mask_cvtepi16_storeu_epi8<T: Is64BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    unsafe { arch::_mm_mask_cvtepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_mask_cvtepi16_storeu_epi8<T: Is128BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    unsafe { arch::_mm256_mask_cvtepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed 16-bit integers in a to packed 8-bit integers with truncation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_mask_cvtepi16_storeu_epi8<T: Is256BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask32,
    a: __m512i,
) {
    unsafe { arch::_mm512_mask_cvtepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtsepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_mask_cvtsepi16_storeu_epi8<T: Is64BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    unsafe { arch::_mm_mask_cvtsepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtsepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_mask_cvtsepi16_storeu_epi8<T: Is128BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    unsafe { arch::_mm256_mask_cvtsepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed signed 16-bit integers in a to packed 8-bit integers with signed saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtsepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_mask_cvtsepi16_storeu_epi8<T: Is256BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask32,
    a: __m512i,
) {
    unsafe { arch::_mm512_mask_cvtsepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_cvtusepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_mask_cvtusepi16_storeu_epi8<T: Is64BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask8,
    a: __m128i,
) {
    unsafe { arch::_mm_mask_cvtusepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_cvtusepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_mask_cvtusepi16_storeu_epi8<T: Is128BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask16,
    a: __m256i,
) {
    unsafe { arch::_mm256_mask_cvtusepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Convert packed unsigned 16-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the active results (those with their respective bit set in writemask k) to unaligned memory at base_addr.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_cvtusepi16_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_mask_cvtusepi16_storeu_epi8<T: Is256BitsUnaligned>(
    base_addr: &mut T,
    k: __mmask32,
    a: __m512i,
) {
    unsafe { arch::_mm512_mask_cvtusepi16_storeu_epi8(ptr::from_mut(base_addr).cast(), k, a) }
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_mask_storeu_epi16<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask8, a: __m128i) {
    unsafe { arch::_mm_mask_storeu_epi16(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 128-bits (composed of 8 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_storeu_epi16<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
    unsafe { arch::_mm_storeu_epi16(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_mask_storeu_epi16<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m256i) {
    unsafe { arch::_mm256_mask_storeu_epi16(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 256-bits (composed of 16 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_storeu_epi16<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
    unsafe { arch::_mm256_storeu_epi16(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 16-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_mask_storeu_epi16<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m512i) {
    unsafe { arch::_mm512_mask_storeu_epi16(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 512-bits (composed of 32 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi16)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_storeu_epi16<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
    unsafe { arch::_mm512_storeu_epi16(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_mask_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_mask_storeu_epi8<T: Is128BitsUnaligned>(mem_addr: &mut T, k: __mmask16, a: __m128i) {
    unsafe { arch::_mm_mask_storeu_epi8(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 128-bits (composed of 16 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm_storeu_epi8<T: Is128BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
    unsafe { arch::_mm_storeu_epi8(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_mask_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_mask_storeu_epi8<T: Is256BitsUnaligned>(mem_addr: &mut T, k: __mmask32, a: __m256i) {
    unsafe { arch::_mm256_mask_storeu_epi8(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 256-bits (composed of 32 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw,avx512vl")]
pub fn _mm256_storeu_epi8<T: Is256BitsUnaligned>(mem_addr: &mut T, a: __m256i) {
    unsafe { arch::_mm256_storeu_epi8(ptr::from_mut(mem_addr).cast(), a) }
}

/// Store packed 8-bit integers from a into memory using writemask k.
/// mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_mask_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_mask_storeu_epi8<T: Is512BitsUnaligned>(mem_addr: &mut T, k: __mmask64, a: __m512i) {
    unsafe { arch::_mm512_mask_storeu_epi8(ptr::from_mut(mem_addr).cast(), k, a) }
}

/// Store 512-bits (composed of 64 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm512_storeu_epi8)
#[inline]
#[target_feature(enable = "avx512bw")]
pub fn _mm512_storeu_epi8<T: Is512BitsUnaligned>(mem_addr: &mut T, a: __m512i) {
    unsafe { arch::_mm512_storeu_epi8(ptr::from_mut(mem_addr).cast(), a) }
}

#[cfg(test)]
mod tests {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::{self as arch, __m128i, __m256i, __m512i};
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::{self as arch, __m128i, __m256i, __m512i};

    use core::hint::black_box;

    // Fail-safe for tests being run on a CPU that doesn't support the instruction set
    static CPU_HAS_AVX512BW: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| is_x86_feature_detected!("avx512bw"));

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
    fn test_mm_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a: [i16; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            let r = super::_mm_loadu_epi16(&a);
            let e = arch::_mm_set_epi16(8, 7, 6, 5, 4, 3, 2, 1);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm_mask_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi16(42);
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11001010;
            let r = super::_mm_mask_loadu_epi16(src, m, black_box(a));
            let e = &[42_i16, 2, 42, 4, 42, 42, 7, 8];
            let e = super::_mm_loadu_epi16(e);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm_maskz_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
            let m = 0b11001010;
            let r = super::_mm_maskz_loadu_epi16(m, black_box(a));
            let e = &[0_i16, 2, 0, 4, 0, 0, 7, 8];
            let e = super::_mm_loadu_epi16(e);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm256_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a: [i16; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let r = super::_mm256_loadu_epi16(&a);
            let e = arch::_mm256_set_epi16(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm256_mask_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi16(42);
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm256_mask_loadu_epi16(src, m, black_box(a));
            let e = &[
                42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
            ];
            let e = super::_mm256_loadu_epi16(e);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm256_maskz_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm256_maskz_loadu_epi16(m, black_box(a));
            let e = &[0_i16, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16];
            let e = super::_mm256_loadu_epi16(e);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm512_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a: [i16; 32] = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let r = super::_mm512_loadu_epi16(&a);
            let e = arch::_mm512_set_epi16(
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            );
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    fn test_mm512_mask_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let src = arch::_mm512_set1_epi16(42);
            let a = &[
                1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b10101010_11001100_11101000_11001010;
            let r = super::_mm512_mask_loadu_epi16(src, m, black_box(a));
            let e = &[
                42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42,
                42, 23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
            ];
            let e = super::_mm512_loadu_epi16(e);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    fn test_mm512_maskz_loadu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a = &[
                1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b10101010_11001100_11101000_11001010;
            let r = super::_mm512_maskz_loadu_epi16(m, black_box(a));
            let e = &[
                0_i16, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16, 0, 0, 19, 20, 0, 0, 23, 24,
                0, 26, 0, 28, 0, 30, 0, 32,
            ];
            let e = super::_mm512_loadu_epi16(e);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    fn test_mm_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let r = super::_mm_loadu_epi8(&a);
            let e = arch::_mm_set_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm_mask_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let src = arch::_mm_set1_epi8(42);
            let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm_mask_loadu_epi8(src, m, black_box(a));
            let e = &[
                42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
            ];
            let e = super::_mm_loadu_epi8(e);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm_maskz_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let m = 0b11101000_11001010;
            let r = super::_mm_maskz_loadu_epi8(m, black_box(a));
            let e = &[0_i8, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16];
            let e = super::_mm_loadu_epi8(e);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm256_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a: [i8; 32] = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let r = super::_mm256_loadu_epi8(&a);
            let e = arch::_mm256_set_epi8(
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            );
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm256_mask_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let src = arch::_mm256_set1_epi8(42);
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b10101010_11001100_11101000_11001010;
            let r = super::_mm256_mask_loadu_epi8(src, m, black_box(a));
            let e = &[
                42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42,
                42, 23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
            ];
            let e = super::_mm256_loadu_epi8(e);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm256_maskz_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let m = 0b10101010_11001100_11101000_11001010;
            let r = super::_mm256_maskz_loadu_epi8(m, black_box(a));
            let e = &[
                0_i8, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16, 0, 0, 19, 20, 0, 0, 23, 24,
                0, 26, 0, 28, 0, 30, 0, 32,
            ];
            let e = super::_mm256_loadu_epi8(e);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm512_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a: [i8; 64] = [
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                24, 25, 26, 27, 28, 29, 30, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let r = super::_mm512_loadu_epi8(&a);
            let e = arch::_mm512_set_epi8(
                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12,
                11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,
                20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            );
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    fn test_mm512_mask_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let src = arch::_mm512_set1_epi8(42);
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            ];
            let m = 0b00000000_11111111_11111111_00000000_10101010_11001100_11101000_11001010;
            let r = super::_mm512_mask_loadu_epi8(src, m, black_box(a));
            let e = &[
                42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42,
                42, 23, 24, 42, 26, 42, 28, 42, 30, 42, 32, 42, 42, 42, 42, 42, 42, 42, 42, 41, 42,
                43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 42, 42, 42, 42, 42, 42, 42,
                42,
            ];
            let e = super::_mm512_loadu_epi8(e);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    fn test_mm512_maskz_loadu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            ];
            let m = 0b00000000_11111111_11111111_00000000_10101010_11001100_11101000_11001010;
            let r = super::_mm512_maskz_loadu_epi8(m, black_box(a));
            let e = &[
                0_i8, 2, 0, 4, 0, 0, 7, 8, 0, 0, 0, 12, 0, 14, 15, 16, 0, 0, 19, 20, 0, 0, 23, 24,
                0, 26, 0, 28, 0, 30, 0, 32, 0, 0, 0, 0, 0, 0, 0, 0, 41, 42, 43, 44, 45, 46, 47, 48,
                49, 50, 51, 52, 53, 54, 55, 56, 0, 0, 0, 0, 0, 0, 0, 0,
            ];
            let e = super::_mm512_loadu_epi8(e);
            assert_eq_m512i(r, e);
        }
    }

    #[test]
    fn test_mm_mask_cvtepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm_set1_epi16(8);
            let mut r = [0u8; 8];
            super::_mm_mask_cvtepi16_storeu_epi8(&mut r, 0b11111111, a);
            let e = [8; 8];
            assert_eq!(r, e);
        }
    }

    #[test]
    fn test_mm256_mask_cvtepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm256_set1_epi16(8);
            let mut r = arch::_mm_undefined_si128();
            super::_mm256_mask_cvtepi16_storeu_epi8(&mut r, 0b11111111_11111111, a);
            let e = arch::_mm_set1_epi8(8);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    fn test_mm512_mask_cvtepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a = arch::_mm512_set1_epi16(8);
            let mut r = arch::_mm256_undefined_si256();
            super::_mm512_mask_cvtepi16_storeu_epi8(
                &mut r,
                0b11111111_11111111_11111111_11111111,
                a,
            );
            let e = arch::_mm256_set1_epi8(8);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_cvtsepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm_set1_epi16(i16::MAX);
            let mut r = [0i8; 8];
            super::_mm_mask_cvtsepi16_storeu_epi8(&mut r, 0b11111111, a);
            let e = [i8::MAX; 8];
            assert_eq!(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_cvtsepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm256_set1_epi16(i16::MAX);
            let mut r = arch::_mm_undefined_si128();
            super::_mm256_mask_cvtsepi16_storeu_epi8(&mut r, 0b11111111_11111111, a);
            let e = arch::_mm_set1_epi8(i8::MAX);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_cvtsepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a = arch::_mm512_set1_epi16(i16::MAX);
            let mut r = arch::_mm256_undefined_si256();
            super::_mm512_mask_cvtsepi16_storeu_epi8(
                &mut r,
                0b11111111_11111111_11111111_11111111,
                a,
            );
            let e = arch::_mm256_set1_epi8(i8::MAX);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm_mask_cvtusepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm_set1_epi16(i16::MAX);
            let mut r = [0u8; 8];
            super::_mm_mask_cvtusepi16_storeu_epi8(&mut r, 0b11111111, a);
            let e = [u8::MAX; 8];
            assert_eq!(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm256_mask_cvtusepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm256_set1_epi16(i16::MAX);
            let mut r = arch::_mm_undefined_si128();
            super::_mm256_mask_cvtusepi16_storeu_epi8(&mut r, 0b11111111_11111111, a);
            let e = arch::_mm_set1_epi8(u8::MAX as i8);
            assert_eq_m128i(r, e);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_mm512_mask_cvtusepi16_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a = arch::_mm512_set1_epi16(i16::MAX);
            let mut r = arch::_mm256_undefined_si256();
            super::_mm512_mask_cvtusepi16_storeu_epi8(
                &mut r,
                0b11111111_11111111_11111111_11111111,
                a,
            );
            let e = arch::_mm256_set1_epi8(u8::MAX as i8);
            assert_eq_m256i(r, e);
        }
    }

    #[test]
    fn test_mm_mask_storeu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let mut r = [42_i16; 8];
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8];
            let a = super::_mm_loadu_epi16(a);
            let m = 0b11001010;
            super::_mm_mask_storeu_epi16(&mut r, m, a);
            let e = &[42_i16, 2, 42, 4, 42, 42, 7, 8];
            let e = super::_mm_loadu_epi16(e);
            assert_eq_m128i(super::_mm_loadu_epi16(&r), e);
        }
    }

    #[test]
    fn test_mm_storeu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm_set1_epi16(9);
            let mut r = arch::_mm_set1_epi32(0);
            super::_mm_storeu_epi16(&mut r, a);
            assert_eq_m128i(r, a);
        }
    }

    #[test]
    fn test_mm256_mask_storeu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let mut r = [42_i16; 16];
            let a = &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let a = super::_mm256_loadu_epi16(a);
            let m = 0b11101000_11001010;
            super::_mm256_mask_storeu_epi16(&mut r, m, a);
            let e = &[
                42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
            ];
            let e = super::_mm256_loadu_epi16(e);
            assert_eq_m256i(super::_mm256_loadu_epi16(&r), e);
        }
    }

    #[test]
    fn test_mm256_storeu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm256_set1_epi16(9);
            let mut r = arch::_mm256_set1_epi32(0);
            super::_mm256_storeu_epi16(&mut r, a);
            assert_eq_m256i(r, a);
        }
    }

    #[test]
    fn test_mm512_mask_storeu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let mut r = [42_i16; 32];
            let a = &[
                1_i16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let a = super::_mm512_loadu_epi16(a);
            let m = 0b10101010_11001100_11101000_11001010;
            super::_mm512_mask_storeu_epi16(&mut r, m, a);
            let e = &[
                42_i16, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42,
                42, 23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
            ];
            let e = super::_mm512_loadu_epi16(e);
            assert_eq_m512i(super::_mm512_loadu_epi16(&r), e);
        }
    }

    #[test]
    fn test_mm512_storeu_epi16() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a = arch::_mm512_set1_epi16(9);
            let mut r = arch::_mm512_undefined_epi32();
            super::_mm512_storeu_epi16(&mut r, a);
            assert_eq_m512i(r, a);
        }
    }

    #[test]
    fn test_mm_mask_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let mut r = [42_i8; 16];
            let a = &[1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
            let a = super::_mm_loadu_epi8(a);
            let m = 0b11101000_11001010;
            super::_mm_mask_storeu_epi8(&mut r, m, a);
            let e = &[
                42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16,
            ];
            let e = super::_mm_loadu_epi8(e);
            assert_eq_m128i(super::_mm_loadu_epi8(&r), e);
        }
    }

    #[test]
    fn test_mm_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm_set1_epi8(9);
            let mut r = arch::_mm_set1_epi32(0);
            super::_mm_storeu_epi8(&mut r, a);
            assert_eq_m128i(r, a);
        }
    }

    #[test]
    fn test_mm256_mask_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let mut r = [42_i8; 32];
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            ];
            let a = super::_mm256_loadu_epi8(a);
            let m = 0b10101010_11001100_11101000_11001010;
            super::_mm256_mask_storeu_epi8(&mut r, m, a);
            let e = &[
                42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42,
                42, 23, 24, 42, 26, 42, 28, 42, 30, 42, 32,
            ];
            let e = super::_mm256_loadu_epi8(e);
            assert_eq_m256i(super::_mm256_loadu_epi8(&r), e);
        }
    }

    #[test]
    fn test_mm256_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw,avx512vl")]
        fn test() {
            let a = arch::_mm256_set1_epi8(9);
            let mut r = arch::_mm256_set1_epi32(0);
            super::_mm256_storeu_epi8(&mut r, a);
            assert_eq_m256i(r, a);
        }
    }

    #[test]
    fn test_mm512_mask_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let mut r = [42_i8; 64];
            let a = &[
                1_i8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
                44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
            ];
            let a = super::_mm512_loadu_epi8(a);
            let m = 0b00000000_11111111_11111111_00000000_10101010_11001100_11101000_11001010;
            super::_mm512_mask_storeu_epi8(&mut r, m, a);
            let e = &[
                42_i8, 2, 42, 4, 42, 42, 7, 8, 42, 42, 42, 12, 42, 14, 15, 16, 42, 42, 19, 20, 42,
                42, 23, 24, 42, 26, 42, 28, 42, 30, 42, 32, 42, 42, 42, 42, 42, 42, 42, 42, 41, 42,
                43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 42, 42, 42, 42, 42, 42, 42,
                42,
            ];
            let e = super::_mm512_loadu_epi8(e);
            assert_eq_m512i(super::_mm512_loadu_epi8(&r), e);
        }
    }

    #[test]
    fn test_mm512_storeu_epi8() {
        assert!(*CPU_HAS_AVX512BW);
        unsafe { test() }

        #[target_feature(enable = "avx512bw")]
        fn test() {
            let a = arch::_mm512_set1_epi8(9);
            let mut r = arch::_mm512_undefined_epi32();
            super::_mm512_storeu_epi8(&mut r, a);
            assert_eq_m512i(r, a);
        }
    }
}

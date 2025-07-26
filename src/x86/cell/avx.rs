#[cfg(target_arch = "x86")]
use core::arch::x86::{self as arch, __m256i};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{self as arch, __m256i};
use core::ptr;

#[cfg(target_arch = "x86")]
use crate::x86::{Is128CellUnaligned, Is256CellUnaligned};
#[cfg(target_arch = "x86_64")]
use crate::x86_64::{Is128CellUnaligned, Is256CellUnaligned};

/// Loads 256-bits of integer data from memory into result.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu_si256)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu_si256<T: Is256CellUnaligned>(mem_addr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu_si256(ptr::from_ref(mem_addr).cast()) }
}

/// Loads two 128-bit values (composed of integer data) from memory, and combine
/// them into a 256-bit value.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_loadu2_m128i)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_loadu2_m128i<T: Is128CellUnaligned>(hiaddr: &T, loaddr: &T) -> __m256i {
    unsafe { arch::_mm256_loadu2_m128i(ptr::from_ref(hiaddr).cast(), ptr::from_ref(loaddr).cast()) }
}

/// Stores 256-bits of integer data from `a` into memory.
/// `mem_addr` does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu_si256)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu_si256<T: Is256CellUnaligned>(mem_addr: &T, a: __m256i) {
    unsafe { arch::_mm256_storeu_si256(ptr::from_ref(mem_addr).cast_mut().cast(), a) }
}

/// Stores the high and low 128-bit halves (each composed of integer data) from
/// `a` into memory two different 128-bit locations.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_storeu2_m128i)
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_storeu2_m128i<T: Is128CellUnaligned>(hiaddr: &T, loaddr: &T, a: __m256i) {
    unsafe {
        arch::_mm256_storeu2_m128i(
            ptr::from_ref(hiaddr).cast_mut().cast(),
            ptr::from_ref(loaddr).cast_mut().cast(),
            a,
        )
    }
}

#[cfg(feature = "_avx_test")]
#[cfg(test)]
mod tests {
    // Fail-safe for tests being run on a CPU that doesn't support `avx`
    static CPU_HAS_AVX: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| is_x86_feature_detected!("avx"));

    #[test]
    fn test_mm256_storeu2_m128i() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let mut st_hi: [i64; 3] = [30, 40, 0];
            let mut st_lo: [i64; 3] = [10, 20, 0];

            let hi = core::cell::Cell::from_mut(&mut st_hi[..]).as_slice_of_cells();
            let lo = core::cell::Cell::from_mut(&mut st_lo[..]).as_slice_of_cells();

            let lhi: &[_; 2] = hi[0..2].try_into().unwrap();
            let llo: &[_; 2] = lo[0..2].try_into().unwrap();

            let shi: &[_; 2] = hi[1..3].try_into().unwrap();
            let slo: &[_; 2] = lo[1..3].try_into().unwrap();

            let a = super::_mm256_loadu2_m128i(lhi, llo);
            super::_mm256_storeu2_m128i(shi, slo, a);

            assert_eq!(st_hi, [30, 30, 40]);
            assert_eq!(st_lo, [10, 10, 20]);
        }
    }

    #[test]
    fn test_mm256_loadu_si256() {
        assert!(*CPU_HAS_AVX);

        unsafe { test() }

        #[target_feature(enable = "avx")]
        fn test() {
            let mut x: [i16; 18] = core::array::from_fn(|i| i as i16);
            let whole_cell = core::cell::Cell::from_mut(&mut x[..]);

            let in_cell: &[_; 16] = whole_cell.as_slice_of_cells()[..16].try_into().unwrap();
            let mm256 = super::_mm256_loadu_si256(in_cell);

            let out_cell: &[_; 16] = whole_cell.as_slice_of_cells()[2..].try_into().unwrap();
            super::_mm256_storeu_si256(out_cell, mm256);

            let y: [i16; 16] = core::array::from_fn(|i| i as i16);
            // We copied it over into this area of the underlying storage
            assert_eq!(y, x[2..]);
        }
    }
}

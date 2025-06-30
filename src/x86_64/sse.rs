use core::arch::x86_64::{self as arch, __m128};

/// Construct a [`__m128`] by duplicating the value read from `mem_addr` into
/// all elements.
///
/// This corresponds to instructions `VMOVSS` / `MOVSS` followed by some
/// shuffling.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load1_ps)
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_load1_ps(mem_addr: &f32) -> __m128 {
    unsafe { arch::_mm_load1_ps(mem_addr) }
}

/// Alias for [`_mm_load1_ps`].
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ps1)
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_load_ps1(mem_addr: &f32) -> __m128 {
    _mm_load1_ps(mem_addr)
}

/// Construct a [`__m128`] with the lowest element read from `mem_addr` and the
/// other elements set to zero.
///
/// This corresponds to instructions `VMOVSS` / `MOVSS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_load_ss)
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_load_ss(mem_addr: &f32) -> __m128 {
    unsafe { arch::_mm_load_ss(mem_addr) }
}

/// Loads four `f32` values from memory into a [`__m128`]. There are no
/// restrictions on memory alignment.
///
/// This corresponds to instructions `VMOVUPS` / `MOVUPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_loadu_ps)
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_loadu_ps(mem_addr: &[f32; 4]) -> __m128 {
    unsafe { arch::_mm_loadu_ps(mem_addr.as_ptr()) }
}

/// Stores the lowest 32-bit float of `a` into memory.
///
/// This intrinsic corresponds to the `MOVSS` instruction.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_store_ss)
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_store_ss(mem_addr: &mut f32, a: __m128) {
    unsafe { arch::_mm_store_ss(mem_addr, a) }
}

/// Stores four 32-bit floats into memory. There are no restrictions on memory
/// alignment.
///
/// This corresponds to instructions `VMOVUPS` / `MOVUPS`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_storeu_ps)
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_storeu_ps(mem_addr: &mut [f32; 4], a: __m128) {
    unsafe { arch::_mm_storeu_ps(mem_addr.as_mut_ptr(), a) }
}

#[cfg(test)]
mod tests {
    use core::arch::x86_64::{self as arch, __m128};

    fn assert_eq_m128(a: __m128, b: __m128) {
        let a: [u8; 16] = unsafe { core::mem::transmute(a) };
        let b: [u8; 16] = unsafe { core::mem::transmute(b) };
        assert_eq!(a, b)
    }

    // SAFETY: The `x86_64` target baseline includes `sse` and `sse2`.

    #[test]
    fn test_mm_load1_ps() {
        let a = 10.0;
        unsafe { test(a) }

        #[target_feature(enable = "sse")]
        fn test(a: f32) {
            let r = super::_mm_load1_ps(&a);
            let target = arch::_mm_setr_ps(a, a, a, a);

            assert_eq_m128(r, target);
        }
    }

    #[test]
    fn test_mm_load_ss() {
        let a = 10.0;
        unsafe { test(a) }

        #[target_feature(enable = "sse")]
        fn test(a: f32) {
            let r = super::_mm_load_ss(&a);
            let target = arch::_mm_setr_ps(a, 0.0, 0.0, 0.0);

            assert_eq_m128(r, target);
        }
    }

    #[test]
    fn test_mm_loadu_ps() {
        let a = [1.0, 2.0, 3.0, 4.0];
        unsafe { test(&a) }

        #[target_feature(enable = "sse")]
        fn test(a: &[f32; 4]) {
            let r = super::_mm_loadu_ps(a);
            let target = arch::_mm_setr_ps(1.0, 2.0, 3.0, 4.0);

            assert_eq_m128(r, target);
        }
    }

    #[test]
    fn test_mm_store_ss() {
        unsafe { test() }

        #[target_feature(enable = "sse")]
        fn test() {
            let a = arch::_mm_setr_ps(1.0, 2.0, 3.0, 4.0);

            let mut mem_addr = 0.0;
            super::_mm_store_ss(&mut mem_addr, a);

            assert_eq!(mem_addr, 1.0);
        }
    }

    #[test]
    fn test_mm_storeu_ps() {
        unsafe { test() }

        #[target_feature(enable = "sse")]
        fn test() {
            let a = arch::_mm_setr_ps(1.0, 2.0, 3.0, 4.0);

            let mut mem_addr = [0.0; 4];
            super::_mm_storeu_ps(&mut mem_addr, a);

            assert_eq!(mem_addr, [1.0, 2.0, 3.0, 4.0]);
        }
    }
}

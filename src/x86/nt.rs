//! Non-temporal store operations.
use core::{marker::PhantomData, ptr};

#[cfg(target_arch = "x86")]
use core::arch::x86::{
    self as arch, __m128, __m128d, __m128i, __m256, __m256d, __m256i, _mm_sfence,
};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{
    self as arch, __m128, __m128d, __m128i, __m256, __m256d, __m256i, _mm_sfence,
};

/// Load from a 32-bit aligned address with non-temporal hint, avoiding filling the cache.
#[inline]
#[cfg(any())]
#[target_feature(enable = "avx2")]
pub fn _mm256_stream_load_si256(addr: &__m256i) -> __m256i {
    unsafe { arch::_mm256_stream_load_si256(addr) }
}

/// Store into a 32-bit aligned address with non-temporal hint, avoiding clobbering the cache.
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_stream_store_256i(addr: &mut NonTemporalStoreable<'_, __m256i>, v: __m256i) {
    unsafe { arch::_mm256_stream_si256(addr.inner.as_ptr(), v) }
}

/// Store a 128-bit floating point vector of `[2 × double]` into a 128-bit aligned memory location.
/// To minimize caching, the data is flagged as non-temporal (unlikely to be used again soon).
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_stream_pd(addr: &mut NonTemporalStoreable<'_, __m128d>, v: __m128d) {
    unsafe { arch::_mm_stream_pd(addr.inner.as_ptr() as *mut f64, v) }
}

/// Store a 128-bit floating point vector of `[4 × f32]` into a 16-byte aligned memory location. To
/// minimize caching, the data is flagged as non-temporal (unlikely to be used again soon).
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_stream_ps(addr: &mut NonTemporalStoreable<'_, __m128>, v: __m128) {
    unsafe { arch::_mm_stream_ps(addr.inner.as_ptr() as *mut f32, v) }
}

/// Store a 64-bit part `v.0` of a 128-bit vector into an aligned memory location. To minimize
/// caching, the data is flagged as non-temporal (unlikely to be used again soon).
#[inline]
#[cfg(any())]
#[target_feature(enable = "sse4a")]
pub fn _mm_stream_sd(addr: &mut NonTemporalStoreable<'_, f64>, v: __m128d) {
    unsafe { arch::_mm_stream_sd(addr.inner.as_ptr(), v) }
}

/// Store a 32-bit value into a memory location. To minimize caching, the data is flagged as
/// non-temporal (unlikely to be used again soon).
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_stream_si32(addr: &mut NonTemporalStoreable<'_, i32>, v: i32) {
    unsafe { arch::_mm_stream_si32(addr.inner.as_ptr(), v) }
}

/// Store a 32-bit value into a memory location. To minimize caching, the data is flagged as
/// non-temporal (unlikely to be used again soon).
#[inline]
#[target_feature(enable = "sse2")]
pub fn _mm_stream_si128(addr: &mut NonTemporalStoreable<'_, __m128i>, v: __m128i) {
    unsafe { arch::_mm_stream_si128(addr.inner.as_ptr(), v) }
}

/// Store a 32-bit part `v.0` of a 128-bit vector into a memory location. To minimize caching, the
/// data is flagged as non-temporal (unlikely to be used again soon).
#[inline]
#[cfg(any())]
#[target_feature(enable = "sse4a")]
pub fn _mm_stream_ss(addr: &mut NonTemporalStoreable<'_, f32>, v: __m128) {
    unsafe { arch::_mm_stream_ss(addr.inner.as_ptr(), v) }
}

/// Store the four double precision floats of a 256-bit vector into a memory location. To minimize
/// caching, the data is flagged as non-temporal (unlikely to be used again soon).
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_stream_pd(addr: &mut NonTemporalStoreable<'_, __m256d>, v: __m256d) {
    unsafe { arch::_mm256_stream_pd(addr.inner.as_ptr() as *mut f64, v) }
}

/// Store eight single precision float values of a 256-bit vector into a memory location. To
/// minimize caching, the data is flagged as non-temporal (unlikely to be used again soon).
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_stream_ps(addr: &mut NonTemporalStoreable<'_, __m256>, v: __m256) {
    unsafe { arch::_mm256_stream_ps(addr.inner.as_ptr() as *mut f32, v) }
}

/// Store a 256-bit vector into an aligned memory location. To minimize caching, the data is
/// flagged as non-temporal (unlikely to be used again soon).
#[inline]
#[target_feature(enable = "avx")]
pub fn _mm256_stream_si256(addr: &mut NonTemporalStoreable<'_, __m256i>, v: __m256i) {
    unsafe { arch::_mm256_stream_si256(addr.inner.as_ptr(), v) }
}

/// A pointer to non-temporally written-to memory.
///
/// The lifetime on this struct means: we can write to the memory within lifetime `'data` while
/// guaranteeing an appropriate fence afterwards, and then that effect appears as if we had written
/// to the data through a mutable reference. The role of this type in this is the guarantee that
/// *no* active reference, mutable or shared, to the memory can exist in the lifetime `'data' which
/// could observe the memory while it is in the non-coherent state between having been written the
/// points in time where it is written-to non-temporally and the fence being issue.
pub struct NonTemporalStoreable<'data, T> {
    inner: ptr::NonNull<T>,
    marker: PhantomData<&'data mut T>,
}

/// A marker for a scope that allows non-temporal writes.
///
/// See [`Self::with`].
pub struct NonTemporalScope<'lt> {
    invariant: PhantomData<fn(&'lt mut ()) -> &'lt ()>,
}

impl<'data> NonTemporalScope<'data> {
    /// Wrap writable memory such that non-temporal stores can be issued to it.
    ///
    /// The scope value certifies an `mfence` instruction is executed after the scope ends and
    /// before any other access to the mutably referenced memory is possible again. This
    /// combination ensures that the observable behavior of stores follows the expected memory
    /// model of Rust. Since we have a unique reference to the memory at the start, no other access
    /// (neither read or write) can occur that does not go through the x86 non-temporal intrinsics.
    ///
    /// Note that the memory must be borrowed for the *whole* duration of the original scope.
    /// Automatic reborrowing can not be used to shorten it for a different scope.
    ///
    /// ```rust,compile_fail
    #[cfg_attr(
        target_arch = "x86",
        doc = "
        use safe_unaligned_simd::x86::NonTemporalScope;
        use core::arch::x86::__m256i;
    "
    )]
    #[cfg_attr(
        target_arch = "x86_64",
        doc = "
        use safe_unaligned_simd::x86_64::NonTemporalScope;
        use core::arch::x86_64::__m256i;
    "
    )]
    ///
    /// #[target_feature(enable = "avx")]
    /// fn zero_data<'d>(scope: NonTemporalScope<'d>, data: &'d mut __m256i) {
    ///     let first = scope.prepare_write(data);
    ///     // Fails!
    ///     let second = scope.prepare_write(data);
    /// }
    /// ```
    pub fn prepare_write<T>(&self, inner: &'data mut T) -> NonTemporalStoreable<'data, T> {
        NonTemporalStoreable {
            inner: ptr::NonNull::from(inner),
            marker: PhantomData,
        }
    }

    /// Run a closure with the guarantee of exiting with an `sfence` instruction. A closure is
    /// invoked within the scope and given a value in reference to the scope. That value allows
    /// qualifying mutably owned memory as memory which can be targeted with non-temporal stores.
    /// When memory is wrapped in such a manner, no other access is allowed until the scope exits.
    ///
    /// ```rust
    #[cfg_attr(
        target_arch = "x86",
        doc = "
        use safe_unaligned_simd::x86::{NonTemporalScope, _mm256_stream_store_256i};
        use core::arch::x86::{__m256i, _mm256_set1_epi8};
    "
    )]
    #[cfg_attr(
        target_arch = "x86_64",
        doc = "
        use safe_unaligned_simd::x86_64::{NonTemporalScope, _mm256_stream_store_256i};
        use core::arch::x86_64::{__m256i, _mm256_set1_epi8};
    "
    )]
    /// #[target_feature(enable = "avx")]
    /// fn zero_data<'d>(scope: NonTemporalScope<'d>, data: &'d mut [__m256i]) {
    ///     let v = _mm256_set1_epi8(0);
    ///
    ///     for part in data {
    ///         let mut storeable = scope.prepare_write(part);
    ///         _mm256_stream_store_256i(&mut storeable, v);
    ///     }
    /// }
    ///
    /// # #[target_feature(enable = "avx")]
    /// # fn _do_main() {
    /// let data: &mut [__m256i] = // ..
    /// # &mut [_mm256_set1_epi8(1); 4];
    ///
    /// NonTemporalScope::with(|scope| {
    ///     zero_data(scope, data);
    /// });
    /// # let a: [u16; 16] = unsafe { core::mem::transmute(data[0]) };
    /// # assert_eq!(a, [0; 16]);
    /// # }
    /// #
    /// # if cfg!(target_feature = "avx") {
    /// #     unsafe { _do_main() }
    /// # }
    /// ```
    #[target_feature(enable = "sse")]
    pub fn with<R>(inner: impl FnOnce(NonTemporalScope<'data>) -> R) -> R {
        struct SFenceOnDrop;

        impl Drop for SFenceOnDrop {
            fn drop(&mut self) {
                // Safety: `SFenceOnDrop` only exists within `with` that has the target_feature
                // `sse` enabled. It is only dropped within that method. So we can assume that
                // target feature to exist and `_mm_sfence` to be available.
                debug_assert!(cfg!(target_feature = "sse"));

                unsafe { _mm_sfence() }
            }
        }

        let _val = SFenceOnDrop;
        inner(NonTemporalScope {
            invariant: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "_avx_test")]
    use super::{_mm256_stream_store_256i, NonTemporalScope};

    #[cfg(target_arch = "x86")]
    #[cfg(feature = "_avx_test")]
    use core::arch::x86::{__m256i, _mm256_set1_epi8};
    #[cfg(target_arch = "x86_64")]
    #[cfg(feature = "_avx_test")]
    use core::arch::x86_64::{__m256i, _mm256_set1_epi8};

    #[cfg(feature = "_avx_test")]
    static CPU_HAS_AVX: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| is_x86_feature_detected!("avx"));

    #[test]
    #[cfg(feature = "_avx_test")]
    fn _mm256_stream_store() {
        #[target_feature(enable = "avx")]
        fn zero_data<'d>(scope: NonTemporalScope<'d>, data: &'d mut [__m256i]) {
            let v = _mm256_set1_epi8(0);

            for part in data {
                let mut storeable = scope.prepare_write(part);
                _mm256_stream_store_256i(&mut storeable, v);
            }
        }

        #[target_feature(enable = "avx")]
        fn test() {
            let data: &mut [__m256i] = &mut [_mm256_set1_epi8(1); 4];
            NonTemporalScope::with(|scope| {
                zero_data(scope, data);
            });
            let a: [u16; 16] = unsafe { core::mem::transmute(data[0]) };
            assert_eq!(a, [0; 16]);
        }

        assert!(*CPU_HAS_AVX);

        unsafe { test() }
    }
}

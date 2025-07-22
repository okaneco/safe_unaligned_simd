//! Non-temporal store operations.
use core::marker::PhantomData;

#[cfg(target_arch = "x86")]
use core::arch::x86::{self as arch, __m256i, _mm_sfence};
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::{self as arch, __m256i, _mm_sfence};

/// Load from a 32-bit aligned address with non-temporal hint, avoiding filling the cache.
#[inline]
#[target_feature(enable = "avx2")]
pub fn _mm256_stream_load_si256(addr: &__m256i) -> __m256i {
    unsafe { arch::_mm256_stream_load_si256(addr) }
}

/// Store into a 32-bit aligned address with non-temporal hint, avoiding clobbering the cache.
#[inline]
#[target_feature(enable = "avx2")]
pub fn _mm256_stream_store_256i(addr: &mut NonTemporalStoreable<'_, __m256i>, v: __m256i) {
    unsafe { arch::_mm256_stream_si256(addr.inner, v) }
}

/// A pointer to non-temporally written-to memory.
pub struct NonTemporalStoreable<'data, T> {
    inner: *mut T,
    marker: PhantomData<&'data mut T>,
}

/// A marker for a scope that allows non-temporal writes.
///
/// See [`Self::with`].
pub struct NonTemporalScope<'lt> {
    inner: PhantomData<&'lt mut ()>,
}

impl<'data> NonTemporalScope<'data> {
    /// Wrap writable memory such that non-temporal stores can be issued to it.
    ///
    /// The scope value certifies an `mfence` instruction is executed after the scope ends and
    /// before any other access to the mutably referenced memory is possible again. This
    /// combination ensures that the observable behavior of stores follows the expected memory
    /// model of Rust. Since we have a unique reference to the memory at the start, no other access
    /// (neither read or write) can occur that does not go through the x86 non-temporal intrinsics.
    pub fn prepare_write<T>(&self, inner: &'data mut T) -> NonTemporalStoreable<'data, T> {
        NonTemporalStoreable {
            inner,
            marker: PhantomData,
        }
    }

    /// Run a closure with the guarantee of exiting with an `sfence` instruction. A closure is
    /// invoked within the scope and given a value in reference to the scope. That value allows
    /// qualifying mutably owned memory as memory which can be targeted with non-temporal stores.
    /// When memory is wrapped in such a manner, no other access is allowed until the scope exits.
    ///
    /// ```rust
    #[cfg_attr(target_arch = "x86", doc = "
        use safe_unaligned_simd::x86::{NonTemporalScope, _mm256_stream_store_256i};
        use core::arch::x86::{__m256i, _mm256_set1_epi8};
    ")]
    #[cfg_attr(target_arch = "x86_64", doc = "
        use safe_unaligned_simd::x86_64::{NonTemporalScope, _mm256_stream_store_256i};
        use core::arch::x86_64::{__m256i, _mm256_set1_epi8};
    ")]
    /// #[target_feature(enable = "avx2")]
    /// fn zero_data<'d>(scope: NonTemporalScope<'d>, data: &'d mut [__m256i]) {
    ///     let v = _mm256_set1_epi8(0);
    ///
    ///     for part in data {
    ///         let mut storeable = scope.prepare_write(part);
    ///         _mm256_stream_store_256i(&mut storeable, v);
    ///     }
    /// }
    ///
    /// # #[target_feature(enable = "avx2")]
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
    /// # unsafe { _do_main() }
    /// ```
    pub fn with<R>(inner: impl FnOnce(NonTemporalScope<'data>) -> R) -> R {
        struct SFenceOnDrop;

        impl Drop for SFenceOnDrop {
            fn drop(&mut self) {
                unsafe { _mm_sfence() }
            }
        }

        let _val = SFenceOnDrop;
        inner(NonTemporalScope { inner: PhantomData })
    }
}

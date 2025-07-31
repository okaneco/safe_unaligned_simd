//! Functions generic over [`Cell`][Cell] array types.
//!
//! These functions enable loading and storing of `&Cell<[T; N]>` and
//! `&[Cell<T>; N]`, shared mutable container types which permit mutability
//! even in the presence of aliasing.
//!
//! This allows for operating on overlapping slices similar to how one can use
//! `std::arch` intrinsics with raw pointers.
//!
//! [Cell]: core::cell::Cell
//!
//! ```rust
//! # unsafe { slide_right() }
//! use core::cell::Cell;
//!
//! #[cfg(target_arch = "x86")]
//! use safe_unaligned_simd::x86::cell;
//! #[cfg(target_arch = "x86_64")]
//! use safe_unaligned_simd::x86_64::cell;
//!
//! #[target_feature(enable = "sse2")]
//! fn slide_right() {
//!    let mut a = [0u16, 1, 2, 3, 4, 5, 6, 7, 8];
//!    let val = Cell::from_mut(&mut a[..]).as_slice_of_cells();
//!
//!    let load: &[_; 8] = val[..8].try_into().unwrap();
//!    let store: &[_; 8] = val[1..].try_into().unwrap();
//!
//!    let r = cell::_mm_loadu_si128(load);
//!    cell::_mm_storeu_si128(store, r);
//!
//!    assert_eq!(a, [0, 0, 1, 2, 3, 4, 5, 6, 7]);
//! }
//! ```

mod sse2;
pub use sse2::*;

mod avx;
pub use avx::*;

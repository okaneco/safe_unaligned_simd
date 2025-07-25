//! Platform-specific intrinsics for the aarch64 platform.
//!
//! Note that instructions are encoded with an optional `<align>` field which none of the intrinsics
//! here set themselves. (It could hint alignment 64/128/256). The field is then 0b00 which is
//!
//! > Whenever `<align>` is omitted, the standard alignment is used, see Unaligned data access,
//!
//! You *could* use all these intrinsics with completely unaligned memory if you set SCTLR, the
//! system control register, which is not part of the guarantees. So we do not allow those. To load
//! unaligned floating point data, use an appropriate u8xN type and reinterpret the vector.
//!
//! See: <https://developer.arm.com/documentation/ddi0597/2025-06/SIMD-FP-Instructions/> on VLD1
#![cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]

// Use all variants of registers.
use core::arch::aarch64::{self as arch, *};

// Most of this is generated via macro due to the respective nature. The macro identifies to which
// kind of internal we want to expand by an introductory keyword (load, store) followed by a
// sequence of wrapper instantiations. To review this, the basic structure is:
//
//   fn vld1_u16_x2(_: &[u16; 4][..2] as [[u16; 4]; 2]) -> uint16x4x2_t;
//
// Here `vld1_u16_x2` identifies the intrinsic. The first type-like syntax is the input type as it
// should be thought of in the context of the vector, the `..2` is the number of vectors involved.
// The second type `[[u16; 4]; 2]` is the actual argument type that the wrapper will have. Then
// finally the return type is used as is.
//
// Each block also expects a `size` macro to perform some compile-time verification of the typing.
// Mostly we verify that types have exactly the register size and thus fit the expected memory
// access. This is only enabled on test/check builds.
macro_rules! vld_n_replicate_k {
    (
        // So we have one unsafe keyword in the pre-expansion.
        unsafe: $kind:ident;
        size: $size:ident;

        $(
            $(#[$meta:meta])* fn $intrinsic:ident(_: &[$base_ty:ty; $n:literal][..$len:literal] as $realty:ty) -> $ret:ty;
        )*
    ) => {
        $(
            vld_n_replicate_k!(
                @ $kind $(#[$meta])* $intrinsic: ([$base_ty; $n][..$len] | $realty) -> $ret [$size]
            );
        )*
    };

    // This macro generates one signature. The basic inputs are:
    //
    // - `base_ty` the register type that underlies each load
    // - `n` the number of elements in one structure
    // - `len` the number of structures being loaded
    // - `ret` the register type to which we may broadcast
    (@ load // Internal expansion for load-like intrinsics.
        $(#[$meta:meta])*
        $intrinsic:ident: ([$base_ty:ty; $n:literal][..$registers:literal] | $realty:ty) -> $ret:ty
        $([$size:ident])?
    ) => {
        $(#[$meta])*
        #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
        #[target_feature(enable = "neon")]
        pub fn $intrinsic(from: &$realty) -> $ret {
            $(
                $size!($registers registers [[$base_ty; $n]; $registers] as $realty);
            )?

            // Safety: Review the macro use and macro construction. We match up types to the
            // intrinsics being used. Sizes are compile-time checked in test builds.
            unsafe { arch::$intrinsic(::core::ptr::from_ref(from).cast()) }
        }
    };

    (@ store // Internal expansion for store-like intrinsics.
        $(#[$meta:meta])*
        $intrinsic:ident: ([$base_ty:ty; $n:literal][..$registers:literal] | $realty:ty) -> $ret:ty
        $([$size:ident])?
    ) => {
        $(#[$meta])*
        #[cfg(any(target_arch = "aarch64", target_arch = "arm64ec"))]
        #[target_feature(enable = "neon")]
        pub fn $intrinsic(into: &mut $realty, from: $ret) {
            $(
                $size!($registers registers [[$base_ty; $n]; $registers] as $realty);
            )?

            // Safety: Review the macro use and macro construction. We match up types to the
            // intrinsics being used. Sizes are compile-time checked in test builds.
            unsafe { arch::$intrinsic(::core::ptr::from_mut(into).cast(), from) }
        }
    };
}

macro_rules! assert_size_8bytes {
    ($n:literal registers $ty:ty as $real:ty) => {
        // Ensure that the type has individual vectors 64 bits wide.
        const _: () = ::core::assert!(::core::mem::size_of::<$ty>() == 8 * $n);
        const _: () = ::core::assert!(::core::mem::size_of::<$real>() == 8 * $n);
    };
}

macro_rules! assert_size_16bytes {
    ($n:literal registers $ty:ty as $real:ty) => {
        // Ensure that the type has individual vectors 128 bits wide.
        const _: () = ::core::assert!(::core::mem::size_of::<$ty>() == 16 * $n);
        const _: () = ::core::assert!(::core::mem::size_of::<$real>() == 16 * $n);
    };
}

#[cfg(test)]
macro_rules! various_sizes {
    ($n:literal registers $ty:ty as $real:ty) => {
        // Only make sure that our annotated structure and the interface type match up
        const _: () = ::core::assert!(core::mem::size_of::<$ty>() == core::mem::size_of::<$real>());
    };
}

#[cfg(not(test))]
macro_rules! various_sizes {
    // No reason to take up compile time for a constant assert.
    ($n:literal registers $ty:ty as $real:ty) => {};
}

// There are four fundamental types of loads:
// - `vldN[q]_<ty>` which loads an array of structures of N elements of type <ty>, as many as
//   fill the 8-byte or with q 16-byte registers. Eg. vld2q_f32 would load 8 total values, each
//   of the two result vectors getting 4 elements for a total of 16-bytes each. (These loads
//   would be interleaved but that's not important for the memory access part).
//
// - `vld1[q]_<ty>_xK` which is performs multiple loads of the type `vld1[q]_<ty>` to
//   K consective registers but in a single instruction, compared to stacking multiple loads.
//
// - vldN[q]_dup_<ty> which loads *one* structure of N elements and fills each of the N
//   registers with the contents of its one element.
//
//- `vldN[q]_lane_<t> which loads *one* structure of N elements and inserts the contents of
//   each registers elements into a specific lane, i.e. the lane must be within the bounds such
//   that `[ty; LANE]` does not exceed 8 or 16-bytes with/without q respectively.
vld_n_replicate_k! {
    unsafe: load;
    // Loads full registers, so 8 bytes per register
    size: assert_size_8bytes;

    /// Load an array of 8 `u8` values to one 8-byte register.
    fn vld1_u8(_: &[u8; 8][..1] as [u8; 8]) -> uint8x8_t;
    /// Load an array of 8 `i8` values to one 8-byte register.
    fn vld1_s8(_: &[i8; 8][..1] as [i8; 8]) -> int8x8_t;
    /// Load an array of 4 `u16` values to one 8-byte register.
    fn vld1_u16(_: &[u16; 4][..1] as [u16; 4]) -> uint16x4_t;
    /// Load an array of 4 `i16` values to one 8-byte register.
    fn vld1_s16(_: &[i16; 4][..1] as [i16; 4]) -> int16x4_t;
    /// Load an array of 2 `u32` values to one 8-byte register.
    fn vld1_u32(_: &[u32; 2][..1] as [u32; 2]) -> uint32x2_t;
    /// Load an array of 2 `i32` values to one 8-byte register.
    fn vld1_s32(_: &[i32; 2][..1] as [i32; 2]) -> int32x2_t;
    /// Load an array of 2 `f32` values to one 8-byte register.
    fn vld1_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x2_t;
    /// Load one `u64` value to one 8-byte register.
    fn vld1_u64(_: &[u64; 1][..1] as u64) -> uint64x1_t;
    /// Load one `i64` value to one 8-byte register.
    fn vld1_s64(_: &[i64; 1][..1] as i64) -> int64x1_t;
    /// Load one `f64` value to one 8-byte register.
    fn vld1_f64(_: &[f64; 1][..1] as f64) -> float64x1_t;

    /// Load arrays of 8 `u8` values to two 8-byte registers.
    fn vld1_u8_x2(_: &[u8; 8][..2] as [[u8; 8]; 2]) -> uint8x8x2_t;
    /// Load arrays of 8 `i8` values to two 8-byte registers.
    fn vld1_s8_x2(_: &[i8; 8][..2] as [[i8; 8]; 2]) -> int8x8x2_t;
    /// Load arrays of 4 `u16` values to two 8-byte registers.
    fn vld1_u16_x2(_: &[u16; 4][..2] as [[u16; 4]; 2]) -> uint16x4x2_t;
    /// Load arrays of 4 `i16` values to two 8-byte registers.
    fn vld1_s16_x2(_: &[i16; 4][..2] as [[i16; 4]; 2]) -> int16x4x2_t;
    /// Load arrays of 2 `u32` values to two 8-byte registers.
    fn vld1_u32_x2(_: &[u32; 2][..2] as [[u32; 2]; 2]) -> uint32x2x2_t;
    /// Load arrays of 2 `i32` values to two 8-byte registers.
    fn vld1_s32_x2(_: &[i32; 2][..2] as [[i32; 2]; 2]) -> int32x2x2_t;
    /// Load arrays of 2 `f32` values to two 8-byte registers.
    fn vld1_f32_x2(_: &[f32; 2][..2] as [[f32; 2]; 2]) -> float32x2x2_t;
    /// Load two `u64` values to two 8-byte registers.
    fn vld1_u64_x2(_: &[u64; 1][..2] as [u64; 2]) -> uint64x1x2_t;
    /// Load two `i64` values to two 8-byte registers.
    fn vld1_s64_x2(_: &[i64; 1][..2] as [i64; 2]) -> int64x1x2_t;
    /// Load two `f64` values to two 8-byte registers.
    fn vld1_f64_x2(_: &[f64; 1][..2] as [f64; 2]) -> float64x1x2_t;

    /// Load arrays of 8 `u8` values to three 8-byte registers.
    fn vld1_u8_x3(_: &[u8; 8][..3] as [[u8; 8]; 3]) -> uint8x8x3_t;
    /// Load arrays of 8 `i8` values to three 8-byte registers.
    fn vld1_s8_x3(_: &[i8; 8][..3] as [[i8; 8]; 3]) -> int8x8x3_t;
    /// Load arrays of 4 `u16` values to three 8-byte registers.
    fn vld1_u16_x3(_: &[u16; 4][..3] as [[u16; 4]; 3]) -> uint16x4x3_t;
    /// Load arrays of 4 `i16` values to three 8-byte registers.
    fn vld1_s16_x3(_: &[i16; 4][..3] as [[i16; 4]; 3]) -> int16x4x3_t;
    /// Load arrays of 2 `u32` values to three 8-byte registers.
    fn vld1_u32_x3(_: &[u32; 2][..3] as [[u32; 2]; 3]) -> uint32x2x3_t;
    /// Load arrays of 2 `i32` values to three 8-byte registers.
    fn vld1_s32_x3(_: &[i32; 2][..3] as [[i32; 2]; 3]) -> int32x2x3_t;
    /// Load arrays of 2 `f32` values to three 8-byte registers.
    fn vld1_f32_x3(_: &[f32; 2][..3] as [[f32; 2]; 3]) -> float32x2x3_t;
    /// Load two `u64` values to three 8-byte registers.
    fn vld1_u64_x3(_: &[u64; 1][..3] as [u64; 3]) -> uint64x1x3_t;
    /// Load two `i64` values to three 8-byte registers.
    fn vld1_s64_x3(_: &[i64; 1][..3] as [i64; 3]) -> int64x1x3_t;
    /// Load two `f64` values to three 8-byte registers.
    fn vld1_f64_x3(_: &[f64; 1][..3] as [f64; 3]) -> float64x1x3_t;

    /// Load arrays of 8 `u8` values to four 8-byte registers.
    fn vld1_u8_x4(_: &[u8; 8][..4] as [[u8; 8]; 4]) -> uint8x8x4_t;
    /// Load arrays of 8 `i8` values to four 8-byte registers.
    fn vld1_s8_x4(_: &[i8; 8][..4] as [[i8; 8]; 4]) -> int8x8x4_t;
    /// Load arrays of 4 `u16` values to four 8-byte registers.
    fn vld1_u16_x4(_: &[u16; 4][..4] as [[u16; 4]; 4]) -> uint16x4x4_t;
    /// Load arrays of 4 `i16` values to four 8-byte registers.
    fn vld1_s16_x4(_: &[i16; 4][..4] as [[i16; 4]; 4]) -> int16x4x4_t;
    /// Load arrays of 2 `u32` values to four 8-byte registers.
    fn vld1_u32_x4(_: &[u32; 2][..4] as [[u32; 2]; 4]) -> uint32x2x4_t;
    /// Load arrays of 2 `i32` values to four 8-byte registers.
    fn vld1_s32_x4(_: &[i32; 2][..4] as [[i32; 2]; 4]) -> int32x2x4_t;
    /// Load arrays of 2 `f32` values to four 8-byte registers.
    fn vld1_f32_x4(_: &[f32; 2][..4] as [[f32; 2]; 4]) -> float32x2x4_t;
    /// Load two `u64` values to four 8-byte registers.
    fn vld1_u64_x4(_: &[u64; 1][..4] as [u64; 4]) -> uint64x1x4_t;
    /// Load two `i64` values to four 8-byte registers.
    fn vld1_s64_x4(_: &[i64; 1][..4] as [i64; 4]) -> int64x1x4_t;
    /// Load two `f64` values to four 8-byte registers.
    fn vld1_f64_x4(_: &[f64; 1][..4] as [f64; 4]) -> float64x1x4_t;
}

vld_n_replicate_k! {
    unsafe: load;
    // Loads full registers, so 16 bytes per register
    size: assert_size_16bytes;

    /// Load an array of 16 `u8` values to one 16-byte register.
    fn vld1q_u8(_: &[u8; 16][..1] as [u8; 16]) -> uint8x16_t;
    /// Load an array of 16 `i8` values to one 16-byte register.
    fn vld1q_s8(_: &[i8; 16][..1] as [i8; 16]) -> int8x16_t;
    /// Load an array of 8 `u16` values to one 16-byte register.
    fn vld1q_u16(_: &[u16; 8][..1] as [u16; 8]) -> uint16x8_t;
    /// Load an array of 8 `i16` values to one 16-byte register.
    fn vld1q_s16(_: &[i16; 8][..1] as [i16; 8]) -> int16x8_t;
    /// Load an array of 4 `u32` values to one 16-byte register.
    fn vld1q_u32(_: &[u32; 4][..1] as [u32; 4]) -> uint32x4_t;
    /// Load an array of 4 `i32` values to one 16-byte register.
    fn vld1q_s32(_: &[i32; 4][..1] as [i32; 4]) -> int32x4_t;
    /// Load an array of 4 `f32` values to one 16-byte register.
    fn vld1q_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x4_t;
    /// Load an array of 2 `u64` value to one 16-byte register.
    fn vld1q_u64(_: &[u64; 2][..1] as [u64; 2]) -> uint64x2_t;
    /// Load an array of 2 `i64` value to one 16-byte register.
    fn vld1q_s64(_: &[i64; 2][..1] as [i64; 2]) -> int64x2_t;
    /// Load an array of 2 `f64` value to one 16-byte register.
    fn vld1q_f64(_: &[f64; 2][..1] as [f64; 2]) -> float64x2_t;

    /// Load two arrays of 16 `u8` values to two 16-byte registers.
    fn vld1q_u8_x2(_: &[u8; 16][..2] as [[u8; 16]; 2]) -> uint8x16x2_t;
    /// Load two arrays of 16 `i8` values to two 16-byte registers.
    fn vld1q_s8_x2(_: &[i8; 16][..2] as [[i8; 16]; 2]) -> int8x16x2_t;
    /// Load two arrays of 8 `u16` values to two 16-byte registers.
    fn vld1q_u16_x2(_: &[u16; 8][..2] as [[u16; 8]; 2]) -> uint16x8x2_t;
    /// Load two arrays of 8 `i16` values to two 16-byte registers.
    fn vld1q_s16_x2(_: &[i16; 8][..2] as [[i16; 8]; 2]) -> int16x8x2_t;
    /// Load two arrays of 4 `u32` values to two 16-byte registers.
    fn vld1q_u32_x2(_: &[u32; 4][..2] as [[u32; 4]; 2]) -> uint32x4x2_t;
    /// Load two arrays of 4 `i32` values to two 16-byte registers.
    fn vld1q_s32_x2(_: &[i32; 4][..2] as [[i32; 4]; 2]) -> int32x4x2_t;
    /// Load two arrays of 4 `f32` values to two 16-byte registers.
    fn vld1q_f32_x2(_: &[f32; 4][..2] as [[f32; 4]; 2]) -> float32x4x2_t;
    /// Load two arrays of 2 `u64` value to two 16-byte registers.
    fn vld1q_u64_x2(_: &[u64; 2][..2] as [[u64; 2]; 2]) -> uint64x2x2_t;
    /// Load two arrays of 2 `i64` value to two 16-byte registers.
    fn vld1q_s64_x2(_: &[i64; 2][..2] as [[i64; 2]; 2]) -> int64x2x2_t;
    /// Load two arrays of 2 `f64` value to two 16-byte registers.
    fn vld1q_f64_x2(_: &[f64; 2][..2] as [[f64; 2]; 2]) -> float64x2x2_t;

    /// Load three arrays of 16 `u8` values to three16-byte registers.
    fn vld1q_u8_x3(_: &[u8; 16][..3] as [[u8; 16]; 3]) -> uint8x16x3_t;
    /// Load three arrays of 16 `i8` values to three16-byte registers.
    fn vld1q_s8_x3(_: &[i8; 16][..3] as [[i8; 16]; 3]) -> int8x16x3_t;
    /// Load three arrays of 8 `u16` values to three16-byte registers.
    fn vld1q_u16_x3(_: &[u16; 8][..3] as [[u16; 8]; 3]) -> uint16x8x3_t;
    /// Load three arrays of 8 `i16` values to three16-byte registers.
    fn vld1q_s16_x3(_: &[i16; 8][..3] as [[i16; 8]; 3]) -> int16x8x3_t;
    /// Load three arrays of 4 `u32` values to three16-byte registers.
    fn vld1q_u32_x3(_: &[u32; 4][..3] as [[u32; 4]; 3]) -> uint32x4x3_t;
    /// Load three arrays of 4 `i32` values to three16-byte registers.
    fn vld1q_s32_x3(_: &[i32; 4][..3] as [[i32; 4]; 3]) -> int32x4x3_t;
    /// Load three arrays of 4 `f32` values to three16-byte registers.
    fn vld1q_f32_x3(_: &[f32; 4][..3] as [[f32; 4]; 3]) -> float32x4x3_t;
    /// Load three arrays of 2 `u64` value to three16-byte registers.
    fn vld1q_u64_x3(_: &[u64; 2][..3] as [[u64; 2]; 3]) -> uint64x2x3_t;
    /// Load three arrays of 2 `i64` value to three16-byte registers.
    fn vld1q_s64_x3(_: &[i64; 2][..3] as [[i64; 2]; 3]) -> int64x2x3_t;
    /// Load three arrays of 2 `f64` value to three16-byte registers.
    fn vld1q_f64_x3(_: &[f64; 2][..3] as [[f64; 2]; 3]) -> float64x2x3_t;

    /// Load four arrays of 16 `u8` values to four 16-byte registers.
    fn vld1q_u8_x4(_: &[u8; 16][..4] as [[u8; 16]; 4]) -> uint8x16x4_t;
    /// Load four arrays of 16 `i8` values to four 16-byte registers.
    fn vld1q_s8_x4(_: &[i8; 16][..4] as [[i8; 16]; 4]) -> int8x16x4_t;
    /// Load four arrays of 8 `u16` values to four 16-byte registers.
    fn vld1q_u16_x4(_: &[u16; 8][..4] as [[u16; 8]; 4]) -> uint16x8x4_t;
    /// Load four arrays of 8 `i16` values to four 16-byte registers.
    fn vld1q_s16_x4(_: &[i16; 8][..4] as [[i16; 8]; 4]) -> int16x8x4_t;
    /// Load four arrays of 4 `u32` values to four 16-byte registers.
    fn vld1q_u32_x4(_: &[u32; 4][..4] as [[u32; 4]; 4]) -> uint32x4x4_t;
    /// Load four arrays of 4 `i32` values to four 16-byte registers.
    fn vld1q_s32_x4(_: &[i32; 4][..4] as [[i32; 4]; 4]) -> int32x4x4_t;
    /// Load four arrays of 4 `f32` values to four 16-byte registers.
    fn vld1q_f32_x4(_: &[f32; 4][..4] as [[f32; 4]; 4]) -> float32x4x4_t;
    /// Load four arrays of 2 `u64` value to four 16-byte registers.
    fn vld1q_u64_x4(_: &[u64; 2][..4] as [[u64; 2]; 4]) -> uint64x2x4_t;
    /// Load four arrays of 2 `i64` value to four 16-byte registers.
    fn vld1q_s64_x4(_: &[i64; 2][..4] as [[i64; 2]; 4]) -> int64x2x4_t;
    /// Load four arrays of 2 `f64` value to four 16-byte registers.
    fn vld1q_f64_x4(_: &[f64; 2][..4] as [[f64; 2]; 4]) -> float64x2x4_t;
}

vld_n_replicate_k! {
    unsafe: store;
    // Stores full registers, so 8 bytes per register
    size: assert_size_8bytes;

    /// Store an array of 8 `u8` values from one 8-byte register.
    fn vst1_u8(_: &[u8; 8][..1] as [u8; 8]) -> uint8x8_t;
    /// Store an array of 8 `i8` values from one 8-byte register.
    fn vst1_s8(_: &[i8; 8][..1] as [i8; 8]) -> int8x8_t;
    /// Store an array of 4 `u16` values from one 8-byte register.
    fn vst1_u16(_: &[u16; 4][..1] as [u16; 4]) -> uint16x4_t;
    /// Store an array of 4 `i16` values from one 8-byte register.
    fn vst1_s16(_: &[i16; 4][..1] as [i16; 4]) -> int16x4_t;
    /// Store an array of 2 `u32` values from one 8-byte register.
    fn vst1_u32(_: &[u32; 2][..1] as [u32; 2]) -> uint32x2_t;
    /// Store an array of 2 `i32` values from one 8-byte register.
    fn vst1_s32(_: &[i32; 2][..1] as [i32; 2]) -> int32x2_t;
    /// Store an array of 2 `f32` values from one 8-byte register.
    fn vst1_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x2_t;
    /// Store one `u64` value from one 8-byte register.
    fn vst1_u64(_: &[u64; 1][..1] as u64) -> uint64x1_t;
    /// Store one `i64` value from one 8-byte register.
    fn vst1_s64(_: &[i64; 1][..1] as i64) -> int64x1_t;
    /// Store one `f64` value from one 8-byte register.
    fn vst1_f64(_: &[f64; 1][..1] as f64) -> float64x1_t;

    /// Store arrays of 8 `u8` values from two 8-byte registers.
    fn vst1_u8_x2(_: &[u8; 8][..2] as [[u8; 8]; 2]) -> uint8x8x2_t;
    /// Store arrays of 8 `i8` values from two 8-byte registers.
    fn vst1_s8_x2(_: &[i8; 8][..2] as [[i8; 8]; 2]) -> int8x8x2_t;
    /// Store arrays of 4 `u16` values from two 8-byte registers.
    fn vst1_u16_x2(_: &[u16; 4][..2] as [[u16; 4]; 2]) -> uint16x4x2_t;
    /// Store arrays of 4 `i16` values from two 8-byte registers.
    fn vst1_s16_x2(_: &[i16; 4][..2] as [[i16; 4]; 2]) -> int16x4x2_t;
    /// Store arrays of 2 `u32` values from two 8-byte registers.
    fn vst1_u32_x2(_: &[u32; 2][..2] as [[u32; 2]; 2]) -> uint32x2x2_t;
    /// Store arrays of 2 `i32` values from two 8-byte registers.
    fn vst1_s32_x2(_: &[i32; 2][..2] as [[i32; 2]; 2]) -> int32x2x2_t;
    /// Store arrays of 2 `f32` values from two 8-byte registers.
    fn vst1_f32_x2(_: &[f32; 2][..2] as [[f32; 2]; 2]) -> float32x2x2_t;
    /// Store two `u64` values from two 8-byte registers.
    fn vst1_u64_x2(_: &[u64; 1][..2] as [u64; 2]) -> uint64x1x2_t;
    /// Store two `i64` values from two 8-byte registers.
    fn vst1_s64_x2(_: &[i64; 1][..2] as [i64; 2]) -> int64x1x2_t;
    /// Store two `f64` values from two 8-byte registers.
    fn vst1_f64_x2(_: &[f64; 1][..2] as [f64; 2]) -> float64x1x2_t;

    /// Store arrays of 8 `u8` values from three 8-byte registers.
    fn vst1_u8_x3(_: &[u8; 8][..3] as [[u8; 8]; 3]) -> uint8x8x3_t;
    /// Store arrays of 8 `i8` values from three 8-byte registers.
    fn vst1_s8_x3(_: &[i8; 8][..3] as [[i8; 8]; 3]) -> int8x8x3_t;
    /// Store arrays of 4 `u16` values from three 8-byte registers.
    fn vst1_u16_x3(_: &[u16; 4][..3] as [[u16; 4]; 3]) -> uint16x4x3_t;
    /// Store arrays of 4 `i16` values from three 8-byte registers.
    fn vst1_s16_x3(_: &[i16; 4][..3] as [[i16; 4]; 3]) -> int16x4x3_t;
    /// Store arrays of 2 `u32` values from three 8-byte registers.
    fn vst1_u32_x3(_: &[u32; 2][..3] as [[u32; 2]; 3]) -> uint32x2x3_t;
    /// Store arrays of 2 `i32` values from three 8-byte registers.
    fn vst1_s32_x3(_: &[i32; 2][..3] as [[i32; 2]; 3]) -> int32x2x3_t;
    /// Store arrays of 2 `f32` values from three 8-byte registers.
    fn vst1_f32_x3(_: &[f32; 2][..3] as [[f32; 2]; 3]) -> float32x2x3_t;
    /// Store two `u64` values from three 8-byte registers.
    fn vst1_u64_x3(_: &[u64; 1][..3] as [u64; 3]) -> uint64x1x3_t;
    /// Store two `i64` values from three 8-byte registers.
    fn vst1_s64_x3(_: &[i64; 1][..3] as [i64; 3]) -> int64x1x3_t;
    /// Store two `f64` values from three 8-byte registers.
    fn vst1_f64_x3(_: &[f64; 1][..3] as [f64; 3]) -> float64x1x3_t;

    /// Store arrays of 8 `u8` values from four 8-byte registers.
    fn vst1_u8_x4(_: &[u8; 8][..4] as [[u8; 8]; 4]) -> uint8x8x4_t;
    /// Store arrays of 8 `i8` values from four 8-byte registers.
    fn vst1_s8_x4(_: &[i8; 8][..4] as [[i8; 8]; 4]) -> int8x8x4_t;
    /// Store arrays of 4 `u16` values from four 8-byte registers.
    fn vst1_u16_x4(_: &[u16; 4][..4] as [[u16; 4]; 4]) -> uint16x4x4_t;
    /// Store arrays of 4 `i16` values from four 8-byte registers.
    fn vst1_s16_x4(_: &[i16; 4][..4] as [[i16; 4]; 4]) -> int16x4x4_t;
    /// Store arrays of 2 `u32` values from four 8-byte registers.
    fn vst1_u32_x4(_: &[u32; 2][..4] as [[u32; 2]; 4]) -> uint32x2x4_t;
    /// Store arrays of 2 `i32` values from four 8-byte registers.
    fn vst1_s32_x4(_: &[i32; 2][..4] as [[i32; 2]; 4]) -> int32x2x4_t;
    /// Store arrays of 2 `f32` values from four 8-byte registers.
    fn vst1_f32_x4(_: &[f32; 2][..4] as [[f32; 2]; 4]) -> float32x2x4_t;
    /// Store two `u64` values from four 8-byte registers.
    fn vst1_u64_x4(_: &[u64; 1][..4] as [u64; 4]) -> uint64x1x4_t;
    /// Store two `i64` values from four 8-byte registers.
    fn vst1_s64_x4(_: &[i64; 1][..4] as [i64; 4]) -> int64x1x4_t;
    /// Store two `f64` values from four 8-byte registers.
    fn vst1_f64_x4(_: &[f64; 1][..4] as [f64; 4]) -> float64x1x4_t;
}

vld_n_replicate_k! {
    unsafe: store;
    // Stores full registers, so 16 bytes per register
    size: assert_size_16bytes;

    /// Store an array of 16 `u8` values to one 16-byte register.
    fn vst1q_u8(_: &[u8; 16][..1] as [u8; 16]) -> uint8x16_t;
    /// Store an array of 16 `i8` values to one 16-byte register.
    fn vst1q_s8(_: &[i8; 16][..1] as [i8; 16]) -> int8x16_t;
    /// Store an array of 8 `u16` values to one 16-byte register.
    fn vst1q_u16(_: &[u16; 8][..1] as [u16; 8]) -> uint16x8_t;
    /// Store an array of 8 `i16` values to one 16-byte register.
    fn vst1q_s16(_: &[i16; 8][..1] as [i16; 8]) -> int16x8_t;
    /// Store an array of 4 `u32` values to one 16-byte register.
    fn vst1q_u32(_: &[u32; 4][..1] as [u32; 4]) -> uint32x4_t;
    /// Store an array of 4 `i32` values to one 16-byte register.
    fn vst1q_s32(_: &[i32; 4][..1] as [i32; 4]) -> int32x4_t;
    /// Store an array of 4 `f32` values to one 16-byte register.
    fn vst1q_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x4_t;
    /// Store an array of 2 `u64` value to one 16-byte register.
    fn vst1q_u64(_: &[u64; 2][..1] as [u64; 2]) -> uint64x2_t;
    /// Store an array of 2 `i64` value to one 16-byte register.
    fn vst1q_s64(_: &[i64; 2][..1] as [i64; 2]) -> int64x2_t;
    /// Store an array of 2 `f64` value to one 16-byte register.
    fn vst1q_f64(_: &[f64; 2][..1] as [f64; 2]) -> float64x2_t;

    /// Store two arrays of 16 `u8` values from two 16-byte registers.
    fn vst1q_u8_x2(_: &[u8; 16][..2] as [[u8; 16]; 2]) -> uint8x16x2_t;
    /// Store two arrays of 16 `i8` values from two 16-byte registers.
    fn vst1q_s8_x2(_: &[i8; 16][..2] as [[i8; 16]; 2]) -> int8x16x2_t;
    /// Store two arrays of 8 `u16` values from two 16-byte registers.
    fn vst1q_u16_x2(_: &[u16; 8][..2] as [[u16; 8]; 2]) -> uint16x8x2_t;
    /// Store two arrays of 8 `i16` values from two 16-byte registers.
    fn vst1q_s16_x2(_: &[i16; 8][..2] as [[i16; 8]; 2]) -> int16x8x2_t;
    /// Store two arrays of 4 `u32` values from two 16-byte registers.
    fn vst1q_u32_x2(_: &[u32; 4][..2] as [[u32; 4]; 2]) -> uint32x4x2_t;
    /// Store two arrays of 4 `i32` values from two 16-byte registers.
    fn vst1q_s32_x2(_: &[i32; 4][..2] as [[i32; 4]; 2]) -> int32x4x2_t;
    /// Store two arrays of 4 `f32` values from two 16-byte registers.
    fn vst1q_f32_x2(_: &[f32; 4][..2] as [[f32; 4]; 2]) -> float32x4x2_t;
    /// Store two arrays of 2 `u64` value from two 16-byte registers.
    fn vst1q_u64_x2(_: &[u64; 2][..2] as [[u64; 2]; 2]) -> uint64x2x2_t;
    /// Store two arrays of 2 `i64` value from two 16-byte registers.
    fn vst1q_s64_x2(_: &[i64; 2][..2] as [[i64; 2]; 2]) -> int64x2x2_t;
    /// Store two arrays of 2 `f64` value from two 16-byte registers.
    fn vst1q_f64_x2(_: &[f64; 2][..2] as [[f64; 2]; 2]) -> float64x2x2_t;

    /// Store three arrays of 16 `u8` values from three16-byte registers.
    fn vst1q_u8_x3(_: &[u8; 16][..3] as [[u8; 16]; 3]) -> uint8x16x3_t;
    /// Store three arrays of 16 `i8` values from three16-byte registers.
    fn vst1q_s8_x3(_: &[i8; 16][..3] as [[i8; 16]; 3]) -> int8x16x3_t;
    /// Store three arrays of 8 `u16` values from three16-byte registers.
    fn vst1q_u16_x3(_: &[u16; 8][..3] as [[u16; 8]; 3]) -> uint16x8x3_t;
    /// Store three arrays of 8 `i16` values from three16-byte registers.
    fn vst1q_s16_x3(_: &[i16; 8][..3] as [[i16; 8]; 3]) -> int16x8x3_t;
    /// Store three arrays of 4 `u32` values from three16-byte registers.
    fn vst1q_u32_x3(_: &[u32; 4][..3] as [[u32; 4]; 3]) -> uint32x4x3_t;
    /// Store three arrays of 4 `i32` values from three16-byte registers.
    fn vst1q_s32_x3(_: &[i32; 4][..3] as [[i32; 4]; 3]) -> int32x4x3_t;
    /// Store three arrays of 4 `f32` values from three16-byte registers.
    fn vst1q_f32_x3(_: &[f32; 4][..3] as [[f32; 4]; 3]) -> float32x4x3_t;
    /// Store three arrays of 2 `u64` value from three16-byte registers.
    fn vst1q_u64_x3(_: &[u64; 2][..3] as [[u64; 2]; 3]) -> uint64x2x3_t;
    /// Store three arrays of 2 `i64` value from three16-byte registers.
    fn vst1q_s64_x3(_: &[i64; 2][..3] as [[i64; 2]; 3]) -> int64x2x3_t;
    /// Store three arrays of 2 `f64` value from three16-byte registers.
    fn vst1q_f64_x3(_: &[f64; 2][..3] as [[f64; 2]; 3]) -> float64x2x3_t;

    /// Store four arrays of 16 `u8` values from four 16-byte registers.
    fn vst1q_u8_x4(_: &[u8; 16][..4] as [[u8; 16]; 4]) -> uint8x16x4_t;
    /// Store four arrays of 16 `i8` values from four 16-byte registers.
    fn vst1q_s8_x4(_: &[i8; 16][..4] as [[i8; 16]; 4]) -> int8x16x4_t;
    /// Store four arrays of 8 `u16` values from four 16-byte registers.
    fn vst1q_u16_x4(_: &[u16; 8][..4] as [[u16; 8]; 4]) -> uint16x8x4_t;
    /// Store four arrays of 8 `i16` values from four 16-byte registers.
    fn vst1q_s16_x4(_: &[i16; 8][..4] as [[i16; 8]; 4]) -> int16x8x4_t;
    /// Store four arrays of 4 `u32` values from four 16-byte registers.
    fn vst1q_u32_x4(_: &[u32; 4][..4] as [[u32; 4]; 4]) -> uint32x4x4_t;
    /// Store four arrays of 4 `i32` values from four 16-byte registers.
    fn vst1q_s32_x4(_: &[i32; 4][..4] as [[i32; 4]; 4]) -> int32x4x4_t;
    /// Store four arrays of 4 `f32` values from four 16-byte registers.
    fn vst1q_f32_x4(_: &[f32; 4][..4] as [[f32; 4]; 4]) -> float32x4x4_t;
    /// Store four arrays of 2 `u64` value from four 16-byte registers.
    fn vst1q_u64_x4(_: &[u64; 2][..4] as [[u64; 2]; 4]) -> uint64x2x4_t;
    /// Store four arrays of 2 `i64` value from four 16-byte registers.
    fn vst1q_s64_x4(_: &[i64; 2][..4] as [[i64; 2]; 4]) -> int64x2x4_t;
    /// Store four arrays of 2 `f64` value from four 16-byte registers.
    fn vst1q_f64_x4(_: &[f64; 2][..4] as [[f64; 2]; 4]) -> float64x2x4_t;
}

vld_n_replicate_k! {
    unsafe: load;
    size: various_sizes;

    /// Load one single-element `f32` and replicate to all lanes.
    fn vld1_dup_f32(_: &[f32; 1][..1] as f32) -> float32x2_t;
    /// Load an array of two `f32` elements and replicate to lanes of two registers.
    fn vld2_dup_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x2x2_t;
    /// Load an array of three `f32` elements and replicate to lanes of three registers.
    fn vld3_dup_f32(_: &[f32; 3][..1] as [f32; 3]) -> float32x2x3_t;
    /// Load an array of four `f32` elements and replicate to lanes of four registers.
    fn vld4_dup_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x2x4_t;

    /// Load one single-element `f64` and replicate to all lanes.
    fn vld1_dup_f64(_: &[f64; 1][..1] as f64) -> float64x1_t;
    /// Load an array of two `f64` elements and replicate to lanes of two registers.
    fn vld2_dup_f64(_: &[f64; 2][..1] as [f64; 2]) -> float64x1x2_t;
    /// Load an array of three `f64` elements and replicate to lanes of three registers.
    fn vld3_dup_f64(_: &[f64; 3][..1] as [f64; 3]) -> float64x1x3_t;
    /// Load an array of four `f64` elements and replicate to lanes of four registers.
    fn vld4_dup_f64(_: &[f64; 4][..1] as [f64; 4]) -> float64x1x4_t;

    /// Load one single-element `f32` and replicate to all lanes.
    fn vld1q_dup_f32(_: &[f32; 1][..1] as f32) -> float32x4_t;
    /// Load an array of two `f32` elements and replicate to lanes of two registers.
    fn vld2q_dup_f32(_: &[f32; 2][..1] as [f32; 2]) -> float32x4x2_t;
    /// Load an array of three `f32` elements and replicate to lanes of three registers.
    fn vld3q_dup_f32(_: &[f32; 3][..1] as [f32; 3]) -> float32x4x3_t;
    /// Load an array of four `f32` elements and replicate to lanes of four registers.
    fn vld4q_dup_f32(_: &[f32; 4][..1] as [f32; 4]) -> float32x4x4_t;
}

#[cfg(test)]
mod tests {
    use core::arch::aarch64 as arch;

    // Generate a test for an intrinsic. The primary use of tests is that they execute under Miri,
    // which eliminates most forms of type confusion we could have inadvertently introduced by
    // mismatching intrinsic types and exposed memory types. The syntax for this macro also formats
    // more consistently under rustfmt (one line each).
    //
    // Safety: `base` must be a Pod (integer) type and `ty` must be a SIMD vector type
    macro_rules! test_vld1_from_slice {
        ($(#[$attr:meta])* fn $testname:ident, $intrinsic:ident, $base:ty, $ty:ty $(, $with:expr)?) => {
            #[test]
            #[cfg(target_feature = "neon")]
            $(#[$attr])*
            fn $testname() {
                fn assert_eq<const N: usize>(v: $ty, val: [$base; N]) {
                    assert!(core::mem::size_of::<$ty>() == core::mem::size_of::<[$base; N]>());
                    // Safety: transmuting a SIMD vector to its array representation which are Pod.
                    // This can not utilized `transmute` since the size of `val` is polymorphic,
                    // and the compiler rejects a transmute it can not statically prove to be
                    // between equivalently sized types, e.g. dependently sized type on `N`.
                    let v = unsafe { core::mem::transmute_copy::<$ty, [$base; N]>(&v) };
                    assert_eq!(v, val);
                }

                #[target_feature(enable = "neon")]
                fn test() {
                    let source = core::array::from_fn(|i| i as $base);
                    let argument = source;
                    $( // optionally we need to change the type from a flat array.
                        let argument = $with(argument);
                    )?
                    let result: $ty = super::$intrinsic(&argument);
                    assert_eq(result, source);
                }

                unsafe { test() }
            }
        };
    }

    test_vld1_from_slice!(fn test_vld1_u8, vld1_u8, u8, arch::uint8x8_t);
    test_vld1_from_slice!(fn test_vld1_i8, vld1_s8, i8, arch::int8x8_t);
    test_vld1_from_slice!(fn test_vld1_u16, vld1_u16, u16, arch::uint16x4_t);
    test_vld1_from_slice!(fn test_vld1_i16, vld1_s16, i16, arch::int16x4_t);
    test_vld1_from_slice!(fn test_vld1_u32, vld1_u32, u32, arch::uint32x2_t);
    test_vld1_from_slice!(fn test_vld1_i32, vld1_s32, i32, arch::int32x2_t);
    test_vld1_from_slice!(fn test_vld1_f32, vld1_f32, f32, arch::float32x2_t);
    test_vld1_from_slice!(fn test_vld1_u64, vld1_u64, u64, arch::uint64x1_t, |[val]: [_; 1]| val);
    test_vld1_from_slice!(fn test_vld1_i64, vld1_s64, i64, arch::int64x1_t, |[val]: [_; 1]| val);
    test_vld1_from_slice!(fn test_vld1_f64, vld1_f64, f64, arch::float64x1_t, |[val]: [_; 1]| val);

    fn as_chunks<T: Copy, const L: usize, const N: usize, const M: usize>(v: [T; N]) -> [[T; M]; L] {
        <[[T; M]; L]>::try_from(v.as_chunks::<M>().0).unwrap()
    }

    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u8_x2, vld1_u8_x2, u8, arch::uint8x8x2_t, as_chunks::<_, 2, 16, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i8_x2, vld1_s8_x2, i8, arch::int8x8x2_t, as_chunks::<_, 2, 16, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u16_x2, vld1_u16_x2, u16, arch::uint16x4x2_t, as_chunks::<_, 2, 8, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i16_x2, vld1_s16_x2, i16, arch::int16x4x2_t, as_chunks::<_, 2, 8, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u32_x2, vld1_u32_x2, u32, arch::uint32x2x2_t, as_chunks::<_, 2, 4, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i32_x2, vld1_s32_x2, i32, arch::int32x2x2_t, as_chunks::<_, 2, 4, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_f32_x2, vld1_f32_x2, f32, arch::float32x2x2_t, as_chunks::<_, 2, 4, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u64_x2, vld1_u64_x2, u64, arch::uint64x1x2_t);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i64_x2, vld1_s64_x2, i64, arch::int64x1x2_t);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_f64_x2, vld1_f64_x2, f64, arch::float64x1x2_t);

    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u8_x3, vld1_u8_x3, u8, arch::uint8x8x3_t, as_chunks::<_, 3, 24, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i8_x3, vld1_s8_x3, i8, arch::int8x8x3_t, as_chunks::<_, 3, 24, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u16_x3, vld1_u16_x3, u16, arch::uint16x4x3_t, as_chunks::<_, 3, 12, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i16_x3, vld1_s16_x3, i16, arch::int16x4x3_t, as_chunks::<_, 3, 12, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u32_x3, vld1_u32_x3, u32, arch::uint32x2x3_t, as_chunks::<_, 3, 6, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i32_x3, vld1_s32_x3, i32, arch::int32x2x3_t, as_chunks::<_, 3, 6, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_f32_x3, vld1_f32_x3, f32, arch::float32x2x3_t, as_chunks::<_, 3, 6, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u64_x3, vld1_u64_x3, u64, arch::uint64x1x3_t);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i64_x3, vld1_s64_x3, i64, arch::int64x1x3_t);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_f64_x3, vld1_f64_x3, f64, arch::float64x1x3_t);

    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u8_x4, vld1_u8_x4, u8, arch::uint8x8x4_t, as_chunks::<_, 4, 32, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i8_x4, vld1_s8_x4, i8, arch::int8x8x4_t, as_chunks::<_, 4, 32, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u16_x4, vld1_u16_x4, u16, arch::uint16x4x4_t, as_chunks::<_, 4, 16, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i16_x4, vld1_s16_x4, i16, arch::int16x4x4_t, as_chunks::<_, 4, 16, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u32_x4, vld1_u32_x4, u32, arch::uint32x2x4_t, as_chunks::<_, 4, 8, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i32_x4, vld1_s32_x4, i32, arch::int32x2x4_t, as_chunks::<_, 4, 8, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_f32_x4, vld1_f32_x4, f32, arch::float32x2x4_t, as_chunks::<_, 4, 8, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_u64_x4, vld1_u64_x4, u64, arch::uint64x1x4_t);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_i64_x4, vld1_s64_x4, i64, arch::int64x1x4_t);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1_f64_x4, vld1_f64_x4, f64, arch::float64x1x4_t);

    test_vld1_from_slice!(fn test_vld1q_u8, vld1q_u8, u8, arch::uint8x16_t);
    test_vld1_from_slice!(fn test_vld1q_i8, vld1q_s8, i8, arch::int8x16_t);
    test_vld1_from_slice!(fn test_vld1q_u16, vld1q_u16, u16, arch::uint16x8_t);
    test_vld1_from_slice!(fn test_vld1q_i16, vld1q_s16, i16, arch::int16x8_t);
    test_vld1_from_slice!(fn test_vld1q_u32, vld1q_u32, u32, arch::uint32x4_t);
    test_vld1_from_slice!(fn test_vld1q_i32, vld1q_s32, i32, arch::int32x4_t);
    test_vld1_from_slice!(fn test_vld1q_f32, vld1q_f32, f32, arch::float32x4_t);
    test_vld1_from_slice!(fn test_vld1q_u64, vld1q_u64, u64, arch::uint64x2_t);
    test_vld1_from_slice!(fn test_vld1q_i64, vld1q_s64, i64, arch::int64x2_t);
    test_vld1_from_slice!(fn test_vld1q_f64, vld1q_f64, f64, arch::float64x2_t);

    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u8_x2, vld1q_u8_x2, u8, arch::uint8x16x2_t, as_chunks::<_, 2, 32, 16>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i8_x2, vld1q_s8_x2, i8, arch::int8x16x2_t, as_chunks::<_, 2, 32, 16>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u16_x2, vld1q_u16_x2, u16, arch::uint16x8x2_t, as_chunks::<_, 2, 16, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i16_x2, vld1q_s16_x2, i16, arch::int16x8x2_t, as_chunks::<_, 2, 16, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u32_x2, vld1q_u32_x2, u32, arch::uint32x4x2_t, as_chunks::<_, 2, 8, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i32_x2, vld1q_s32_x2, i32, arch::int32x4x2_t, as_chunks::<_, 2, 8, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_f32_x2, vld1q_f32_x2, f32, arch::float32x4x2_t, as_chunks::<_, 2, 8, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u64_x2, vld1q_u64_x2, u64, arch::uint64x2x2_t, as_chunks::<_, 2, 4, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i64_x2, vld1q_s64_x2, i64, arch::int64x2x2_t, as_chunks::<_, 2, 4, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_f64_x2, vld1q_f64_x2, f64, arch::float64x2x2_t, as_chunks::<_, 2, 4, 2>);

    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u8_x3, vld1q_u8_x3, u8, arch::uint8x16x3_t,as_chunks::<_, 3, 48, 16>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i8_x3, vld1q_s8_x3, i8, arch::int8x16x3_t, as_chunks::<_, 3, 48, 16>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u16_x3, vld1q_u16_x3, u16, arch::uint16x8x3_t, as_chunks::<_, 3, 24, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i16_x3, vld1q_s16_x3, i16, arch::int16x8x3_t, as_chunks::<_, 3, 24, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u32_x3, vld1q_u32_x3, u32, arch::uint32x4x3_t, as_chunks::<_, 3, 12, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i32_x3, vld1q_s32_x3, i32, arch::int32x4x3_t, as_chunks::<_, 3, 12, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_f32_x3, vld1q_f32_x3, f32, arch::float32x4x3_t, as_chunks::<_, 3, 12, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u64_x3, vld1q_u64_x3, u64, arch::uint64x2x3_t, as_chunks::<_, 3, 6, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i64_x3, vld1q_s64_x3, i64, arch::int64x2x3_t, as_chunks::<_, 3, 6, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_f64_x3, vld1q_f64_x3, f64, arch::float64x2x3_t, as_chunks::<_, 3, 6, 2>);

    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u8_x4, vld1q_u8_x4, u8, arch::uint8x16x4_t, as_chunks::<_, 4, 64, 16>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i8_x4, vld1q_s8_x4, i8, arch::int8x16x4_t, as_chunks::<_, 4, 64, 16>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u16_x4, vld1q_u16_x4, u16, arch::uint16x8x4_t, as_chunks::<_, 4, 32, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i16_x4, vld1q_s16_x4, i16, arch::int16x8x4_t, as_chunks::<_, 4, 32, 8>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u32_x4, vld1q_u32_x4, u32, arch::uint32x4x4_t, as_chunks::<_, 4, 16, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i32_x4, vld1q_s32_x4, i32, arch::int32x4x4_t, as_chunks::<_, 4, 16, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_f32_x4, vld1q_f32_x4, f32, arch::float32x4x4_t, as_chunks::<_, 4, 16, 4>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_u64_x4, vld1q_u64_x4, u64, arch::uint64x2x4_t, as_chunks::<_, 4, 8, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_i64_x4, vld1q_s64_x4, i64, arch::int64x2x4_t, as_chunks::<_, 4, 8, 2>);
    test_vld1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vld1q_f64_x4, vld1q_f64_x4, f64, arch::float64x2x4_t, as_chunks::<_, 4, 8, 2>);

    // Generate a test for an intrinsic. The primary use of tests is that they execute under Miri,
    // which eliminates most forms of type confusion we could have inadvertently introduced by
    // mismatching intrinsic types and exposed memory types. The syntax for this macro also formats
    // more consistently under rustfmt (one line each).
    //
    // Safety: `base` must be a Pod (integer) type and `ty` must be a SIMD vector type
    macro_rules! test_vst1_from_slice {
        ($(#[$attr:meta])* fn $testname:ident, $intrinsic:ident, $base:ty, $ty:ty $(, $with:expr)?) => {
            #[test]
            #[cfg(target_feature = "neon")]
            $(#[$attr])*
            fn $testname() {
                fn generate<const N: usize>(val: &[$base; N]) -> $ty {
                    assert!(core::mem::size_of::<$ty>() == core::mem::size_of::<[$base; N]>());
                    // Safety: transmuting array representation to a SIMD vector, both are Pod.
                    // This can not utilize `transmute` since the size of `val` is polymorphic,
                    // and the compiler rejects a transmute it can not statically prove to be
                    // between equivalently sized types, e.g. dependently sized type on `N`.
                    unsafe { core::mem::transmute_copy::<[$base; N], $ty>(val) }
                }

                fn result_init<T>() -> T {
                    // Safety: only called on arrays out of `Pod` types. (See use below). This does
                    // not escape the macro hence only local use must be reviewed.
                    unsafe { core::mem::zeroed() }
                }

                // Help the type unification between source and result.
                fn assert_eq<T: PartialEq + core::fmt::Debug, const N: usize>(a: &[T; N], b: &[T; N]) {
                    assert_eq!(a, b);
                }

                #[target_feature(enable = "neon")]
                fn test() {
                    let ground_truth = core::array::from_fn(|i| i as $base);
                    let argument = generate(&ground_truth);

                    let mut result = result_init();
                    super::$intrinsic(&mut result, argument);

                    $( // optionally we need to change the type from a flat array.
                        let result = $with(result);
                    )?

                    assert_eq(&result, &ground_truth);
                }

                unsafe { test() }
            }
        };
    }

    test_vst1_from_slice!(fn test_vst1_u8, vst1_u8, u8, arch::uint8x8_t);
    test_vst1_from_slice!(fn test_vst1_i8, vst1_s8, i8, arch::int8x8_t);
    test_vst1_from_slice!(fn test_vst1_u16, vst1_u16, u16, arch::uint16x4_t);
    test_vst1_from_slice!(fn test_vst1_i16, vst1_s16, i16, arch::int16x4_t);
    test_vst1_from_slice!(fn test_vst1_u32, vst1_u32, u32, arch::uint32x2_t);
    test_vst1_from_slice!(fn test_vst1_i32, vst1_s32, i32, arch::int32x2_t);
    test_vst1_from_slice!(fn test_vst1_f32, vst1_f32, f32, arch::float32x2_t);
    test_vst1_from_slice!(fn test_vst1_u64, vst1_u64, u64, arch::uint64x1_t, |val| [val]);
    test_vst1_from_slice!(fn test_vst1_i64, vst1_s64, i64, arch::int64x1_t, |val| [val]);
    test_vst1_from_slice!(fn test_vst1_f64, vst1_f64, f64, arch::float64x1_t, |val| [val]);

    fn flatten<T: Copy, const L: usize, const N: usize, const M: usize>(v: [[T; M]; L]) -> [T; N] {
        <[T; N]>::try_from(v.as_flattened()).unwrap()
    }

    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u8_x2, vst1_u8_x2, u8, arch::uint8x8x2_t, flatten::<_, 2, 16, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i8_x2, vst1_s8_x2, i8, arch::int8x8x2_t, flatten::<_, 2, 16, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u16_x2, vst1_u16_x2, u16, arch::uint16x4x2_t, flatten::<_, 2, 8, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i16_x2, vst1_s16_x2, i16, arch::int16x4x2_t, flatten::<_, 2, 8, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u32_x2, vst1_u32_x2, u32, arch::uint32x2x2_t, flatten::<_, 2, 4, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i32_x2, vst1_s32_x2, i32, arch::int32x2x2_t, flatten::<_, 2, 4, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_f32_x2, vst1_f32_x2, f32, arch::float32x2x2_t, flatten::<_, 2, 4, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u64_x2, vst1_u64_x2, u64, arch::uint64x1x2_t);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i64_x2, vst1_s64_x2, i64, arch::int64x1x2_t);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_f64_x2, vst1_f64_x2, f64, arch::float64x1x2_t);

    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u8_x3, vst1_u8_x3, u8, arch::uint8x8x3_t, flatten::<_, 3, 24, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i8_x3, vst1_s8_x3, i8, arch::int8x8x3_t, flatten::<_, 3, 24, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u16_x3, vst1_u16_x3, u16, arch::uint16x4x3_t, flatten::<_, 3, 12, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i16_x3, vst1_s16_x3, i16, arch::int16x4x3_t, flatten::<_, 3, 12, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u32_x3, vst1_u32_x3, u32, arch::uint32x2x3_t, flatten::<_, 3, 6, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i32_x3, vst1_s32_x3, i32, arch::int32x2x3_t, flatten::<_, 3, 6, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_f32_x3, vst1_f32_x3, f32, arch::float32x2x3_t, flatten::<_, 3, 6, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u64_x3, vst1_u64_x3, u64, arch::uint64x1x3_t);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i64_x3, vst1_s64_x3, i64, arch::int64x1x3_t);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_f64_x3, vst1_f64_x3, f64, arch::float64x1x3_t);

    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u8_x4, vst1_u8_x4, u8, arch::uint8x8x4_t, flatten::<_, 4, 32, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i8_x4, vst1_s8_x4, i8, arch::int8x8x4_t, flatten::<_, 4, 32, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u16_x4, vst1_u16_x4, u16, arch::uint16x4x4_t, flatten::<_, 4, 16, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i16_x4, vst1_s16_x4, i16, arch::int16x4x4_t, flatten::<_, 4, 16, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u32_x4, vst1_u32_x4, u32, arch::uint32x2x4_t, flatten::<_, 4, 8, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i32_x4, vst1_s32_x4, i32, arch::int32x2x4_t, flatten::<_, 4, 8, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_f32_x4, vst1_f32_x4, f32, arch::float32x2x4_t, flatten::<_, 4, 8, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_u64_x4, vst1_u64_x4, u64, arch::uint64x1x4_t);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_i64_x4, vst1_s64_x4, i64, arch::int64x1x4_t);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1_f64_x4, vst1_f64_x4, f64, arch::float64x1x4_t);

    test_vst1_from_slice!(fn test_vst1q_u8, vst1q_u8, u8, arch::uint8x16_t);
    test_vst1_from_slice!(fn test_vst1q_i8, vst1q_s8, i8, arch::int8x16_t);
    test_vst1_from_slice!(fn test_vst1q_u16, vst1q_u16, u16, arch::uint16x8_t);
    test_vst1_from_slice!(fn test_vst1q_i16, vst1q_s16, i16, arch::int16x8_t);
    test_vst1_from_slice!(fn test_vst1q_u32, vst1q_u32, u32, arch::uint32x4_t);
    test_vst1_from_slice!(fn test_vst1q_i32, vst1q_s32, i32, arch::int32x4_t);
    test_vst1_from_slice!(fn test_vst1q_f32, vst1q_f32, f32, arch::float32x4_t);
    test_vst1_from_slice!(fn test_vst1q_u64, vst1q_u64, u64, arch::uint64x2_t);
    test_vst1_from_slice!(fn test_vst1q_i64, vst1q_s64, i64, arch::int64x2_t);
    test_vst1_from_slice!(fn test_vst1q_f64, vst1q_f64, f64, arch::float64x2_t);

    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u8_x2, vst1q_u8_x2, u8, arch::uint8x16x2_t, flatten::<_, 2, 32, 16>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i8_x2, vst1q_s8_x2, i8, arch::int8x16x2_t, flatten::<_, 2, 32, 16>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u16_x2, vst1q_u16_x2, u16, arch::uint16x8x2_t, flatten::<_, 2, 16, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i16_x2, vst1q_s16_x2, i16, arch::int16x8x2_t, flatten::<_, 2, 16, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u32_x2, vst1q_u32_x2, u32, arch::uint32x4x2_t, flatten::<_, 2, 8, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i32_x2, vst1q_s32_x2, i32, arch::int32x4x2_t, flatten::<_, 2, 8, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_f32_x2, vst1q_f32_x2, f32, arch::float32x4x2_t, flatten::<_, 2, 8, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u64_x2, vst1q_u64_x2, u64, arch::uint64x2x2_t, flatten::<_, 2, 4, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i64_x2, vst1q_s64_x2, i64, arch::int64x2x2_t, flatten::<_, 2, 4, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_f64_x2, vst1q_f64_x2, f64, arch::float64x2x2_t, flatten::<_, 2, 4, 2>);

    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u8_x3, vst1q_u8_x3, u8, arch::uint8x16x3_t, flatten::<_, 3, 48, 16>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i8_x3, vst1q_s8_x3, i8, arch::int8x16x3_t, flatten::<_, 3, 48, 16>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u16_x3, vst1q_u16_x3, u16, arch::uint16x8x3_t, flatten::<_, 3, 24, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i16_x3, vst1q_s16_x3, i16, arch::int16x8x3_t, flatten::<_, 3, 24, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u32_x3, vst1q_u32_x3, u32, arch::uint32x4x3_t, flatten::<_, 3, 12, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i32_x3, vst1q_s32_x3, i32, arch::int32x4x3_t, flatten::<_, 3, 12, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_f32_x3, vst1q_f32_x3, f32, arch::float32x4x3_t, flatten::<_, 3, 12, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u64_x3, vst1q_u64_x3, u64, arch::uint64x2x3_t, flatten::<_, 3, 6, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i64_x3, vst1q_s64_x3, i64, arch::int64x2x3_t, flatten::<_, 3, 6, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_f64_x3, vst1q_f64_x3, f64, arch::float64x2x3_t, flatten::<_, 3, 6, 2>);

    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u8_x4, vst1q_u8_x4, u8, arch::uint8x16x4_t, flatten::<_, 4, 64, 16>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i8_x4, vst1q_s8_x4, i8, arch::int8x16x4_t, flatten::<_, 4, 64, 16>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u16_x4, vst1q_u16_x4, u16, arch::uint16x8x4_t, flatten::<_, 4, 32, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i16_x4, vst1q_s16_x4, i16, arch::int16x8x4_t, flatten::<_, 4, 32, 8>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u32_x4, vst1q_u32_x4, u32, arch::uint32x4x4_t, flatten::<_, 4, 16, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i32_x4, vst1q_s32_x4, i32, arch::int32x4x4_t, flatten::<_, 4, 16, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_f32_x4, vst1q_f32_x4, f32, arch::float32x4x4_t, flatten::<_, 4, 16, 4>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_u64_x4, vst1q_u64_x4, u64, arch::uint64x2x4_t, flatten::<_, 4, 8, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_i64_x4, vst1q_s64_x4, i64, arch::int64x2x4_t, flatten::<_, 4, 8, 2>);
    test_vst1_from_slice!(#[cfg_attr(miri, ignore)] fn test_vst1q_f64_x4, vst1q_f64_x4, f64, arch::float64x2x4_t, flatten::<_, 4, 8, 2>);
}

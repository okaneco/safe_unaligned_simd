//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -C llvm-args=-x86-asm-syntax=intel
//@ compile-flags: -Copt-level=3
//@ only: x86_64

extern crate safe_unaligned_simd;

use safe_unaligned_simd::x86_64 as simd;
use std::arch::x86_64::{__m128d, __m128i};

// See the note in `sse2-loads.rs`.

// // Manual expects: movsd
// CHECK-LABEL: _mm_store_sd
// CHECK: movsd
#[no_mangle]
pub fn _mm_store_sd(mem_addr: &mut f64, a: __m128d) {
    unsafe { simd::_mm_store_sd(mem_addr, a) }
}

// FIXME: Test may require blackbox or more code
// Manual expects: movhpd
// _mm_storeh_pd
// movhps
// #[no_mangle]
// pub fn _mm_storeh_pd(mem_addr: &mut f64, a: __m128d) {
//     unsafe { simd::_mm_storeh_pd(mem_addr, a) }
// }

// No particular instruction
// pub fn _mm_storel_epi64(mem_addr: &mut [u8; 16], a: __m128i) {
//     unsafe { simd::_mm_storel_epi64(mem_addr, a) }
// }

// Manual expects: movlpd
// CHECK-LABEL: _mm_storel_pd
// CHECK: movlps
#[no_mangle]
pub fn _mm_storel_pd(mem_addr: &mut f64, a: __m128d) {
    unsafe { simd::_mm_storel_pd(mem_addr, a) }
}

// FIXME: Function seems to get optimized out, extract to own test
// Manual expects: movupd
// _mm_storeu_pd
// movups
// #[no_mangle]
// pub fn _mm_storeu_pd(mem_addr: &mut [f64; 2], a: __m128d) {
//     unsafe { simd::_mm_storeu_pd(mem_addr, a) }
// }

// Manual expects: movdqu
// CHECK-LABEL: _mm_storeu_si128
// CHECK: movups
#[no_mangle]
pub fn _mm_storeu_si128(mem_addr: &mut [i64; 2], a: __m128i) {
    unsafe { simd::_mm_storeu_si128(mem_addr, a) }
}

// Sequence, no particular instruction to test
// pub fn _mm_storeu_si16<T: Is16BitsUnaligned>(mem_addr: &mut T, a: __m128i) {
//     unsafe { simd::_mm_storeu_si16(mem_addr, a) }
// }

// No particular instruction
// pub fn _mm_storeu_si32(mem_addr: &mut [i32; 1], a: __m128i) {
//     unsafe { simd::_mm_storeu_si32(mem_addr, a) }
// }

// No particular instruction
// pub fn _mm_storeu_si64(mem_addr: &mut [u8; 8], a: __m128i) {
//     unsafe { simd::_mm_storeu_si64(mem_addr, a) }
// }

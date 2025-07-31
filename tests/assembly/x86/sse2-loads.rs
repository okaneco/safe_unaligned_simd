//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -C llvm-args=-x86-asm-syntax=intel
//@ compile-flags: -Copt-level=3
//@ only: x86_64

extern crate safe_unaligned_simd;

use safe_unaligned_simd::x86_64 as simd;
use std::arch::x86_64::{__m128d, __m128i};

// Note, LLVM doesn't emit the same instructions as those listed in the
// intrinsics manual. For example, many double types emit single instructions.

// CHECK-LABEL: _mm_load_sd
// CHECK: movsd
#[no_mangle]
pub fn _mm_load_sd(mem_addr: &f64) -> __m128d {
    unsafe { simd::_mm_load_sd(mem_addr) }
}

// Listed as an LLVM FIXME in std::arch, differing codegen than expected
// pub fn _mm_load1_pd(mem_addr: &f64) -> __m128d {
//     unsafe { simd::_mm_load1_pd(mem_addr) }
// }

// Manual expects: movhpd
// CHECK-LABEL: _mm_loadh_pd
// CHECK: movhps
#[no_mangle]
pub fn _mm_loadh_pd(a: __m128d, mem_addr: &f64) -> __m128d {
    unsafe { simd::_mm_loadh_pd(a, mem_addr) }
}

// No particular instruction
// pub fn _mm_loadl_epi64(mem_addr: &[u16; 8]) -> __m128i {
//     unsafe { simd::_mm_loadl_epi64(mem_addr) }
// }

// Manual expects: movlpd
// CHECK-LABEL: _mm_loadl_pd
// CHECK: movlps
#[no_mangle]
pub fn _mm_loadl_pd(a: __m128d, mem_addr: &f64) -> __m128d {
    unsafe { simd::_mm_loadl_pd(a, mem_addr) }
}

// Manual expects: movupd
// CHECK-LABEL: _mm_loadu_pd
// CHECK: movups
#[no_mangle]
pub fn _mm_loadu_pd(mem_addr: &[f64; 2]) -> __m128d {
    unsafe { simd::_mm_loadu_pd(mem_addr) }
}

// Manual expects: movdqu
// CHECK-LABEL: _mm_loadu_si128
// CHECK: movups
#[no_mangle]
pub fn _mm_loadu_si128(mem_addr: &[i32; 4]) -> __m128i {
    unsafe { simd::_mm_loadu_si128(mem_addr) }
}

// Sequence, no particular instruction to test
// pub fn _mm_loadu_si16<T: Is16BitsUnaligned>(mem_addr: &T) -> __m128i {
//     unsafe { simd::_mm_loadu_si16(mem_addr) }
// }

// No particular instruction
// pub fn _mm_loadu_si32(mem_addr: &[i8; 4]) -> __m128i {
//     unsafe { simd::_mm_loadu_si32(mem_addr) }
// }

// No particular instruction
// pub fn _mm_loadu_si64(mem_addr: &[i16; 4]) -> __m128i {
//     unsafe { simd::_mm_loadu_si64(mem_addr) }
// }

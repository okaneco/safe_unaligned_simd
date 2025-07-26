//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -C llvm-args=-x86-asm-syntax=intel
//@ compile-flags: -Copt-level=3 --target=x86_64-unknown-linux-gnu

extern crate safe_unaligned_simd;

use safe_unaligned_simd::x86_64 as simd;
use std::arch::x86_64::__m128;

// CHECK-LABEL: _mm_load1_ps:
// CHECK: movss
#[no_mangle]
pub fn _mm_load1_ps(mem_addr: &f32) -> __m128 {
    unsafe { simd::_mm_load1_ps(mem_addr) }
}

// CHECK-LABEL: _mm_load_ss:
// CHECK: movss
#[no_mangle]
pub fn _mm_load_ss(mem_addr: &f32) -> __m128 {
    unsafe { simd::_mm_load_ss(mem_addr) }
}

// CHECK-LABEL: _mm_loadu_ps:
// CHECK: movups
#[no_mangle]
pub fn _mm_loadu_ps(mem_addr: &[f32; 4]) -> __m128 {
    unsafe { simd::_mm_loadu_ps(mem_addr) }
}

// CHECK-LABEL: _mm_store_ss:
// CHECK: movss
#[no_mangle]
pub fn _mm_store_ss(mem_addr: &mut f32, a: __m128) {
    unsafe { simd::_mm_store_ss(mem_addr, a) }
}

// CHECK-LABEL: _mm_storeu_ps:
// CHECK: movups
#[no_mangle]
#[target_feature(enable = "sse")]
pub fn _mm_storeu_ps(mem_addr: &mut [f32; 4], a: __m128) {
    unsafe { simd::_mm_storeu_ps(mem_addr, a) }
}

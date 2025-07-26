//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -C llvm-args=-x86-asm-syntax=intel
//@ compile-flags: -Copt-level=3
//@ only-x86_64

extern crate safe_unaligned_simd;

use safe_unaligned_simd::x86_64 as simd;
use std::arch::x86_64::__m128;

// SAFETY: x86_64 requires `sse` and `sse2` so they are safe to call.

// CHECK-LABEL: _mm_load1_ps:
// CHECK: movss
#[no_mangle]
pub fn _mm_load1_ps(mem_addr: &f32) -> __m128 {
    unsafe { simd::_mm_load1_ps(mem_addr) }
}

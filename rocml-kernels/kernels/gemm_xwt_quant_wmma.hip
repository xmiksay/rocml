// Default-tile-config WMMA kernels: TILE_M=128, WARPS_M=4, WARPS_N=4 (2x2
// accumulator fragments/warp) — the per-wave-efficiency round's winning
// config for every real ornith-9b `m` except `m=1024`. See
// `gemm_xwt_wmma_impl.h` (the shared implementation both this file and
// `gemm_xwt_quant_wmma_narrow.hip` instantiate) for the full algorithm,
// fragment-layout, and tuning-round doc history. `rocml/src/forward/
// kernels_quant.rs` dispatches here for `m >= 2048`; below that it uses
// `gemm_xwt_quant_wmma_narrow.hip`'s kernels instead.
#include "gemm_xwt_wmma_impl.h"

extern "C" __global__ void gemm_xwt_wmma_q8_0(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<32, 34, 128, 128, 4, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ8_0{});
}

extern "C" __global__ void gemm_xwt_wmma_q4_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 144, 128, 128, 4, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ4K{});
}

extern "C" __global__ void gemm_xwt_wmma_q5_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 176, 128, 128, 4, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ5K{});
}

extern "C" __global__ void gemm_xwt_wmma_q6_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 210, 128, 128, 4, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ6K{});
}

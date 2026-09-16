// Narrow-`m` WMMA kernels: TILE_M=64, WARPS_M=8, WARPS_N=2 (the original
// 1x2-accumulator-fragment config from before the per-wave-efficiency
// round's TILE_M=128 rewrite). Shape-aware dispatch round (issue #6): the
// per-wave-efficiency round's 2x2/TILE_M=128 config (`gemm_xwt_quant_wmma.
// hip`) regressed ornith-9b's smallest real WMMA-eligible `m` (1024,
// attn_k/attn_v) by ~20% — TILE_M=128 nearly halves `grid.x` there, and the
// GPU is too grid-starved to fill. This file keeps the narrower config
// alive as a real dispatch target instead of just a historical footnote.
//
// Crossover measured via a standalone interleaved hipcc harness (rows=512,
// q4_k, both configs including the dequant-off-critical-path round's
// vec4-X-load lever) sweeping m in {512,768,1024,1280,1536,1792,2048,3072,
// 4096}:
//
// | m    | narrow (this file) | wide (TILE_M=128) | narrow wins by |
// |------|---------------------|--------------------|----------------|
// | 512  | 5200.1              | 4878.1             | +6.6%          |
// | 768  | 8217.2              | 7650.0             | +7.4%          |
// | 1024 | 10338.9             | 7896.2             | +30.9%         |
// | 1280 | 12858.8             | 9901.0             | +29.9%         |
// | 1536 | 13250.9             | 11581.3            | +14.4%         |
// | 1792 | 15087.0             | 13004.2            | +16.0%         |
// | 2048 | 12490.2             | 12680.0            | -1.5% (wide wins) |
// | 3072 | 14755.0             | 16114.4            | -8.4% (wide wins) |
// | 4096 | 15057.6             | 15465.5            | -2.6% (wide wins) |
//
// Crossover falls cleanly between 1792 and 2048 — `rocml/src/forward/
// kernels_quant.rs` dispatches to this file's kernels for `m < 2048`,
// matching the round's own working hypothesis. See `gemm_xwt_wmma_impl.h`
// for the shared algorithm both this file and `gemm_xwt_quant_wmma.hip`
// instantiate.
#include "gemm_xwt_wmma_impl.h"

extern "C" __global__ void gemm_xwt_wmma_q8_0_narrow(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<32, 34, 128, 64, 8, 2>(x, (const unsigned char*)w, out, rows, m, n, DequantQ8_0{});
}

extern "C" __global__ void gemm_xwt_wmma_q4_k_narrow(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 144, 128, 64, 8, 2>(x, (const unsigned char*)w, out, rows, m, n, DequantQ4K{});
}

extern "C" __global__ void gemm_xwt_wmma_q5_k_narrow(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 176, 128, 64, 8, 2>(x, (const unsigned char*)w, out, rows, m, n, DequantQ5K{});
}

extern "C" __global__ void gemm_xwt_wmma_q6_k_narrow(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 210, 128, 64, 8, 2>(x, (const unsigned char*)w, out, rows, m, n, DequantQ6K{});
}

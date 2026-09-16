// Micro-row-tile WMMA kernels (qwen35moe M4, lever 1): TILE_ROWS=16 (WMMA's
// minimum — one 16-row fragment), TILE_M=64, WARPS_M=1, WARPS_N=4. Targets
// qwen35moe's grouped-by-expert batched GEMM (`moe_chunk.rs`): a 512-token
// chunk's `chunk_len * top_k` (token, expert) assignments spread over up to
// 256 experts average only ~2-16 rows per expert group, far below the
// default/narrow configs' `rows >= 128` dispatch floor — every such call
// fell through to the scalar `gemm_xwt_q*` kernel even though the model's
// two expert-projection shapes (gate/up `m=expert_ff_len`=512, down
// `m=hidden`=2048 on Ornith-1.5-35B-A3B) are otherwise WMMA-friendly.
//
// `WM`=1 is forced: `TR/16`=1 has no divisor other than 1 (see
// `gemm_xwt_wmma_impl.h`'s module doc and its `static_assert`s). `TM`=64/
// `WN`=4 (rather than `TM`=128/`WN`=8) was picked to maximize `grid.x` at
// this shape's small `m`s (512/2048) — more, smaller blocks fill the GPU
// better than fewer, larger ones when `grid.y` (`rows_e.div_ceil(16)`) is
// already tiny for a typical expert group. `rocml/src/forward/
// kernels_quant_dispatch.rs`'s `gemm_micro` dispatches here only when the
// caller explicitly opts in (`allow_micro`, set only by qwen35moe's grouped
// GEMM call site) and the shape doesn't already qualify for the default/
// narrow WMMA tile — see that function's module doc for the measured
// crossover and why this stays opt-in rather than a blanket small-`rows`
// dispatch for every caller in the codebase.
#include "gemm_xwt_wmma_impl.h"

extern "C" __global__ void gemm_xwt_wmma_q8_0_micro(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<32, 34, 16, 64, 1, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ8_0{});
}

extern "C" __global__ void gemm_xwt_wmma_q4_k_micro(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 144, 16, 64, 1, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ4K{});
}

extern "C" __global__ void gemm_xwt_wmma_q5_k_micro(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 176, 16, 64, 1, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ5K{});
}

extern "C" __global__ void gemm_xwt_wmma_q6_k_micro(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 210, 16, 64, 1, 4>(x, (const unsigned char*)w, out, rows, m, n, DequantQ6K{});
}

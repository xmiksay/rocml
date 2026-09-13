// Split-K instantiations of `gemm_xwt_wmma_splitk_impl.h`, using the default
// (TILE_M=128, WARPS_M=4, WARPS_N=4) tile config — the same one
// `gemm_xwt_quant_wmma.hip` uses for `m >= 2048`, since split-K's own
// eligibility condition (`rocml/src/forward/kernels_quant_dispatch.rs`)
// only ever engages for shapes already routed to that config (ffn-down's
// class: `m` big enough for the default tile, `n` large, but too few grid
// blocks — see that header's module doc). `partial_out` is
// `[num_splits, rows, m]` f32; `gemm_splitk_reduce_f32`
// (`kernels/gemm_splitk_reduce.hip`) sums it down to the real `[rows, m]`
// output.
#include "gemm_xwt_wmma_splitk_impl.h"

extern "C" __global__ void gemm_xwt_wmma_splitk_q8_0(
    const float* x, const void* w, float* partial_out, unsigned rows, unsigned m, unsigned n,
    unsigned num_splits) {
    gemm_xwt_wmma_splitk_impl<32, 34, 128, 4, 4>(
        x, (const unsigned char*)w, partial_out, rows, m, n, num_splits, DequantQ8_0{});
}

extern "C" __global__ void gemm_xwt_wmma_splitk_q4_k(
    const float* x, const void* w, float* partial_out, unsigned rows, unsigned m, unsigned n,
    unsigned num_splits) {
    gemm_xwt_wmma_splitk_impl<256, 144, 128, 4, 4>(
        x, (const unsigned char*)w, partial_out, rows, m, n, num_splits, DequantQ4K{});
}

extern "C" __global__ void gemm_xwt_wmma_splitk_q5_k(
    const float* x, const void* w, float* partial_out, unsigned rows, unsigned m, unsigned n,
    unsigned num_splits) {
    gemm_xwt_wmma_splitk_impl<256, 176, 128, 4, 4>(
        x, (const unsigned char*)w, partial_out, rows, m, n, num_splits, DequantQ5K{});
}

extern "C" __global__ void gemm_xwt_wmma_splitk_q6_k(
    const float* x, const void* w, float* partial_out, unsigned rows, unsigned m, unsigned n,
    unsigned num_splits) {
    gemm_xwt_wmma_splitk_impl<256, 210, 128, 4, 4>(
        x, (const unsigned char*)w, partial_out, rows, m, n, num_splits, DequantQ6K{});
}

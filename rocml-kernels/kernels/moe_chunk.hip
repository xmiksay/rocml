// Chunked-prefill batching for qwen35moe (M3): grouped-by-expert GEMM
// support. M1/M2 (`moe.hip`) only ever process one token (row) at a time —
// this file adds the pieces a whole-chunk, tokens-sorted-by-expert design
// needs that `moe.hip`/`gemm_xwt_quant.hip` don't already provide:
//
// - `gemm_xwt_f32`: batched `out[rows,m] = X[rows,n] * W[m,n]^T`, both
//   operands plain f32 — the router (`ffn_gate_inp`, [expert_count,hidden])
//   and shared-expert gate (`ffn_gate_inp_shexp`, [1,hidden]) are the only
//   two weights in this architecture stored as raw f32 rather than a
//   quantized/f16 `LinearWeight`, so neither `gemm_xwt_f16` (expects
//   `__half`) nor `gemm_xwt_q*` (expects a quant block layout) applies.
//   Mirrors `gemm_xwt_f16.hip`'s TILE=16 shared-memory tiling exactly, minus
//   the f16->f32 conversion.
// - `moe_gather_rows_f32`/`moe_scatter_weighted_accum_f32`: the
//   llama.cpp-`mul_mat_id`-style gather/scatter a grouped-by-expert batched
//   GEMM needs — gather the subset of a chunk's rows routed to one expert
//   into a contiguous buffer before the batched GEMM, then scatter each
//   expert's weighted output back into the chunk-wide accumulator at the
//   original row positions (a token can appear in more than one expert's
//   group, so this must be `+=`, never a plain store).
// - `moe_shared_gate_write_chunk_f32`: batched sibling of `moe.hip`'s
//   `moe_shared_gate_write_f32` — one sigmoid gate value per row instead of
//   a single scalar shared by every row.
#include <hip/hip_runtime.h>

#define MOE_CHUNK_TILE 16

extern "C" __global__ void gemm_xwt_f32(
    const float* x, const float* w, float* out, unsigned rows, unsigned m, unsigned n) {
    __shared__ float xs[MOE_CHUNK_TILE][MOE_CHUNK_TILE];
    __shared__ float ws[MOE_CHUNK_TILE][MOE_CHUNK_TILE];

    unsigned row = blockIdx.y * MOE_CHUNK_TILE + threadIdx.y;
    unsigned col = blockIdx.x * MOE_CHUNK_TILE + threadIdx.x;

    float acc = 0.0f;
    unsigned num_tiles = (n + MOE_CHUNK_TILE - 1) / MOE_CHUNK_TILE;
    for (unsigned t = 0; t < num_tiles; ++t) {
        unsigned k_x = t * MOE_CHUNK_TILE + threadIdx.x;
        unsigned k_w = t * MOE_CHUNK_TILE + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] = (row < rows && k_x < n) ? x[(size_t)row * n + k_x] : 0.0f;
        ws[threadIdx.y][threadIdx.x] = (col < m && k_w < n) ? w[(size_t)col * n + k_w] : 0.0f;
        __syncthreads();

        for (unsigned kk = 0; kk < MOE_CHUNK_TILE; ++kk) {
            acc += xs[threadIdx.y][kk] * ws[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && col < m) {
        out[(size_t)row * m + col] = acc;
    }
}

// dst[i, :] = src[row_idx[i], :] for i in [0, rows) — gathers one expert
// group's assigned chunk rows out of the chunk-wide `[chunk_len, cols]`
// activation buffer into a contiguous `[rows, cols]` staging buffer the
// batched GEMM can read.
extern "C" __global__ void moe_gather_rows_f32(
    const float* src, const unsigned* row_idx, float* dst, unsigned rows, unsigned cols) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = rows * cols;
    if (idx >= total) {
        return;
    }
    unsigned c = idx % cols;
    unsigned i = idx / cols;
    dst[(size_t)i * cols + c] = src[(size_t)row_idx[i] * cols + c];
}

// dst[row_idx[i], :] += weight[i] * src[i, :] for i in [0, rows) — scatters
// one expert group's weighted GEMM output back into the chunk-wide
// accumulator. Must be `+=`: top-k routing means the same destination row
// can receive contributions from up to `top_k` different expert groups
// (never from two rows of the *same* group, so no cross-thread race on a
// single destination element within one launch).
extern "C" __global__ void moe_scatter_weighted_accum_f32(
    const float* src, const unsigned* row_idx, const float* weight, float* dst, unsigned rows,
    unsigned cols) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = rows * cols;
    if (idx >= total) {
        return;
    }
    unsigned c = idx % cols;
    unsigned i = idx / cols;
    dst[(size_t)row_idx[i] * cols + c] += weight[i] * src[(size_t)i * cols + c];
}

// out[r, :] = y[r, :] * sigmoid(gate_logit[r]) — batched sibling of
// `moe.hip`'s `moe_shared_gate_write_f32` (one gate value per row instead of
// a single scalar).
extern "C" __global__ void moe_shared_gate_write_chunk_f32(
    const float* y, const float* gate_logit, float* out, unsigned rows, unsigned cols) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = rows * cols;
    if (idx >= total) {
        return;
    }
    unsigned c = idx % cols;
    unsigned r = idx / cols;
    float w = 1.0f / (1.0f + expf(-gate_logit[r]));
    out[(size_t)r * cols + c] = y[(size_t)r * cols + c] * w;
}

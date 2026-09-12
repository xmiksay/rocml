// Flash-attention-2-style tiled prefill attention with split-K over the
// KV/depth axis (issue #6 flash-prefill round) — `attn_prefill.hip`'s
// ROW_TILE row-tiling design (proven: shares one K/V-tile load across
// `ROW_TILE` query rows via hardware threads, `blockDim.z`, not a register
// loop) extended with `attn_decode`'s split-K/online-softmax-merge pattern.
//
// **Why split-K instead of a bigger ROW_TILE alone**: the row-tiling round
// already measured that growing `ROW_TILE` past 4 *loses* more grid
// parallelism (fewer row-tiles per chunk = fewer independent workgroups)
// than the extra K/V-tile reuse buys back in bandwidth (see
// `attn_prefill.hip`'s own module doc: "ROW_TILE=8... reduces grid
// parallelism enough to roughly cancel the bandwidth win"). Splitting the
// (causal, per-row-varying) KV range into `n_splits` independent
// partial-softmax passes manufactures back that grid parallelism —
// multiplying, not trading off against, `ROW_TILE`'s reuse win — the same
// trade `attn_decode` already made for single-token decode when
// `n_kv_heads` alone wasn't enough workgroups.
//
// **A register-tiled first draft of this file was tried and measured
// worse than the ROW_TILE=4 baseline it was meant to replace** (ornith-9b
// prefill 485->334 tok/s @ depth 2048): giving every query row in the tile
// its own *register-resident* (m, l, acc[head_dim/32]) state (`BR`=8 rows
// unrolled in one warp) cost roughly 8x the accumulator registers of the
// thread-parallel design (~64-144 VGPR/lane vs ~18), which throttled
// occupancy far more than the extra K/V reuse saved — the exact same
// "bigger tile trades away more grid/wave parallelism than it buys in
// bandwidth" trap the row-tiling round already hit, just paid in registers
// instead of blocks. Reverted in favor of this file's current design,
// which keeps the original kernel's proven-cheap per-lane register
// footprint and gets its parallelism-at-depth entirely from split-K.
//
// **WMMA was evaluated and rejected for this kernel** (see the issue's own
// design outline, which asks for it to be "sweep[t]/justif[ied]", not
// assumed): baseline profiling on real ornith-9b @ depth 8192/16384 shows
// prefill attention at 68-72% of the *bandwidth* roofline but only 4.9-5.1%
// of the *FLOP* roofline — this op is decisively memory-bound, not
// compute-bound, so swapping the QK^T/P*V scalar dot products for matrix
// core ops would speed up an ALU stage that was never the bottleneck.
//
// **Dispatch**: `attention_chunk.rs` picks this kernel only when the
// chunk's deepest row's causal bound (`pos_base + chunk_len`) clears
// `USE_FLASH_PREFILL_MIN_DEPTH`, falling back to `attn_prefill.hip`'s
// single-pass kernel for shallow chunks, where the extra reduce launch
// isn't worth it.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_common.h"

#define MAX_LANE_ELEMS 8
// Rows of K/V staged into LDS at a time — same role as `attn_prefill.hip`'s
// `TILE_T`. Must match the Rust launcher's `ATTN_PREFILL_FLASH_BC` constant.
#define BC 16
// Consecutive query rows sharing one workgroup's K/V-tile load, handled via
// `blockDim.z` (hardware threads, not a register loop — see the module doc
// above for why). Must match the Rust launcher's `ATTN_PREFILL_FLASH_BR`
// constant.
#define BR 8

template <typename T>
__device__ __forceinline__ float load_kv(const T* plane, size_t idx);

template <>
__device__ __forceinline__ float load_kv<float>(const float* plane, size_t idx) {
    return plane[idx];
}

template <>
__device__ __forceinline__ float load_kv<__half>(const __half* plane, size_t idx) {
    return __half2float(plane[idx]);
}

// One workgroup per (kv head, row-tile of up to BR query rows, KV split).
// `q` is `[chunk_len, n_heads, head_dim]`; `k_layer`/`v_layer` are one
// layer's whole KV-cache buffer, `[n_kv_heads, max_seq, head_dim]`.
// `partial_out`/`partial_m`/`partial_l` are `[chunk_len, n_heads, n_splits,
// ...]`, row-major over the *global* row index `i` — `split_len` is derived
// from the *chunk's* deepest row (`pos_base + chunk_len`), not this
// row-tile's own, so every row-tile in the launch shares the same split
// boundaries; shallow row-tiles just see most of their high-numbered
// splits contribute nothing (`start >= end` below), which is correct by
// construction, not a special case — identical in spirit to
// `attn_decode_partial_f32`'s own handling of the same situation.
template <typename T>
__device__ __forceinline__ void attn_prefill_flash_partial_impl(
    const float* q, const T* k_layer, const T* v_layer, float* partial_out, float* partial_m,
    float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim, unsigned max_seq,
    unsigned chunk_len, unsigned pos_base, unsigned split_len, unsigned n_splits, float scale) {
    unsigned kvh = blockIdx.x;
    unsigned row_base = blockIdx.y * BR;
    unsigned s = blockIdx.z;
    if (kvh >= n_kv_heads || row_base >= chunk_len || s >= n_splits) {
        return;
    }
    unsigned n_rows = min((unsigned)BR, chunk_len - row_base);

    unsigned lane = threadIdx.x;
    unsigned g = threadIdx.y;
    unsigned r = threadIdx.z;
    unsigned tid = (r * blockDim.y + g) * 32 + lane;
    unsigned total_threads = blockDim.x * blockDim.y * blockDim.z;
    unsigned h = kvh * group + g;
    unsigned n_heads = n_kv_heads * group;
    bool active = r < n_rows;
    unsigned i = row_base + r;

    // This row's own causal bound, clipped to this split's assigned range.
    // The tile-load loop below must run the same number of outer
    // iterations for every thread in the block, so it's bounded by the
    // *deepest* active row's clipped end, not this thread's own — same
    // structure as `attn_prefill.hip`'s `end`/`end_max`.
    unsigned row_end = pos_base + i + 1;
    unsigned end_max = pos_base + row_base + n_rows;
    unsigned start = s * split_len;
    unsigned end = min(start + split_len, end_max);
    unsigned row_end_clipped = min(row_end, end);

    extern __shared__ float smem[];
    float* k_tile = smem;
    float* v_tile = smem + (size_t)BC * head_dim;

    const float* q_h = q + ((size_t)i * n_heads + h) * head_dim;
    const T* k_plane = k_layer + (size_t)kvh * max_seq * head_dim;
    const T* v_plane = v_layer + (size_t)kvh * max_seq * head_dim;

    float q_reg[MAX_LANE_ELEMS];
    unsigned n_elems = 0;
    if (active) {
        for (unsigned d = lane; d < head_dim; d += 32) {
            q_reg[n_elems++] = q_h[d];
        }
    }

    float acc[MAX_LANE_ELEMS];
    for (unsigned e = 0; e < MAX_LANE_ELEMS; ++e) {
        acc[e] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (unsigned tile_base = start; tile_base < end; tile_base += BC) {
        unsigned tile_len = min((unsigned)BC, end - tile_base);

        // Cooperative load spans every thread in the block regardless of
        // (g, r) — one shared copy for the whole (row tile x split), not
        // per-row.
        for (unsigned idx = tid; idx < tile_len * head_dim; idx += total_threads) {
            unsigned row = idx / head_dim;
            unsigned col = idx % head_dim;
            size_t src = (size_t)(tile_base + row) * head_dim + col;
            k_tile[idx] = load_kv(k_plane, src);
            v_tile[idx] = load_kv(v_plane, src);
        }
        __syncthreads();

        if (active) {
            for (unsigned t = 0; t < tile_len; ++t) {
                if (tile_base + t >= row_end_clipped) {
                    break;
                }
                const float* k_row = k_tile + (size_t)t * head_dim;
                float partial = 0.0f;
                unsigned e = 0;
                for (unsigned d = lane; d < head_dim; d += 32, ++e) {
                    partial += q_reg[e] * k_row[d];
                }
                float score = warp_reduce_sum(partial) * scale;

                float new_m = fmaxf(m, score);
                float correction = expf(m - new_m);
                float p = expf(score - new_m);
                l = l * correction + p;

                const float* v_row = v_tile + (size_t)t * head_dim;
                e = 0;
                for (unsigned d = lane; d < head_dim; d += 32, ++e) {
                    acc[e] = acc[e] * correction + p * v_row[d];
                }
                m = new_m;
            }
        }
        __syncthreads();
    }

    if (active) {
        size_t base = ((size_t)i * n_heads + h) * n_splits + s;
        if (lane == 0) {
            partial_m[base] = m;
            partial_l[base] = l;
        }
        unsigned e = 0;
        float* out_row = partial_out + base * head_dim;
        for (unsigned d = lane; d < head_dim; d += 32, ++e) {
            out_row[d] = acc[e];
        }
    }
}

extern "C" __global__ void attn_prefill_flash_partial_f32(
    const float* q, const float* k_layer, const float* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_impl<float>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, chunk_len, pos_base, split_len, n_splits, scale);
}

extern "C" __global__ void attn_prefill_flash_partial_f16(
    const float* q, const __half* k_layer, const __half* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_impl<__half>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, chunk_len, pos_base, split_len, n_splits, scale);
}

// Merges one (row, head)'s per-split partials with the log-sum-exp rule —
// the same algebra as `attn_decode_reduce_f32`, with an added leading row
// axis (`blockIdx.x`) since a prefill chunk covers many rows per launch,
// not decode's single token. Dtype-independent (only reads the f32 partial
// scratch buffers this kernel's partial siblings write), shared by both KV
// dtypes.
extern "C" __global__ void attn_prefill_flash_reduce_f32(
    const float* partial_out, const float* partial_m, const float* partial_l, float* out,
    unsigned n_heads, unsigned head_dim, unsigned n_splits) {
    unsigned i = blockIdx.x;
    unsigned h = blockIdx.y;
    unsigned d = threadIdx.x;
    size_t row_base = ((size_t)i * n_heads + h) * n_splits;

    __shared__ float global_m;
    __shared__ float global_l;
    if (d == 0) {
        float m = -INFINITY;
        for (unsigned s = 0; s < n_splits; ++s) {
            m = fmaxf(m, partial_m[row_base + s]);
        }
        float l = 0.0f;
        for (unsigned s = 0; s < n_splits; ++s) {
            l += partial_l[row_base + s] * expf(partial_m[row_base + s] - m);
        }
        global_m = m;
        // Every real row has at least one valid causal position (row i's
        // own newly-appended K/V), so l == 0 can't happen in practice; the
        // fallback only guards a synthetic all-empty test input from 0/0.
        global_l = (l > 0.0f) ? l : 1.0f;
    }
    __syncthreads();

    if (d >= head_dim) {
        return;
    }
    float acc = 0.0f;
    for (unsigned s = 0; s < n_splits; ++s) {
        float w = expf(partial_m[row_base + s] - global_m);
        acc += partial_out[(row_base + s) * head_dim + d] * w;
    }
    out[((size_t)i * n_heads + h) * head_dim + d] = acc / global_l;
}

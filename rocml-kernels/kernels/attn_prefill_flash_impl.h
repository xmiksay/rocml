// Shared implementation for `attn_prefill_flash.hip`'s (BR=8) and
// `attn_prefill_flash_narrow.hip`'s (BR=4) partial-softmax kernels — see
// `attn_prefill_flash.hip`'s own module doc for the full flash-attention
// design (row-tiling x split-K) this implements; this header only exists
// so both BR variants share one algebra instead of two copies drifting
// apart. `BR`/`BC` are template non-type parameters (not `#define`s, the
// original single-BR design's approach) specifically so a `.hip` file can
// instantiate the identical body at a different `BR` without duplicating
// it — see `attention_chunk.rs`'s dispatch for *why* a second `BR` exists:
// gfx1101's 1024-thread/block limit caps `32 * group * BR`, and a wide-GQA
// checkpoint (e.g. Ornith-1.5-35B-A3B's `group=8`) overflows that at
// `BR=8`, so it must fall back to `BR=4`.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_common.h"

#define MAX_LANE_ELEMS 8

template <typename T>
__device__ __forceinline__ float load_kv_flash(const T* plane, size_t idx);

template <>
__device__ __forceinline__ float load_kv_flash<float>(const float* plane, size_t idx) {
    return plane[idx];
}

template <>
__device__ __forceinline__ float load_kv_flash<__half>(const __half* plane, size_t idx) {
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
template <typename T, unsigned BR, unsigned BC>
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
            k_tile[idx] = load_kv_flash(k_plane, src);
            v_tile[idx] = load_kv_flash(v_plane, src);
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

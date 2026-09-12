// Batched causal attention for a prefill chunk: T new query rows (positions
// `pos_base .. pos_base+T-1`) against the KV cache, which already holds this
// chunk's own K/V (the caller batch-appends before calling this kernel — see
// the Rust launcher's doc comment). Extends `attn_decode.hip`'s single-token
// design to T rows in one launch instead of T serial decode-style launches:
// v1, pragmatic (see issue #6's staging) — one workgroup per (kv head, query
// row) reuses `attn_decode_partial_f32`'s online-softmax K/V-tile-staging
// loop verbatim, just with a per-row causal bound instead of a shared
// `cur_len`, and skips the split-K/reduce dance entirely (T query rows
// already supply `n_kv_heads * T` independent workgroups — ample occupancy
// without needing to also split the sequence axis, unlike a lone decode
// step's `n_kv_heads`-only parallelism).
//
// Not yet doing: sharing one K/V tile load across multiple query rows in the
// same workgroup (real flash-attention batching, listed in the issue as a
// stretch goal) — each query row's workgroup reads the K/V range it needs
// independently, so this kernel's bandwidth profile for the *shared* prefix
// of two adjacent rows' ranges is not improved over decode; the win here is
// occupancy/launch-count, not K/V bandwidth. A follow-up tiling pass (tile
// QUERY_TILE rows per workgroup, causal-mask each row's contribution from a
// shared LDS tile) would close that gap.
//
// Causal semantics: query row i (global position `pos_base+i`) attends to
// `[0, pos_base+i]` inclusive, i.e. `end = pos_base + i + 1` cached
// positions — exactly `attn_decode`'s bound with `cur_len` replaced by a
// per-row value.
#include <hip/hip_runtime.h>

#include "attn_common.h"

#define MAX_LANE_ELEMS 8
#define TILE_T 8

// See `attn_decode.hip`'s identical seam: the one place a K/V element is
// read off the cache plane, so a future quantized-KV cache only touches this
// function.
__device__ __forceinline__ float load_kv_f32(const float* plane, size_t idx) {
    return plane[idx];
}

// One workgroup per (kv head, query row within the chunk). `q` is
// `[chunk_len, n_heads, head_dim]` (already roped/qk-normed by the caller);
// `k_layer`/`v_layer` are one layer's whole KV-cache buffer, `[n_kv_heads,
// max_seq, head_dim]`, already holding this chunk's appended K/V. `out` is
// `[chunk_len, n_heads, head_dim]`.
extern "C" __global__ void attn_prefill_f32(
    const float* q, const float* k_layer, const float* v_layer, float* out, unsigned n_kv_heads,
    unsigned group, unsigned head_dim, unsigned max_seq, unsigned chunk_len, unsigned pos_base,
    float scale) {
    unsigned kvh = blockIdx.x;
    unsigned i = blockIdx.y;
    if (kvh >= n_kv_heads || i >= chunk_len) {
        return;
    }

    unsigned lane = threadIdx.x;
    unsigned g = threadIdx.y;
    unsigned tid = g * 32 + lane;
    unsigned total_threads = blockDim.x * blockDim.y;
    unsigned h = kvh * group + g;

    unsigned end = pos_base + i + 1;

    extern __shared__ float smem[];
    float* k_tile = smem;
    float* v_tile = smem + (size_t)TILE_T * head_dim;

    const float* q_h = q + ((size_t)i * (n_kv_heads * group) + h) * head_dim;
    const float* k_plane = k_layer + (size_t)kvh * max_seq * head_dim;
    const float* v_plane = v_layer + (size_t)kvh * max_seq * head_dim;

    float q_reg[MAX_LANE_ELEMS];
    unsigned n_elems = 0;
    for (unsigned d = lane; d < head_dim; d += 32) {
        q_reg[n_elems++] = q_h[d];
    }

    float acc[MAX_LANE_ELEMS];
    for (unsigned e = 0; e < n_elems; ++e) {
        acc[e] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (unsigned tile_base = 0; tile_base < end; tile_base += TILE_T) {
        unsigned tile_len = min((unsigned)TILE_T, end - tile_base);

        for (unsigned idx = tid; idx < tile_len * head_dim; idx += total_threads) {
            unsigned row = idx / head_dim;
            unsigned col = idx % head_dim;
            size_t src = (size_t)(tile_base + row) * head_dim + col;
            k_tile[idx] = load_kv_f32(k_plane, src);
            v_tile[idx] = load_kv_f32(v_plane, src);
        }
        __syncthreads();

        for (unsigned t = 0; t < tile_len; ++t) {
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
        // Every thread in the block ran the same tile_len iterations above
        // (block-uniform), so this barrier is reached uniformly before the
        // next iteration's load loop overwrites k_tile/v_tile.
        __syncthreads();
    }

    unsigned e = 0;
    float* out_row = out + ((size_t)i * (n_kv_heads * group) + h) * head_dim;
    for (unsigned d = lane; d < head_dim; d += 32, ++e) {
        out_row[d] = acc[e] / l;
    }
}

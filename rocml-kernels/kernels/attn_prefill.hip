// Batched causal attention for a prefill chunk: T new query rows (positions
// `pos_base .. pos_base+T-1`) against the KV cache, which already holds this
// chunk's own K/V (the caller batch-appends before calling this kernel — see
// the Rust launcher's doc comment). Extends `attn_decode.hip`'s single-token
// design to T rows in one launch instead of T serial decode-style launches:
// one workgroup per (kv head, query-row tile) reuses
// `attn_decode_partial_f32`'s online-softmax K/V-tile-staging loop, with a
// per-row causal bound instead of a shared `cur_len`, and skips the
// split-K/reduce dance entirely (T query rows already supply ample
// workgroups — no need to also split the sequence axis, unlike a lone
// decode step's `n_kv_heads`-only parallelism).
//
// **Row-tiling** (issue #6 prefill-cleanup round): a v1 design gave every
// query row its own workgroup — already sharing K/V-tile loads across a
// row's whole GQA group via `blockDim.y`=`group`, but re-reading the K/V
// range from scratch for *every row*, even though adjacent rows' causal
// ranges overlap almost entirely (row `i` and `i+1` differ by exactly one
// token). At depth (large `pos_base`), that's `chunk_len`-fold redundant
// K/V bandwidth for the shared prefix — measured as the single biggest
// prefill cost at depth 8192 (32% of prefill time, only 44.7% of the BW
// roofline despite this op being essentially pure bandwidth-bound). Now
// `ROW_TILE` consecutive query rows share one workgroup and one K/V-tile
// load per `tile_base` step: the tile is staged once for the *deepest* row
// in the group (the one with the largest causal bound) and every row reads
// from the same LDS copy, causally masking off (via an early per-row
// `break`, not a load-side skip — the tile is still loaded whole so the
// load loop stays block-uniform) whatever lies past its own bound. LDS
// footprint is unchanged (`2*TILE_T*head_dim` floats, independent of
// `ROW_TILE`) — only the number of threads cooperating on that same load,
// and the number of independent (lane, group, row) softmax accumulators,
// grows. `ROW_TILE`=4 was the empirical pick (swept against {2,4,8} on
// real `bench --depth 8192` prefill tok/s — a fixed thread-count ceiling
// means the block is `32*group*ROW_TILE` threads, and gfx1101's 1024
// threads/block ceiling caps how far this can grow before *reducing* grid
// parallelism costs more than the bandwidth win — see the kernel source's
// module doc for numbers at the acceptance shapes).
//
// **Miscompile trap found while landing this** (rocml-kernels/tests
// caught it, `gqa_group_and_causal_bound_vary_per_row` specifically):
// precomputing the per-row inner-loop trip count as a plain local
// (`unsigned local_len = (tile_base >= end) ? 0 : min(tile_len, end -
// tile_base);` then `for (t = 0; t < local_len; ++t)`) produced silently
// wrong results for some rows — deterministically wrong (same bad value
// every run, so not a data race), reproducing correctly for `ROW_TILE`=1
// and for the tile every row's own `row_base` starts at (`r`==0 in
// `blockIdx.y`==0) but not other tiles' shallow rows. Rewriting the exact
// same bound as a `for (t = 0; t < tile_len; ++t) { if (tile_base + t >=
// end) break; ... }` — mathematically identical, just expressed as an
// early `break` instead of a precomputed trip count — fixed it and has
// passed every run since (including repeats). Never seen a HIP/AMDGPU
// backend miscompile before or since in this codebase; not chased further
// given the reproducing fix was immediate and free, but worth remembering
// if a future kernel here shows the same "some warps' results are wrong,
// deterministically, only for a precomputed-trip-count divergent loop"
// signature — try the `break` form before assuming a logic bug.
//
// Causal semantics: query row i (global position `pos_base+i`) attends to
// `[0, pos_base+i]` inclusive, i.e. `end = pos_base + i + 1` cached
// positions — exactly `attn_decode`'s bound with `cur_len` replaced by a
// per-row value.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_common.h"

#define MAX_LANE_ELEMS 8
#define TILE_T 8
// Consecutive query rows sharing one workgroup's K/V-tile load; see the
// module doc above for the tuning rationale. Must match the Rust launcher's
// `ATTN_PREFILL_ROW_TILE` constant (`kernels.rs`/`kernels_kv.rs`).
#define ROW_TILE 4

// See `attn_decode.hip`'s identical templated seam: `T` is the cache's
// on-device element type (`float` for the f32 parity-test path, `__half`
// for the default f16 path — issue #3), widened to `float` here for the
// online-softmax math. A future quantized-KV read (issue #2) would add
// further specializations the same way.
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

// One workgroup per (kv head, row-tile of up to ROW_TILE query rows). `q` is
// `[chunk_len, n_heads, head_dim]` (already roped/qk-normed by the caller);
// `k_layer`/`v_layer` are one layer's whole KV-cache buffer, `[n_kv_heads,
// max_seq, head_dim]`, already holding this chunk's appended K/V. `out` is
// `[chunk_len, n_heads, head_dim]`.
template <typename T>
__device__ __forceinline__ void attn_prefill_impl(
    const float* q, const T* k_layer, const T* v_layer, float* out, unsigned n_kv_heads,
    unsigned group, unsigned head_dim, unsigned max_seq, unsigned chunk_len, unsigned pos_base,
    float scale) {
    unsigned kvh = blockIdx.x;
    unsigned row_base = blockIdx.y * ROW_TILE;
    if (kvh >= n_kv_heads || row_base >= chunk_len) {
        return;
    }
    unsigned n_rows = min((unsigned)ROW_TILE, chunk_len - row_base);

    unsigned lane = threadIdx.x;
    unsigned g = threadIdx.y;
    unsigned r = threadIdx.z;
    unsigned tid = (r * blockDim.y + g) * 32 + lane;
    unsigned total_threads = blockDim.x * blockDim.y * blockDim.z;
    unsigned h = kvh * group + g;
    bool active = r < n_rows;
    unsigned i = row_base + r;

    // The tile-load loop below must run the same number of outer iterations
    // for every thread in the block (it's guarded by __syncthreads()), so
    // it's bounded by the *deepest* active row's causal end, not this
    // thread's own row.
    unsigned end = pos_base + i + 1;
    unsigned end_max = pos_base + row_base + n_rows;

    extern __shared__ float smem[];
    float* k_tile = smem;
    float* v_tile = smem + (size_t)TILE_T * head_dim;

    const float* q_h = q + ((size_t)i * (n_kv_heads * group) + h) * head_dim;
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

    for (unsigned tile_base = 0; tile_base < end_max; tile_base += TILE_T) {
        unsigned tile_len = min((unsigned)TILE_T, end_max - tile_base);

        // Cooperative load spans every thread in the block regardless of
        // (g, r) — the tile is one shared copy for the whole row tile, not
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
            // Shallower rows in the tile stop consuming the loaded tile
            // once past their own causal bound (`end`, <= `end_max`); the
            // load above already covers the full `tile_len` uniformly, so
            // breaking early here only skips wasted score/accumulate work.
            for (unsigned t = 0; t < tile_len; ++t) {
                if (tile_base + t >= end) {
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
        // Every thread in the block ran the same tile_len iterations above
        // (block-uniform), so this barrier is reached uniformly before the
        // next iteration's load loop overwrites k_tile/v_tile.
        __syncthreads();
    }

    if (active) {
        unsigned e = 0;
        float* out_row = out + ((size_t)i * (n_kv_heads * group) + h) * head_dim;
        for (unsigned d = lane; d < head_dim; d += 32, ++e) {
            out_row[d] = acc[e] / l;
        }
    }
}

// f32 KV storage: the parity-test reference path (see
// `crate::model::LoadOptions`) — never selected by the CLI/server.
extern "C" __global__ void attn_prefill_f32(
    const float* q, const float* k_layer, const float* v_layer, float* out, unsigned n_kv_heads,
    unsigned group, unsigned head_dim, unsigned max_seq, unsigned chunk_len, unsigned pos_base,
    float scale) {
    attn_prefill_impl<float>(
        q, k_layer, v_layer, out, n_kv_heads, group, head_dim, max_seq, chunk_len, pos_base,
        scale);
}

// f16 KV storage: the default path (issue #3).
extern "C" __global__ void attn_prefill_f16(
    const float* q, const __half* k_layer, const __half* v_layer, float* out, unsigned n_kv_heads,
    unsigned group, unsigned head_dim, unsigned max_seq, unsigned chunk_len, unsigned pos_base,
    float scale) {
    attn_prefill_impl<__half>(
        q, k_layer, v_layer, out, n_kv_heads, group, head_dim, max_seq, chunk_len, pos_base,
        scale);
}

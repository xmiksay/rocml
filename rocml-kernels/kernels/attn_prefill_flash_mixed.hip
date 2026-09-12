// Chunked-prefill sibling of `attn_decode_mixed.hip` for a KIVI-style mixed
// KV layer (issue #2's chunked-prefill follow-up): the exact same `BR`-row
// query tile x split-K design `attn_prefill_flash.hip` uses for a dense KV
// plane, except each K/V element is read through this file's own
// `attn_common.h`-shared `load_mixed_kv` helper instead of a single flat
// `[n_kv_heads, max_seq, head_dim]` buffer — see that header's doc comment
// for the sink/bulk/window dispatch, shared bit-for-bit with
// `attn_decode_mixed.hip`'s decode-attention kernel.
//
// **Ordering contract this kernel relies on**: by the time this launches,
// `qwen35::cache_mixed::MixedAttnPlane::append_chunk` has already written
// the *whole* current chunk's K/V into the sink/window/bulk regions (the
// same append-before-attend order `attention_chunk.rs`'s dense path already
// uses for `scatter_kv_chunk`) — so a query row `i`'s causal read range
// `[0, pos_base+i]` is always fully resolvable through the cache alone; this
// kernel never reads the chunk's own live Q/K/V scratch buffers for
// anything but `q`.
//
// **No separate non-flash (single-pass `attn_prefill.hip`-style) mixed
// kernel exists.** Issue #2's mixed cache targets long-context (150K)
// workloads where depth is always well past `attn_prefill.hip`'s
// `attention_chunk.rs`-side `USE_FLASH_PREFILL_MIN_DEPTH`(512) threshold in
// practice; `attention_chunk_mixed.rs` always dispatches here regardless of
// depth rather than duplicating a second mixed-read kernel for the shallow
// case, since a shallow chunk still runs correctly here (split-K degrades
// gracefully to `n_splits==1`, just paying one extra (cheap) reduce launch
// it wouldn't otherwise need — not a correctness concern, and not measured
// as a throughput one either since every acceptance-relevant depth in issue
// #2's own report is >= 2048).
//
// **WMMA/register-tiling are not re-evaluated here** — see
// `attn_prefill_flash.hip`'s own module doc for why both were rejected for
// the dense kernel this mirrors; the same memory-bound-not-compute-bound
// and occupancy-vs-reuse arguments apply unchanged (if anything more so:
// mixed reads do *more* per-element ALU work per byte fetched than the
// dense kernel's plain load, which only strengthens the "not compute bound"
// case for skipping WMMA here).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_common.h"

#define MAX_LANE_ELEMS 8
// Must match `BC` in `attn_prefill_flash.hip` and the Rust launcher's
// `ATTN_PREFILL_FLASH_BC` constant (shared between the dense and mixed
// launch paths).
#define BC 16
// Must match `BR` in `attn_prefill_flash.hip` / `ATTN_PREFILL_FLASH_BR`.
#define BR 8

// One workgroup per (kv head, row-tile of up to BR query rows, KV split) —
// identical grid/block shape to `attn_prefill_flash_partial_impl`, see that
// kernel's own doc comment for the row-tiling/split-K design this reuses
// unchanged. The only difference is the K/V tile-load loop below, which
// dequantizes through `load_mixed_kv` instead of a flat buffer read.
template <int V_BITS>
__device__ __forceinline__ void attn_prefill_flash_partial_mixed_impl(
    const float* q, const __half* sink_k, const __half* sink_v, const __half* window_k,
    const __half* window_v, const signed char* bulk_k_codes, const float* bulk_k_scales,
    const void* bulk_v_codes, const float* bulk_v_scales, float* partial_out, float* partial_m,
    float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim, unsigned sink_len,
    unsigned window_len, unsigned window_base, unsigned bulk_cap, unsigned num_blocks_total,
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

    // Same split/row-bound structure as `attn_prefill_flash_partial_impl`:
    // every thread in the block runs the same number of outer tile-load
    // iterations, bounded by the deepest active row's clipped end.
    unsigned row_end = pos_base + i + 1;
    unsigned end_max = pos_base + row_base + n_rows;
    unsigned start = s * split_len;
    unsigned end = min(start + split_len, end_max);
    unsigned row_end_clipped = min(row_end, end);

    extern __shared__ float smem[];
    float* k_tile = smem;
    float* v_tile = smem + (size_t)BC * head_dim;

    const float* q_h = q + ((size_t)i * n_heads + h) * head_dim;
    const __half* sink_k_h = sink_k + (size_t)kvh * sink_len * head_dim;
    const __half* sink_v_h = sink_v + (size_t)kvh * sink_len * head_dim;
    const __half* window_k_h = window_k + (size_t)kvh * window_len * head_dim;
    const __half* window_v_h = window_v + (size_t)kvh * window_len * head_dim;
    const signed char* bulk_k_h = bulk_k_codes + (size_t)kvh * bulk_cap * head_dim;
    const float* bulk_k_scales_h = bulk_k_scales + (size_t)kvh * num_blocks_total * head_dim;
    const float* bulk_v_scales_h = bulk_v_scales + (size_t)kvh * bulk_cap;
    const void* bulk_v_codes_h = (V_BITS == 8)
        ? (const void*)((const signed char*)bulk_v_codes + (size_t)kvh * bulk_cap * head_dim)
        : (const void*)((const unsigned char*)bulk_v_codes
                         + (size_t)kvh * bulk_cap * (head_dim / 2));

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

        for (unsigned idx = tid; idx < tile_len * head_dim; idx += total_threads) {
            unsigned row = idx / head_dim;
            unsigned col = idx % head_dim;
            unsigned pos = tile_base + row;
            float kv_k, kv_v;
            load_mixed_kv<V_BITS>(
                sink_k_h, sink_v_h, window_k_h, window_v_h, bulk_k_h, bulk_k_scales_h,
                bulk_v_codes_h, bulk_v_scales_h, pos, col, head_dim, sink_len, window_len,
                window_base, &kv_k, &kv_v);
            k_tile[idx] = kv_k;
            v_tile[idx] = kv_v;
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

extern "C" __global__ void attn_prefill_flash_partial_mixed_q8(
    const float* q, const __half* sink_k, const __half* sink_v, const __half* window_k,
    const __half* window_v, const signed char* bulk_k_codes, const float* bulk_k_scales,
    const signed char* bulk_v_codes, const float* bulk_v_scales, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned sink_len, unsigned window_len, unsigned window_base, unsigned bulk_cap,
    unsigned num_blocks_total, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_mixed_impl<8>(
        q, sink_k, sink_v, window_k, window_v, bulk_k_codes, bulk_k_scales,
        (const void*)bulk_v_codes, bulk_v_scales, partial_out, partial_m, partial_l, n_kv_heads,
        group, head_dim, sink_len, window_len, window_base, bulk_cap, num_blocks_total, chunk_len,
        pos_base, split_len, n_splits, scale);
}

extern "C" __global__ void attn_prefill_flash_partial_mixed_q4(
    const float* q, const __half* sink_k, const __half* sink_v, const __half* window_k,
    const __half* window_v, const signed char* bulk_k_codes, const float* bulk_k_scales,
    const unsigned char* bulk_v_codes, const float* bulk_v_scales, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned sink_len, unsigned window_len, unsigned window_base, unsigned bulk_cap,
    unsigned num_blocks_total, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_mixed_impl<4>(
        q, sink_k, sink_v, window_k, window_v, bulk_k_codes, bulk_k_scales,
        (const void*)bulk_v_codes, bulk_v_scales, partial_out, partial_m, partial_l, n_kv_heads,
        group, head_dim, sink_len, window_len, window_base, bulk_cap, num_blocks_total, chunk_len,
        pos_base, split_len, n_splits, scale);
}

// The reduce pass is dtype/source-independent (only reads the f32 partial
// scratch buffers the kernels above write) — reuses
// `attn_prefill_flash_reduce_f32` from `attn_prefill_flash.hip` unchanged;
// no reduce kernel is defined in this file.

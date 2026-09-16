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
// isn't worth it — and, as of the qwen35moe wide-GQA fix (M2), also when
// `32 * group * BR` would exceed gfx1101's 1024-thread/block limit, in
// which case `attn_prefill_flash_narrow.hip`'s `BR=4` variant runs instead
// (same algebra, see `attn_prefill_flash_impl.h`, the header both files
// share).
//
// This file itself just instantiates the shared `attn_prefill_flash_impl.h`
// template at `BR=8`/`BC=16` — see that header for the actual kernel body.
#include "attn_prefill_flash_impl.h"

// Rows of K/V staged into LDS at a time — same role as `attn_prefill.hip`'s
// `TILE_T`. Must match the Rust launcher's `ATTN_PREFILL_FLASH_BC` constant.
#define BC 16
// Consecutive query rows sharing one workgroup's K/V-tile load, handled via
// `blockDim.z` (hardware threads, not a register loop — see the module doc
// above for why). Must match the Rust launcher's `ATTN_PREFILL_FLASH_BR`
// constant.
#define BR 8

extern "C" __global__ void attn_prefill_flash_partial_f32(
    const float* q, const float* k_layer, const float* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_impl<float, BR, BC>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, chunk_len, pos_base, split_len, n_splits, scale);
}

extern "C" __global__ void attn_prefill_flash_partial_f16(
    const float* q, const __half* k_layer, const __half* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_impl<__half, BR, BC>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, chunk_len, pos_base, split_len, n_splits, scale);
}

// Merges one (row, head)'s per-split partials with the log-sum-exp rule —
// the same algebra as `attn_decode_reduce_f32`, with an added leading row
// axis (`blockIdx.x`) since a prefill chunk covers many rows per launch,
// not decode's single token. Dtype- *and* BR/BC-independent (only reads the
// f32 partial scratch buffers this kernel's partial siblings write, by
// `n_splits`/`head_dim` alone) — shared by both KV dtypes and both BR
// variants (`attn_prefill_flash_narrow.hip` reuses this exact symbol
// rather than defining its own).
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

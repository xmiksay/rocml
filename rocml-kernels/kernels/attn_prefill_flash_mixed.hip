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
//
// This file instantiates the shared `attn_prefill_flash_mixed_impl.h`
// template at `BR=8`/`BC=16` (the default — see
// `attn_prefill_flash_mixed_narrow.hip` for the `BR=4` wide-GQA fallback
// the qwen35moe wide-GQA fix (M2) added, and `attn_prefill_flash.hip`'s own
// doc for why a second `BR` exists at all).
#include "attn_prefill_flash_mixed_impl.h"

// Must match `BC` in `attn_prefill_flash.hip` and the Rust launcher's
// `ATTN_PREFILL_FLASH_BC` constant (shared between the dense and mixed
// launch paths).
#define BC 16
// Must match `BR` in `attn_prefill_flash.hip` / `ATTN_PREFILL_FLASH_BR`.
#define BR 8

extern "C" __global__ void attn_prefill_flash_partial_mixed_q8(
    const float* q, const __half* sink_k, const __half* sink_v, const __half* window_k,
    const __half* window_v, const signed char* bulk_k_codes, const float* bulk_k_scales,
    const signed char* bulk_v_codes, const float* bulk_v_scales, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned sink_len, unsigned window_len, unsigned window_base, unsigned bulk_cap,
    unsigned num_blocks_total, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_mixed_impl<8, BR, BC>(
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
    attn_prefill_flash_partial_mixed_impl<4, BR, BC>(
        q, sink_k, sink_v, window_k, window_v, bulk_k_codes, bulk_k_scales,
        (const void*)bulk_v_codes, bulk_v_scales, partial_out, partial_m, partial_l, n_kv_heads,
        group, head_dim, sink_len, window_len, window_base, bulk_cap, num_blocks_total, chunk_len,
        pos_base, split_len, n_splits, scale);
}

// The reduce pass is dtype/source-independent (only reads the f32 partial
// scratch buffers the kernels above write) — reuses
// `attn_prefill_flash_reduce_f32` from `attn_prefill_flash.hip` unchanged;
// no reduce kernel is defined in this file.

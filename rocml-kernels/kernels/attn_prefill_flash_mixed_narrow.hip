// `BR=4` sibling of `attn_prefill_flash_mixed.hip` (same algebra — see that
// file's module doc and `attn_prefill_flash_mixed_impl.h`, the header both
// instantiate) for wide-GQA checkpoints where `32 * group * BR` would
// exceed gfx1101's 1024-thread/block limit at the default `BR=8` — see
// `attn_prefill_flash_narrow.hip`'s own doc comment for the Ornith-1.5-35B
// discovery this fixes (`group=8`); this is the mixed-KV-cache sibling of
// that same fix, needed because `attention_chunk_mixed.rs` has no
// shallow-depth fallback to fall back to (it always dispatches to a flash
// kernel — see `attn_prefill_flash_mixed.hip`'s own doc comment).
#include "attn_prefill_flash_mixed_impl.h"

#define BC 16
#define BR 4

extern "C" __global__ void attn_prefill_flash_partial_mixed_q8_br4(
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

extern "C" __global__ void attn_prefill_flash_partial_mixed_q4_br4(
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

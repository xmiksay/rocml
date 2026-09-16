// `BR=4` sibling of `attn_prefill_flash.hip` (same algebra — see that
// file's module doc and `attn_prefill_flash_impl.h`, the header both
// instantiate) for wide-GQA checkpoints where `32 * group * BR` would
// exceed gfx1101's 1024-thread/block limit at the default `BR=8`.
//
// Found via Ornith-1.5-35B-A3B (M2): `head_count=16`/`head_count_kv=2` ->
// `group=8`, so `32 * 8 * 8 = 2048` threads/block — `hipModuleLaunchKernel`
// fails with `hipErrorInvalidValue` ("invalid argument") the first time a
// chunk's deepest row reaches `USE_FLASH_PREFILL_MIN_DEPTH` and dispatch
// tries to launch the default kernel. Every previously-tested qwen35/
// qwen35moe checkpoint (Qwen3.5-2B, Ornith-1.0-9B, Ornith-1.5-9B) has
// `group <= 4`, so `32 * group * 8 <= 1024` and this was never hit before.
// `BR=4` halves the block size (`32 * group * 4`), safe up to `group=8`;
// `attention_chunk.rs`'s dispatch picks this over the default kernel
// whenever `group > 4`.
#include "attn_prefill_flash_impl.h"

#define BC 16
#define BR 4

extern "C" __global__ void attn_prefill_flash_partial_f32_br4(
    const float* q, const float* k_layer, const float* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_impl<float, BR, BC>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, chunk_len, pos_base, split_len, n_splits, scale);
}

extern "C" __global__ void attn_prefill_flash_partial_f16_br4(
    const float* q, const __half* k_layer, const __half* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned chunk_len, unsigned pos_base, unsigned split_len,
    unsigned n_splits, float scale) {
    attn_prefill_flash_partial_impl<__half, BR, BC>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, chunk_len, pos_base, split_len, n_splits, scale);
}

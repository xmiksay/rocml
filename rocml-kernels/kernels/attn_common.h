// Shared device-side helper for the attention kernels (`attn_decode.hip`,
// `attn_prefill.hip`, `attn_decode_mixed.hip`, `attn_prefill_flash_mixed.hip`).
// Kept in a header (not a .hip file — build.rs only compiles kernels/*.hip)
// so kernels sharing this logic don't duplicate it.
#pragma once

// Butterfly (XOR) reduction across one 32-lane wavefront: after the full
// 5-step exchange every lane holds the total, so callers need no separate
// broadcast. gfx1101 is wave32, so 5 halving steps (16,8,4,2,1) cover the
// whole wavefront.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, 32);
    }
    return v;
}

// KIVI-style mixed KV cache (issue #2): shared dequant-on-load helpers for
// any kernel reading a `qwen35::cache_mixed::MixedAttnPlane` layer directly
// (`attn_decode_mixed.hip`'s decode-attention kernel, and
// `attn_prefill_flash_mixed.hip`'s chunked-prefill sibling) — one source of
// truth for the sink/bulk/window region dispatch and the V
// dequant-per-bit-width math, so the two kernels can never drift apart on
// what a given cached position actually decodes to. See
// `rocml::kv_quant::layout`'s module doc for the region model and
// `rocml::kv_quant::quant_math`'s for the exact quantization math this must
// invert bit-for-bit.

// `V_BITS` is 8 or 4 — see `MixedAttnPlane`'s doc comment for the packed-
// nibble layout at 4 bits (even channel low nibble, odd channel high
// nibble). `bulk_v_codes` is `const signed char*` at `V_BITS==8` and `const
// unsigned char*` (packed) at `V_BITS==4`, passed through as `const void*`
// since callers already know the right pointee type for their own entry
// points.
template <int V_BITS>
__device__ __forceinline__ float load_bulk_v(
    const void* bulk_v_codes, const float* bulk_v_scale_row, unsigned bulk_pos, unsigned head_dim,
    unsigned col) {
    float scale = bulk_v_scale_row[bulk_pos];
    if (V_BITS == 8) {
        const signed char* codes = (const signed char*)bulk_v_codes;
        return (float)codes[(size_t)bulk_pos * head_dim + col] * scale;
    } else {
        const unsigned char* codes = (const unsigned char*)bulk_v_codes;
        unsigned half_dim = head_dim / 2;
        unsigned char byte = codes[(size_t)bulk_pos * half_dim + col / 2];
        int nibble = (col % 2 == 0) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        int signed_nibble = nibble >= 8 ? nibble - 16 : nibble;
        return (float)signed_nibble * scale;
    }
}

// Dequantizes/reads position `pos`'s (K, V) pair at channel `col` from
// whichever of the three mixed-layer regions it currently lives in — the
// single shared implementation of the branch every mixed-KV-reading kernel
// needs (see `qwen35::cache_mixed`'s module doc for why this exact
// dispatch — sink/bulk/window boundaries at `sink_len`/`window_base` — is
// what a caller must reproduce to stay consistent with the Rust-side
// eviction bookkeeping). All pointers are already offset to the current kv
// head's own row (`kvh`) by the caller, matching `attn_decode_mixed.hip`'s
// existing per-head pointer setup.
template <int V_BITS>
__device__ __forceinline__ void load_mixed_kv(
    const __half* sink_k_h, const __half* sink_v_h, const __half* window_k_h,
    const __half* window_v_h, const signed char* bulk_k_h, const float* bulk_k_scales_h,
    const void* bulk_v_codes_h, const float* bulk_v_scales_h, unsigned pos, unsigned col,
    unsigned head_dim, unsigned sink_len, unsigned window_len, unsigned window_base, float* out_k,
    float* out_v) {
    if (pos < sink_len) {
        *out_k = __half2float(sink_k_h[(size_t)pos * head_dim + col]);
        *out_v = __half2float(sink_v_h[(size_t)pos * head_dim + col]);
    } else if (pos < window_base) {
        unsigned rel = pos - sink_len;
        unsigned block = rel / window_len;
        unsigned offset = rel % window_len;
        float kscale = bulk_k_scales_h[(size_t)block * head_dim + col];
        signed char kcode =
            bulk_k_h[(size_t)block * window_len * head_dim + (size_t)offset * head_dim + col];
        *out_k = (float)kcode * kscale;

        unsigned bulk_pos = block * window_len + offset;
        *out_v = load_bulk_v<V_BITS>(bulk_v_codes_h, bulk_v_scales_h, bulk_pos, head_dim, col);
    } else {
        unsigned widx = pos - window_base;
        *out_k = __half2float(window_k_h[(size_t)widx * head_dim + col]);
        *out_v = __half2float(window_v_h[(size_t)widx * head_dim + col]);
    }
}

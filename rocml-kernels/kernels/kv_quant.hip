// Quantize-on-evict kernels for the KIVI-style mixed KV cache (issue #2):
// one batch launch per evicted `WINDOW_LEN`-position block, never a
// per-position quantize (see `rocml::kv_quant::layout`'s module doc for why
// eviction is whole-window). K is per-channel (one scale per (kv head,
// head_dim channel), computed from that channel's max-abs across the whole
// evicted block); V is per-token (one scale per (kv head, position)),
// either 8-bit (`quantize_evict_v_f16_to_q8`) or 4-bit packed 2 codes/byte
// (`quantize_evict_v_f16_to_q4`) — see `rocml::kv_quant::quant_math`'s
// module doc for the exact packing this must match bit-for-bit (checked by
// `rocml-kernels/tests/kv_quant.rs` against that CPU reference).
//
// Bulk storage layout (owned by `qwen35::cache::MixedAttnPlane`):
// - K codes: `[n_kv_heads, bulk_cap, head_dim]` int8; scales:
//   `[n_kv_heads, num_blocks_total, head_dim]` f32.
// - V codes: `[n_kv_heads, bulk_cap, head_dim]` int8 (q8) or
//   `[n_kv_heads, bulk_cap, head_dim/2]` packed nibbles (q4); scales:
//   `[n_kv_heads, bulk_cap]` f32 either way (one scale per (head, absolute
//   evicted position), addressed directly by `block_idx*window_len + t` —
//   no separate per-block dimension needed since V's granularity is
//   already per-position).
//
// Inputs are always the window's f16 buffer (issue #3's default KV dtype
// extended to the sink/window regions of a mixed layer) at its *physical*
// indices `[0, window_len)` — eviction always happens with the window
// exactly full, so no window_base offset is needed here (see
// `MixedLayout::prepare_append`'s doc comment).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_common.h"

// Max per-lane elements this file's V kernels stage — head_dim <= 256 / 32
// lanes = 8, same ceiling as the attention kernels.
#define MAX_LANE_ELEMS 8

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor(v, offset, 32));
    }
    return v;
}

// One block per kv_head, `head_dim` threads (one channel each) — computes
// that channel's scale from all `window_len` positions, then writes every
// position's code. `bulk_cap` sizes a head's row stride in `bulk_codes`;
// `num_blocks_total` sizes a head's row stride in `bulk_scales`.
extern "C" __global__ void quantize_evict_k_f16_to_q8(
    const __half* window_k, signed char* bulk_codes, float* bulk_scales, unsigned n_kv_heads,
    unsigned window_len, unsigned head_dim, unsigned bulk_cap, unsigned num_blocks_total,
    unsigned block_idx) {
    unsigned h = blockIdx.x;
    unsigned d = threadIdx.x;
    if (h >= n_kv_heads || d >= head_dim) {
        return;
    }

    const __half* win = window_k + (size_t)h * window_len * head_dim;
    float max_abs = 0.0f;
    for (unsigned t = 0; t < window_len; ++t) {
        max_abs = fmaxf(max_abs, fabsf(__half2float(win[(size_t)t * head_dim + d])));
    }
    float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    bulk_scales[((size_t)h * num_blocks_total + block_idx) * head_dim + d] = scale;

    signed char* out = bulk_codes + (size_t)h * bulk_cap * head_dim
        + (size_t)block_idx * window_len * head_dim + d;
    for (unsigned t = 0; t < window_len; ++t) {
        float q = roundf(__half2float(win[(size_t)t * head_dim + d]) / scale);
        out[(size_t)t * head_dim] = (signed char)fminf(fmaxf(q, -127.0f), 127.0f);
    }
}

// One warp (32 threads) per (kv_head, token) — computes that position's
// scale from all `head_dim` channels, then writes every channel's code.
extern "C" __global__ void quantize_evict_v_f16_to_q8(
    const __half* window_v, signed char* bulk_codes, float* bulk_scales, unsigned n_kv_heads,
    unsigned window_len, unsigned head_dim, unsigned bulk_cap, unsigned block_idx) {
    unsigned h = blockIdx.x;
    unsigned t = blockIdx.y;
    if (h >= n_kv_heads || t >= window_len) {
        return;
    }
    unsigned lane = threadIdx.x;
    const __half* row = window_v + ((size_t)h * window_len + t) * head_dim;

    float local_max = 0.0f;
    for (unsigned d = lane; d < head_dim; d += 32) {
        local_max = fmaxf(local_max, fabsf(__half2float(row[d])));
    }
    float scale = warp_reduce_max(local_max);
    scale = scale > 0.0f ? scale / 127.0f : 1.0f;
    if (lane == 0) {
        bulk_scales[(size_t)h * bulk_cap + (size_t)block_idx * window_len + t] = scale;
    }

    signed char* out = bulk_codes + ((size_t)h * bulk_cap + (size_t)block_idx * window_len + t) * head_dim;
    for (unsigned d = lane; d < head_dim; d += 32) {
        float q = roundf(__half2float(row[d]) / scale);
        out[d] = (signed char)fminf(fmaxf(q, -127.0f), 127.0f);
    }
}

// Same shape as `quantize_evict_v_f16_to_q8` but packs two signed 4-bit
// codes per byte: channel `2*p` in the low nibble, `2*p+1` in the high
// nibble (must match `rocml::kv_quant::quant_math::pack_nibbles` exactly).
extern "C" __global__ void quantize_evict_v_f16_to_q4(
    const __half* window_v, unsigned char* bulk_codes, float* bulk_scales, unsigned n_kv_heads,
    unsigned window_len, unsigned head_dim, unsigned bulk_cap, unsigned block_idx) {
    unsigned h = blockIdx.x;
    unsigned t = blockIdx.y;
    if (h >= n_kv_heads || t >= window_len) {
        return;
    }
    unsigned lane = threadIdx.x;
    const __half* row = window_v + ((size_t)h * window_len + t) * head_dim;

    float local_max = 0.0f;
    for (unsigned d = lane; d < head_dim; d += 32) {
        local_max = fmaxf(local_max, fabsf(__half2float(row[d])));
    }
    float scale = warp_reduce_max(local_max);
    scale = scale > 0.0f ? scale / 7.0f : 1.0f;
    if (lane == 0) {
        bulk_scales[(size_t)h * bulk_cap + (size_t)block_idx * window_len + t] = scale;
    }

    unsigned half_dim = head_dim / 2;
    unsigned char* out =
        bulk_codes + ((size_t)h * bulk_cap + (size_t)block_idx * window_len + t) * half_dim;
    for (unsigned p = lane; p < half_dim; p += 32) {
        float lo_f = fminf(fmaxf(__half2float(row[2 * p]) / scale, -8.0f), 7.0f);
        float hi_f = fminf(fmaxf(__half2float(row[2 * p + 1]) / scale, -8.0f), 7.0f);
        int lo = (int)roundf(lo_f);
        int hi = (int)roundf(hi_f);
        out[p] = (unsigned char)((lo & 0x0F) | ((hi & 0x0F) << 4));
    }
}

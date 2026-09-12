// Fused decode attention for a KIVI-style mixed KV layer (issue #2):
// extends `attn_decode.hip`'s online-softmax split-K design with a
// per-position dequant-on-load that reads from whichever of three regions
// a cached position currently lives in (see `rocml::kv_quant::layout`'s
// module doc for the sink/bulk/window model this mirrors exactly):
//
// - `pos < sink_len`: fp16 attention sink, read directly.
// - `sink_len <= pos < window_base`: quantized bulk — K dequantized via its
//   per-(kv head, block, channel) scale, V via its per-(kv head, absolute
//   position) scale, either 8-bit (`V_BITS=8`) or 4-bit-packed (`V_BITS=4`,
//   two codes/byte: even channel low nibble, odd channel high nibble).
// - `window_base <= pos < cur_len`: fp16 recent window, read directly
//   (physical index `pos - window_base` — the window always restarts at
//   physical index 0 right after an eviction, see `MixedLayout`'s doc
//   comment, so no modulo/ring arithmetic is needed here).
//
// The dequantized value is never written back to any buffer — it only ever
// exists in the same per-tile LDS staging array `attn_decode_partial_f16`
// uses, for exactly as long as this tile's score/weighted-V accumulation
// needs it. This is the single most important property of this kernel:
// materializing a dequantized copy back to global memory would erase the
// entire bandwidth win quantization is for.
//
// `attn_decode_reduce_f32` (unchanged, from attn_decode.hip) still merges
// this kernel's per-split partials — it only ever touches the f32 partial
// scratch buffers, never the cache itself, so one reduce kernel serves
// every KV storage scheme in this codebase.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_common.h"

#define MAX_LANE_ELEMS 8
#define TILE_T 8

// `V_BITS` is 8 (`attn_decode_partial_mixed_q8`) or 4
// (`attn_decode_partial_mixed_q4`) — the only difference between the two
// entry points below. `bulk_v_codes` is `const signed char*` for V_BITS==8
// and `const unsigned char*` (packed) for V_BITS==4, passed through as
// `const void*` since the two entry points' own signatures already carry
// the right pointee type for their callers.
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

template <int V_BITS>
__device__ __forceinline__ void attn_decode_partial_mixed_impl(
    const float* q, const __half* sink_k, const __half* sink_v, const __half* window_k,
    const __half* window_v, const signed char* bulk_k_codes, const float* bulk_k_scales,
    const void* bulk_v_codes, const float* bulk_v_scales, float* partial_out, float* partial_m,
    float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim, unsigned sink_len,
    unsigned window_len, unsigned window_base, unsigned bulk_cap, unsigned num_blocks_total,
    unsigned cur_len, unsigned split_len, unsigned n_splits, float scale) {
    unsigned kvh = blockIdx.x;
    unsigned s = blockIdx.y;
    if (kvh >= n_kv_heads || s >= n_splits) {
        return;
    }

    unsigned lane = threadIdx.x;
    unsigned g = threadIdx.y;
    unsigned tid = g * 32 + lane;
    unsigned total_threads = blockDim.x * blockDim.y;
    unsigned h = kvh * group + g;

    unsigned start = s * split_len;
    unsigned end = min(start + split_len, cur_len);

    extern __shared__ float smem[];
    float* k_tile = smem;
    float* v_tile = smem + (size_t)TILE_T * head_dim;

    const float* q_h = q + (size_t)h * head_dim;
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
    for (unsigned d = lane; d < head_dim; d += 32) {
        q_reg[n_elems++] = q_h[d];
    }

    float acc[MAX_LANE_ELEMS];
    for (unsigned i = 0; i < n_elems; ++i) {
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (unsigned tile_base = start; tile_base < end; tile_base += TILE_T) {
        unsigned tile_len = min((unsigned)TILE_T, end - tile_base);

        for (unsigned idx = tid; idx < tile_len * head_dim; idx += total_threads) {
            unsigned row = idx / head_dim;
            unsigned col = idx % head_dim;
            unsigned pos = tile_base + row;

            float kv_k, kv_v;
            if (pos < sink_len) {
                kv_k = __half2float(sink_k_h[(size_t)pos * head_dim + col]);
                kv_v = __half2float(sink_v_h[(size_t)pos * head_dim + col]);
            } else if (pos < window_base) {
                unsigned rel = pos - sink_len;
                unsigned block = rel / window_len;
                unsigned offset = rel % window_len;
                float kscale = bulk_k_scales_h[(size_t)block * head_dim + col];
                signed char kcode =
                    bulk_k_h[(size_t)block * window_len * head_dim + (size_t)offset * head_dim + col];
                kv_k = (float)kcode * kscale;

                unsigned bulk_pos = block * window_len + offset;
                kv_v = load_bulk_v<V_BITS>(bulk_v_codes_h, bulk_v_scales_h, bulk_pos, head_dim, col);
            } else {
                unsigned widx = pos - window_base;
                kv_k = __half2float(window_k_h[(size_t)widx * head_dim + col]);
                kv_v = __half2float(window_v_h[(size_t)widx * head_dim + col]);
            }

            k_tile[idx] = kv_k;
            v_tile[idx] = kv_v;
        }
        __syncthreads();

        for (unsigned t = 0; t < tile_len; ++t) {
            const float* k_row = k_tile + (size_t)t * head_dim;
            float partial = 0.0f;
            unsigned i = 0;
            for (unsigned d = lane; d < head_dim; d += 32, ++i) {
                partial += q_reg[i] * k_row[d];
            }
            float score = warp_reduce_sum(partial) * scale;

            float new_m = fmaxf(m, score);
            float correction = expf(m - new_m);
            float p = expf(score - new_m);
            l = l * correction + p;

            const float* v_row = v_tile + (size_t)t * head_dim;
            i = 0;
            for (unsigned d = lane; d < head_dim; d += 32, ++i) {
                acc[i] = acc[i] * correction + p * v_row[d];
            }
            m = new_m;
        }
        __syncthreads();
    }

    if (lane == 0) {
        partial_m[h * n_splits + s] = m;
        partial_l[h * n_splits + s] = l;
    }
    unsigned i = 0;
    float* out_row = partial_out + (size_t)(h * n_splits + s) * head_dim;
    for (unsigned d = lane; d < head_dim; d += 32, ++i) {
        out_row[d] = acc[i];
    }
}

extern "C" __global__ void attn_decode_partial_mixed_q8(
    const float* q, const __half* sink_k, const __half* sink_v, const __half* window_k,
    const __half* window_v, const signed char* bulk_k_codes, const float* bulk_k_scales,
    const signed char* bulk_v_codes, const float* bulk_v_scales, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned sink_len, unsigned window_len, unsigned window_base, unsigned bulk_cap,
    unsigned num_blocks_total, unsigned cur_len, unsigned split_len, unsigned n_splits,
    float scale) {
    attn_decode_partial_mixed_impl<8>(
        q, sink_k, sink_v, window_k, window_v, bulk_k_codes, bulk_k_scales,
        (const void*)bulk_v_codes, bulk_v_scales, partial_out, partial_m, partial_l, n_kv_heads,
        group, head_dim, sink_len, window_len, window_base, bulk_cap, num_blocks_total, cur_len,
        split_len, n_splits, scale);
}

extern "C" __global__ void attn_decode_partial_mixed_q4(
    const float* q, const __half* sink_k, const __half* sink_v, const __half* window_k,
    const __half* window_v, const signed char* bulk_k_codes, const float* bulk_k_scales,
    const unsigned char* bulk_v_codes, const float* bulk_v_scales, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned sink_len, unsigned window_len, unsigned window_base, unsigned bulk_cap,
    unsigned num_blocks_total, unsigned cur_len, unsigned split_len, unsigned n_splits,
    float scale) {
    attn_decode_partial_mixed_impl<4>(
        q, sink_k, sink_v, window_k, window_v, bulk_k_codes, bulk_k_scales,
        (const void*)bulk_v_codes, bulk_v_scales, partial_out, partial_m, partial_l, n_kv_heads,
        group, head_dim, sink_len, window_len, window_base, bulk_cap, num_blocks_total, cur_len,
        split_len, n_splits, scale);
}

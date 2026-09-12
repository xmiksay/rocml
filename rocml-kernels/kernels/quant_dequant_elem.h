// Shared per-element dequant functors for the batched prefill-path
// `gemm_xwt_q*` kernels (scalar-FMA in `gemm_xwt_quant.hip`, WMMA in
// `gemm_xwt_quant_wmma.hip`) — split out so both share one bit-for-bit
// definition instead of drifting. Each functor reproduces rocml-core's
// `quant::q{4,5,6}_k`/`q8_0::dequantize` output order exactly (see each
// functor's comment for the index algebra) so every kernel including this
// header agrees with the CPU reference element-for-element.
#pragma once

#include <hip/hip_fp16.h>

#include <cstring>

#include "quant_gemv_common.h"

struct DequantQ8_0 {
    __device__ __forceinline__ float operator()(const unsigned char* bptr, unsigned e) const {
        __half d_h;
        std::memcpy(&d_h, bptr, sizeof(__half));
        float d = __half2float(d_h);
        signed char q = (signed char)bptr[2 + e];
        return (float)q * d;
    }
};

// Q4_K: `e` in [0,256). Mirrors rocml-core's `for j in (0..256).step_by(64)`
// outer loop (`is` steps by 2 alongside it) with an inner 32-wide low-nibble
// pass then a 32-wide high-nibble pass.
struct DequantQ4K {
    __device__ __forceinline__ float operator()(const unsigned char* bptr, unsigned e) const {
        __half d_h, dmin_h;
        std::memcpy(&d_h, bptr, sizeof(__half));
        std::memcpy(&dmin_h, bptr + 2, sizeof(__half));
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);
        const unsigned char* scales = bptr + 4;
        const unsigned char* qs = bptr + 16;

        unsigned j = (e / 64) * 64;
        unsigned local = e - j;
        unsigned is = (j / 64) * 2;
        unsigned char sc, mn;
        if (local < 32) {
            get_scale_min_k4(is, scales, &sc, &mn);
            unsigned char b = qs[j / 2 + local];
            return d * (float)sc * (float)(b & 0xF) - dmin * (float)mn;
        }
        get_scale_min_k4(is + 1, scales, &sc, &mn);
        unsigned char b = qs[j / 2 + (local - 32)];
        return d * (float)sc * (float)(b >> 4) - dmin * (float)mn;
    }
};

// Q5_K: like Q4_K plus the `u1`/`u2` high-bit mask, rotated left by 2 bits
// per `j` step (`step = j/64`) — mirrors rocml-core's `u1 <<= 2; u2 <<= 2`
// per outer iteration.
struct DequantQ5K {
    __device__ __forceinline__ float operator()(const unsigned char* bptr, unsigned e) const {
        __half d_h, dmin_h;
        std::memcpy(&d_h, bptr, sizeof(__half));
        std::memcpy(&dmin_h, bptr + 2, sizeof(__half));
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);
        const unsigned char* scales = bptr + 4;
        const unsigned char* qh = bptr + 16;
        const unsigned char* qs = bptr + 48;

        unsigned j = (e / 64) * 64;
        unsigned local = e - j;
        unsigned step = j / 64;
        unsigned is = step * 2;
        unsigned char u1 = (unsigned char)(1u << (2 * step));
        unsigned char u2 = (unsigned char)(2u << (2 * step));
        unsigned char sc, mn;
        if (local < 32) {
            get_scale_min_k4(is, scales, &sc, &mn);
            unsigned char b = qs[j / 2 + local];
            float hi = (qh[local] & u1) ? 16.0f : 0.0f;
            return d * (float)sc * ((float)(b & 0xF) + hi) - dmin * (float)mn;
        }
        unsigned loc2 = local - 32;
        get_scale_min_k4(is + 1, scales, &sc, &mn);
        unsigned char b = qs[j / 2 + loc2];
        float hi = (qh[loc2] & u2) ? 16.0f : 0.0f;
        return d * (float)sc * ((float)(b >> 4) + hi) - dmin * (float)mn;
    }
};

// Q6_K: `e` in [0,256) splits into `half_idx = e/128` (ggml's two 128-wide
// halves), then within a half, `group in {0,1,2,3}` (ggml's q1..q4) and
// `l in [0,32)`. Mirrors rocml-core's nested `half_idx`/`l` loop with the
// four `q1..q4` expressions unrolled algebraically by `group`.
struct DequantQ6K {
    __device__ __forceinline__ float operator()(const unsigned char* bptr, unsigned e) const {
        const unsigned char* ql_full = bptr;
        const unsigned char* qh_full = bptr + 128;
        const signed char* sc_full = (const signed char*)(bptr + 192);
        __half d_h;
        std::memcpy(&d_h, bptr + 208, sizeof(__half));
        float d = __half2float(d_h);

        unsigned half_idx = e / 128;
        unsigned local = e % 128;
        unsigned group = local / 32;
        unsigned l = local % 32;
        unsigned is = l / 16;

        const unsigned char* ql = ql_full + 64 * half_idx;
        const unsigned char* qh = qh_full + 32 * half_idx;
        const signed char* sc = sc_full + 8 * half_idx;

        unsigned ql_idx = l + ((group & 1) ? 32 : 0);
        bool high_nibble = group >= 2;
        unsigned qh_shift = group * 2;
        int nib = high_nibble ? (ql[ql_idx] >> 4) : (ql[ql_idx] & 0xF);
        int qv = (nib | (((qh[l] >> qh_shift) & 3) << 4)) - 32;
        float scale = (float)sc[is + group * 2];
        return d * scale * (float)qv;
    }
};

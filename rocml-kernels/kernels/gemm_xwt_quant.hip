// Prefill-path batched linear layer for GGUF-quantized weights: out[rows, m]
// = X[rows, n] * dequant(W)^T, where W's rows are raw GGUF quantized blocks
// (Q8_0/Q4_K/Q5_K/Q6_K), staying in that block format in VRAM exactly like
// the decode-path `gemv_q*` kernels — this is the batched sibling of those,
// not a replacement (`gemv_q*` stays the fast path for rows==1 decode).
//
// Design: one weight row (`col`) is shared by `TILE_ROWS` (8) consecutive
// output rows per workgroup (grid = (m, ceil(rows/8)), block = (32, 8)) — a
// wavefront per output row (`threadIdx.y` picks the row, `threadIdx.x` the
// lane within it; HIP linearizes a block's threads as
// `threadIdx.y*blockDim.x+threadIdx.x`, so each row's 32 lanes are exactly
// one gfx1101 wavefront, matching `attn_decode.hip`'s block=(32,group)
// warp-per-unit design). Per outer iteration the whole 256-thread block
// cooperatively dequantizes one `TILE_ELEMS`-wide (256) chunk of the shared
// weight row into LDS once (`w_tile`), then every row-warp reduces its own
// dot product against that shared tile — the weight's global-memory read and
// dequant cost is paid once per 8 output rows instead of once per row, which
// is the whole point of batching prefill's X*W^T over `gemv_q*`'s one-row-
// at-a-time X*w. `TILE_ELEMS` is fixed at 256 regardless of dtype so every
// kernel below requests the same (1KiB) dynamic shared memory: Q8_0's native
// block is 32 elements (8 blocks staged per outer iteration), the K-quants'
// native super-block already is 256 (one staged per iteration).
//
// Per-element dequant (`Dequant*::operator()`) reproduces rocml-core's
// `quant::q{4,5,6}_k`/`q8_0::dequantize` output order exactly (see each
// functor's comment for the index algebra) so this kernel and the CPU
// reference agree element-for-element; only the *reduction order* differs
// (strided-by-lane partial sums, tree-combined via `__shfl_xor`, accumulated
// across outer iterations) — same contract `gemv_q*` already documents.
//
// Correctness first: no X-tile LDS staging (each lane's `x` reads are
// already contiguous across the wavefront, i.e. coalesced, without it) and
// no attempt at multi-column tiling — see the module's issue for the
// staged-tuning follow-up.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

namespace {

constexpr unsigned TILE_ELEMS = 256;
constexpr unsigned TILE_ROWS = 8;

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

// `QK`: the dtype's native block width (elements dequantized together from
// one `BLOCK_BYTES`-byte block sharing one scale). `TILE_ELEMS/QK` such
// blocks are staged into `w_tile` per outer iteration.
template <unsigned QK, unsigned BLOCK_BYTES, typename DequantFn>
__device__ __forceinline__ void gemm_xwt_quant_impl(
    const float* x, const unsigned char* w, float* out, unsigned rows, unsigned m, unsigned n,
    DequantFn dequant_elem) {
    constexpr unsigned GROUP_BLOCKS = TILE_ELEMS / QK;

    unsigned col = blockIdx.x;
    if (col >= m) {
        return;
    }
    unsigned row = blockIdx.y * TILE_ROWS + threadIdx.y;
    unsigned lane = threadIdx.x;
    unsigned tid = threadIdx.y * 32 + lane;

    unsigned blocks_per_row = n / QK;
    const unsigned char* w_row = w + (size_t)col * blocks_per_row * BLOCK_BYTES;

    extern __shared__ float w_tile[];
    float acc = 0.0f;

    for (unsigned bg = 0; bg * GROUP_BLOCKS < blocks_per_row; ++bg) {
        unsigned blocks_here = min(GROUP_BLOCKS, blocks_per_row - bg * GROUP_BLOCKS);
        unsigned elems_here = blocks_here * QK;
        if (tid < elems_here) {
            unsigned local_blk = tid / QK;
            unsigned local_off = tid % QK;
            const unsigned char* bptr = w_row + (size_t)(bg * GROUP_BLOCKS + local_blk) * BLOCK_BYTES;
            w_tile[tid] = dequant_elem(bptr, local_off);
        }
        __syncthreads();

        if (row < rows) {
            const float* x_row = x + (size_t)row * n + (size_t)bg * TILE_ELEMS;
            float partial = 0.0f;
            for (unsigned d = lane; d < elems_here; d += 32) {
                partial += w_tile[d] * x_row[d];
            }
#pragma unroll
            for (unsigned offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_xor(partial, offset, 32);
            }
            acc += partial;
        }
        __syncthreads();
    }

    if (lane == 0 && row < rows) {
        out[(size_t)row * m + col] = acc;
    }
}

} // namespace

extern "C" __global__ void gemm_xwt_q8_0(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_quant_impl<32, 34>(x, (const unsigned char*)w, out, rows, m, n, DequantQ8_0{});
}

extern "C" __global__ void gemm_xwt_q4_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_quant_impl<256, 144>(x, (const unsigned char*)w, out, rows, m, n, DequantQ4K{});
}

extern "C" __global__ void gemm_xwt_q5_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_quant_impl<256, 176>(x, (const unsigned char*)w, out, rows, m, n, DequantQ5K{});
}

extern "C" __global__ void gemm_xwt_q6_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_quant_impl<256, 210>(x, (const unsigned char*)w, out, rows, m, n, DequantQ6K{});
}

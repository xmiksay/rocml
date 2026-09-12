// Prefill-path batched linear layer for GGUF-quantized weights: out[rows, m]
// = X[rows, n] * dequant(W)^T, where W's rows are raw GGUF quantized blocks
// (Q8_0/Q4_K/Q5_K/Q6_K), staying in that block format in VRAM exactly like
// the decode-path `gemv_q*` kernels — this is the batched sibling of those,
// not a replacement (`gemv_q*` stays the fast path for rows==1 decode).
//
// Design: one weight row (`col`) is shared by `TILE_ROWS` (64) consecutive
// output rows per workgroup (grid = (m, ceil(rows/64)), block = (32, 8)) — a
// wavefront per `ROWS_PER_WARP` (8) output rows (`threadIdx.y` picks the
// warp, `threadIdx.x` the lane within it; HIP linearizes a block's threads
// as `threadIdx.y*blockDim.x+threadIdx.x`, so each warp's 32 lanes are
// exactly one gfx1101 wavefront, matching `attn_decode.hip`'s
// block=(32,group) warp-per-unit design). Per outer iteration the whole
// 256-thread block cooperatively dequantizes one `TILE_ELEMS`-wide (256)
// chunk of the shared weight row into LDS once (`w_tile`), then every warp
// register-accumulates `ROWS_PER_WARP` independent dot products against that
// same LDS-resident tile before moving on — the weight's global-memory read
// and dequant cost is paid once per `TILE_ROWS` output rows instead of once
// per row, which is the whole point of batching prefill's X*W^T over
// `gemv_q*`'s one-row-at-a-time X*w. An earlier revision fixed
// `ROWS_PER_WARP` at 1 (i.e. `TILE_ROWS` = warp count = 8): at a 128-row
// chunked-prefill batch that re-read and re-dequantized every weight row 16x
// (`rows / TILE_ROWS`); register-blocking 8 rows per warp instead of adding
// more warps keeps the LDS footprint and the dequant work identical while
// amortizing both over 8x more output rows. `TILE_ELEMS` is fixed at 256
// regardless of dtype so every kernel below requests the same (1KiB)
// dynamic shared memory: Q8_0's native block is 32 elements (8 blocks
// staged per outer iteration), the K-quants' native super-block already is
// 256 (one staged per iteration).
//
// Two further points beyond the row-count itself turned out to matter more
// than the weight-reread count for measured throughput (issue #6's
// benchmarking): (1) the k-reduction loop is `d`-outer/`row`-inner — each
// LDS load of `w_tile[d]` is reused across all `ROWS_PER_WARP` rows' FMAs in
// registers, instead of a `row`-outer/`d`-inner order that reloads the same
// LDS element once per row for zero reuse; and (2) the `__shfl_xor`
// cross-lane reduction runs once per row *after* the whole `bg` loop
// (accumulating each lane's unreduced partial sum across every outer
// iteration first), not once per row per `bg` — for n=4096 (16 outer
// iterations) that cut shuffle-instruction count 16x for the same result.
// Combined, these took the measured kernel from ~750 GFLOP/s to ~1700-1800
// GFLOP/s at the issue's target chunk-128 shapes (~2.3x, well short of the
// ~10x aspiration — see the crate-level perf notes in
// `tests/gemm_xwt_quant_perf.rs` for the full before/after numbers and why
// a scalar-FMA/LDS/shuffle dot product plateaus here: closing the rest of
// the gap to the f32 FLOP roofline needs RDNA3 WMMA matrix-core intrinsics,
// a materially larger rewrite left as follow-up).
//
// Per-element dequant (`Dequant*::operator()`) reproduces rocml-core's
// `quant::q{4,5,6}_k`/`q8_0::dequantize` output order exactly (see each
// functor's comment for the index algebra) so this kernel and the CPU
// reference agree element-for-element; only the *reduction order* differs
// (strided-by-lane partial sums accumulated across the whole k-reduction,
// then tree-combined via `__shfl_xor` once per row) — same contract
// `gemv_q*` already documents, just with the cross-lane combine deferred
// later than earlier revisions.
//
// No X-tile LDS staging: each lane's `x` reads are already contiguous
// across the wavefront (i.e. coalesced) without it, and with `ROWS_PER_WARP`
// rows live per warp those reads already have plenty of independent
// in-flight loads to hide latency. No multi-column tiling either — see the
// module's issue for that follow-up if weight-row reuse across columns is
// ever needed too.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

namespace {

constexpr unsigned TILE_ELEMS = 256;
// Must equal TILE_ELEMS/32 so the block has exactly TILE_ELEMS threads —
// the cooperative dequant below stages one element per thread per outer
// iteration in a single pass (see `gemm_xwt_quant_impl`'s `tid < elems_here`
// staging step).
constexpr unsigned WARPS_PER_BLOCK = 8;
// Output rows one warp carries in registers, reusing the same LDS-resident
// weight tile for all of them before the block moves to the next tile.
constexpr unsigned ROWS_PER_WARP = 8;
constexpr unsigned TILE_ROWS = WARPS_PER_BLOCK * ROWS_PER_WARP;

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
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    unsigned tid = warp * 32 + lane;
    unsigned row_base = blockIdx.y * TILE_ROWS + warp * ROWS_PER_WARP;

    unsigned blocks_per_row = n / QK;
    const unsigned char* w_row = w + (size_t)col * blocks_per_row * BLOCK_BYTES;

    // Rows this warp owns, clamped into `[0, rows)` so every lane can read
    // `x` unconditionally in the hot loop below (no per-element branch) —
    // `row_ok[r]` gates the corresponding output write instead, so a
    // clamped/duplicated read never corrupts another row's result.
    unsigned row_idx[ROWS_PER_WARP];
    bool row_ok[ROWS_PER_WARP];
#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WARP; ++r) {
        unsigned row = row_base + r;
        row_ok[r] = row < rows;
        row_idx[r] = row_ok[r] ? row : (rows - 1);
    }
    // Precomputed once (not per-`bg`): each row's byte offset into `x` is
    // constant across the whole k-reduction, only the `bg*TILE_ELEMS`
    // column offset moves.
    const float* x_base[ROWS_PER_WARP];
#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WARP; ++r) {
        x_base[r] = x + (size_t)row_idx[r] * n;
    }

    extern __shared__ float w_tile[];
    // Per-lane, not-yet-cross-lane-reduced running sum for each row, kept
    // live across the *whole* `bg` loop (not reset/reduced per iteration):
    // the `__shfl_xor` butterfly reduction only needs to run once per row,
    // after every `bg` has contributed, instead of once per `bg` — for
    // n=4096 (16 outer iterations) that's a 16x cut in shuffle instructions
    // for the same result (lane `l`'s partial sum spans the same strided
    // indices `{l, l+32, l+64, ...}` either way; only *when* the cross-lane
    // combine happens changes, which changes only the reduction order —
    // already documented as differing from the CPU reference regardless).
    float partial[ROWS_PER_WARP];
#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WARP; ++r) {
        partial[r] = 0.0f;
    }

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

        // `d`-outer / `r`-inner: each LDS load of `w_tile[d]` is reused
        // across all `ROWS_PER_WARP` rows' FMAs instead of being re-fetched
        // once per row — this is the register-blocking step that actually
        // raises arithmetic intensity (a row-outer/`d`-inner ordering
        // re-reads the same LDS element once per row for no reuse).
        unsigned col_off = bg * TILE_ELEMS;
        for (unsigned d = lane; d < elems_here; d += 32) {
            float wv = w_tile[d];
#pragma unroll
            for (unsigned r = 0; r < ROWS_PER_WARP; ++r) {
                partial[r] += wv * x_base[r][col_off + d];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WARP; ++r) {
        float p = partial[r];
#pragma unroll
        for (unsigned offset = 16; offset > 0; offset >>= 1) {
            p += __shfl_xor(p, offset, 32);
        }
        if (lane == 0) {
            unsigned row = row_base + r;
            if (row < rows) {
                out[(size_t)row * m + col] = p;
            }
        }
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

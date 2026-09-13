// int8 MMQ-style batched prefill-path linear layer for Q6_K-quantized
// weights (int8-MMQ-integration round's mechanical extension of
// `gemm_xwt_quant_mmq_q8_0.hip` — see that file's module doc for the shared
// fragment layout/tiling this file reuses via `mmq_common.h`). Q6_K is
// **symmetric** like Q8_0 (`d*scale*qv`, `qv` a signed value in [-32,31] —
// no affine min-offset term, so this kernel needs no `x_sum`/`block_sum`
// input, unlike the Q4_K/Q5_K variants), but its scale granularity is 16
// elements (`quant_dequant_elem.h`'s `DequantQ6K`: `is = l/16` inside each
// 32-wide `group`), not 32 — so unlike every other variant here, a single
// `K_STAGE`(32)-wide outer iteration spans *two* independent scales, one
// per chained `k=[0,16)`/`k=[16,32)` WMMA half. Those two halves are kept
// as two separate `int32x8` accumulators (`acc_i0`/`acc_i1`, each scaled by
// its own `d*scale` before being summed into the running per-(row,col)
// total) rather than Q8_0's/Q4_K's single chained-then-scaled-once
// accumulator — the only structural difference from the rest of this
// family. A block-aligned 32-wide MMQ iteration always lands fully inside
// one `DequantQ6K` `group` (32-wide, `group = local/32`), so `group`,
// `half_idx` (which 128-wide ggml "half" this superblock quadrant is in)
// and the two 16-wide sub-scale indices (`group*2`, `group*2+1`) are all
// constant across the whole iteration — only the per-element `ql`/`qh`
// nibble/bit lookup varies, decomposed the same way `DequantQ6K` does.
#include "mmq_common.h"

namespace {
constexpr unsigned Q6_K_BLOCK_BYTES = 210;
} // namespace

extern "C" __global__ void gemm_xwt_mmq_q6_k(
    const signed char* x_codes_in, const float* x_scale_in, const void* w, float* out,
    unsigned rows, unsigned m, unsigned n) {
    unsigned col_base = blockIdx.x * TILE_M;
    unsigned row_base = blockIdx.y * TILE_ROWS;
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    unsigned tid = warp * 32 + lane;
    unsigned wm = warp / WARPS_N;
    unsigned wn = warp % WARPS_N;
    unsigned n_a_blocks = n / K_STAGE;
    unsigned superblocks_per_row = n / 256;
    const unsigned char* wbytes = (const unsigned char*)w;

    extern __shared__ unsigned char smem_mmq_q6k[];
    signed char* x_codes = (signed char*)smem_mmq_q6k;
    signed char* w_codes = x_codes + TILE_ROWS * K_STAGE;
    float* d_a_lds = (float*)(w_codes + TILE_M * K_STAGE);
    // Two scales per column per iteration (first/second 16-wide half) —
    // see the module doc for why Q6_K can't share one `d_w` value across a
    // whole 32-wide K_STAGE block the way every other variant here does.
    float* d_w_lds = d_a_lds + TILE_ROWS;

    float acc_f[SUBCOLS_PER_WARP][8];
#pragma unroll
    for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
#pragma unroll
        for (unsigned e = 0; e < 8; ++e) {
            acc_f[c][e] = 0.0f;
        }
    }

    for (unsigned k0 = 0; k0 < n; k0 += K_STAGE) {
        unsigned blk = k0 / K_STAGE;
        unsigned superblock = k0 / 256;
        unsigned e_base = k0 % 256;
        unsigned half_idx = e_base / 128;
        unsigned local = e_base % 128;
        unsigned group = local / 32;

        for (unsigned idx = tid; idx < TILE_ROWS * K_STAGE; idx += WARPS_PER_BLOCK * 32) {
            unsigned r = idx / K_STAGE;
            unsigned e = idx % K_STAGE;
            unsigned row = row_base + r;
            unsigned row_c = row < rows ? row : rows - 1;
            x_codes[idx] = x_codes_in[(size_t)row_c * n + k0 + e];
        }
        for (unsigned idx = tid; idx < TILE_ROWS; idx += WARPS_PER_BLOCK * 32) {
            unsigned row = row_base + idx;
            unsigned row_c = row < rows ? row : rows - 1;
            d_a_lds[idx] = x_scale_in[(size_t)row_c * n_a_blocks + blk];
        }
        for (unsigned idx = tid; idx < TILE_M * K_STAGE; idx += WARPS_PER_BLOCK * 32) {
            unsigned r = idx / K_STAGE;
            unsigned e = idx % K_STAGE; // this block's local element offset, 0..31
            unsigned col = col_base + r;
            unsigned col_c = col < m ? col : m - 1;
            const unsigned char* bptr =
                wbytes + ((size_t)col_c * superblocks_per_row + superblock) * Q6_K_BLOCK_BYTES;
            const unsigned char* ql = bptr + 64 * half_idx;
            const unsigned char* qh = bptr + 128 + 32 * half_idx;
            unsigned ql_idx = e + ((group & 1) ? 32 : 0);
            bool high_nibble = group >= 2;
            unsigned qh_shift = group * 2;
            int nib = high_nibble ? (ql[ql_idx] >> 4) : (ql[ql_idx] & 0xF);
            int qv = (nib | (((qh[e] >> qh_shift) & 3) << 4)) - 32;
            w_codes[idx] = (signed char)qv;
        }
        for (unsigned idx = tid; idx < TILE_M; idx += WARPS_PER_BLOCK * 32) {
            unsigned col = col_base + idx;
            unsigned col_c = col < m ? col : m - 1;
            const unsigned char* bptr =
                wbytes + ((size_t)col_c * superblocks_per_row + superblock) * Q6_K_BLOCK_BYTES;
            __half d_h;
            std::memcpy(&d_h, bptr + 208, sizeof(__half));
            float d = __half2float(d_h);
            const signed char* sc = (const signed char*)(bptr + 192) + 8 * half_idx;
            d_w_lds[idx * 2 + 0] = d * (float)sc[group * 2 + 0];
            d_w_lds[idx * 2 + 1] = d * (float)sc[group * 2 + 1];
        }
        __syncthreads();

#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            unsigned subcol = wn * SUBCOLS_PER_WARP + c;
            int4x a0 = load_row_frag_i8(x_codes, wm * 16, 0, lane);
            int4x b0 = load_row_frag_i8(w_codes, subcol * 16, 0, lane);
            int8x acc_i0 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a0, true, b0, int8x{0, 0, 0, 0, 0, 0, 0, 0}, false);
            int4x a1 = load_row_frag_i8(x_codes, wm * 16, 16, lane);
            int4x b1 = load_row_frag_i8(w_codes, subcol * 16, 16, lane);
            int8x acc_i1 = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a1, true, b1, int8x{0, 0, 0, 0, 0, 0, 0, 0}, false);

            unsigned wcol = subcol * 16 + lane % 16;
            float d_w0 = d_w_lds[wcol * 2 + 0];
            float d_w1 = d_w_lds[wcol * 2 + 1];
#pragma unroll
            for (unsigned ele = 0; ele < 8; ++ele) {
                unsigned i = 2 * ele + lane / 16;
                float d_a_val = d_a_lds[wm * 16 + i];
                acc_f[c][ele] += d_a_val * (d_w0 * (float)acc_i0[ele] + d_w1 * (float)acc_i1[ele]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
        unsigned subcol = wn * SUBCOLS_PER_WARP + c;
        unsigned j = lane % 16;
        unsigned col = col_base + subcol * 16 + j;
        if (col >= m) {
            continue;
        }
#pragma unroll
        for (unsigned ele = 0; ele < 8; ++ele) {
            unsigned i = 2 * ele + lane / 16;
            unsigned row = row_base + wm * 16 + i;
            if (row < rows) {
                out[(size_t)row * m + col] = acc_f[c][ele];
            }
        }
    }
}

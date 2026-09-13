// int8 MMQ-style batched prefill-path linear layer for Q5_K-quantized
// weights (int8-MMQ-integration round's mechanical extension of
// `gemm_xwt_quant_mmq_q4_k.hip` — see that file's module doc for the shared
// affine-dequant K-reduction structure this file reuses verbatim). Q5_K
// adds one high bit per element on top of Q4_K's 4-bit nibble (a 5-bit
// unsigned code, 0..31 — still fed to the iu8 op unsigned/widened, same as
// Q4_K's nibble), packed the same way `quant_dequant_elem.h`'s `DequantQ5K`
// unpacks it: a 12-byte `(scale,min)` table (`get_scale_min_k4`, identical
// to Q4_K) plus a 32-byte `qh` high-bit-plane and a 128-byte `qs` low-nibble
// plane (176 bytes total).
//
// Per-superblock indexing mirrors `DequantQ5K` exactly, specialized to this
// kernel's `K_STAGE`(32)-wide, block-aligned iteration (so — like Q4_K — no
// element within one outer iteration ever needs a *different* `(sc, mn,
// mask)` triple than its block-mates): `sub = (k0 % 256) / 32` is `DequantQ5K`'s
// `is` directly (same identity as Q4_K's own `sub`/`is` coincidence — see
// that file's module doc), decomposed further into `step = sub / 2` (which
// 64-wide `j`-group, `DequantQ5K`'s `j/64`) and `half = sub % 2` (low nibble
// / `u1` vs high nibble / `u2` within that group). `qh`'s bit position
// rotates left by 2 per `step` exactly as `DequantQ5K` does (`u1 <<= 2` per
// outer iteration there); `qs`'s byte index (`step*32 + e`) and `qh`'s byte
// index (`e`, `e` in `[0,32)` this block's local offset) are shared between
// both halves of a `step` — the *same* `qs` byte holds one element's low
// nibble for `half=0` and a different element's high nibble for `half=1`,
// identical to Q4_K's own nibble packing.
#include "mmq_common.h"

extern "C" __global__ void gemm_xwt_mmq_q5_k(
    const signed char* x_codes_in, const float* x_scale_in, const float* x_sum_in, const void* w,
    float* out, unsigned rows, unsigned m, unsigned n) {
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

    extern __shared__ unsigned char smem_mmq_q5k[];
    signed char* x_codes = (signed char*)smem_mmq_q5k;
    signed char* w_codes = x_codes + TILE_ROWS * K_STAGE;
    float* d_a_lds = (float*)(w_codes + TILE_M * K_STAGE);
    float* s_a_lds = d_a_lds + TILE_ROWS;
    float* d_w_lds = s_a_lds + TILE_ROWS;
    float* off_w_lds = d_w_lds + TILE_M;

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
        unsigned sub = (k0 % 256) / 32;
        unsigned step = sub / 2;
        unsigned half = sub % 2;
        unsigned char mask = half == 0 ? (unsigned char)(1u << (2 * step))
                                        : (unsigned char)(2u << (2 * step));

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
            s_a_lds[idx] = x_sum_in[(size_t)row_c * n_a_blocks + blk];
        }
        for (unsigned idx = tid; idx < TILE_M * K_STAGE; idx += WARPS_PER_BLOCK * 32) {
            unsigned r = idx / K_STAGE;
            unsigned e = idx % K_STAGE;
            unsigned col = col_base + r;
            unsigned col_c = col < m ? col : m - 1;
            const unsigned char* bptr =
                wbytes + ((size_t)col_c * superblocks_per_row + superblock) * 176;
            const unsigned char* qh = bptr + 16;
            const unsigned char* qs = bptr + 48;
            unsigned char qs_byte = qs[step * 32 + e];
            unsigned char nib = (half == 0) ? (qs_byte & 0xF) : (qs_byte >> 4);
            unsigned char hi = (qh[e] & mask) ? 16 : 0;
            w_codes[idx] = (signed char)(nib + hi);
        }
        for (unsigned idx = tid; idx < TILE_M; idx += WARPS_PER_BLOCK * 32) {
            unsigned col = col_base + idx;
            unsigned col_c = col < m ? col : m - 1;
            const unsigned char* bptr =
                wbytes + ((size_t)col_c * superblocks_per_row + superblock) * 176;
            __half d_h, dmin_h;
            std::memcpy(&d_h, bptr, sizeof(__half));
            std::memcpy(&dmin_h, bptr + 2, sizeof(__half));
            float d = __half2float(d_h);
            float dmin = __half2float(dmin_h);
            const unsigned char* scales = bptr + 4;
            unsigned char sc, mn;
            get_scale_min_k4(sub, scales, &sc, &mn);
            d_w_lds[idx] = d * (float)sc;
            off_w_lds[idx] = dmin * (float)mn;
        }
        __syncthreads();

#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            unsigned subcol = wn * SUBCOLS_PER_WARP + c;
            int4x a0 = load_row_frag_i8(x_codes, wm * 16, 0, lane);
            int4x b0 = load_row_frag_i8(w_codes, subcol * 16, 0, lane);
            int8x acc_i = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a0, false, b0, int8x{0, 0, 0, 0, 0, 0, 0, 0}, false);
            int4x a1 = load_row_frag_i8(x_codes, wm * 16, 16, lane);
            int4x b1 = load_row_frag_i8(w_codes, subcol * 16, 16, lane);
            acc_i = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, false, b1, acc_i, false);

            float d_w_val = d_w_lds[subcol * 16 + lane % 16];
            float off_w_val = off_w_lds[subcol * 16 + lane % 16];
#pragma unroll
            for (unsigned ele = 0; ele < 8; ++ele) {
                unsigned i = 2 * ele + lane / 16;
                float d_a_val = d_a_lds[wm * 16 + i];
                float s_a_val = s_a_lds[wm * 16 + i];
                acc_f[c][ele] += d_w_val * d_a_val * (float)acc_i[ele] - off_w_val * s_a_val;
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

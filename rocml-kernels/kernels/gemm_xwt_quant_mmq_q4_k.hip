// int8 MMQ-style batched prefill-path linear layer for Q4_K-quantized
// weights — the affine sibling of `gemm_xwt_quant_mmq_q8_0.hip` (see that
// file's module doc for the shared fragment layout / K-reduction structure
// this file reuses verbatim via `mmq_common.h`).
//
// Q4_K's affine dequant: a Q4_K weight element is `d*sc_j*q[e] - dmin*mn_j`
// (`q[e]` a 4-bit *unsigned* nibble, `sc_j`/`mn_j` a 6-bit (scale, min) pair
// per 32-element sub-block within the 256-element superblock) — an
// **affine**, not linear, dequant, unlike Q8_0's plain `d*q[e]`. The nibble
// itself needs no re-quantization (fed to the iu8 op unsigned/widened as-is,
// same as `gemm_xwt_mmq_q8_0` does for Q8_0's signed byte), but the
// reduction no longer telescopes into one scale multiply:
// `sum_e w[e]*a[e] = d*sc_j*IntDot(q, qa) - dmin*mn_j*sum_e(a[e])`. The
// second term is exactly why `quantize_act_q8_blk` computes a Q8_1-style
// `block_sum = d_a * sum_e(qa[e])` output, not just a scale — `sum_e(a[e])`
// falls out of it directly, no extra activation read needed. Every
// sub-block's contribution needs two additional per-(col, sub-block)
// scalars (`d*sc_j` in `d_w_lds`, `dmin*mn_j` in `off_w_lds`) alongside the
// per-row `d_a_lds`/`s_a_lds` (`s_a_lds` holding `block_sum`).
// `K_STAGE`(32) exactly matches Q4_K's own sub-block width
// (`get_scale_min_k4`'s granularity, ported from
// `quant_gemv_common.h`/`quant_dequant_elem.h`'s `DequantQ4K` — same bit
// algebra, just producing `(scale, offset)` scalars instead of a
// dequantized element), so — like the Q8_0 kernel — no cross-block K-tail
// handling is ever needed. A Q4_K *superblock* spans 8 sub-blocks
// (256 elements, 144 bytes), so the raw byte offset needs both a
// superblock index (`k0 / 256`) and a sub-block index within it
// (`(k0 % 256) / 32`, which is exactly `get_scale_min_k4`'s `is` argument —
// confirmed to coincide with `k0`'s own 32-wide block index by
// construction: both step by 64 elements per `j`/`is`-pair, and a 32-wide
// MMQ block never straddles that boundary).
#include "mmq_common.h"

extern "C" __global__ void gemm_xwt_mmq_q4_k(
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

    extern __shared__ unsigned char smem_mmq_q4k[];
    signed char* x_codes = (signed char*)smem_mmq_q4k;
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
                wbytes + ((size_t)col_c * superblocks_per_row + superblock) * 144;
            const unsigned char* qs = bptr + 16;
            unsigned byte_off = (sub / 2) * 32 + e;
            unsigned char byte = qs[byte_off];
            unsigned char nib = (sub % 2 == 0) ? (byte & 0xF) : (byte >> 4);
            w_codes[idx] = (signed char)nib;
        }
        for (unsigned idx = tid; idx < TILE_M; idx += WARPS_PER_BLOCK * 32) {
            unsigned col = col_base + idx;
            unsigned col_c = col < m ? col : m - 1;
            const unsigned char* bptr =
                wbytes + ((size_t)col_c * superblocks_per_row + superblock) * 144;
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

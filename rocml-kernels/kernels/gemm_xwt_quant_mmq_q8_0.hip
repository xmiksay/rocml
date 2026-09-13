// int8 MMQ-style batched prefill-path linear layer for Q8_0-quantized
// weights (issue #6's WMMA-pipeline round, lever 2 prototype; wired into
// the model's chunked-prefill dispatch and extended to Q5_K/Q6_K in the
// int8-MMQ-integration round — see `mmq_common.h` and the sibling
// `gemm_xwt_quant_mmq_q{4_k,5_k,6_k}.hip` files, split out of one combined
// file once Q5_K/Q6_K pushed it past the 400-line cap): `out[rows, m] =
// X[rows, n] * dequant(W)^T`, computed via RDNA3's integer matrix unit
// (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`) fed directly with the
// weight's *native* stored codes (Q8_0's signed byte) and a quantized-on-
// the-fly int8 copy of the activation on the other operand — unlike
// `gemm_xwt_wmma_q8_0` (`gemm_xwt_quant_wmma.hip`), no operand is ever
// dequantized to f16. This doubles the matrix unit's per-cycle throughput
// (`amd_matrix_instruction_calculator -a gfx1101 -i
// v_wmma_i32_16x16x16_iu8 -d`: same 32-cycle/16x16x16 execution shape as
// the f16 op but 8192 int8 MACs/wave vs 4096 f16 MACs/wave) and halves the
// LDS traffic per K-element (1 byte vs 2).
//
// Activation quantization is a separate kernel
// (`quantize_act_q8.hip`'s `quantize_act_q8_blk`, host-launched once per
// layer-matmul just like this kernel) producing llama.cpp Q8_1-style
// per-32-element-block `(code: i8, scale: f32)` pairs — block width 32
// exactly matches Q8_0's own native block, so every K-reduction step here
// covers one whole native block on *both* operands, with no cross-block
// splitting ever (unlike `gemm_xwt_wmma_q8_0`'s `K_STAGE`=16 needing a
// zero-padded tail step when `n` isn't a multiple of 16 — here `n` must
// already be a multiple of 32 for a Q8_0 tensor, so it's also automatically
// a multiple of this kernel's `K_STAGE`).
//
// Fragment layout confirmed with
// `amd_matrix_instruction_calculator -a gfx1101 -i
// v_wmma_i32_16x16x16_iu8 -w 32 -d` (mirrors the f16 kernel's own
// probe-and-confirm process): `A[i][k]`/`B[k][j]` pack 4 consecutive K
// elements per 32-bit lane register (`GPR = floor(k/4)`, byte `k%4` within
// it — i.e. a lane's 16-element K-slice is 4 `int32`s, each holding 4
// sequential int8 codes low-byte-first), with the same `i = lane % 16`
// (replicated across the two 16-lane halves) row/column addressing as the
// f16 op; the `int32x8` accumulator `D[i][j]` uses the exact same
// `i = 2*ele + lane/16`, `j = lane % 16` layout as `gemm_xwt_wmma_q8_0`'s
// `float8`.
//
// K-reduction structure: since the per-(row,block)/(col,block) *scale*
// changes every 32 elements, the two chained `k=[0,16)`/`k=[16,32)` WMMA
// ops per outer iteration accumulate into a fresh `int32x8` (reset every
// iteration, not carried across the whole K reduction the way the f16
// kernel's `float8` is) — the raw integer dot product is converted to f32
// and scaled by `d_w[col,blk] * d_a[row,blk]` immediately, then added into
// a running per-(row,col) `float8` that *is* carried across the whole
// reduction. `d_w`/`d_a` are staged into small per-iteration LDS lookup
// tables (`TILE_M`/`TILE_ROWS` floats) alongside the int8 tiles, since a
// WMMA lane's 8 accumulator elements span 8 different output *rows*
// (`i = 2*ele + lane/16`) needing 8 different `d_a` values, not a single
// per-lane broadcast.
#include "mmq_common.h"

extern "C" __global__ void gemm_xwt_mmq_q8_0(
    const signed char* x_codes_in, const float* x_scale_in, const void* w, float* out,
    unsigned rows, unsigned m, unsigned n) {
    unsigned col_base = blockIdx.x * TILE_M;
    unsigned row_base = blockIdx.y * TILE_ROWS;
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    unsigned tid = warp * 32 + lane;
    unsigned wm = warp / WARPS_N;
    unsigned wn = warp % WARPS_N;
    unsigned blocks_per_row = n / K_STAGE;
    unsigned n_a_blocks = n / K_STAGE;
    const unsigned char* wbytes = (const unsigned char*)w;

    extern __shared__ unsigned char smem_mmq_q8_0[];
    signed char* x_codes = (signed char*)smem_mmq_q8_0;
    signed char* w_codes = x_codes + TILE_ROWS * K_STAGE;
    float* d_a_lds = (float*)(w_codes + TILE_M * K_STAGE);
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
            unsigned e = idx % K_STAGE;
            unsigned col = col_base + r;
            unsigned col_c = col < m ? col : m - 1;
            const unsigned char* bptr =
                wbytes + ((size_t)col_c * blocks_per_row + blk) * Q8_0_BLOCK_BYTES;
            w_codes[idx] = (signed char)bptr[2 + e];
        }
        for (unsigned idx = tid; idx < TILE_M; idx += WARPS_PER_BLOCK * 32) {
            unsigned col = col_base + idx;
            unsigned col_c = col < m ? col : m - 1;
            const unsigned char* bptr =
                wbytes + ((size_t)col_c * blocks_per_row + blk) * Q8_0_BLOCK_BYTES;
            __half d_h;
            std::memcpy(&d_h, bptr, sizeof(__half));
            d_w_lds[idx] = __half2float(d_h);
        }
        __syncthreads();

#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            unsigned subcol = wn * SUBCOLS_PER_WARP + c;
            int4x a0 = load_row_frag_i8(x_codes, wm * 16, 0, lane);
            int4x b0 = load_row_frag_i8(w_codes, subcol * 16, 0, lane);
            int8x acc_i = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
                true, a0, true, b0, int8x{0, 0, 0, 0, 0, 0, 0, 0}, false);
            int4x a1 = load_row_frag_i8(x_codes, wm * 16, 16, lane);
            int4x b1 = load_row_frag_i8(w_codes, subcol * 16, 16, lane);
            acc_i = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a1, true, b1, acc_i, false);

            float d_w_val = d_w_lds[subcol * 16 + lane % 16];
#pragma unroll
            for (unsigned ele = 0; ele < 8; ++ele) {
                unsigned i = 2 * ele + lane / 16;
                float d_a_val = d_a_lds[wm * 16 + i];
                acc_f[c][ele] += d_w_val * d_a_val * (float)acc_i[ele];
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

// int8 MMQ-style batched prefill-path linear layer for Q8_0- and
// Q4_K-quantized weights (issue #6's WMMA-pipeline round, lever 2
// prototype): `out[rows, m] = X[rows, n] * dequant(W)^T`, computed via
// RDNA3's integer matrix unit
// (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`) fed directly with the
// weight's *native* stored codes (Q8_0's signed byte, Q4_K's unsigned
// nibble widened to int8 — neither needs re-quantization, they're already
// 8-bit-representable integers) and a quantized-on-the-fly int8 copy of
// the activation on the other operand — unlike `gemm_xwt_wmma_q8_0`/
// `gemm_xwt_wmma_q4_k` (`gemm_xwt_quant_wmma.hip`), no operand is ever
// dequantized to f16. This doubles the matrix unit's per-cycle throughput
// (`amd_matrix_instruction_calculator -a gfx1101 -i
// v_wmma_i32_16x16x16_iu8 -d`: same 32-cycle/16x16x16 execution shape as
// the f16 op but 8192 int8 MACs/wave vs 4096 f16 MACs/wave) and halves the
// LDS traffic per K-element (1 byte vs 2).
//
// **Q5_K/Q6_K are not implemented by this file** (only Q8_0/Q4_K, per the
// WMMA-pipeline round's brief) — Q5_K is a straightforward extension of
// the Q4_K kernel below (same affine `d*sc*q - dmin*mn` shape, one extra
// high-bit mask per element per `DequantQ5K`) but wasn't needed to validate
// or measure this round's numbers; Q6_K has no min-shift term but a
// different (4 groups of 32, not 8 sub-blocks of 32) scale layout and
// wasn't prioritized either. Both are mechanical follow-ups, not a
// different algorithm.
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
//
// Q4_K's affine dequant (`gemm_xwt_mmq_q4_k` below): a Q4_K weight element
// is `d*sc_j*q[e] - dmin*mn_j` (`q[e]` a 4-bit *unsigned* nibble, `sc_j`/
// `mn_j` a 6-bit (scale, min) pair per 32-element sub-block within the
// 256-element superblock) — an **affine**, not linear, dequant, unlike
// Q8_0's plain `d*q[e]`. The nibble itself needs no re-quantization (fed to
// the iu8 op unsigned/widened as-is, same as this file does for Q8_0's
// signed byte), but the reduction no longer telescopes into one scale
// multiply: `sum_e w[e]*a[e] = d*sc_j*IntDot(q, qa) - dmin*mn_j*sum_e(a[e])`.
// The second term is exactly why `quantize_act_q8_blk` computes a Q8_1-style
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
// handling is ever needed.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

namespace {

typedef int int4x __attribute__((ext_vector_type(4)));
typedef int int8x __attribute__((ext_vector_type(8)));

constexpr unsigned TILE_ROWS = 128;
constexpr unsigned TILE_M = 64;
constexpr unsigned K_STAGE = 32; // == Q8_0's QK and the activation quantizer's block width
constexpr unsigned WARPS_M = TILE_ROWS / 16;
constexpr unsigned WARPS_N = 2;
constexpr unsigned SUBCOLS_PER_WARP = (TILE_M / 16) / WARPS_N;
constexpr unsigned WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr unsigned Q8_0_BLOCK_BYTES = 34; // 2-byte f16 scale + 32 int8 codes

// Reads 16 contiguous rows of a `[rows][K_STAGE]` row-major int8 LDS tile
// into a `v_wmma_i32_16x16x16_iu8` operand fragment: lane `L`'s fragment
// packs codes `[row_base + L%16][k_off, k_off+16)` 4-per-`int32` GPR
// (`GPR floor(k/4)`, byte `k%4`) — see the module doc's fragment-layout
// paragraph.
__device__ __forceinline__ int4x load_row_frag_i8(
    const signed char* tile, unsigned row_base, unsigned k_off, unsigned lane) {
    unsigned row = row_base + lane % 16;
    const signed char* p = tile + (size_t)row * K_STAGE + k_off;
    int4x frag;
#pragma unroll
    for (unsigned w = 0; w < 4; ++w) {
        int packed;
        std::memcpy(&packed, p + w * 4, sizeof(int));
        frag[w] = packed;
    }
    return frag;
}

} // namespace

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

    extern __shared__ unsigned char smem_mmq[];
    signed char* x_codes = (signed char*)smem_mmq;
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

// Q4_K variant: the nibble `q[e]` needs no re-quantization (already an
// 8-bit-representable unsigned integer in [0,15] — fed to the iu8 op
// unsigned/widened, same as Q8_0's signed byte above), but a Q4_K element
// is `d*sc*q[e] - dmin*mn` (affine, not linear — see the module doc's "Why
// Q4_K is harder" section), so each 32-element sub-block's contribution
// needs the extra `-dmin*mn*sum_e(a[e])` term. `sum_e(a[e])` is exactly
// `quantize_act_q8_blk`'s `block_sum` output (`x_sum_in` here), already
// scaled by the activation's own per-block `d_a` — no extra activation
// read needed. `K_STAGE`(32) exactly matches Q4_K's own sub-block width
// (`get_scale_min_k4`'s granularity), so — like the Q8_0 kernel — no
// cross-block K-tail handling is ever needed (`n` is already required to
// be a multiple of 256, hence of 32, by `LinearWeight::load`'s policy);
// unlike Q8_0, a Q4_K *superblock* spans 8 sub-blocks (256 elements, 144
// bytes), so the raw byte offset needs both a superblock index
// (`k0 / 256`) and a sub-block index within it (`(k0 % 256) / 32`, which
// is exactly `get_scale_min_k4`'s `is` argument).
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

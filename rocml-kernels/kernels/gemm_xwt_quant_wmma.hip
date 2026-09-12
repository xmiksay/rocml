// RDNA3 WMMA-based batched prefill-path linear layer for GGUF-quantized
// weights: out[rows, m] = X[rows, n] * dequant(W)^T, using gfx11 matrix-core
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 for the K-loop instead of
// scalar FMA (`gemm_xwt_quant.hip`, kept as the dispatch fallback for shapes
// this kernel can't tile cleanly — see that file's module doc). Issue #6's
// follow-up: closing the rest of the gap to the RX 7800 XT's f16 matrix-core
// roofline (~74 TFLOP/s) needs feeding the wave32 matrix unit real f16
// fragments instead of the scalar-FMA/LDS/shuffle dot product the previous
// revision plateaued at (~1.7 TFLOP/s).
//
// Fragment layout (RDNA3/gfx11, wave32; confirmed against AMD's
// `amd_matrix_instruction_calculator` tool's `--detail-instruction` output
// for `v_wmma_f32_16x16x16_f16` and cross-checked against a CPU reference
// with a standalone probe before wiring this in): for
// `D = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, C)`, all three
// 16x16 tiles are addressed `[i][k]`/`[k][j]`/`[i][j]` (row, col) with:
//   - `A[i][k]` fragment: lane `L`'s `<16 x half>` holds row `i = L % 16`
//     (replicated across `L` and `L+16`), one element per `k`.
//   - `B[k][j]` fragment: lane `L`'s `<16 x half>` holds column `j = L % 16`
//     (also replicated), one element per `k` — i.e. `b_frag[k] = B[k][j]`,
//     which for our `B = W^T` is just `W`'s row `j` read exactly the same
//     way as `A`'s row (`W^T[k][j] = W[j][k]`, so no transpose is needed —
//     see `load_row_frag` below, shared by both operands).
//   - `C`/`D[i][j]`: lane `L` holds 8 `f32` values (`ele` in [0,8)) with
//     `j = L % 16`, `i = 2*ele + L/16` — the wave's two 16-lane halves own
//     alternating output rows.
//
// Tiling: a workgroup (`block = (32, WARPS_PER_BLOCK, 1)`) computes a
// `TILE_ROWS x TILE_M` output tile (`grid = (ceil(m/TILE_M),
// ceil(rows/TILE_ROWS), 1)`). Each warp owns one 16-row subrow slice
// (`WARPS_M = TILE_ROWS/16` slices total) and `SUBCOLS_PER_WARP`
// 16-column subtiles within it, accumulating all of them in f32 registers
// across the whole K reduction. The K axis is staged through LDS
// `K_STAGE` elements at a time: cooperative dequant+f16-convert fills
// `x_tile` (`TILE_ROWS x K_STAGE`, from `X`) and `w_tile` (`TILE_M x
// K_STAGE`, from dequantized `W`) once per outer iteration, then every warp
// issues `K_STAGE/16` WMMA ops per column-subtile against that shared LDS
// tile before the block moves on — same "pay the weight dequant cost once,
// reuse across the whole row tile" idea as the scalar kernel, just feeding
// a matrix unit instead of scalar FMA.
//
// Row/column tail handling mirrors the scalar kernel: an out-of-range `X`
// row or `W` row (`col`) is clamped to the last valid one for the LDS fill
// (branch-free) and the corresponding output write is gated instead —
// reading duplicate data for a row that's never written is harmless. The K
// axis is different: a short last stage (`n` not a multiple of `K_STAGE`)
// is zero-padded in LDS instead of clamped, because K is the reduction axis
// every output row/col sums over — garbage there would corrupt every valid
// output, not just a discarded one. `n`/`m` not a multiple of 16 (WMMA's
// native tile width) or `rows` too small to amortize a whole `TILE_ROWS`
// tile are refused by the dispatch layer, not handled here — see
// `rocml/src/forward/kernels_quant.rs`.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "quant_dequant_elem.h"

namespace {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// Tile sizes: chosen empirically (see the perf notes in
// `tests/gemm_xwt_quant_wmma_perf.rs`) by sweeping TILE_ROWS/TILE_M/K_STAGE
// against issue #6's acceptance shapes with a standalone hipcc harness.
// `TILE_ROWS` = 128 exactly matches `PREFILL_CHUNK_SIZE`
// (`qwen35::forward::chunk_forward`), so a full prefill chunk fills one
// row-tile exactly (`grid.y` = 1, no wasted row padding) — this is also why
// the dispatch layer's WMMA/scalar threshold is `rows >= 128`, not some
// smaller "big enough" cutoff. `K_STAGE` = 16 (WMMA's minimum possible,
// one `ks` step per stage) beat every larger value tried (32/64/256) by a
// wide margin: a bigger `K_STAGE` raises LDS-per-block enough to cut how
// many blocks fit resident per WGP, and losing occupancy costs far more
// than the extra `__syncthreads`/staging-loop overhead saves.
constexpr unsigned TILE_ROWS = 128; // output rows (X rows) per workgroup
constexpr unsigned TILE_M = 64;     // output cols (W rows) per workgroup
constexpr unsigned K_STAGE = 16;    // reduction elements staged into LDS per outer iter
constexpr unsigned WARPS_M = TILE_ROWS / 16;                    // 8 subrow slices
constexpr unsigned WARPS_N = 2;                                 // subcol slices split 2 warps wide
constexpr unsigned SUBCOLS_PER_WARP = (TILE_M / 16) / WARPS_N;  // 2
constexpr unsigned WARPS_PER_BLOCK = WARPS_M * WARPS_N;         // 16

// Reads 16 contiguous rows of a `[rows][K_STAGE]` row-major LDS tile into a
// WMMA operand fragment: lane `L`'s fragment holds row `row_base + L % 16`
// (replicated across `L` and `L+16`), one element per `k` in
// `[k_off, k_off+16)`. Shared by both the A (`X` tile) and B (`W` tile)
// loads — per the module doc, `B[k][j] = W[j][k]` means B's fragment is
// just W's row `j` read exactly like A's row, no transpose needed.
__device__ __forceinline__ half16 load_row_frag(
    const _Float16* tile, unsigned row_base, unsigned k_off, unsigned lane) {
    unsigned row = row_base + lane % 16;
    const _Float16* p = tile + (size_t)row * K_STAGE + k_off;
    half16 frag;
#pragma unroll
    for (unsigned k = 0; k < 16; ++k) {
        frag[k] = p[k];
    }
    return frag;
}

template <unsigned QK, unsigned BLOCK_BYTES, typename DequantFn>
__device__ __forceinline__ void gemm_xwt_wmma_impl(
    const float* x, const unsigned char* w, float* out, unsigned rows, unsigned m, unsigned n,
    DequantFn dequant_elem) {
    unsigned col_base = blockIdx.x * TILE_M;
    unsigned row_base = blockIdx.y * TILE_ROWS;
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    unsigned tid = warp * 32 + lane;
    unsigned wm = warp / WARPS_N;
    unsigned wn = warp % WARPS_N;
    unsigned blocks_per_row = n / QK;

    extern __shared__ _Float16 smem[];
    _Float16* x_tile = smem;
    _Float16* w_tile = smem + (size_t)TILE_ROWS * K_STAGE;

    float8 acc[SUBCOLS_PER_WARP];
#pragma unroll
    for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
        acc[c] = {0, 0, 0, 0, 0, 0, 0, 0};
    }

    for (unsigned k0 = 0; k0 < n; k0 += K_STAGE) {
        unsigned elems_here = min(K_STAGE, n - k0);

        // Cooperative stage: X tile (TILE_ROWS x K_STAGE) and W tile
        // (TILE_M x K_STAGE, dequantized on the fly) — a few elements per
        // thread each at the tile sizes above (2048/1024 elements over 512
        // threads).
        for (unsigned idx = tid; idx < TILE_ROWS * K_STAGE; idx += WARPS_PER_BLOCK * 32) {
            unsigned r = idx / K_STAGE;
            unsigned koff = idx % K_STAGE;
            unsigned row = row_base + r;
            unsigned row_c = row < rows ? row : rows - 1;
            float v = koff < elems_here ? x[(size_t)row_c * n + k0 + koff] : 0.0f;
            x_tile[idx] = (_Float16)v;
        }
        for (unsigned idx = tid; idx < TILE_M * K_STAGE; idx += WARPS_PER_BLOCK * 32) {
            unsigned r = idx / K_STAGE;
            unsigned koff = idx % K_STAGE;
            unsigned col = col_base + r;
            unsigned col_c = col < m ? col : m - 1;
            float v = 0.0f;
            if (koff < elems_here) {
                unsigned k_global = k0 + koff;
                unsigned blk = k_global / QK;
                unsigned e = k_global % QK;
                const unsigned char* bptr =
                    w + ((size_t)col_c * blocks_per_row + blk) * BLOCK_BYTES;
                v = dequant_elem(bptr, e);
            }
            w_tile[idx] = (_Float16)v;
        }
        __syncthreads();

        for (unsigned ks = 0; ks < K_STAGE; ks += 16) {
            half16 a_frag = load_row_frag(x_tile, wm * 16, ks, lane);
#pragma unroll
            for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
                unsigned subcol = wn * SUBCOLS_PER_WARP + c;
                half16 b_frag = load_row_frag(w_tile, subcol * 16, ks, lane);
                acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc[c]);
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
                out[(size_t)row * m + col] = acc[c][ele];
            }
        }
    }
}

} // namespace

extern "C" __global__ void gemm_xwt_wmma_q8_0(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<32, 34>(x, (const unsigned char*)w, out, rows, m, n, DequantQ8_0{});
}

extern "C" __global__ void gemm_xwt_wmma_q4_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 144>(x, (const unsigned char*)w, out, rows, m, n, DequantQ4K{});
}

extern "C" __global__ void gemm_xwt_wmma_q5_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 176>(x, (const unsigned char*)w, out, rows, m, n, DequantQ5K{});
}

extern "C" __global__ void gemm_xwt_wmma_q6_k(
    const float* x, const void* w, float* out, unsigned rows, unsigned m, unsigned n) {
    gemm_xwt_wmma_impl<256, 210>(x, (const unsigned char*)w, out, rows, m, n, DequantQ6K{});
}

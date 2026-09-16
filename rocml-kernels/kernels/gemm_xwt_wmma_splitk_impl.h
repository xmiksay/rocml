// Split-K WMMA variant of `gemm_xwt_wmma_impl.h`'s GEMM, for shapes whose
// plain grid (`ceil(m/TM) * ceil(rows/TILE_ROWS)`) is too narrow to fill
// gfx1101's 60 CUs even though the K reduction (`n`) is deep — ffn-down's
// class of shape (m=hidden, n=intermediate; see docs/prefill-gap-analysis.md's
// diagnosis: same total FLOPs as ffn-gate-up, a third the grid blocks,
// because the narrower output dimension is the grid-tiling axis and the
// wider one is only ever a K_STAGE loop trip count). Adding a K-split grid
// axis (`blockIdx.z`) multiplies block count by `num_splits` independent of
// `m`, without touching the tile/fragment shape that's already tuned.
//
// Each block reduces only its own `[k_start, k_end)` K-slice and writes a
// *partial* sum to `partial_out[split, row, col]` — no cross-block
// accumulation, so no atomics and no race. `gemm_splitk_reduce_f32`
// (`kernels/gemm_splitk_reduce.hip`) sums the fixed `num_splits` partials in
// a second pass, always in ascending split order — deterministic summation
// order for a given `(m, n, num_splits)` shape, run to run (the whole reason
// this is a two-pass reduce and not an atomicAdd-into-`out` accumulation:
// atomic float adds land in whatever order warps happen to retire in, which
// is not reproducible across runs and would make the exact-greedy parity
// gates flaky instead of just numerically shifted).
//
// The dispatch layer (`rocml/src/forward/kernels_quant_dispatch.rs`) only
// ever enables split-K when `n` is evenly divisible by `num_splits *
// K_STAGE` — so every split's `[k_start, k_end)` range is itself a multiple
// of `K_STAGE`, and `stage_k_tile`'s per-iteration `elems_here` is always
// exactly `K_STAGE` (same invariant the plain kernel relies on via its own
// `n % 16 == 0` dispatch precondition) — this file never exercises the
// short-stage zero-pad path. Reuses `gemm_xwt_wmma_impl.h`'s
// `stage_k_tile`/`load_row_frag`/fragment-layout verbatim; only the
// reduction-axis loop bounds and the output destination differ from
// `gemm_xwt_wmma_impl`.
#pragma once

#include "gemm_xwt_wmma_impl.h"

namespace {

template <
    unsigned QK, unsigned BLOCK_BYTES, unsigned TR, unsigned TM, unsigned WM, unsigned WN,
    typename DequantFn>
__device__ __forceinline__ void gemm_xwt_wmma_splitk_impl(
    const float* x, const unsigned char* w, float* partial_out, unsigned rows, unsigned m,
    unsigned n, unsigned num_splits, DequantFn dequant_elem) {
    constexpr unsigned SUBROWS_PER_WARP = (TR / 16) / WM;
    constexpr unsigned SUBCOLS_PER_WARP = (TM / 16) / WN;

    unsigned col_base = blockIdx.x * TM;
    unsigned row_base = blockIdx.y * TR;
    unsigned split = blockIdx.z;
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    unsigned tid = warp * 32 + lane;
    unsigned wm = warp / WN;
    unsigned wn = warp % WN;
    unsigned blocks_per_row = n / QK;

    unsigned split_size = n / num_splits;
    unsigned k_start = split * split_size;
    unsigned k_end = k_start + split_size;

    constexpr unsigned X_TILE_ELEMS = TR * ROW_STRIDE;
    constexpr unsigned W_TILE_ELEMS = TM * ROW_STRIDE;
    extern __shared__ _Float16 smem[];
    _Float16* x_base = smem;
    _Float16* w_base = smem + 2 * X_TILE_ELEMS;

    float8 acc[SUBROWS_PER_WARP][SUBCOLS_PER_WARP];
#pragma unroll
    for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            acc[r][c] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    // Prologue: fill buffer 0 with this split's first K-stage.
    stage_k_tile<QK, BLOCK_BYTES, TR, TM, WM, WN>(
        x_base, w_base, x, w, row_base, col_base, rows, m, n, blocks_per_row, k_start, tid,
        dequant_elem);
    __syncthreads();

    unsigned cur = 0;
    for (unsigned k0 = k_start; k0 < k_end; k0 += K_STAGE) {
        unsigned next_k0 = k0 + K_STAGE;
        unsigned next = cur ^ 1;
        if (next_k0 < k_end) {
            stage_k_tile<QK, BLOCK_BYTES, TR, TM, WM, WN>(
                x_base + next * X_TILE_ELEMS, w_base + next * W_TILE_ELEMS, x, w, row_base,
                col_base, rows, m, n, blocks_per_row, next_k0, tid, dequant_elem);
        }

        _Float16* x_tile = x_base + cur * X_TILE_ELEMS;
        _Float16* w_tile = w_base + cur * W_TILE_ELEMS;
        for (unsigned ks = 0; ks < K_STAGE; ks += 16) {
            half16 a_frags[SUBROWS_PER_WARP];
#pragma unroll
            for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
                unsigned subrow = wm * SUBROWS_PER_WARP + r;
                a_frags[r] = load_row_frag(x_tile, subrow * 16, ks, lane);
            }
            half16 b_frags[SUBCOLS_PER_WARP];
#pragma unroll
            for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
                unsigned subcol = wn * SUBCOLS_PER_WARP + c;
                b_frags[c] = load_row_frag(w_tile, subcol * 16, ks, lane);
            }
#pragma unroll
            for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
#pragma unroll
                for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
                    acc[r][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                        a_frags[r], b_frags[c], acc[r][c]);
                }
            }
        }
        __syncthreads();
        cur = next;
    }

    // Write this split's own partial contribution to
    // `partial_out[split, row, col]` — every (split, row, col) triple is
    // written by exactly one block, so this is a plain store, never an
    // atomic or read-modify-write.
#pragma unroll
    for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
        unsigned subcol = wn * SUBCOLS_PER_WARP + c;
        unsigned j = lane % 16;
        unsigned col = col_base + subcol * 16 + j;
        if (col >= m) {
            continue;
        }
#pragma unroll
        for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
            unsigned subrow = wm * SUBROWS_PER_WARP + r;
#pragma unroll
            for (unsigned ele = 0; ele < 8; ++ele) {
                unsigned i = 2 * ele + lane / 16;
                unsigned row = row_base + subrow * 16 + i;
                if (row < rows) {
                    partial_out[((size_t)split * rows + row) * m + col] = acc[r][c][ele];
                }
            }
        }
    }
}

} // namespace

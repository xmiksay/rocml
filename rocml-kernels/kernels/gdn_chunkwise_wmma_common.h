// Shared WMMA fragment loaders for the chunkwise GDN recurrence's matrix-
// core kernels (`gdn_chunkwise_wmma.hip`, gdn-wmma round, issue #6). Kept in
// a header (not a `.hip` file — `build.rs` only compiles `kernels/*.hip`),
// mirroring `gemm_xwt_wmma_impl.h`'s reasoning for the batched-linear-layer
// WMMA kernels.
//
// Every loader here reads directly from **global** memory (no LDS staging):
// unlike `gemm_xwt_wmma_impl.h`'s weight tiles (megabyte-scale, re-read by
// every row-tile block in the grid, hence worth staging/dequanting once per
// block), these operands are per-(head,tile) scratch buffers already
// resident in L2 (at most `tile_len*tile_len*4` bytes per head, `tile_len`
// <= 128) — a block only ever touches one head's slice once, so there is no
// reuse to buy back with a staging pass. Every element is read straight
// from the existing f32 scratch buffer and rounded to `_Float16` in
// registers.
//
// Fragment layout convention (RDNA3/gfx11, wave32 — full derivation and
// citation in `gemm_xwt_wmma_impl.h`'s module doc, reused verbatim here):
// for `D = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(A, B, C)`, lane `L`'s
// `A`/`B` fragment holds one row (`A`) or column (`B`) — `L % 16`,
// replicated across `L` and `L+16` — with one element per `k` in the
// current 16-wide reduction step; the `float8` result `D[i][j]` has
// `j = L % 16`, `i = 2*ele + L/16`.
//
// Every GDN operand here is a plain row-major `[rows, cols]` f32 buffer
// (never the "W stored transposed" trick `gemm_xwt_wmma_impl.h`'s B operand
// exploits), so which loader an operand needs depends only on which axis —
// its own row axis or its own column axis — is the WMMA contraction (`k`)
// axis for that particular matmul:
//   - `load_frag_contig`: the buffer's **row** axis is the fragment's `i`/
//     `j` (replicated-per-lane) axis, so 16 contiguous columns starting at
//     `k_off` are the fragment's `k`-run for one row — a straight
//     contiguous read. Used for an `A`-operand whose contraction axis is
//     its own column axis (`q_norm`, `kq`).
//   - `load_frag_strided`: the buffer's **column** axis is the fragment's
//     `i`/`j` axis, so the `k`-run walks 16 different *rows* at one fixed
//     column — a strided read (`ld` elements apart). Used for a `B`-operand
//     whose contraction axis is its own row axis (`state`, `v_new`), and
//     for `k_norm` in the state-update kernel where the contraction axis
//     (`t`) is `k_norm`'s row axis but the *output* axis (`head_k_dim`) is
//     its column axis.
// Both zero-pad any `k >= k_dim` (a short last reduction step — GDN's
// `tile_len`, `head_k_dim` are never guaranteed multiples of 16, e.g. a
// prompt's partial final tile) and any out-of-range free-axis index (an
// `M`/`N` tile that runs past `n_free`), exactly like `gemm_xwt_wmma_impl.h`
// clamps its own row/col tails — this is what lets the WMMA kernels below
// stay correct (not just fast) for every dimension the existing f64-
// reference test suite already exercises (head dims 8/16/32/128, tile
// lengths down to 1), not only Ornith's real 128/128/128 shape. `row_scale`/
// `k_scale` let a caller fold a bounded (`<=1`, verified per call site — see
// `gdn_chunkwise_wmma.hip`'s doc comments) per-row or per-`k` decay factor
// into the f16 rounding for free, instead of a separate elementwise pass.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

// `base` is `[[>=k_dim rows], n_free]` row-major, contiguous in `n_free`.
// Fragment row = `free_base + lane % 16`; one element per `k` in
// `[k_off, k_off+16)` of that row, scaled by `row_scale[row]` if given.
__device__ __forceinline__ half16 load_frag_contig(
    const float* base, unsigned ld, unsigned n_free, unsigned free_base, unsigned k_off,
    unsigned k_dim, unsigned lane, const float* row_scale) {
    unsigned row = free_base + lane % 16;
    half16 frag;
    if (row < n_free) {
        float scale = row_scale ? row_scale[row] : 1.0f;
        const float* p = base + (size_t)row * ld + k_off;
        unsigned lim = min(16u, k_dim - k_off);
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            frag[k] = (k < lim) ? (_Float16)(p[k] * scale) : (_Float16)0.0f;
        }
    } else {
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            frag[k] = (_Float16)0.0f;
        }
    }
    return frag;
}

// `base` is `[k_dim, n_free]` row-major, contiguous in `n_free` (so the
// contraction axis `k` is `base`'s row axis, `ld` elements apart). Fragment
// column = `free_base + lane % 16`; one element per `k` in
// `[k_off, k_off+16)`, scaled by `k_scale[k]` if given (a per-contraction-
// step, not per-output-row, scale — see `gdn_chunkwise_state_wmma_f32`).
__device__ __forceinline__ half16 load_frag_strided(
    const float* base, unsigned ld, unsigned n_free, unsigned free_base, unsigned k_off,
    unsigned k_dim, unsigned lane, const float* k_scale) {
    unsigned col = free_base + lane % 16;
    half16 frag;
    if (col < n_free) {
        unsigned lim = min(16u, k_dim - k_off);
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            if (k < lim) {
                unsigned kk = k_off + k;
                float v = base[(size_t)kk * ld + col];
                float s = k_scale ? k_scale[kk] : 1.0f;
                frag[k] = (_Float16)(v * s);
            } else {
                frag[k] = (_Float16)0.0f;
            }
        }
    } else {
#pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            frag[k] = (_Float16)0.0f;
        }
    }
    return frag;
}

} // namespace

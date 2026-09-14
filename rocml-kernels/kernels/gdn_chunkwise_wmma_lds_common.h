// LDS-staged WMMA helpers for the GDN chunkwise-recurrence follow-up
// (gdn-wmma-lds round, issue #6): the naive one-warp-per-16x16-tile design in
// `gdn_chunkwise_wmma.hip`/`gdn_chunkwise_wmma_common.h` re-reads its shared
// operands (state/v_new/k_norm/q_norm) from **global** memory once per output
// tile with no cross-block reuse, which measured 2.4x-8.2x *slower* than
// scalar for stages B (`ut_build`)/F (`output`) despite the matrix unit's raw
// throughput advantage (see that file's module doc). This header fixes that:
// one workgroup per `head` stages every operand it needs into LDS **once**,
// shared across every warp's WMMA fragments for that whole head — the same
// "pay the read cost once, reuse across the tile" idea `gemm_xwt_wmma_impl.h`
// uses for the batched-linear-layer GEMM, adapted to this pipeline's smaller,
// runtime-sized (`tile_len`, `head_k_dim`, `head_v_dim`) operands.
//
// **LDS budget** (gfx1101: 64KB/workgroup): every operand this path stages is
// capped at `LDS_FREE`(128) elements on its free (non-reduction) axis and
// `K_SLICE`(64) on its reduction axis per outer iteration — `128*64*2` bytes
// (f16) = 16KB/operand. `gdn_chunkwise_output_wmma_lds_f32` stages 2 operands
// at a time (32KB); `gdn_chunkwise_ut_build_wmma_lds_f32` stages 3 (48KB) —
// both comfortably under the cap without needing double-buffering (this
// pipeline's whole reduction axis is <=128 elements, 1-2 `K_SLICE` steps, so
// there's little steady-state latency to hide compared to the main GEMM's
// up-to-12288-deep reduction). `LDS_FREE`=128 is a **correctness precondition
// of this path**, not just a tuning choice — a `tile_len`/`head_k_dim`/
// `head_v_dim` bigger than 128 would silently leave part of the output
// uncomputed (the fixed `WARPS_M*WARPS_N` tiling below only ever covers a
// 128x128 output region). `tile_len` is architecturally always <=128
// (`GDN_RECUR_TILE`, `gdn_chunkwise.rs`'s sub-chunking); `head_k_dim`/
// `head_v_dim` are per-model load-time constants checked at dispatch
// (`gdn_chunkwise.rs`) — never dynamic per call — so this is a dispatch-time
// gate, not a runtime check in the kernel itself, mirroring the existing
// multiple-of-16 WMMA-eligibility gate.
//
// Every operand in this pipeline is a plain row-major `[rows, cols]` f32
// buffer (see `gdn_chunkwise_wmma_common.h`'s doc for the same point); which
// staging function an operand needs depends only on whether its **own** row
// axis or column axis is the WMMA contraction (`k`) axis for that matmul —
// identical classification to that file's `load_frag_contig`/
// `load_frag_strided`, just staged into LDS once instead of read from global
// per fragment:
//   - `stage_contig_slice`: the buffer's row axis is the free (`i`/`j`) axis
//     — a straight copy into the `[free][k]`-shaped LDS tile.
//   - `stage_transposed_slice`: the buffer's row axis is the contraction
//     axis — the copy transposes so every later WMMA fragment read off LDS
//     stays a simple contiguous load (`load_lds_frag`, mirroring
//     `gemm_xwt_wmma_impl.h`'s `load_row_frag`) instead of a strided,
//     bank-conflict-prone one. Thread mapping puts consecutive threads on
//     consecutive `free` (contiguous in the source buffer) for a coalesced
//     global read; the one-time LDS write absorbs the transpose's stride.
// Both zero-pad past `n_free`/`k_dim`, exactly like the naive path's loaders,
// so `tile_len`/`head_k_dim` need not be multiples of 16 for *this* header's
// helpers to stay correct (only the dispatch-time `LDS_FREE` bound and the
// separate 16-multiple WMMA-eligibility gate matter) — verified by
// `gdn_chunkwise_wmma_lds.rs`'s test suite, which mirrors the naive path's
// non-16-multiple/partial-tile cases.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float float8 __attribute__((ext_vector_type(8)));

constexpr unsigned LDS_FREE = 128;
constexpr unsigned K_SLICE = 64;
constexpr unsigned WARPS_M = 4;
constexpr unsigned WARPS_N = 4;
constexpr unsigned SUBROWS_PER_WARP = (LDS_FREE / 16) / WARPS_M;
constexpr unsigned SUBCOLS_PER_WARP = (LDS_FREE / 16) / WARPS_N;

// `base` is `[>=n_free rows, >=k_dim cols]` row-major (free = row axis).
// Writes an `[LDS_FREE][K_SLICE]` f16 LDS tile, zero-padded past `n_free`/
// this slice's real width. `row_scale`, if given, folds a bounded per-free
// decay factor into the f16 rounding (see each kernel's own doc comment for
// which factors are provably safe this way).
__device__ __forceinline__ void stage_contig_slice(
    _Float16* lds, const float* base, unsigned ld, unsigned n_free, unsigned k_dim, unsigned k0,
    unsigned tid, unsigned nthreads, const float* row_scale) {
    unsigned lim = (k_dim > k0) ? min(K_SLICE, k_dim - k0) : 0;
    for (unsigned idx = tid; idx < LDS_FREE * K_SLICE; idx += nthreads) {
        unsigned free = idx / K_SLICE;
        unsigned k = idx % K_SLICE;
        float v = 0.0f;
        if (free < n_free && k < lim) {
            float s = row_scale ? row_scale[free] : 1.0f;
            v = base[(size_t)free * ld + k0 + k] * s;
        }
        lds[idx] = (_Float16)v;
    }
}

// `base` is `[>=k_dim rows, >=n_free cols]` row-major (free = column axis,
// contraction = row axis) — the transposing counterpart of
// `stage_contig_slice` above. `k_scale`, if given, folds a bounded
// per-contraction-step decay factor into the f16 rounding.
__device__ __forceinline__ void stage_transposed_slice(
    _Float16* lds, const float* base, unsigned ld, unsigned n_free, unsigned k_dim, unsigned k0,
    unsigned tid, unsigned nthreads, const float* k_scale) {
    unsigned lim = (k_dim > k0) ? min(K_SLICE, k_dim - k0) : 0;
    for (unsigned idx = tid; idx < LDS_FREE * K_SLICE; idx += nthreads) {
        // `free` varies fastest so consecutive threads read consecutive
        // elements of `base`'s row (coalesced); the LDS write below is what
        // absorbs the transpose.
        unsigned free = idx % LDS_FREE;
        unsigned k = idx / LDS_FREE;
        float v = 0.0f;
        if (free < n_free && k < lim) {
            unsigned kk = k0 + k;
            float s = k_scale ? k_scale[kk] : 1.0f;
            v = base[(size_t)kk * ld + free] * s;
        }
        lds[(size_t)free * K_SLICE + k] = (_Float16)v;
    }
}

// Reads a fragment from an already-staged `[LDS_FREE][K_SLICE]` LDS tile:
// lane `L`'s fragment holds row `free_base + L % 16` (replicated across `L`
// and `L+16`), one element per `k` in `[k_off, k_off+16)` — a plain
// contiguous load, since staging above already put every operand into this
// `[free][k]` orientation regardless of its source layout.
__device__ __forceinline__ half16 load_lds_frag(
    const _Float16* lds, unsigned free_base, unsigned k_off, unsigned lane) {
    unsigned free = free_base + lane % 16;
    const _Float16* p = lds + (size_t)free * K_SLICE + k_off;
    half16 frag;
#pragma unroll
    for (unsigned k = 0; k < 16; ++k) {
        frag[k] = p[k];
    }
    return frag;
}

// Runs one `K_SLICE`-wide (or shorter, for a short last slice) reduction
// step's worth of WMMA ops against two already-staged, already-synced LDS
// tiles, accumulating into `acc` — shared by every kernel/phase in this
// header's family (`output`'s two phases, `ut_build`'s two accumulators).
__device__ __forceinline__ void wmma_from_lds(
    const _Float16* a_tile, const _Float16* b_tile, unsigned lim, unsigned wm, unsigned wn,
    unsigned lane, float8 acc[SUBROWS_PER_WARP][SUBCOLS_PER_WARP]) {
    for (unsigned ks = 0; ks < lim; ks += 16) {
        half16 a_frags[SUBROWS_PER_WARP];
#pragma unroll
        for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
            a_frags[r] = load_lds_frag(a_tile, (wm * SUBROWS_PER_WARP + r) * 16, ks, lane);
        }
        half16 b_frags[SUBCOLS_PER_WARP];
#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            b_frags[c] = load_lds_frag(b_tile, (wn * SUBCOLS_PER_WARP + c) * 16, ks, lane);
        }
#pragma unroll
        for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
#pragma unroll
            for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
                acc[r][c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frags[r], b_frags[c], acc[r][c]);
            }
        }
    }
}

} // namespace

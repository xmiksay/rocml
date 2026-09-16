// RDNA3 WMMA-based batched prefill-path linear layer for GGUF-quantized
// weights: out[rows, m] = X[rows, n] * dequant(W)^T, using gfx11 matrix-core
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 for the K-loop instead of
// scalar FMA (`gemm_xwt_quant.hip`, kept as the dispatch fallback for shapes
// this kernel can't tile cleanly). Issue #6's follow-up: closing the rest
// of the gap to the RX 7800 XT's f16 matrix-core roofline (~74 TFLOP/s)
// needs feeding the wave32 matrix unit real f16 fragments instead of the
// scalar-FMA/LDS/shuffle dot product the previous revision plateaued at
// (~1.7 TFLOP/s).
//
// Kept in a header (not a `.hip` file — `build.rs` only compiles
// `kernels/*.hip`) so the two tile-config `.hip` files below share one
// bit-for-bit algorithm instead of drifting — the shape-aware dispatch
// round (below) turned `TILE_M`/`WARPS_M`/`WARPS_N` from file-scope
// constants into template parameters for exactly this reason. Each `.hip`
// file just instantiates this header's templates with its own tile config
// and exports the `extern "C"` wrappers.
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
// Tiling: a workgroup (`block = (32, WM*WN, 1)`) computes a `TILE_ROWS x
// TM` output tile (`grid = (ceil(m/TM), ceil(rows/TILE_ROWS), 1)`). Each
// warp owns a `SUBROWS_PER_WARP x SUBCOLS_PER_WARP` block of 16x16 subtiles
// within it (`WM x WN` warps total tile the full `TILE_ROWS x TM` output),
// accumulating all of them in f32 registers across the whole K reduction.
// The K axis is staged through LDS `K_STAGE` elements at a time (each row
// padded to `ROW_STRIDE` halves — see that constant's doc): cooperative
// dequant+f16-convert fills `x_tile` (`TILE_ROWS x ROW_STRIDE`, from `X`)
// and `w_tile` (`TM x ROW_STRIDE`, from dequantized `W`) once per outer
// iteration, then every warp issues `K_STAGE/16` WMMA ops per subtile
// against that shared LDS tile before the block moves on — same "pay the
// weight dequant cost once, reuse across the whole row tile" idea as the
// scalar kernel, just feeding a matrix unit instead of scalar FMA.
//
// Per-wave-efficiency round (issue #6 second follow-up): the WMMA rewrite
// above closed most of the gap to the *practical* scalar-FMA-ceiling
// roofline (11.58/10.41 TFLOP/s measured on ornith-9b's ffn-gate-up/
// ffn-down, 66%/59% of it) but only ~15% of gfx1101's *true* 75 TFLOP/s WMMA
// peak, at 71% measured occupancy — i.e. the gap was per-wave math density,
// not occupancy. Two levers closed a meaningful chunk of it: (1) growing
// each warp's accumulator tile from 1x2 to 2x2 16x16 fragments so every LDS
// fragment load is reused across more WMMA ops (`TM`=128/`WM`=4/`WN`=4,
// `gemm_xwt_quant_wmma.hip`'s default config below), and (2) padding the
// LDS row stride (`ROW_STRIDE`'s doc) to break a bank-conflict pattern in
// those same fragment loads. Measured together at that config: ornith-9b
// prefill 596->684.7 tok/s (+14.9%) at depth 2048, 497->557.1 (+12.1%) at
// depth 8192, 406->445.4 (+9.7%) at depth 16384; decode unaffected (WMMA
// never engages at decode's `rows=1`). A `K_STAGE=32` variant was tried and
// rejected: -10.8% at m=12288. See `tests/gemm_xwt_quant_wmma_perf.rs`'s
// module doc for the full numbers behind all of that round's levers.
//
// Shape-aware dispatch round (issue #6, fixing the per-wave-efficiency
// round's known `m=1024` regression): that round's 2x2/`TM`=128 config won
// at every real ornith-9b `m` except `m=1024` (attn_k/attn_v, -20%, `TM`=128
// nearly halving `grid.x` at the smallest WMMA-eligible `m`) — accepted then
// since every alternative *fixed* tile config regressed that shape as much
// or more. Making `TM`/`WM`/`WN` template parameters (this header) instead
// of file-scope constants lets both configs share one algorithm: the
// original 1x2/`TM`=64 config survives as `gemm_xwt_quant_wmma_narrow.hip`,
// dispatched by `rocml/src/forward/kernels_quant.rs` whenever `m < 2048`
// (crossover measured via a standalone interleaved hipcc harness sweeping
// m in {512,768,...,4096} at rows=512/q4_k: the narrow config wins by
// 6%-64% for every `m <= 1792`, the wide config wins by 1.5%-7% for every
// `m >= 2048` — see that file's own module doc for the full table and
// `.claude/CLAUDE.md` for the end-to-end numbers).
//
// Dequant-off-critical-path round (issue #6, lever 3 deferred by the
// per-wave-efficiency round): 3 levers prototyped via a standalone
// interleaved hipcc harness (rows=512, q4_k/q6_k, m in {1024,4096,12288},
// n in {4096,12288}). Landed: vectorize the X-tile stage to one float4
// read + half4 LDS store/thread (legal: X, unlike W's gather over
// scattered quant blocks, is one contiguous row-major buffer) — +20-26%
// GFLOP/s (q4_k), +1-26% (q6_k), bit-identical output. Rejected: (a)
// sharing the W tile's per-element scale/min decode across the 16 lanes
// covering one row's K_STAGE window via `__shfl(.., 0, 16)` — bit-
// identical but flat-to-worse (-1% q4_k, -8-14% q6_k: broadcast cost beat
// the mostly-cached reads it removed); (b) an 8-producer/8-consumer warp
// split, -34% to -55% (halving WMMA warps while doubling per-warp
// accumulator regs cost more occupancy than it saved — same trap
// `attn_prefill_flash.hip`'s reverted register-tiled draft hit). Sweep
// tables: `tests/gemm_xwt_quant_wmma_perf.rs`, `.claude/CLAUDE.md`.
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
//
// Small-`rows`-tile follow-up (qwen35moe M4, lever 1): `TILE_ROWS` (128,
// below) was a namespace-scope constant until this round because every
// caller processed a whole `PREFILL_CHUNK_SIZE` chunk (or the model's own
// dense/attention layers) at once. qwen35moe's grouped-by-expert batched
// GEMM (`moe_chunk.rs`) is a fundamentally different shape: a 512-token
// chunk's `chunk_len * top_k` (token, expert) assignments spread over up to
// 256 experts average only ~2-16 rows per expert group — far short of even
// one 128-row tile, so every such call fell through to the scalar
// `gemm_xwt_q*` kernel regardless of this file's WMMA path existing at all.
// `TILE_ROWS` is now a template parameter (`TR`) like `TM`/`WM`/`WN`, and
// `gemm_xwt_quant_wmma_micro.hip` instantiates it at `TR=16` (WMMA's
// minimum row-tile — one 16x16 fragment row, the floor below which the
// matrix unit can't be fed at all) specifically for this small-group shape.
// `WM` must divide `TR/16` exactly and `WN` must divide `TM/16` exactly (a
// real correctness requirement — unlike the `stage_k_tile` load-balance
// static_assert below, these two ratios size the accumulator array and an
// inexact division would leave part of the block's own output tile
// uncomputed), which forces `WM=1` at `TR=16` (`TR/16==1` has no other
// integer divisor) — see that file for the micro tile's exact `TM`/`WN`.
//
// Software pipelining (WMMA-pipeline round, issue #6 follow-up): the
// straight-line version above staged the K-tile into one LDS buffer, then
// paid *two* `__syncthreads()` per outer iteration — one so every thread's
// dequant-to-LDS stores were visible before any WMMA op read them, a second
// so every thread's WMMA reads were done before the next iteration
// overwrote the same buffer. Neither the dequant/staging work nor the WMMA
// math depends on the other tile's *data*, only on write-then-read
// ordering into the *same* addresses — so double-buffering (`x_tile`/
// `w_tile` each become a `[2]` array) lets the next stage's global loads +
// dequant + LDS stores for buffer `1-cur` run declared right next to the
// current stage's WMMA math against buffer `cur`, with only one
// `__syncthreads()` per iteration instead of two. A one-time prologue fills
// buffer 0 before the loop starts.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "quant_dequant_elem.h"

namespace {

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float4_t __attribute__((ext_vector_type(4)));

// `K_STAGE` is fixed across every tile config (`TILE_ROWS`/`TILE_M`/
// `WARPS_M`/`WARPS_N` are all template parameters — see this header's module
// doc's "shape-aware dispatch round" and "small-rows-tile follow-up"). The
// default/narrow configs' `TILE_ROWS`=128 exactly matches
// `PREFILL_CHUNK_SIZE`, so a full prefill chunk fills one row-tile exactly
// (`grid.y` = 1) for those two — also why the dispatch layer's WMMA/scalar
// threshold is `rows >= 128` for them; the micro config's `TR`=16 has no
// such correspondence (it targets MoE's per-expert row groups, not a whole
// chunk). `K_STAGE` = 16 (WMMA's minimum) beat every larger value tried
// (32/64/256): a bigger `K_STAGE` raises LDS-per-block enough to cut how
// many blocks fit resident per WGP, costing more occupancy than the extra
// `__syncthreads`/staging overhead saves.
constexpr unsigned K_STAGE = 16;

// LDS row stride, padded past `K_STAGE` (per-wave-efficiency round, lever
// 2): `load_row_frag` below has each of 16 lanes read a *different* row's
// full `K_STAGE` run — at `ROW_STRIDE == K_STAGE == 16` (8 dwords), the
// stride's gcd with the 32-bank LDS width is 8, so those 16 rows only ever
// land on 4 distinct banks (4-way conflict on every read). Padding to 24
// halves (12 dwords, the classic "+8 halves" fix) spreads them out.
// Measured with a standalone hipcc harness sweeping PAD in {0,2,4,8}: +2/+4
// *regressed* sharply (-45 to -50%), while +8 won cleanly (+7.5% at
// m=12288, +8.3% at m=1024) — see `tests/gemm_xwt_quant_wmma_perf.rs`'s
// module doc. LDS headroom is not a concern either way: even at PAD=8 the
// default config uses 24KB/block against gfx1101's 64KB budget.
constexpr unsigned LDS_PAD = 8;
constexpr unsigned ROW_STRIDE = K_STAGE + LDS_PAD;

// Reads 16 contiguous rows of a `[rows][ROW_STRIDE]` row-major LDS tile
// (only the first `K_STAGE` halves of each row are real data — the rest is
// bank-conflict padding, see `ROW_STRIDE`'s doc) into a WMMA operand
// fragment: lane `L`'s fragment holds row `row_base + L % 16` (replicated
// across `L` and `L+16`), one element per `k` in `[k_off, k_off+16)`.
// Shared by both the A (`X` tile) and B (`W` tile) loads — per the module
// doc, `B[k][j] = W[j][k]` means B's fragment is just W's row `j` read
// exactly like A's row, no transpose needed.
__device__ __forceinline__ half16 load_row_frag(
    const _Float16* tile, unsigned row_base, unsigned k_off, unsigned lane) {
    unsigned row = row_base + lane % 16;
    const _Float16* p = tile + (size_t)row * ROW_STRIDE + k_off;
    half16 frag;
#pragma unroll
    for (unsigned k = 0; k < 16; ++k) {
        frag[k] = p[k];
    }
    return frag;
}

// Dequant-and-stage one K-tile (`x_tile`/`w_tile`, whichever buffer the
// caller passes) starting at reduction offset `k0`. Pulled out of
// `gemm_xwt_wmma_impl` so the pipelined loop below can call it identically
// for the prologue fill and every steady-state next-buffer fill. `TR`/`TM`/
// `WM`/`WN` are this instantiation's tile config (see the `.hip` files that
// include this header).
template <
    unsigned QK, unsigned BLOCK_BYTES, unsigned TR, unsigned TM, unsigned WM, unsigned WN,
    typename DequantFn>
__device__ __forceinline__ void stage_k_tile(
    _Float16* x_tile, _Float16* w_tile, const float* x, const unsigned char* w, unsigned row_base,
    unsigned col_base, unsigned rows, unsigned m, unsigned n, unsigned blocks_per_row,
    unsigned k0, unsigned tid, DequantFn dequant_elem) {
    constexpr unsigned WPB = WM * WN;
    unsigned elems_here = min(K_STAGE, n - k0);

    // X tile: one float4 read + packed half4 LDS store per thread (see the
    // "dequant off the critical path" round's doc above). `elems_here ==
    // K_STAGE` always holds here (same `n % 16 == 0` guarantee the W-tile
    // tail comment below relies on). Unlike the `WM`/`WN` divisibility
    // requirements above, this grid-stride loop (`idx4 += WPB * 32`) is
    // correct for *any* `TR`/`WPB` combination — a thread simply stops once
    // `idx4` passes the element count, whether or not that count is an
    // exact multiple of the block's thread count. `TR`=128 (the
    // default/narrow configs) happens to divide evenly and was asserted as
    // a load-balance sanity check; `TR`=16 (the micro config) does not, and
    // is simply less thread-balanced for this one small staging step —
    // never a correctness concern, so the assert was dropped rather than
    // special-cased per `TR`.
    for (unsigned idx4 = tid; idx4 < TR * K_STAGE / 4; idx4 += WPB * 32) {
        unsigned xg_row = idx4 / (K_STAGE / 4), xg_off = (idx4 % (K_STAGE / 4)) * 4;
        unsigned row_c = min(row_base + xg_row, rows - 1);
        float4_t v4;
        memcpy(&v4, x + (size_t)row_c * n + k0 + xg_off, sizeof(float4_t));
        half4 h4 = {(_Float16)v4[0], (_Float16)v4[1], (_Float16)v4[2], (_Float16)v4[3]};
        memcpy(&x_tile[(size_t)xg_row * ROW_STRIDE + xg_off], &h4, sizeof(half4));
    }
    for (unsigned idx = tid; idx < TM * K_STAGE; idx += WPB * 32) {
        unsigned r = idx / K_STAGE;
        unsigned koff = idx % K_STAGE;
        unsigned col = col_base + r;
        unsigned col_c = col < m ? col : m - 1;
        float v = 0.0f;
        if (koff < elems_here) {
            unsigned k_global = k0 + koff;
            unsigned blk = k_global / QK;
            unsigned e = k_global % QK;
            const unsigned char* bptr = w + ((size_t)col_c * blocks_per_row + blk) * BLOCK_BYTES;
            v = dequant_elem(bptr, e);
        }
        w_tile[(size_t)r * ROW_STRIDE + koff] = (_Float16)v;
    }
}

template <
    unsigned QK, unsigned BLOCK_BYTES, unsigned TR, unsigned TM, unsigned WM, unsigned WN,
    typename DequantFn>
__device__ __forceinline__ void gemm_xwt_wmma_impl(
    const float* x, const unsigned char* w, float* out, unsigned rows, unsigned m, unsigned n,
    DequantFn dequant_elem) {
    static_assert(TR % (16 * WM) == 0, "WM must divide TR/16 exactly (see module doc)");
    static_assert(TM % (16 * WN) == 0, "WN must divide TM/16 exactly (see module doc)");
    constexpr unsigned SUBROWS_PER_WARP = (TR / 16) / WM;
    constexpr unsigned SUBCOLS_PER_WARP = (TM / 16) / WN;

    unsigned col_base = blockIdx.x * TM;
    unsigned row_base = blockIdx.y * TR;
    unsigned warp = threadIdx.y;
    unsigned lane = threadIdx.x;
    unsigned tid = warp * 32 + lane;
    unsigned wm = warp / WN;
    unsigned wn = warp % WN;
    unsigned blocks_per_row = n / QK;

    constexpr unsigned X_TILE_ELEMS = TR * ROW_STRIDE;
    constexpr unsigned W_TILE_ELEMS = TM * ROW_STRIDE;
    extern __shared__ _Float16 smem[];
    // Double-buffered: buffer 0 then buffer 1, X tile then W tile within
    // each — see the module doc's "Software pipelining" section. Plain base
    // pointers plus `buf * ELEMS` offsets rather than a `_Float16*[2]`
    // array: hipcc/LLVM tried to promote a local array-of-pointers-into-LDS
    // to a static initializer and failed to lower the resulting
    // addrspace(3) addrspacecast ("unsupported expression in static
    // initializer") — this form sidesteps that entirely.
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

    // Prologue: fill buffer 0 with the first K-stage before the loop starts.
    stage_k_tile<QK, BLOCK_BYTES, TR, TM, WM, WN>(
        x_base, w_base, x, w, row_base, col_base, rows, m, n, blocks_per_row, 0, tid,
        dequant_elem);
    __syncthreads();

    unsigned cur = 0;
    for (unsigned k0 = 0; k0 < n; k0 += K_STAGE) {
        unsigned next_k0 = k0 + K_STAGE;
        unsigned next = cur ^ 1;
        // Issue the next stage's global loads/dequant/LDS stores now, ahead
        // of consuming the current stage below — the hardware overlaps this
        // stage's WMMA math with the load/dequant latency instead of
        // stalling on it every iteration.
        if (next_k0 < n) {
            stage_k_tile<QK, BLOCK_BYTES, TR, TM, WM, WN>(
                x_base + next * X_TILE_ELEMS, w_base + next * W_TILE_ELEMS, x, w, row_base,
                col_base, rows, m, n, blocks_per_row, next_k0, tid, dequant_elem);
        }

        _Float16* x_tile = x_base + cur * X_TILE_ELEMS;
        _Float16* w_tile = w_base + cur * W_TILE_ELEMS;
        for (unsigned ks = 0; ks < K_STAGE; ks += 16) {
            // Load every A-row and B-row fragment this warp needs once, then
            // compute the full SUBROWS_PER_WARP x SUBCOLS_PER_WARP cross
            // product against them — each LDS fragment load is reused
            // instead of being re-read per accumulator.
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
        // Single sync per iteration: publishes `next`'s stores for the next
        // iteration's WMMA reads, and guarantees this iteration's WMMA
        // reads of `cur` finished before `cur` is overwritten as the
        // following iteration's `next` buffer.
        __syncthreads();
        cur = next;
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
        for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
            unsigned subrow = wm * SUBROWS_PER_WARP + r;
#pragma unroll
            for (unsigned ele = 0; ele < 8; ++ele) {
                unsigned i = 2 * ele + lane / 16;
                unsigned row = row_base + subrow * 16 + i;
                if (row < rows) {
                    out[(size_t)row * m + col] = acc[r][c][ele];
                }
            }
        }
    }
}

} // namespace

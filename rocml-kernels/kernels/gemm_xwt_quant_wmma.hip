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
// ceil(rows/TILE_ROWS), 1)`). Each warp owns a `SUBROWS_PER_WARP x
// SUBCOLS_PER_WARP` block of 16x16 subtiles within it (`WARPS_M x WARPS_N`
// warps total tile the full `TILE_ROWS x TILE_M` output), accumulating all
// of them in f32 registers across the whole K reduction. The K axis is
// staged through LDS `K_STAGE` elements at a time (each row padded to
// `ROW_STRIDE` halves — see that constant's doc): cooperative
// dequant+f16-convert fills `x_tile` (`TILE_ROWS x ROW_STRIDE`, from `X`)
// and `w_tile` (`TILE_M x ROW_STRIDE`, from dequantized `W`) once per outer
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
// each warp's accumulator tile from 1x2 to 2x2 16x16 fragments (`WARPS_M`/
// `WARPS_N`/`SUBROWS_PER_WARP`/`SUBCOLS_PER_WARP`'s doc below) so every LDS
// fragment load is reused across more WMMA ops, and (2) padding the LDS row
// stride (`ROW_STRIDE`'s doc) to break a bank-conflict pattern in those same
// fragment loads. Measured together: ornith-9b prefill 596->684.7 tok/s
// (+14.9%) at depth 2048, 497->557.1 (+12.1%) at depth 8192, 406->445.4
// (+9.7%) at depth 16384; decode unaffected (WMMA never engages at
// decode's `rows=1`). A `K_STAGE=32` variant (deeper reduction per LDS
// round-trip, halving `__syncthreads()` frequency) was also tried and
// rejected: -10.8% at m=12288, a wash at m=1024 — see
// `tests/gemm_xwt_quant_wmma_perf.rs`'s module doc for the full numbers
// behind all of this round's levers, landed and rejected.
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
// `__syncthreads()` per iteration instead of two: it both publishes the
// just-filled `1-cur` buffer for the *next* iteration's WMMA reads, and
// guarantees this iteration's WMMA reads of `cur` are done before `cur`
// gets overwritten as the *next* next-buffer two iterations later. A
// one-time prologue fills buffer 0 before the loop starts. Measured impact:
// see `tests/gemm_xwt_quant_wmma_perf.rs`'s module doc for before/after
// GFLOP/s at rows={128,512}, m=12288, n=4096.
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
// Output cols (W rows) per workgroup — doubled from 64 to 128 in the
// per-wave-efficiency round alongside the 2x2 per-warp accumulator tile
// below; see that constant's doc for why the two changes are paired.
constexpr unsigned TILE_M = 128;
constexpr unsigned K_STAGE = 16; // reduction elements staged into LDS per outer iter
// Per-wave accumulator tile (per-wave-efficiency round, issue #6 follow-up):
// each warp now owns `SUBROWS_PER_WARP x SUBCOLS_PER_WARP` = 2x2 16x16
// output fragments instead of the original 1x2, amortizing every LDS
// fragment load (both the A/X row and the B/W row) across more of the other
// operand's fragments — 2 independent A-frag loads are each reused against
// 2 B-frags and vice versa, so LDS reads grow as `SUBROWS_PER_WARP +
// SUBCOLS_PER_WARP` (4) while WMMA math grows as their product (4x the
// per-load math density of the original 1x2's 2). `WARPS_PER_BLOCK` (16) is
// unchanged from the original 1x2 tiling — `TILE_M` doubled alongside
// `WARPS_N` (2->4) specifically to keep it that way, since a standalone
// hipcc sweep (below) found that shrinking warps/block at a *fixed* `TILE_M`
// costs occupancy at small `m` shapes with little upside, while growing
// `TILE_M` to add fragments per warp at a *fixed* warp count doesn't.
//
// Swept {1x2 (orig, TILE_M=64), 2x2 (TILE_M=64, 8 warps/block), 1x4
// (TILE_M=128, 16 warps/block), 2x2 (TILE_M=128, 16 warps/block), 4x2
// (TILE_M=64, 4 warps/block), 4x2 (TILE_M=128, 8 warps/block)} with a
// standalone hipcc harness (interleaved in one process to cancel out
// cross-process GPU clock/contention noise — a real effect measured on this
// machine: identical `cargo test` binaries invoked back-to-back showed up
// to 2x swings from ambient GPU power state) against real production shapes
// including ornith-9b's actual `m`s (12288 FFN, 8192, 4096 attn-q/o/ffn-down,
// 2048, 1024 attn-k/v). 2x2/TILE_M=128 won at every shape except `m=1024`
// (attn_k/attn_v — the smallest real WMMA-eligible `m`, where `TILE_M=128`
// nearly halves `grid.x` and the GPU is too grid-starved to fill): +7 to
// +19% everywhere else, -20% at `m=1024`, still the best (least-bad) of
// every enlarged-tile option there. Every 4x-fragment (`4x2`) variant lost
// at every shape (VGPR pressure collapsing occupancy — matches the
// flash-prefill round's documented register-tiling trap), confirming 2x2 as
// the ceiling for this tile-size regime. Accepted net-positive: `m=1024`
// is ornith-9b's attn_k/attn_v projections, ~7% of the model's total
// per-layer linear-GEMM FLOPs (`docs/prefill-gap-analysis.md`'s FFN GEMMs
// alone are ~42% of prefill wall time, ~7x `attn_k`+`attn_v`'s combined
// share) — see `tests/gemm_xwt_quant_wmma_perf.rs`'s module doc for the
// full before/after GFLOP/s table including this shape.
constexpr unsigned WARPS_M = 4;
constexpr unsigned WARPS_N = 4;
constexpr unsigned SUBROWS_PER_WARP = (TILE_ROWS / 16) / WARPS_M; // 2
constexpr unsigned SUBCOLS_PER_WARP = (TILE_M / 16) / WARPS_N;    // 2
constexpr unsigned WARPS_PER_BLOCK = WARPS_M * WARPS_N;           // 16

// LDS row stride, padded past `K_STAGE` (per-wave-efficiency round, lever
// 2): `load_row_frag` below has each of 16 lanes read a *different* row's
// full `K_STAGE` run — at `ROW_STRIDE == K_STAGE == 16` (8 dwords), the
// stride's gcd with the 32-bank LDS width is 8, so those 16 rows only ever
// land on 4 distinct banks (4-way conflict on every read). Padding to 24
// halves (12 dwords, the classic "+8 halves" fix) spreads them out.
// Measured with a standalone hipcc harness (interleaved same-process to
// cancel cross-run GPU clock noise) sweeping PAD in {0,2,4,8}: +2/+4
// *regressed* sharply (-45 to -50%, likely a non-power-of-2 multiply in the
// address-calc hot loop costing more than the conflict savings), while +8
// won cleanly (+7.5% at m=12288, +8.3% at m=1024) — see
// `tests/gemm_xwt_quant_wmma_perf.rs`'s module doc for the full numbers.
// LDS headroom is not a concern either way: even at PAD=8 this kernel uses
// 24KB/block against gfx1101's 64KB budget.
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
// for the prologue fill and every steady-state next-buffer fill.
template <unsigned QK, unsigned BLOCK_BYTES, typename DequantFn>
__device__ __forceinline__ void stage_k_tile(
    _Float16* x_tile, _Float16* w_tile, const float* x, const unsigned char* w, unsigned row_base,
    unsigned col_base, unsigned rows, unsigned m, unsigned n, unsigned blocks_per_row,
    unsigned k0, unsigned tid, DequantFn dequant_elem) {
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
        x_tile[(size_t)r * ROW_STRIDE + koff] = (_Float16)v;
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
            const unsigned char* bptr = w + ((size_t)col_c * blocks_per_row + blk) * BLOCK_BYTES;
            v = dequant_elem(bptr, e);
        }
        w_tile[(size_t)r * ROW_STRIDE + koff] = (_Float16)v;
    }
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

    constexpr unsigned X_TILE_ELEMS = TILE_ROWS * ROW_STRIDE;
    constexpr unsigned W_TILE_ELEMS = TILE_M * ROW_STRIDE;
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
    stage_k_tile<QK, BLOCK_BYTES>(
        x_base, w_base, x, w, row_base, col_base, rows, m, n, blocks_per_row, 0, tid,
        dequant_elem);
    __syncthreads();

    unsigned cur = 0;
    for (unsigned k0 = 0; k0 < n; k0 += K_STAGE) {
        unsigned next_k0 = k0 + K_STAGE;
        unsigned next = cur ^ 1;
        // Issue the next stage's global loads/dequant/LDS stores now, ahead
        // of consuming the current stage below — the hardware overlaps this
        // stage's WMMA math (independent registers, independent LDS
        // addresses) with the load/dequant latency instead of stalling on
        // it every iteration. No sync needed here: nothing below reads
        // `next`, only `cur` (already synced from the previous iteration or
        // the prologue).
        if (next_k0 < n) {
            stage_k_tile<QK, BLOCK_BYTES>(
                x_base + next * X_TILE_ELEMS, w_base + next * W_TILE_ELEMS, x, w, row_base,
                col_base, rows, m, n, blocks_per_row, next_k0, tid, dequant_elem);
        }

        _Float16* x_tile = x_base + cur * X_TILE_ELEMS;
        _Float16* w_tile = w_base + cur * W_TILE_ELEMS;
        for (unsigned ks = 0; ks < K_STAGE; ks += 16) {
            // Load every A-row and B-row fragment this warp needs once, then
            // compute the full SUBROWS_PER_WARP x SUBCOLS_PER_WARP cross
            // product against them — each LDS fragment load is reused
            // SUBCOLS_PER_WARP (A) or SUBROWS_PER_WARP (B) times instead of
            // being re-read per accumulator. The nested nature also makes
            // each acc[r][c] chain an independent WMMA dependency chain (no
            // serial accumulator), giving the scheduler multiple in-flight
            // chains to interleave and hide the op's multi-cycle latency
            // behind (lever 5 of the per-wave-efficiency round).
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

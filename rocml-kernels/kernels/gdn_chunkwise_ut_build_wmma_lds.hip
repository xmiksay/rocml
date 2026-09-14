// GDN chunkwise-recurrence stage B, LDS-staged WMMA variant (gdn-wmma-lds
// round, issue #6, follow-up to `gdn_chunkwise_wmma.hip`'s perf-rejected
// `gdn_chunkwise_ut_build_wmma_f32`). Same exact algebra as that kernel and
// the scalar `gdn_chunkwise_ut_build_f32` (`gdn_chunkwise.hip`) — the raw,
// unscaled `k_beta . k_j` / `q_i . k_j` dot products are computed via WMMA,
// then `decay(i,j) = exp(g_cum_i - g_cum_j)` and the strictly-lower/
// lower-or-diag masks are applied as a genuine post-accumulator f32 pass
// (see `gdn_chunkwise_wmma.hip`'s doc comment for why `decay` is not
// per-factor-bounded and can't be folded into either f16 operand). Only the
// memory-access pattern changes — see `gdn_chunkwise_wmma_lds_common.h`'s
// module doc.
//
// One workgroup per `head`, `block = (32, WARPS_M*WARPS_N, 1)` = `(32,16,1)`,
// computing the entire `[tile_len, tile_len]` `kb`/`kq` output for that head.
// Three operands share the `head_k_dim` reduction axis (`q_norm`/`k_beta` as
// the two `A` operands, `k_norm` as the shared `B` operand for both
// accumulators) — all three are staged with `stage_contig_slice` (their own
// column axis is the contraction axis, see the common header's doc), 3 x
// 16KB = 48KB peak LDS, under the 64KB cap with no double-buffering needed.
// Dispatched only when `head_k_dim` is a multiple of 16 *and* <=128
// (`LDS_FREE`) — see `gdn_chunkwise.rs`'s dispatch gate.
#include "gdn_chunkwise_wmma_lds_common.h"

extern "C" __global__ void gdn_chunkwise_ut_build_wmma_lds_f32(
    const float* q_norm, const float* k_norm, const float* k_beta, const float* g_cum, float* kb,
    float* kq, unsigned num_v_heads, unsigned head_k_dim, unsigned tile_len) {
    unsigned head = blockIdx.x;
    if (head >= num_v_heads) {
        return;
    }
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    unsigned tid = warp * 32 + lane;
    unsigned nthreads = blockDim.x * blockDim.y;
    unsigned wm = warp / WARPS_N;
    unsigned wn = warp % WARPS_N;

    const float* head_q = q_norm + (size_t)head * tile_len * head_k_dim;
    const float* head_kb_in = k_beta + (size_t)head * tile_len * head_k_dim;
    const float* head_k = k_norm + (size_t)head * tile_len * head_k_dim;
    const float* head_g_cum = g_cum + (size_t)head * tile_len;

    extern __shared__ _Float16 smem[];
    _Float16* k_tile = smem;
    _Float16* kb_a_tile = smem + LDS_FREE * K_SLICE;
    _Float16* kq_a_tile = smem + 2 * LDS_FREE * K_SLICE;

    float8 acc_kb[SUBROWS_PER_WARP][SUBCOLS_PER_WARP];
    float8 acc_kq[SUBROWS_PER_WARP][SUBCOLS_PER_WARP];
#pragma unroll
    for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            acc_kb[r][c] = {0, 0, 0, 0, 0, 0, 0, 0};
            acc_kq[r][c] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    for (unsigned k0 = 0; k0 < head_k_dim; k0 += K_SLICE) {
        stage_contig_slice(k_tile, head_k, head_k_dim, tile_len, head_k_dim, k0, tid, nthreads, nullptr);
        stage_contig_slice(
            kb_a_tile, head_kb_in, head_k_dim, tile_len, head_k_dim, k0, tid, nthreads, nullptr);
        stage_contig_slice(kq_a_tile, head_q, head_k_dim, tile_len, head_k_dim, k0, tid, nthreads, nullptr);
        __syncthreads();
        unsigned lim = min(K_SLICE, head_k_dim - k0);
        wmma_from_lds(kb_a_tile, k_tile, lim, wm, wn, lane, acc_kb);
        wmma_from_lds(kq_a_tile, k_tile, lim, wm, wn, lane, acc_kq);
        __syncthreads();
    }

#pragma unroll
    for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
        unsigned subcol = wn * SUBCOLS_PER_WARP + c;
        unsigned j = subcol * 16 + lane % 16;
        float g_cum_j = (j < tile_len) ? head_g_cum[j] : 0.0f;
#pragma unroll
        for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
            unsigned subrow = wm * SUBROWS_PER_WARP + r;
#pragma unroll
            for (unsigned ele = 0; ele < 8; ++ele) {
                unsigned i = subrow * 16 + 2 * ele + lane / 16;
                if (i >= tile_len || j >= tile_len) {
                    continue;
                }
                float decay = expf(head_g_cum[i] - g_cum_j);
                size_t out_idx = ((size_t)head * tile_len + i) * tile_len + j;
                kb[out_idx] = (j < i) ? acc_kb[r][c][ele] * decay : 0.0f;
                kq[out_idx] = (j <= i) ? acc_kq[r][c][ele] * decay : 0.0f;
            }
        }
    }
}

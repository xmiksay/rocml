// GDN chunkwise-recurrence stage F, LDS-staged WMMA variant (gdn-wmma-lds
// round, issue #6, follow-up to `gdn_chunkwise_wmma.hip`'s perf-rejected
// `gdn_chunkwise_output_wmma_f32`). Same exact algebra as that kernel and its
// scalar counterpart (`gdn_chunkwise_output_f32`, `gdn_chunkwise.hip`) — only
// the memory-access pattern changes, see `gdn_chunkwise_wmma_lds_common.h`'s
// module doc for why (naive per-16x16-tile global reads vs. this file's
// once-per-head LDS staging) and its LDS budget accounting.
//
// One workgroup per `head`, `block = (32, WARPS_M*WARPS_N, 1)` = `(32,16,1)`,
// computing the entire `[tile_len, head_v_dim]` output tile for that head in
// two LDS-staged phases feeding the *same* accumulator (algebraically just a
// reduction axis widened from `head_k_dim` to `head_k_dim + tile_len` — see
// `gdn_chunkwise_wmma.hip`'s `gdn_chunkwise_output_wmma_f32` doc comment for
// why both phases are safe to fold together with no post-accumulator mask):
//   phase 1: `(q_norm * cum_decay_exp) @ state`, reduces over `head_k_dim`.
//   phase 2: `kq @ v_new`, reduces over `tile_len` (already causally masked
//            to zero above the diagonal in `kq`, so an unmasked matmul
//            reproduces the causal sum for free).
// Dispatched only when `head_k_dim`/`head_v_dim` are both multiples of 16
// *and* <=128 (`LDS_FREE`) — see `gdn_chunkwise.rs`'s dispatch gate and
// `gdn_chunkwise_wmma_lds_common.h`'s module doc for why the second bound is
// a correctness precondition of this staged design, not just a perf choice.
#include "gdn_chunkwise_wmma_lds_common.h"

extern "C" __global__ void gdn_chunkwise_output_wmma_lds_f32(
    const float* q_norm, const float* cum_decay_exp, const float* state, const float* kq,
    const float* v_new, float* y, unsigned num_v_heads, unsigned head_k_dim, unsigned head_v_dim,
    unsigned tile_len) {
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
    const float* head_cde = cum_decay_exp + (size_t)head * tile_len;
    const float* head_state = state + (size_t)head * head_k_dim * head_v_dim;
    const float* head_kq = kq + (size_t)head * tile_len * tile_len;
    const float* head_vnew = v_new + (size_t)head * tile_len * head_v_dim;

    extern __shared__ _Float16 smem[];
    _Float16* a_tile = smem;
    _Float16* b_tile = smem + LDS_FREE * K_SLICE;

    float8 acc[SUBROWS_PER_WARP][SUBCOLS_PER_WARP];
#pragma unroll
    for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            acc[r][c] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    for (unsigned k0 = 0; k0 < head_k_dim; k0 += K_SLICE) {
        stage_contig_slice(a_tile, head_q, head_k_dim, tile_len, head_k_dim, k0, tid, nthreads, head_cde);
        stage_transposed_slice(
            b_tile, head_state, head_v_dim, head_v_dim, head_k_dim, k0, tid, nthreads, nullptr);
        __syncthreads();
        wmma_from_lds(a_tile, b_tile, min(K_SLICE, head_k_dim - k0), wm, wn, lane, acc);
        __syncthreads();
    }
    for (unsigned k0 = 0; k0 < tile_len; k0 += K_SLICE) {
        stage_contig_slice(a_tile, head_kq, tile_len, tile_len, tile_len, k0, tid, nthreads, nullptr);
        stage_transposed_slice(
            b_tile, head_vnew, head_v_dim, head_v_dim, tile_len, k0, tid, nthreads, nullptr);
        __syncthreads();
        wmma_from_lds(a_tile, b_tile, min(K_SLICE, tile_len - k0), wm, wn, lane, acc);
        __syncthreads();
    }

#pragma unroll
    for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
        unsigned subcol = wn * SUBCOLS_PER_WARP + c;
        unsigned d = subcol * 16 + lane % 16;
        if (d >= head_v_dim) {
            continue;
        }
#pragma unroll
        for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
            unsigned subrow = wm * SUBROWS_PER_WARP + r;
#pragma unroll
            for (unsigned ele = 0; ele < 8; ++ele) {
                unsigned t = subrow * 16 + 2 * ele + lane / 16;
                if (t < tile_len) {
                    y[((size_t)t * num_v_heads + head) * head_v_dim + d] = acc[r][c][ele];
                }
            }
        }
    }
}

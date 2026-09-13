// Gated Delta Net chunkwise-recurrence pipeline, stage A (`gdn-recur-occupancy`
// round): the per-tile prep kernels, split out of `gdn_chunkwise.hip` into
// their own file purely for the 400-line cap once stage A itself split in
// two (see below) — `gdn_chunkwise.hip`'s module doc has the full seven-
// kernel pipeline overview (stages A1/A2 here, B through G there) and the
// shared numeric contract (`g` already log-decay, tiled grouped-head
// broadcast, head-major scratch layout).
//
// Stage A used to be one kernel (`gdn_chunkwise_prep_f32`, still described
// in `gdn_chunkwise.hip`'s history for context): one block per value head,
// `tile_len` threads, each thread owning one token's whole L2-norm(Q,K) +
// `k_beta` computation *and* participating in a per-head Hillis-Steele scan
// of `g`. That grid is exactly `num_v_heads` blocks (32 at Ornith's shape)
// regardless of `tile_len` — measured 4.75% `OccupancyPercent` on real
// hardware (`rocprofv3 --pmc OccupancyPercent`), the worst of the pipeline's
// six original kernels precisely because the L2-norm work has no cross-token
// dependency at all and could have used a much wider grid. Split here into:
//
//   A1. `gdn_chunkwise_prep_point_f32`  — per-(head,token) L2-norm(Q,K),
//                                         k_beta = K*beta. Embarrassingly
//                                         parallel, one warp per (head,
//                                         token) — grid `(num_v_heads,
//                                         tile_len, 1)`, up to 4096 blocks at
//                                         Ornith's shape.
//   A2. `gdn_chunkwise_prep_cumsum_f32` — inclusive cumsum of g -> g_cum,
//                                         cum_decay_exp = exp(g_cum). This
//                                         half *does* need every token of a
//                                         head's tile in one block for the
//                                         scan, so it keeps the original
//                                         one-block-per-head grid (32
//                                         blocks) — but now carries none of
//                                         A1's per-token work, so its own
//                                         cost shrinks to just the scan.
//                                         Independent of A1 (reads only `g`);
//                                         the two may run in either order.
//
// Measured (rocprofv3 kernel trace, ornith-9b @ depth 2048, n=384 calls
// each): the old fused kernel's 158.66us/call (60.93ms total) became
// prep_point 10.485us/call (4.03ms) + prep_cumsum 2.391us/call (0.92ms) —
// 4.95ms total, a 12.3x reduction for this stage.
#include <hip/hip_runtime.h>

// ── A1: per-(head,token) L2-norm(Q,K), k_beta ──────────────────────────────
//
// `conv_out` is `[tile_len, conv_dim]` (`gdn_layer_step`'s `[Q|K|V]` layout,
// `key_dim = num_k_heads*head_k_dim`, token-major — this kernel is the seam
// between that layout and every downstream kernel's head-major one);
// `beta` is `[tile_len, num_v_heads]` (`gdn_gate_chunk_f32`'s existing output
// layout). Grid `(num_v_heads, tile_len, 1)`, block `(32, 1, 1)` — one warp
// per (head, token) pair, lanes striding over `head_k_dim` and reducing via
// `__shfl_xor` (same cooperative-reduction shape `gdn_chunkwise_ut_build_f32`
// in `gdn_chunkwise.hip` already uses).
extern "C" __global__ void gdn_chunkwise_prep_point_f32(
    const float* conv_out, const float* beta, float* q_norm, float* k_norm, float* k_beta,
    unsigned num_v_heads, unsigned num_k_heads, unsigned head_k_dim, unsigned conv_dim,
    unsigned key_dim, unsigned tile_len, float l2_eps) {
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned lane = threadIdx.x;
    if (head >= num_v_heads || t >= tile_len) {
        return;
    }
    unsigned key_head = head % num_k_heads;
    float q_scale = rsqrtf((float)head_k_dim);

    const float* row = conv_out + (size_t)t * conv_dim;
    const float* q_h = row + (size_t)key_head * head_k_dim;
    const float* k_h = row + key_dim + (size_t)key_head * head_k_dim;

    float sq = 0.0f, sk = 0.0f;
    for (unsigned d = lane; d < head_k_dim; d += 32) {
        float qv = q_h[d];
        float kv = k_h[d];
        sq += qv * qv;
        sk += kv * kv;
    }
    for (unsigned offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_xor(sq, offset, 32);
        sk += __shfl_xor(sk, offset, 32);
    }
    float inv_q = rsqrtf(sq + l2_eps) * q_scale;
    float inv_k = rsqrtf(sk + l2_eps);
    float beta_t = beta[(size_t)t * num_v_heads + head];

    float* q_out = q_norm + ((size_t)head * tile_len + t) * head_k_dim;
    float* k_out = k_norm + ((size_t)head * tile_len + t) * head_k_dim;
    float* kb_out = k_beta + ((size_t)head * tile_len + t) * head_k_dim;
    for (unsigned d = lane; d < head_k_dim; d += 32) {
        float qn = q_h[d] * inv_q;
        float kn = k_h[d] * inv_k;
        q_out[d] = qn;
        k_out[d] = kn;
        kb_out[d] = kn * beta_t;
    }
}

// ── A2: per-head inclusive cumsum of g ──────────────────────────────────────
//
// `g` is `[tile_len, num_v_heads]`. Grid: one block per value head, block
// size `tile_len` (one thread per token) — the Hillis-Steele inclusive scan
// needs every token of a head's tile resident in one block's shared memory,
// so `blockDim.x` must equal `tile_len` exactly (not just `>=`). Independent
// of `gdn_chunkwise_prep_point_f32` above (reads only `g`) — the two may run
// in either order.
// `state_decay[head,t] = exp(g_last - g_cum[t])` (per-wave-efficiency lever,
// gdn round): the state-update kernel (`gdn_chunkwise_state_f32`,
// `gdn_chunkwise.hip`) used to recompute this exact quantity itself, once per
// thread per reduction step — but the value depends only on `(head, t)`, not
// on that kernel's `d`/`k` thread and block indices, so it was being
// recomputed `head_v_dim * head_k_dim` (16384 at Ornith's shape) times more
// often than the `num_v_heads * tile_len` (4096) distinct values that
// actually exist. Computing it once here (this kernel already holds both
// `sh_g[t]` and `sh_g[tile_len-1]` in shared memory from the scan above) and
// having the state kernel just read it back is a pure redundant-computation
// elimination, not a reformulation: same two operands, same subtraction,
// same `expf` — bit-identical to what the state kernel used to compute
// inline, not merely numerically close.
extern "C" __global__ void gdn_chunkwise_prep_cumsum_f32(
    const float* g, float* g_cum, float* cum_decay_exp, float* state_decay,
    unsigned num_v_heads, unsigned tile_len) {
    unsigned head = blockIdx.x;
    unsigned t = threadIdx.x;
    if (head >= num_v_heads || t >= tile_len) {
        return;
    }

    // Inclusive prefix sum of g[:, head] over the tile, Hillis-Steele in
    // shared memory (tile_len <= 128, so <= 7 doubling steps).
    extern __shared__ float sh_g[];
    sh_g[t] = g[(size_t)t * num_v_heads + head];
    __syncthreads();
    for (unsigned offset = 1; offset < tile_len; offset <<= 1) {
        float add = (t >= offset) ? sh_g[t - offset] : 0.0f;
        __syncthreads();
        sh_g[t] += add;
        __syncthreads();
    }
    float g_last = sh_g[tile_len - 1];
    g_cum[(size_t)head * tile_len + t] = sh_g[t];
    cum_decay_exp[(size_t)head * tile_len + t] = expf(sh_g[t]);
    state_decay[(size_t)head * tile_len + t] = expf(g_last - sh_g[t]);
}

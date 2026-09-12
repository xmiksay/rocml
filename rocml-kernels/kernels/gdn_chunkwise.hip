// Gated Delta Net prefill-chunk kernels, chunkwise (blocked delta-rule)
// formulation (issue #6 part 3 follow-up): replaces the token-serial-inside-
// chunk `gdn_recurrence_chunk_f32` (still present in `gdn_chunk.hip` for
// reference/tests) with the standard chunked-delta-rule algorithm used by
// FLA/HF transformers' `torch_chunk_gated_delta_rule` and llama.cpp's
// `build_delta_net_chunking` — see those two references for the algebra
// ground truth this is a direct HIP transcription of. The old kernel's cost
// was O(chunk_len) *sequential* launches-worth of work squeezed into one
// launch (still O(chunk_len) serial dependency, just no launch overhead);
// this formulation turns that into O(chunk_len^2 * head_dim) *parallel*
// matmul-shaped work per chunk, at the cost of a short (`chunk_len`-long)
// serial dependency chain only inside the triangular-inverse kernel below.
//
// Numeric contract carried over unchanged from `gdn_recurrence_chunk_f32`
// (see that kernel's doc comment for the full citation): `g` is already the
// per-token-per-head log-decay (`decay = exp(g)`, ggml's `ssm_a` is
// pre-baked `-exp(A_log)`, no extra exponentiation here), and value head `h`
// reads key head `h % num_k_heads` (llama.cpp's GGUF-conversion-time
// *tiled* broadcast, not HF's raw `repeat_interleave`).
//
// Pipeline (one call per <=128-token tile — `gdn_chunkwise_step`, the host
// wrapper in `rocml/src/qwen35/forward/gdn_chunkwise.rs`, sub-chunks any
// longer request into <=128-token tiles run in sequence, since the
// per-block LDS budget below only fits a 128x128 f32 tile exactly):
//
//   A. `gdn_chunkwise_prep_f32`      — per-token L2-norm(Q,K), k_beta = K*beta,
//                                      inclusive cumsum of g -> g_cum, and
//                                      cum_decay_exp = exp(g_cum).
//   B. `gdn_chunkwise_ut_build_f32`  — kb[i,j] = strictly-lower((k_beta_i . k_j) * decay(i,j))
//                                      kq[i,j] = lower-or-diag((q_i . k_j) * decay(i,j))
//                                      decay(i,j) = exp(g_cum[i] - g_cum[j]).
//   C. `gdn_chunkwise_tinv_f32`      — inverts the unit-lower-triangular
//                                      T = I + kb in place via forward
//                                      substitution (the "UT transform" —
//                                      avoids an iterative/recursive solver
//                                      by exploiting that T's inverse is
//                                      itself unit lower triangular, so each
//                                      row's inverse only needs previously
//                                      finalized rows: this is the "matmul-
//                                      heavy formulation to avoid an
//                                      explicit generic solve" the module
//                                      docs above call out).
//   D+E (fused). `gdn_chunkwise_uv_vnew_f32` — v_new = (Tinv @ (V*beta)) -
//                                      (Tinv @ (k_beta * cum_decay_exp)) @ S_in
//                                      (the only place the carried
//                                      cross-chunk state enters before the
//                                      output/update step; the original
//                                      two-kernel split's `new_values`/
//                                      `k_cumdecay` intermediates are never
//                                      read anywhere else, so fusing them
//                                      away drops a launch and two global
//                                      round trips — see that kernel's doc
//                                      comment).
//   F. `gdn_chunkwise_output_f32`    — y = (q*cum_decay_exp) @ S_in + kq @ v_new.
//   G. `gdn_chunkwise_state_f32`     — S_out = S_in * exp(g_last) + K_gdiff^T @ v_new,
//                                      K_gdiff[t] = k_norm[t] * exp(g_last - g_cum[t]).
//                                      F and G both only depend on `v_new`
//                                      (already complete) and read `S_in`/
//                                      write `S_out` respectively — fusing
//                                      them would race (a block doing G's
//                                      write could run before another
//                                      block's F read of the old state, with
//                                      no ordering guarantee between blocks
//                                      in one launch), so unlike D+E they
//                                      stay separate kernels.
//
// Every intermediate scratch buffer (`ChunkScratch`'s `gdn_cw_*` fields) is
// **head-major**: `[num_v_heads, tile_len, dim]` (`q_norm`/`k_norm`/
// `k_beta`/`v_new`), `[num_v_heads, tile_len]`
// (`g_cum`/`cum_decay_exp`), or `[num_v_heads, tile_len, tile_len]`
// (`kb`/`kq`) — every kernel below processes one head per block
// (`blockIdx.x`), so head-major makes that head's whole working set one
// contiguous slab, which is what makes kernel B's `tile_len` threads (one
// per reduction-partner row `j`) read *adjacent* cache lines instead of
// jumping `num_v_heads * dim` elements between threads. (An earlier
// token-major `[tile_len, dim, num_v_heads]` layout, chosen to mirror
// `conv_out`'s existing per-token contract, measured ~10x slower on kernel
// B alone — a real 30%-of-prefill regression this rewrite exists to fix,
// not a micro-optimization: always measure per-kernel wall time with a
// realistic shape before trusting an FMA-count estimate, see
// `gdn_chunkwise_perf.rs`.) The one exception is `y` (this pipeline's final
// output), which stays `[tile_len, num_v_heads, head_v_dim]` — the
// token-major layout `gdn_recurrence_chunk_f32` used and downstream host
// code (the per-head RMS norm + SiLU-gated output projection) still
// expects. `tile_len` (<=128, the actual chunk length passed in, not a
// fixed constant) is a runtime parameter throughout so the last,
// possibly-partial, chunk of a prompt works unchanged.
#include <hip/hip_runtime.h>

// ── A: per-token L2-norm(Q,K), k_beta, g cumsum ────────────────────────────
//
// `conv_out` is `[tile_len, conv_dim]` (`gdn_layer_step`'s `[Q|K|V]` layout,
// `key_dim = num_k_heads*head_k_dim`, token-major — this kernel is the seam
// between that layout and every downstream kernel's head-major one);
// `beta`/`g` are `[tile_len, num_v_heads]` (`gdn_gate_chunk_f32`'s existing
// output layout). Grid: one block per value head, block size `tile_len`
// (one thread per token) — the cumsum phase additionally uses that block's
// threads for a Hillis-Steele inclusive scan over `g`, so `blockDim.x` must
// equal `tile_len` exactly (not just `>=`).
extern "C" __global__ void gdn_chunkwise_prep_f32(
    const float* conv_out, const float* beta, const float* g, float* q_norm, float* k_norm,
    float* k_beta, float* g_cum, float* cum_decay_exp, unsigned num_v_heads, unsigned num_k_heads,
    unsigned head_k_dim, unsigned conv_dim, unsigned key_dim, unsigned tile_len, float l2_eps) {
    unsigned head = blockIdx.x;
    unsigned t = threadIdx.x;
    if (head >= num_v_heads || t >= tile_len) {
        return;
    }
    unsigned key_head = head % num_k_heads;
    float q_scale = rsqrtf((float)head_k_dim);

    const float* row = conv_out + (size_t)t * conv_dim;
    const float* q_h = row + (size_t)key_head * head_k_dim;
    const float* k_h = row + key_dim + (size_t)key_head * head_k_dim;

    float sq = 0.0f, sk = 0.0f;
    for (unsigned d = 0; d < head_k_dim; ++d) {
        sq += q_h[d] * q_h[d];
        sk += k_h[d] * k_h[d];
    }
    float inv_q = rsqrtf(sq + l2_eps) * q_scale;
    float inv_k = rsqrtf(sk + l2_eps);
    float beta_t = beta[(size_t)t * num_v_heads + head];

    float* q_out = q_norm + ((size_t)head * tile_len + t) * head_k_dim;
    float* k_out = k_norm + ((size_t)head * tile_len + t) * head_k_dim;
    float* kb_out = k_beta + ((size_t)head * tile_len + t) * head_k_dim;
    for (unsigned d = 0; d < head_k_dim; ++d) {
        float qn = q_h[d] * inv_q;
        float kn = k_h[d] * inv_k;
        q_out[d] = qn;
        k_out[d] = kn;
        kb_out[d] = kn * beta_t;
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
    g_cum[(size_t)head * tile_len + t] = sh_g[t];
    cum_decay_exp[(size_t)head * tile_len + t] = expf(sh_g[t]);
}

// Columns (`j`) a block's warps cover per outer-loop iteration in
// `gdn_chunkwise_ut_build_f32` below — `blockDim.y`, chosen so
// `blockDim.x*blockDim.y` (32*8=256) is a comfortable occupancy-friendly
// block size regardless of `tile_len`.
#define UT_BUILD_J_PER_BLOCK 8

// ── B: intra-chunk UT-system and attention matrices ────────────────────────
//
// One block per `(head, row i)`, `blockDim = (32, UT_BUILD_J_PER_BLOCK, 1)`
// — each of the `UT_BUILD_J_PER_BLOCK` warps owns one column `j` at a time
// (warp `w`'s 32 lanes all share `threadIdx.y == w`, so `j = j_base + w` is
// warp-uniform) and reduces `head_k_dim` via `32`-wide coalesced chunks
// (`k_norm[j, lane], k_norm[j, lane+32], ...` — consecutive lanes read
// consecutive floats) plus a `__shfl_xor` butterfly, looping over `j_base`
// to cover all `tile_len` columns. This replaced a one-thread-per-column
// version where consecutive threads read *entire different rows* of
// `k_norm` (`head_k_dim*4` bytes apart) with no lane-to-lane address
// locality at all — ~2.7ms/call at Ornith's shape (32 heads, dims 128),
// found by `gdn_chunkwise_perf.rs`'s per-stage probe and >90% of this
// pipeline's total cost, vs every other stage's <0.3ms. `kb`/`kq` are
// `[num_v_heads, tile_len, tile_len]`, zero outside their mask.
extern "C" __global__ void gdn_chunkwise_ut_build_f32(
    const float* q_norm, const float* k_norm, const float* k_beta, const float* g_cum, float* kb,
    float* kq, unsigned num_v_heads, unsigned head_k_dim, unsigned tile_len) {
    unsigned head = blockIdx.x;
    unsigned i = blockIdx.y;
    unsigned lane = threadIdx.x;
    unsigned warp = threadIdx.y;
    if (head >= num_v_heads || i >= tile_len) {
        return;
    }

    const float* head_g_cum = g_cum + (size_t)head * tile_len;
    const float* head_k_norm = k_norm + (size_t)head * tile_len * head_k_dim;
    const float* kb_i = k_beta + ((size_t)head * tile_len + i) * head_k_dim;
    const float* q_i = q_norm + ((size_t)head * tile_len + i) * head_k_dim;
    float g_cum_i = head_g_cum[i];

    for (unsigned j_base = 0; j_base < tile_len; j_base += UT_BUILD_J_PER_BLOCK) {
        unsigned j = j_base + warp;
        float dot_kb = 0.0f, dot_kq = 0.0f;
        if (j < tile_len) {
            const float* k_j = head_k_norm + (size_t)j * head_k_dim;
            for (unsigned d = lane; d < head_k_dim; d += 32) {
                float kjd = k_j[d];
                dot_kb += kb_i[d] * kjd;
                dot_kq += q_i[d] * kjd;
            }
        }
        for (unsigned offset = 16; offset > 0; offset >>= 1) {
            dot_kb += __shfl_xor(dot_kb, offset, 32);
            dot_kq += __shfl_xor(dot_kq, offset, 32);
        }
        if (lane == 0 && j < tile_len) {
            float decay = expf(g_cum_i - head_g_cum[j]);
            size_t out_idx = ((size_t)head * tile_len + i) * tile_len + j;
            kb[out_idx] = (j < i) ? dot_kb * decay : 0.0f;
            kq[out_idx] = (j <= i) ? dot_kq * decay : 0.0f;
        }
    }
}

// ── C: triangular inverse (the "UT transform") ─────────────────────────────
//
// In-place: `kb` (strictly lower, `[tile_len, tile_len]` per head, already
// contiguous per head in this head-major layout) becomes `Tinv = (I +
// kb)^-1` (also unit lower triangular). One block per head, `blockDim.x ==
// tile_len`, dynamic shared memory `tile_len*tile_len*4` bytes — at
// `tile_len == 128` this is exactly gfx1101's 64KB LDS budget, which is why
// `tile_len` is capped at 128 (see this file's module doc).
//
// Forward substitution from `T @ X = I`, row i: `X[i,i] = 1`, and for
// `j < i`, `X[i,j] = -sum_{k=j}^{i-1} T[i,k] * X[k,j]` — row i only reads
// its own (unmodified-this-far) values and rows `< i`, which are already
// finalized, so the whole thing runs in place with one barrier pair per
// row (compute every row's new values into registers, sync, then publish).
extern "C" __global__ void gdn_chunkwise_tinv_f32(
    float* kb, unsigned num_v_heads, unsigned tile_len) {
    unsigned head = blockIdx.x;
    if (head >= num_v_heads) {
        return;
    }
    unsigned tid = threadIdx.x;
    extern __shared__ float mat[]; // [tile_len, tile_len], row-major
    float* base = kb + (size_t)head * tile_len * tile_len;

    for (unsigned idx = tid; idx < tile_len * tile_len; idx += blockDim.x) {
        mat[idx] = base[idx];
    }
    __syncthreads();

    for (unsigned i = 0; i < tile_len; ++i) {
        float val = 0.0f;
        if (tid < i) {
            float acc = 0.0f;
            for (unsigned k = tid; k < i; ++k) {
                acc += mat[i * tile_len + k] * mat[k * tile_len + tid];
            }
            val = -acc;
        }
        __syncthreads();
        if (tid < i) {
            mat[i * tile_len + tid] = val;
        } else if (tid == i) {
            mat[i * tile_len + i] = 1.0f;
        }
        __syncthreads();
    }

    for (unsigned idx = tid; idx < tile_len * tile_len; idx += blockDim.x) {
        base[idx] = mat[idx];
    }
}

// ── D+E (fused): v_new = Tinv @ (beta-scaled V) - (Tinv @ (k_beta *
//    cum_decay_exp)) @ S_in ─────────────────────────────────────────────────
//
// The original two-kernel split (`new_values`/`k_cumdecay` = Tinv @ ..., then
// `v_new` = `new_values` - `k_cumdecay` @ `S_in`) materialized both
// intermediates to global scratch even though neither is read anywhere else
// in the pipeline — pure launch-count and bandwidth overhead. Fused here:
// one block per `(head, row t)` computes both `Tinv`-weighted sums (the
// causal `k <= t` reduction, same as the old D) directly into registers,
// stages `k_cumdecay`'s row into shared memory (needed in full — the state
// dot product below reduces over *all* `head_k_dim`, not just this thread's
// own `d`), then every thread finishes with its own `v_new[t,d]` in one
// pass. `blockDim.x == max(head_k_dim, head_v_dim)`; dynamic shared memory
// `head_k_dim * sizeof(f32)` bytes.
extern "C" __global__ void gdn_chunkwise_uv_vnew_f32(
    const float* tinv, const float* conv_out, const float* beta, const float* k_beta,
    const float* cum_decay_exp, const float* state, float* v_new, unsigned num_v_heads,
    unsigned num_k_heads, unsigned head_k_dim, unsigned head_v_dim, unsigned conv_dim,
    unsigned key_dim, unsigned tile_len) {
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned d = threadIdx.x;
    if (head >= num_v_heads || t >= tile_len) {
        return;
    }

    const float* tinv_row = tinv + ((size_t)head * tile_len + t) * tile_len;
    const float* head_k_beta = k_beta + (size_t)head * tile_len * head_k_dim;
    const float* head_cde = cum_decay_exp + (size_t)head * tile_len;

    float acc_v = 0.0f, acc_kcd = 0.0f;
    bool do_v = d < head_v_dim;
    bool do_kcd = d < head_k_dim;
    for (unsigned k = 0; k <= t; ++k) {
        float w = tinv_row[k];
        if (do_v) {
            const float* v_h = conv_out + (size_t)k * conv_dim + 2 * key_dim +
                                (size_t)head * head_v_dim;
            float beta_k = beta[(size_t)k * num_v_heads + head];
            acc_v += w * v_h[d] * beta_k;
        }
        if (do_kcd) {
            acc_kcd += w * head_k_beta[k * head_k_dim + d] * head_cde[k];
        }
    }

    extern __shared__ float kcd_row[]; // [head_k_dim]
    if (do_kcd) {
        kcd_row[d] = acc_kcd;
    }
    __syncthreads();

    if (do_v) {
        const float* s = state + (size_t)head * head_k_dim * head_v_dim;
        float acc_state = 0.0f;
        for (unsigned k = 0; k < head_k_dim; ++k) {
            acc_state += kcd_row[k] * s[k * head_v_dim + d];
        }
        v_new[((size_t)head * tile_len + t) * head_v_dim + d] = acc_v - acc_state;
    }
    (void)num_k_heads;
}

// ── F: y = (q * cum_decay_exp) @ S_in + kq @ v_new ─────────────────────────
//
// One block per `(head, row t)`, `blockDim.x == head_v_dim`. `y` is
// `[tile_len, num_v_heads, head_v_dim]` — the same *token-major* layout
// `gdn_recurrence_chunk_f32` wrote, so downstream host code (the per-head
// RMS norm + SiLU-gated output projection) is unchanged; every other
// operand here is head-major.
extern "C" __global__ void gdn_chunkwise_output_f32(
    const float* q_norm, const float* cum_decay_exp, const float* state, const float* kq,
    const float* v_new, float* y, unsigned num_v_heads, unsigned head_k_dim, unsigned head_v_dim,
    unsigned tile_len) {
    unsigned head = blockIdx.x;
    unsigned t = blockIdx.y;
    unsigned d = threadIdx.x;
    if (head >= num_v_heads || t >= tile_len || d >= head_v_dim) {
        return;
    }
    const float* q_t = q_norm + ((size_t)head * tile_len + t) * head_k_dim;
    float cde_t = cum_decay_exp[(size_t)head * tile_len + t];
    const float* s = state + (size_t)head * head_k_dim * head_v_dim;
    const float* kq_row = kq + ((size_t)head * tile_len + t) * tile_len;
    const float* head_v_new = v_new + (size_t)head * tile_len * head_v_dim;

    float inter = 0.0f;
    for (unsigned k = 0; k < head_k_dim; ++k) {
        inter += q_t[k] * cde_t * s[k * head_v_dim + d];
    }
    float intra = 0.0f;
    for (unsigned k = 0; k <= t; ++k) {
        intra += kq_row[k] * head_v_new[k * head_v_dim + d];
    }
    y[((size_t)t * num_v_heads + head) * head_v_dim + d] = inter + intra;
}

// ── G: S_out = S_in * exp(g_last) + K_gdiff^T @ v_new ──────────────────────
//
// One block per `(head, row k)` of the state (`blockIdx.y` indexes
// `head_k_dim`), `blockDim.x == head_v_dim`. `state` is mutated in place —
// safe since each thread only ever reads/writes its own `[head, k, d]`
// element (the `v_new`/`k_norm` reduction over `t` never touches `state`).
extern "C" __global__ void gdn_chunkwise_state_f32(
    const float* k_norm, const float* g_cum, const float* v_new, float* state,
    unsigned num_v_heads, unsigned head_k_dim, unsigned head_v_dim, unsigned tile_len) {
    unsigned head = blockIdx.x;
    unsigned k = blockIdx.y;
    unsigned d = threadIdx.x;
    if (head >= num_v_heads || k >= head_k_dim || d >= head_v_dim) {
        return;
    }
    const float* head_g_cum = g_cum + (size_t)head * tile_len;
    const float* head_k_norm = k_norm + (size_t)head * tile_len * head_k_dim;
    const float* head_v_new = v_new + (size_t)head * tile_len * head_v_dim;
    float g_last = head_g_cum[tile_len - 1];
    float chunk_decay = expf(g_last);

    float acc = 0.0f;
    for (unsigned t = 0; t < tile_len; ++t) {
        float kn = head_k_norm[t * head_k_dim + k];
        float gdiff = expf(g_last - head_g_cum[t]);
        acc += kn * gdiff * head_v_new[t * head_v_dim + d];
    }
    float* s = state + (size_t)head * head_k_dim * head_v_dim + (size_t)k * head_v_dim + d;
    *s = *s * chunk_decay + acc;
}

// GDN chunkwise-recurrence WMMA follow-up (gdn-wmma round, issue #6): matrix-
// core reformulations for stages B (`gdn_chunkwise_ut_build_f32`), F
// (`gdn_chunkwise_output_f32`) and G (`gdn_chunkwise_state_f32`) — three of
// the four non-`tinv` stages by measured wall time (`gdn_chunkwise_perf.rs`,
// Ornith shape: F 0.3049ms/call, D+E 0.2915ms, B 0.2045ms, G 0.1934ms).
// `tinv` stays scalar (serial forward-substitution, already evaluated and
// rejected for a block-recursive rewrite in the `gdn-recur-occupancy` round
// — see `gdn_chunkwise.hip`'s module doc). D+E (`uv_vnew`) was evaluated
// (see the round's report) but not implemented: it chains three dependent
// matmuls through one shared-memory intermediate and its output (`v_new`)
// feeds the *recurrent* state carried across every tile of a prompt.
//
// All three kernels below pass their correctness gate (f64-reference,
// `rocml-kernels/tests/gdn_chunkwise_wmma.rs`) — but only G
// (`gdn_chunkwise_state_wmma_f32`) is actually wired into the host dispatch
// (`rocml/src/qwen35/forward/gdn_chunkwise_kernels_wmma.rs`). Real-shape
// `rocprofv3` measurement on ornith-9b (depth 8192) found B and F *slower*
// than the scalar kernels they'd replace (B 197.2us->1610.2us/call, 8.2x
// slower; F 294.0us->694.7us/call, 2.4x slower) despite G winning cleanly
// (186.6us->92.3us/call, ~2x) — see `gdn_chunkwise_kernels_wmma.rs`'s module
// doc for the numbers and root cause (B/F each read 2-3 operands shared
// across many output tiles straight from global memory, once per 16x16
// tile, with no LDS staging or multi-warp reuse; G's single matmul has
// minimal such sharing). B and F are kept here as correctness-proven,
// perf-rejected prototypes — the same disposition this codebase already
// gives the int8 MMQ GEMM (`.claude/CLAUDE.md`'s WMMA-pipeline round) —
// rather than deleted, since the algebra derivation below is reused
// verbatim by the LDS-staged follow-up that *is* wired into production
// (gdn-wmma-lds round: `kernels/gdn_chunkwise_{ut_build,output}_wmma_lds.hip`
// stage the same shared operands into LDS once per head instead of
// re-reading them from global per 16x16 tile — see those files' module docs
// and `gdn_chunkwise_kernels_wmma.rs`'s doc for the measured 2.33x/2.97x
// wins over scalar that fixed exactly the redundant-traffic problem noted
// above).
//
// Every kernel below is an *exact* algebraic reformulation of its scalar
// counterpart as a `[16-multiple, 16-multiple]`-friendly matmul — see each
// kernel's own comment for why every mask/decay term in the original scalar
// loop is either (a) already zero in the source buffer (so an unmasked full
// matmul reproduces it for free), (b) a bounded (`<=1`) per-row/per-`k`
// scale folded into the f16 rounding without a magnitude-blowup risk, or
// (c) — `ut_build` only, `decay(i,j) = exp(g_cum_i - g_cum_j)` is NOT
// factor-bounded on its own (only the *ratio* is) — applied as a genuine
// post-accumulator elementwise pass in f32, exactly the "mask on the
// accumulator" escape hatch this round's brief called for. Grid is
// `(num_v_heads, ceil(M/16), ceil(N/16))`, one warp (32 threads, one 16x16
// output tile) per block — `blockDim = (32,1,1)`. This trades the wider
// fragment-reuse tiling `gemm_xwt_wmma_impl.h` uses for simplicity (no LDS
// staging at all — see `gdn_chunkwise_wmma_common.h`'s module doc) since
// correctness, not peak matmul throughput, was this round's initial
// hypothesis for the binding constraint on a small-`K`-class problem; that
// hypothesis held for G but not for B/F, per the measurement above.
//
// Host dispatch only routes G to this file's kernel, and only when
// `head_k_dim`/`head_v_dim` are both multiples of 16 (a per-model load-time
// constant, always true for Ornith/Qwen3.5's 128) — the fragment loaders
// zero-pad any short last `k`-step or out-of-range `M`/`N` tail regardless,
// so `tile_len` itself (dynamic, includes a prompt's partial last tile)
// never needs to be a multiple of 16 — verified against every `tile_len`
// the existing f64-reference suite already covers (1, 5, 13, 37, 127, 128,
// `rocml-kernels/tests/gdn_chunkwise.rs`).
#include "gdn_chunkwise_wmma_common.h"

// ── F: y = (q * cum_decay_exp) @ S_in + kq @ v_new, as one 2-phase matmul ──
//
// Both terms already avoid needing a post-matmul mask/decay pass:
//   - `cum_decay_exp[t] = exp(g_cum[t]) <= 1` always (`g_cum` is a cumsum of
//     non-positive log-decays, `gdn_chunkwise.hip`'s module doc) — folding
//     it into `q_norm`'s f16 rounding as a per-row (`t`) scale is a bounded,
//     safe rescale, not a decomposition that could blow up in f16 (contrast
//     `ut_build`'s `decay(i,j) = exp(g_cum_i - g_cum_j)`, which is NOT
//     bounded per-factor and is why that kernel — not implemented in WMMA
//     this round — would need a post-accumulator mask instead).
//   - `kq[t,k]` is already exactly zero for `k > t` (built that way in
//     `gdn_chunkwise_ut_build_f32`: `(j <= i) ? dot_kq * decay : 0.0f`), so
//     a full, unmasked `[tile_len, tile_len] @ [tile_len, head_v_dim]`
//     matmul reproduces the causal `sum_{k<=t}` reduction exactly — the
//     zero entries contribute nothing, no post-processing needed.
// Phase 1 contracts over `head_k_dim` (`q_norm` scaled by `cum_decay_exp` as
// operand A, `state` as operand B); phase 2 contracts over `tile_len` (`kq`
// as A, `v_new` as B) into the *same* accumulator — algebraically just
// widening the reduction axis from `head_k_dim` to `head_k_dim + tile_len`.
extern "C" __global__ void gdn_chunkwise_output_wmma_f32(
    const float* q_norm, const float* cum_decay_exp, const float* state, const float* kq,
    const float* v_new, float* y, unsigned num_v_heads, unsigned head_k_dim, unsigned head_v_dim,
    unsigned tile_len) {
    unsigned head = blockIdx.x;
    if (head >= num_v_heads) {
        return;
    }
    unsigned row_base = blockIdx.y * 16;
    unsigned col_base = blockIdx.z * 16;
    unsigned lane = threadIdx.x;

    const float* head_q = q_norm + (size_t)head * tile_len * head_k_dim;
    const float* head_cde = cum_decay_exp + (size_t)head * tile_len;
    const float* head_state = state + (size_t)head * head_k_dim * head_v_dim;
    const float* head_kq = kq + (size_t)head * tile_len * tile_len;
    const float* head_vnew = v_new + (size_t)head * tile_len * head_v_dim;

    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};

    for (unsigned k0 = 0; k0 < head_k_dim; k0 += 16) {
        half16 a = load_frag_contig(head_q, head_k_dim, tile_len, row_base, k0, head_k_dim, lane, head_cde);
        half16 b = load_frag_strided(head_state, head_v_dim, head_v_dim, col_base, k0, head_k_dim, lane, nullptr);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }
    for (unsigned k0 = 0; k0 < tile_len; k0 += 16) {
        half16 a = load_frag_contig(head_kq, tile_len, tile_len, row_base, k0, tile_len, lane, nullptr);
        half16 b = load_frag_strided(head_vnew, head_v_dim, head_v_dim, col_base, k0, tile_len, lane, nullptr);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }

#pragma unroll
    for (unsigned ele = 0; ele < 8; ++ele) {
        unsigned t = row_base + 2 * ele + lane / 16;
        unsigned d = col_base + lane % 16;
        if (t < tile_len && d < head_v_dim) {
            y[((size_t)t * num_v_heads + head) * head_v_dim + d] = acc[ele];
        }
    }
}

// ── G: S_out = S_in * exp(g_last) + K_gdiff^T @ v_new ──────────────────────
//
// `K_gdiff[t,k] = k_norm[t,k] * state_decay[t]`, `state_decay[t] =
// exp(g_last - g_cum[t]) <= 1` always (`g_cum` is non-increasing, `g_last`
// is its final, smallest value — see `gdn_chunkwise_prep.hip`'s doc comment
// for the identical bound already relied on there) — a safe per-`k`-step
// (here `k` is the reduction index `t`) scale folded into `k_norm`'s f16
// rounding via `load_frag_strided`'s `k_scale`, no post-accumulator pass
// needed. `chunk_decay = cum_decay_exp[tile_len-1] <= 1` (same bound) scales
// the carried-in state and stays a plain f32 elementwise op on the
// accumulator, exactly like the scalar kernel it replaces.
//
// Both operands contract over `t` (`tile_len`) on their own *row* axis
// (`k_norm`/`v_new` are both `[tile_len, dim]` row-major) — `head_k_dim`
// (this matmul's row/`M` axis) and `head_v_dim` (`N`) are both the free
// *column* axis of their respective source buffers, so both fragments use
// `load_frag_strided` (see that function's doc comment for why).
extern "C" __global__ void gdn_chunkwise_state_wmma_f32(
    const float* k_norm, const float* cum_decay_exp, const float* state_decay, const float* v_new,
    float* state, unsigned num_v_heads, unsigned head_k_dim, unsigned head_v_dim,
    unsigned tile_len) {
    unsigned head = blockIdx.x;
    if (head >= num_v_heads) {
        return;
    }
    unsigned row_base = blockIdx.y * 16; // head_k_dim tile
    unsigned col_base = blockIdx.z * 16; // head_v_dim tile
    unsigned lane = threadIdx.x;

    const float* head_k_norm = k_norm + (size_t)head * tile_len * head_k_dim;
    const float* head_vnew = v_new + (size_t)head * tile_len * head_v_dim;
    const float* head_state_decay = state_decay + (size_t)head * tile_len;
    float chunk_decay = cum_decay_exp[(size_t)head * tile_len + tile_len - 1];

    float8 acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned k0 = 0; k0 < tile_len; k0 += 16) {
        half16 a = load_frag_strided(
            head_k_norm, head_k_dim, head_k_dim, row_base, k0, tile_len, lane, head_state_decay);
        half16 b = load_frag_strided(head_vnew, head_v_dim, head_v_dim, col_base, k0, tile_len, lane, nullptr);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, acc);
    }

    float* base = state + (size_t)head * head_k_dim * head_v_dim;
#pragma unroll
    for (unsigned ele = 0; ele < 8; ++ele) {
        unsigned k = row_base + 2 * ele + lane / 16;
        unsigned d = col_base + lane % 16;
        if (k < head_k_dim && d < head_v_dim) {
            float* s = base + (size_t)k * head_v_dim + d;
            *s = *s * chunk_decay + acc[ele];
        }
    }
}

// ── B: kb[i,j] = strictly-lower((k_beta_i . k_j) * decay(i,j)),
//      kq[i,j] = lower-or-diag((q_i . k_j) * decay(i,j)) ──────────────────
//
// `decay(i,j) = exp(g_cum_i - g_cum_j)` is not itself magnitude-bounded (only
// the *product* with a strictly-lower/lower-or-diag mask is, since `g_cum`
// is non-increasing so `decay(i,j) <= 1` exactly where the mask keeps it —
// off-mask it can be arbitrarily large, which is precisely why it is
// unsafe to pre-scale either f16 operand by it) — so unlike F/G above, this
// kernel computes the *raw*, unscaled dot products via WMMA (`k_beta . k_j`
// and `q_i . k_j`, both contracting over `head_k_dim`) and applies
// `decay(i,j)` plus the triangular mask as a genuine post-accumulator f32
// pass, reading `g_cum` directly rather than folding a decomposed factor
// into either operand — bit-for-bit the same formula the scalar kernel
// computes, just after a matrix-core dot product instead of a warp-shuffle
// reduction. `k_norm` is used as the shared `B` operand for both `kb`'s and
// `kq`'s dot product (`k_beta`/`q_norm` are the two `A` operands) — one
// `load_frag_contig` load of `k_norm`'s column-`j` fragment feeds both
// accumulators per `k`-step, since `head_k_dim` is `kb`/`kq`'s only
// reduction axis and `k_norm[j,:]` (this kernel's `B[k][j]`, per
// `gdn_chunkwise_wmma_common.h`'s "lucky" contiguous-load case — `k_norm`
// is stored `[tile_len, head_k_dim]` row-major, so reading row `j`
// contiguously already gives exactly the `B[k][j]` fragment the WMMA
// convention wants) doesn't depend on which of `kb`/`kq` is being computed.
extern "C" __global__ void gdn_chunkwise_ut_build_wmma_f32(
    const float* q_norm, const float* k_norm, const float* k_beta, const float* g_cum, float* kb,
    float* kq, unsigned num_v_heads, unsigned head_k_dim, unsigned tile_len) {
    unsigned head = blockIdx.x;
    if (head >= num_v_heads) {
        return;
    }
    unsigned row_base = blockIdx.y * 16; // i
    unsigned col_base = blockIdx.z * 16; // j
    unsigned lane = threadIdx.x;

    const float* head_q = q_norm + (size_t)head * tile_len * head_k_dim;
    const float* head_kb_in = k_beta + (size_t)head * tile_len * head_k_dim;
    const float* head_k = k_norm + (size_t)head * tile_len * head_k_dim;
    const float* head_g_cum = g_cum + (size_t)head * tile_len;

    float8 acc_kb = {0, 0, 0, 0, 0, 0, 0, 0};
    float8 acc_kq = {0, 0, 0, 0, 0, 0, 0, 0};
    for (unsigned k0 = 0; k0 < head_k_dim; k0 += 16) {
        half16 b = load_frag_contig(head_k, head_k_dim, tile_len, col_base, k0, head_k_dim, lane, nullptr);
        half16 a_kb =
            load_frag_contig(head_kb_in, head_k_dim, tile_len, row_base, k0, head_k_dim, lane, nullptr);
        half16 a_kq = load_frag_contig(head_q, head_k_dim, tile_len, row_base, k0, head_k_dim, lane, nullptr);
        acc_kb = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_kb, b, acc_kb);
        acc_kq = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_kq, b, acc_kq);
    }

    unsigned j = col_base + lane % 16;
    float g_cum_j = (j < tile_len) ? head_g_cum[j] : 0.0f;
#pragma unroll
    for (unsigned ele = 0; ele < 8; ++ele) {
        unsigned i = row_base + 2 * ele + lane / 16;
        if (i >= tile_len || j >= tile_len) {
            continue;
        }
        float decay = expf(head_g_cum[i] - g_cum_j);
        size_t out_idx = ((size_t)head * tile_len + i) * tile_len + j;
        kb[out_idx] = (j < i) ? acc_kb[ele] * decay : 0.0f;
        kq[out_idx] = (j <= i) ? acc_kq[ele] * decay : 0.0f;
    }
}

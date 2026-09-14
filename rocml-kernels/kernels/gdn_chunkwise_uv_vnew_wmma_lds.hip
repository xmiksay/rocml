// GDN chunkwise-recurrence stages D+E (fused), LDS-staged WMMA variant
// (gdn-uvvnew round, issue #6 — the last item on the structured-prefill
// worklist: `tinv`'s serial forward-substitution was rejected for rewrite
// twice, `ut_build`/`output`/`state` are already WMMA-ized, this closes
// `uv_vnew`, the only other scalar kernel in the seven-stage pipeline).
//
// ── Derived algebra (from `gdn_chunkwise_uv_vnew_f32`, `gdn_chunkwise.hip`) ─
//
// The scalar kernel computes, per `(head, row t)`:
//   acc_v[t,d]   = sum_{k=0}^{t} Tinv[t,k] * V[k,d] * beta[k]
//   acc_kcd[t,c] = sum_{k=0}^{t} Tinv[t,k] * k_beta[k,c] * cum_decay_exp[k]
//   v_new[t,d]   = acc_v[t,d] - sum_{c=0}^{head_k_dim-1} acc_kcd[t,c] * state[c,d]
// i.e. in matrix form (row t = free axis, `V`/`k_beta`/`state` all read as
// their natural `[[email protected]]` layout):
//   VB      = V * beta            (elementwise per-row scale, [tile_len, head_v_dim])
//   KB      = k_beta * cum_decay_exp   (elementwise per-row scale, [tile_len, head_k_dim])
//   K_cd    = Tinv @ KB            ([tile_len, head_k_dim] — the mid-kernel
//                                   intermediate that must become a fresh
//                                   WMMA operand below)
//   v_new   = Tinv @ VB - K_cd @ state
// This is the exact same "Tinv @ (beta-scaled V)" / "(Tinv @ (k_beta *
// cum_decay_exp)) @ S_in" the scalar kernel's own doc comment already names
// — the derivation above only makes the three-matmul chain explicit so each
// factor's boundedness can be checked before choosing how to feed it to
// f16 matrix cores:
//   (1) `Tinv` is unit lower triangular, exactly zero above the diagonal (the
//       `tinv` kernel constructs it that way) — precisely the same
//       "unmasked WMMA reproduces the causal sum for free" case
//       `gdn_chunkwise_wmma.hip`'s `output`/`uv_vnew` write-up already
//       established for `output`'s `kq`. Both `Tinv @ VB` and `Tinv @ KB`
//       reduce over the *full* `tile_len` axis with no post-accumulator mask.
//   (2) `beta = sigmoid(b_raw)` (`gdn_gate_chunk_f32`, `chunk_reshape.hip`)
//       is bounded in (0,1); `cum_decay_exp = exp(g_cum)` is bounded in
//       (0,1] (`g_cum` is a cumsum of non-positive log-decays — see
//       `gdn_chunkwise.hip`'s own module doc). Both are safe to fold into
//       the f16 rounding of `VB`/`KB` as a per-contraction-step pre-scale,
//       the same case `gdn_chunkwise_wmma.hip` used for `output`'s
//       `cum_decay_exp` and `state`'s `state_decay`.
//   (3) `K_cd` itself is *not* a global-memory operand — it's produced by
//       this kernel's own first WMMA reduction and only exists in
//       accumulator registers. Making it a fresh WMMA operand for the
//       `K_cd @ state` matmul needs writing those f32 accumulator values out
//       to LDS as f16 fragments (negated, so a second accumulation phase
//       computes `acc_v - K_cd@state` via a plain MAC instead of a separate
//       subtraction pass) and `__syncthreads()` before consuming them — the
//       "accumulator -> LDS(f16) -> fragment" pattern flagged as novel by
//       both the `gdn-wmma`/`gdn-wmma-lds` rounds' own uv_vnew write-ups.
//       `K_cd`'s columns beyond `head_k_dim` are structurally zero already
//       (its `KB` operand is zero-padded past `head_k_dim` at staging time),
//       so writing the full 128-wide accumulator range with no extra
//       bounds check reproduces the same zero-padding this header's other
//       staging helpers already rely on — no separate LDS-clear pass needed.
//
// ── Design ──────────────────────────────────────────────────────────────
//
// One workgroup per `head`, `block = (32, WARPS_M*WARPS_N, 1)` = `(32,16,1)`,
// three phases feeding two accumulators (`acc_v` = the final `v_new`,
// `acc_kcd` = the mid-kernel intermediate):
//   phase 1+2 (same k0-loop over `tile_len`, `Tinv` staged once, shared by
//              both accumulators — mirrors `ut_build`'s shared-`k_tile`
//              pattern): `acc_v += Tinv @ VB`, `acc_kcd += Tinv @ KB`.
//   phase 3 (k0-loop over `head_k_dim`): write this slice of `-acc_kcd` into
//              LDS (reusing phase 1+2's `tinv_tile` slot), stage `state`'s
//              matching slice (reusing the `vb_tile` slot), then
//              `acc_v += (-K_cd) @ state` — the same accumulator as phase 1,
//              so the running total is exactly `Tinv@VB - K_cd@state`.
//
// **LDS budget** (checked before coding, per this round's brief): phase 1+2
// stages 3 operands (`tinv_tile`/`vb_tile`/`kb_tile`, `gdn_chunkwise_wmma_lds_
// common.h`'s `LDS_FREE(128)*K_SLICE(64)*sizeof(f16)` = 16KB each) = 48KB
// peak, identical to `ut_build`'s already-proven budget. Phase 3 reuses the
// `tinv_tile`/`vb_tile` slots as `kcd_tile`/`state_tile` (the intermediate
// plus the state tile) — 32KB of the same 48KB allocation, `kb_tile`'s slot
// just unused by then. Total declared shared memory: `3 * LDS_TILE_BYTES` =
// 48KB, comfortably under gfx1101's 64KB/workgroup cap with 16KB headroom.
//
// Dispatched only when `head_k_dim`/`head_v_dim` are both multiples of 16
// *and* <=128 (`LDS_FREE`) — same eligibility bound as `output_lds`, since
// this kernel shares both of `output`'s free axes (`head_k_dim` for the
// `K_cd`/state phase, `head_v_dim` for the final `v_new` output). See
// `gdn_chunkwise.rs`'s dispatch gate.
#include "gdn_chunkwise_wmma_lds_common.h"

extern "C" __global__ void gdn_chunkwise_uv_vnew_wmma_lds_f32(
    const float* tinv, const float* conv_out, const float* beta, const float* k_beta,
    const float* cum_decay_exp, const float* state, float* v_new, unsigned num_v_heads,
    unsigned num_k_heads, unsigned head_k_dim, unsigned head_v_dim, unsigned conv_dim,
    unsigned key_dim, unsigned tile_len) {
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

    const float* head_tinv = tinv + (size_t)head * tile_len * tile_len;
    // `V`'s per-token row lives inside `conv_out`'s wider per-token layout
    // (`[tile_len, conv_dim]`), offset past the Q/K portion — see
    // `gdn_chunkwise.hip`'s `gdn_chunkwise_uv_vnew_f32` for the same offset.
    const float* head_v = conv_out + 2 * (size_t)key_dim + (size_t)head * head_v_dim;
    // `beta` is `[tile_len, num_v_heads]` (token-major — see
    // `gdn_chunkwise_wmma_lds_common.h`'s `stage_transposed_slice` doc for
    // why this needs a strided scale lookup, unlike every other bounded
    // factor this header's callers fold).
    const float* head_beta_col = beta + head;
    const float* head_kb_in = k_beta + (size_t)head * tile_len * head_k_dim;
    const float* head_cde = cum_decay_exp + (size_t)head * tile_len;
    const float* head_state = state + (size_t)head * head_k_dim * head_v_dim;

    extern __shared__ _Float16 smem[];
    _Float16* tinv_tile = smem;
    _Float16* vb_tile = smem + LDS_FREE * K_SLICE;
    _Float16* kb_tile = smem + 2 * LDS_FREE * K_SLICE;

    float8 acc_v[SUBROWS_PER_WARP][SUBCOLS_PER_WARP];
    float8 acc_kcd[SUBROWS_PER_WARP][SUBCOLS_PER_WARP];
#pragma unroll
    for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            acc_v[r][c] = {0, 0, 0, 0, 0, 0, 0, 0};
            acc_kcd[r][c] = {0, 0, 0, 0, 0, 0, 0, 0};
        }
    }

    // Phase 1+2: acc_v = Tinv @ (V*beta), acc_kcd = Tinv @ (k_beta*cum_decay_exp).
    for (unsigned k0 = 0; k0 < tile_len; k0 += K_SLICE) {
        stage_contig_slice(tinv_tile, head_tinv, tile_len, tile_len, tile_len, k0, tid, nthreads, nullptr);
        stage_transposed_slice(
            vb_tile, head_v, conv_dim, head_v_dim, tile_len, k0, tid, nthreads, head_beta_col,
            num_v_heads);
        stage_transposed_slice(
            kb_tile, head_kb_in, head_k_dim, head_k_dim, tile_len, k0, tid, nthreads, head_cde);
        __syncthreads();
        unsigned lim = min(K_SLICE, tile_len - k0);
        wmma_from_lds(tinv_tile, vb_tile, lim, wm, wn, lane, acc_v);
        wmma_from_lds(tinv_tile, kb_tile, lim, wm, wn, lane, acc_kcd);
        __syncthreads();
    }

    // Phase 3: acc_v -= K_cd @ state, one head_k_dim-slice at a time. `kcd_tile`/
    // `state_tile` reuse phase 1+2's `tinv_tile`/`vb_tile` LDS (no longer
    // needed — both matmuls above already fully reduced over `tile_len`).
    _Float16* kcd_tile = tinv_tile;
    _Float16* state_tile = vb_tile;
    for (unsigned k0 = 0; k0 < head_k_dim; k0 += K_SLICE) {
#pragma unroll
        for (unsigned c = 0; c < SUBCOLS_PER_WARP; ++c) {
            unsigned subcol = wn * SUBCOLS_PER_WARP + c;
            unsigned dk = subcol * 16 + lane % 16;
            if (dk < k0 || dk >= k0 + K_SLICE) {
                continue;
            }
            unsigned local_k = dk - k0;
#pragma unroll
            for (unsigned r = 0; r < SUBROWS_PER_WARP; ++r) {
                unsigned subrow = wm * SUBROWS_PER_WARP + r;
#pragma unroll
                for (unsigned ele = 0; ele < 8; ++ele) {
                    unsigned t = subrow * 16 + 2 * ele + lane / 16;
                    // Every (t, local_k) in this tile is written by exactly
                    // one thread across the whole block, so this doubles as
                    // the LDS zero-pad for `t >= tile_len` — no separate
                    // clear pass needed (see the module doc's point (3)).
                    float val = (t < tile_len) ? -acc_kcd[r][c][ele] : 0.0f;
                    kcd_tile[(size_t)t * K_SLICE + local_k] = (_Float16)val;
                }
            }
        }
        stage_transposed_slice(
            state_tile, head_state, head_v_dim, head_v_dim, head_k_dim, k0, tid, nthreads, nullptr);
        __syncthreads();
        unsigned lim = min(K_SLICE, head_k_dim - k0);
        wmma_from_lds(kcd_tile, state_tile, lim, wm, wn, lane, acc_v);
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
                    v_new[((size_t)head * tile_len + t) * head_v_dim + d] = acc_v[r][c][ele];
                }
            }
        }
    }
    (void)num_k_heads;
}

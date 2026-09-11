// Gated Delta Net recurrence, one decode step, fused state-update + readout.
// One block per value head (`num_heads` == num_v_heads); `head_v_dim`
// threads do the real work (`head_k_dim` threads additionally help load Q/K
// into shared memory), so launch with block = max(head_k_dim, head_v_dim).
// State is `[head_k_dim, head_v_dim]` row-major per value head, persistent
// across calls (mutated in place).
//
// `num_k_heads` may be less than `num_heads` (grouped Q/K, as used by
// Qwen3.5/Qwen3-Next sizes above 2B): value head `h`'s query/key come from
// key head `h % num_k_heads` — a *tiled* broadcast (`[K0,K1,...,K0,K1,...]`),
// NOT HF transformers' raw per-checkpoint `repeat_interleave` layout
// (`[K0,K0,...,K1,K1,...]`, i.e. `h / group`). llama.cpp's GGUF converter
// permutes V (and the matching Z/beta/alpha/A_log/dt_bias/conv1d-V rows,
// plus out_proj's input columns) from HF's grouped order into this tiled
// order at conversion time specifically so a cheap modulo broadcast works
// at inference (see llama.cpp's `_LinearAttentionVReorderBase._reorder_v_heads`,
// convert_hf_to_gguf's `conversion/qwen.py`) — since every GDN tensor this
// kernel reads comes from that already-reordered GGUF, `%` here is what
// actually lines Q/K back up with them; `/` silently uses the wrong key
// head whenever `num_k_heads < num_heads` and produces well-formed-looking
// but nonsensical output (finite, deterministic, just wrong). `num_heads`
// must be a multiple of `num_k_heads` (checked by `Qwen35Config`, not
// here). When `num_k_heads == num_heads` both formulas coincide.
//
// Per head: decay the state by `exp(g)`, read `kv_mem = decayed_S^T . k`,
// form the delta-rule residual `delta = beta * (v - kv_mem)`, write
// `S = decayed_S + outer(k, delta)`, then read out `y = S^T . q`. Matches
// Crane's portable `gated_delta_rule_recurrence` (crane-core/src/ops/gdn/backend.rs)
// for a single timestep with an in-place state instead of a `[B,H,S,*]` batch.
#include <hip/hip_runtime.h>

extern "C" __global__ void gdn_recurrence_decode_f32(
    float* state, const float* q, const float* k, const float* v, const float* beta,
    const float* g, float* y, unsigned num_heads, unsigned num_k_heads, unsigned head_k_dim,
    unsigned head_v_dim) {
    unsigned head = blockIdx.x;
    if (head >= num_heads) {
        return;
    }
    unsigned tid = threadIdx.x;
    unsigned key_head = head % num_k_heads;

    extern __shared__ float shared[];
    float* q_s = shared;
    float* k_s = shared + head_k_dim;
    float* v_s = shared + 2 * head_k_dim;
    float* delta_s = shared + 2 * head_k_dim + head_v_dim;

    const float* q_h = q + (size_t)key_head * head_k_dim;
    const float* k_h = k + (size_t)key_head * head_k_dim;
    const float* v_h = v + (size_t)head * head_v_dim;
    float* y_h = y + (size_t)head * head_v_dim;
    float* s = state + (size_t)head * head_k_dim * head_v_dim;

    if (tid < head_k_dim) {
        q_s[tid] = q_h[tid];
        k_s[tid] = k_h[tid];
    }
    if (tid < head_v_dim) {
        v_s[tid] = v_h[tid];
    }
    __syncthreads();

    float decay = expf(g[head]);
    float beta_h = beta[head];

    if (tid < head_v_dim) {
        float kv_mem = 0.0f;
        for (unsigned kk = 0; kk < head_k_dim; ++kk) {
            unsigned idx = kk * head_v_dim + tid;
            float decayed = s[idx] * decay;
            s[idx] = decayed;
            kv_mem += decayed * k_s[kk];
        }
        delta_s[tid] = beta_h * (v_s[tid] - kv_mem);
    }
    __syncthreads();

    if (tid < head_v_dim) {
        float y_acc = 0.0f;
        for (unsigned kk = 0; kk < head_k_dim; ++kk) {
            unsigned idx = kk * head_v_dim + tid;
            float updated = s[idx] + k_s[kk] * delta_s[tid];
            s[idx] = updated;
            y_acc += updated * q_s[kk];
        }
        y_h[tid] = y_acc;
    }
}

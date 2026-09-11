// Gated Delta Net recurrence, one decode step, fused state-update + readout.
// One block per head; `head_v_dim` threads do the real work (`head_k_dim`
// threads additionally help load Q/K into shared memory), so launch with
// block = max(head_k_dim, head_v_dim). State is `[head_k_dim, head_v_dim]`
// row-major, persistent across calls (mutated in place).
//
// Per head: decay the state by `exp(g)`, read `kv_mem = decayed_S^T . k`,
// form the delta-rule residual `delta = beta * (v - kv_mem)`, write
// `S = decayed_S + outer(k, delta)`, then read out `y = S^T . q`. Matches
// Crane's portable `gated_delta_rule_recurrence` (crane-core/src/ops/gdn/backend.rs)
// for a single timestep with an in-place state instead of a `[B,H,S,*]` batch.
#include <hip/hip_runtime.h>

extern "C" __global__ void gdn_recurrence_decode_f32(
    float* state, const float* q, const float* k, const float* v, const float* beta,
    const float* g, float* y, unsigned num_heads, unsigned head_k_dim, unsigned head_v_dim) {
    unsigned head = blockIdx.x;
    if (head >= num_heads) {
        return;
    }
    unsigned tid = threadIdx.x;

    extern __shared__ float shared[];
    float* q_s = shared;
    float* k_s = shared + head_k_dim;
    float* v_s = shared + 2 * head_k_dim;
    float* delta_s = shared + 2 * head_k_dim + head_v_dim;

    const float* q_h = q + (size_t)head * head_k_dim;
    const float* k_h = k + (size_t)head * head_k_dim;
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

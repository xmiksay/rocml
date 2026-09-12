// Gated Delta Net prefill-chunk kernels: batched causal conv1d and the
// gated-delta-rule recurrence, each processing a whole T-token chunk in one
// launch instead of T serial decode-style launches (issue #6 part 3, v1
// pragmatic — the true chunkwise-matmul GDN formulation is a later
// optimization, not attempted here). Both reproduce their `_decode_f32`
// sibling's per-step math exactly, just looped over the chunk inside the
// kernel with the recurrence *state* carried in registers/global memory
// across iterations the same way T separate launches would carry it via
// `conv_state`/`state` — so this changes launch count, not arithmetic.
#include <hip/hip_runtime.h>

// Bounds the local sliding-window register array below; GDN's conv kernel
// size is 4 in every model this codebase loads (window 4, per issue #6), so
// 8 leaves headroom without costing registers anywhere that matters.
#define MAX_CONV_HIST 8

// Batched causal depthwise conv1d + SiLU over a `chunk_len`-token chunk.
// `x` is `[chunk_len, channels]`; `conv_state` (`[channels, kernel_size-1]`,
// oldest first) is the same carried-state buffer `causal_conv1d_decode_f32`
// uses, read once into a per-thread register window and written back once at
// the end — equivalent to `chunk_len` sequential decode calls' worth of
// shift-in-place updates, just without the global-memory round trip between
// tokens. `out` is `[chunk_len, channels]`.
extern "C" __global__ void causal_conv1d_chunk_f32(
    const float* x, float* conv_state, const float* weight, float* out, unsigned channels,
    unsigned kernel_size, unsigned chunk_len) {
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    unsigned hist_len = kernel_size - 1;
    const float* w = weight + (size_t)c * kernel_size;
    float* state = conv_state + (size_t)c * hist_len;

    float hist[MAX_CONV_HIST];
    for (unsigned j = 0; j < hist_len; ++j) {
        hist[j] = state[j];
    }

    for (unsigned t = 0; t < chunk_len; ++t) {
        float xt = x[(size_t)t * channels + c];
        float acc = 0.0f;
        for (unsigned j = 0; j < hist_len; ++j) {
            acc += hist[j] * w[j];
        }
        acc += xt * w[hist_len];

        float sig = 1.0f / (1.0f + expf(-acc));
        out[(size_t)t * channels + c] = acc * sig;

        for (unsigned j = 0; j + 1 < hist_len; ++j) {
            hist[j] = hist[j + 1];
        }
        if (hist_len > 0) {
            hist[hist_len - 1] = xt;
        }
    }

    for (unsigned j = 0; j < hist_len; ++j) {
        state[j] = hist[j];
    }
}

// Batched Gated Delta Net recurrence over a `chunk_len`-token chunk: same
// per-head state update/readout as `gdn_recurrence_decode_f32` (see that
// kernel's doc comment for the delta-rule math and the `num_k_heads`
// tiled-broadcast contract), but additionally folds in the per-head L2 norm
// `forward::gdn`'s decode path applies to Q/K *before* calling the decode
// kernel (see that module's doc comment for the `rmsnorm`-as-L2-norm
// derivation) — reproduced here directly (`sum-of-squares` then
// `rsqrtf(.. + l2_eps)`) rather than requiring a separate batched-rmsnorm
// pass over a non-contiguous per-head view, since `conv_out`'s natural
// per-token `[Q|K|V]` layout doesn't give Q (or K) a contiguous `[chunk_len *
// num_k_heads, head_k_dim]` view a plain `rmsnorm_f32` call could use
// directly. `q_scale = 1/sqrt(head_k_dim)` is the recurrence's own query
// scale (folded into Q's norm here, same as the decode path's `l2_alpha_q`).
//
// `conv_out` is `[chunk_len, conv_dim]`, `conv_dim`-strided rows of
// `[Q(key_dim) | K(key_dim) | V(value_dim)]` (`key_dim = num_k_heads *
// head_k_dim`, `value_dim = num_heads * head_v_dim`) — `gdn::gdn_layer_step`'s
// per-token `gdn_conv_out` layout, batched over tokens with no reshape.
// `beta`/`g` are `[chunk_len, num_heads]`; `y` is `[chunk_len, num_heads,
// head_v_dim]`. Launch contract identical to `gdn_recurrence_decode_f32`:
// one block per value head, block size `max(head_k_dim, head_v_dim)`,
// `2*(head_k_dim+head_v_dim)*sizeof(f32)` bytes of dynamic shared memory.
extern "C" __global__ void gdn_recurrence_chunk_f32(
    float* state, const float* conv_out, const float* beta, const float* g, float* y,
    unsigned num_heads, unsigned num_k_heads, unsigned head_k_dim, unsigned head_v_dim,
    unsigned conv_dim, unsigned key_dim, unsigned chunk_len, float l2_eps) {
    unsigned head = blockIdx.x;
    if (head >= num_heads) {
        return;
    }
    unsigned tid = threadIdx.x;
    unsigned key_head = head % num_k_heads;
    float q_scale = rsqrtf((float)head_k_dim);

    extern __shared__ float shared[];
    float* q_s = shared;
    float* k_s = shared + head_k_dim;
    float* v_s = shared + 2 * head_k_dim;
    float* delta_s = shared + 2 * head_k_dim + head_v_dim;
    __shared__ float inv_q, inv_k;

    float* s = state + (size_t)head * head_k_dim * head_v_dim;

    for (unsigned t = 0; t < chunk_len; ++t) {
        const float* row = conv_out + (size_t)t * conv_dim;
        const float* q_h = row + (size_t)key_head * head_k_dim;
        const float* k_h = row + key_dim + (size_t)key_head * head_k_dim;
        const float* v_h = row + 2 * key_dim + (size_t)head * head_v_dim;
        float* y_h = y + ((size_t)t * num_heads + head) * head_v_dim;
        float beta_h = beta[(size_t)t * num_heads + head];
        float decay = expf(g[(size_t)t * num_heads + head]);

        if (tid < head_k_dim) {
            q_s[tid] = q_h[tid];
            k_s[tid] = k_h[tid];
        }
        if (tid < head_v_dim) {
            v_s[tid] = v_h[tid];
        }
        __syncthreads();

        // Serial per-block reduction: correctness-first (see the module
        // doc's "v1 pragmatic" note) — `num_heads` blocks already give this
        // kernel plenty of cross-block parallelism, so a per-block O(head_k_dim)
        // serial sum-of-squares by one lane is not the bottleneck here.
        if (tid == 0) {
            float sq = 0.0f, sk = 0.0f;
            for (unsigned j = 0; j < head_k_dim; ++j) {
                sq += q_s[j] * q_s[j];
                sk += k_s[j] * k_s[j];
            }
            inv_q = rsqrtf(sq + l2_eps) * q_scale;
            inv_k = rsqrtf(sk + l2_eps);
        }
        __syncthreads();
        if (tid < head_k_dim) {
            q_s[tid] *= inv_q;
            k_s[tid] *= inv_k;
        }
        __syncthreads();

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
        // Reached uniformly every iteration (chunk_len is block-uniform), so
        // this barrier is safe before the next iteration's load overwrites
        // q_s/k_s/v_s/delta_s.
        __syncthreads();
    }
}

// Mixture-of-experts routing + accumulate building blocks (qwen35moe, M1).
//
// `moe_route_topk_f32`: one block per row (token). Computes a full softmax
// over `expert_count` logits, then does `top_k` sequential block-wide
// argmax-reductions against the softmax'd probabilities (marking each
// selected slot as consumed with a `-1.0f` sentinel — always strictly below
// any real probability, so it can never be re-selected), then renormalizes
// the `top_k` selected weights so they sum to 1 (llama.cpp's
// `build_moe_ffn` with `norm_w=true`: sum clamped to `min_sum` first, so a
// pathological near-zero sum can't blow the weights up). `expert_count` must
// be <= MOE_MAX_EXPERTS (this model's 256 comfortably fits); block size is
// fixed at MOE_BLOCK and grid-strides over `expert_count`.
//
// `moe_shared_gate_write_f32`/`moe_weighted_accum_f32`: small elementwise
// ops (mirroring `elementwise.hip`'s `add_inplace_f32`/`sigmoid_mul_f32`
// idiom) that let the host-side per-expert loop build up the FFN output
// accumulator without a `memset` — the shared expert's contribution is
// always the *first* write (sigmoid-gated), every routed expert after it
// is a weighted accumulate whose weight lives in device memory (no host
// round-trip needed for the actual float value, only the expert *index*).
#include <hip/hip_runtime.h>

#define MOE_BLOCK 256
#define MOE_MAX_EXPERTS 1024

extern "C" __global__ void moe_route_topk_f32(
    const float* logits,
    int* out_idx,
    float* out_weight,
    unsigned rows,
    unsigned expert_count,
    unsigned top_k,
    float min_sum) {
    unsigned row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    __shared__ float probs[MOE_MAX_EXPERTS];
    __shared__ float red_val[MOE_BLOCK];
    __shared__ int red_idx[MOE_BLOCK];
    __shared__ float row_max;
    __shared__ float row_sum;

    const float* lrow = logits + (size_t)row * expert_count;

    float local_max = -INFINITY;
    for (unsigned i = threadIdx.x; i < expert_count; i += blockDim.x) {
        local_max = fmaxf(local_max, lrow[i]);
    }
    red_val[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red_val[threadIdx.x] = fmaxf(red_val[threadIdx.x], red_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_max = red_val[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned i = threadIdx.x; i < expert_count; i += blockDim.x) {
        float p = expf(lrow[i] - row_max);
        probs[i] = p;
        local_sum += p;
    }
    red_val[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red_val[threadIdx.x] += red_val[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_sum = red_val[0];
    }
    __syncthreads();

    for (unsigned i = threadIdx.x; i < expert_count; i += blockDim.x) {
        probs[i] /= row_sum;
    }
    __syncthreads();

    for (unsigned k = 0; k < top_k; ++k) {
        float local_best_val = -1.0f;
        int local_best_idx = -1;
        for (unsigned i = threadIdx.x; i < expert_count; i += blockDim.x) {
            float v = probs[i];
            if (v > local_best_val) {
                local_best_val = v;
                local_best_idx = (int)i;
            }
        }
        red_val[threadIdx.x] = local_best_val;
        red_idx[threadIdx.x] = local_best_idx;
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s && red_val[threadIdx.x + s] > red_val[threadIdx.x]) {
                red_val[threadIdx.x] = red_val[threadIdx.x + s];
                red_idx[threadIdx.x] = red_idx[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            int best_idx = red_idx[0];
            out_idx[(size_t)row * top_k + k] = best_idx;
            out_weight[(size_t)row * top_k + k] = red_val[0];
            if (best_idx >= 0) {
                probs[best_idx] = -1.0f;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned k = 0; k < top_k; ++k) {
            s += out_weight[(size_t)row * top_k + k];
        }
        if (s < min_sum) {
            s = min_sum;
        }
        for (unsigned k = 0; k < top_k; ++k) {
            out_weight[(size_t)row * top_k + k] /= s;
        }
    }
}

// out[i] = y[i] * sigmoid(*gate_logit) — the shared expert's contribution,
// always the accumulator's first write (never a `+=`).
extern "C" __global__ void moe_shared_gate_write_f32(
    const float* y, const float* gate_logit, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float w = 1.0f / (1.0f + expf(-gate_logit[0]));
        out[i] = y[i] * w;
    }
}

// out[i] += y[i] * (*weight) — one routed expert's weighted accumulate.
extern "C" __global__ void moe_weighted_accum_f32(
    const float* y, const float* weight, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] += y[i] * weight[0];
    }
}

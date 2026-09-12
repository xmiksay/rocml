// Gated Delta Net prefill-chunk causal-conv1d kernel: batched depthwise
// conv1d + SiLU over a whole T-token chunk in one launch instead of T serial
// decode-style launches, reproducing `causal_conv1d_decode_f32`'s per-step
// math exactly (state carried in registers/global memory across iterations
// the same way T separate launches would carry it via `conv_state` — so
// this changes launch count, not arithmetic). The recurrence's chunk kernel
// used to live here too (`gdn_recurrence_chunk_f32`, a v1 pragmatic
// token-serial-inside-chunk loop); it has been replaced by the true
// chunkwise (blocked delta-rule) formulation, `kernels/gdn_chunkwise.hip`
// (issue #6's chunkwise rewrite) — see that file for the recurrence.
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

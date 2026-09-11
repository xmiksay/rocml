// Gated Delta Net causal depthwise Conv1D, one decode step. Each of
// `channels` gets its own `kernel_size`-tap filter (`weight`, row-major
// `[channels, kernel_size]`, oldest tap first) applied over its own
// `kernel_size - 1` history slots (`conv_state`, row-major
// `[channels, kernel_size - 1]`, oldest first) plus the new input sample,
// then SiLU. `conv_state` is shifted in place so the next call sees this
// step's input as history. No block-size constraint (plain grid-stride,
// no reduction).
#include <hip/hip_runtime.h>

extern "C" __global__ void causal_conv1d_decode_f32(
    const float* x_new, float* conv_state, const float* weight, float* out, unsigned channels,
    unsigned kernel_size) {
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    unsigned hist_len = kernel_size - 1;
    float* hist = conv_state + (size_t)c * hist_len;
    const float* w = weight + (size_t)c * kernel_size;

    float acc = 0.0f;
    for (unsigned j = 0; j < hist_len; ++j) {
        acc += hist[j] * w[j];
    }
    acc += x_new[c] * w[hist_len];

    float sig = 1.0f / (1.0f + expf(-acc));
    out[c] = acc * sig;

    for (unsigned j = 0; j + 1 < hist_len; ++j) {
        hist[j] = hist[j + 1];
    }
    if (hist_len > 0) {
        hist[hist_len - 1] = x_new[c];
    }
}

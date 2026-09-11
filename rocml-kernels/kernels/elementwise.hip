// Small elementwise ops shared across the dense-transformer pipeline:
// in-place residual add, and f16<->f32 casts for moving between compute
// (f32) and storage (f16) precision.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void add_inplace_f32(float* acc, const float* x, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        acc[i] += x[i];
    }
}

extern "C" __global__ void cast_f16_f32(const __half* in, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __half2float(in[i]);
    }
}

extern "C" __global__ void cast_f32_f16(const float* in, __half* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2half(in[i]);
    }
}

// qwen35 full-attention output gate: out = x * sigmoid(gate).
extern "C" __global__ void sigmoid_mul_f32(
    const float* x, const float* gate, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = x[i] / (1.0f + expf(-gate[i]));
    }
}

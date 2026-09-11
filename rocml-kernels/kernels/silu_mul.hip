// SwiGLU-style gate: out = silu(gate) * up, silu(v) = v * sigmoid(v).
#include <hip/hip_runtime.h>

extern "C" __global__ void silu_mul_f32(
    const float* gate, const float* up, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

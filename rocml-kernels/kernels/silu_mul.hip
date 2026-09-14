// Gated-FFN elementwise activations: out = act(gate) * up. Both kernels
// share this file (and its one compiled code object) since they're the same
// tiny shape, differing only in the activation function — see
// `Kernels::silu_mul`/`Kernels::gelu_mul`'s doc comments for which
// architectures select which (issue #16's config-driven-activation seam:
// `ModelConfig::activation` picks the kernel, never a hardcoded choice
// baked into `ffn::ffn_step`/`ffn_chunk::ffn_chunk_step`).
#include <hip/hip_runtime.h>

// SwiGLU-style gate: silu(v) = v * sigmoid(v). Qwen3/Qwen3.5's FFN
// activation.
extern "C" __global__ void silu_mul_f32(
    const float* gate, const float* up, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

// GeGLU-style gate using the tanh approximation of GELU (HF's
// "gelu_pytorch_tanh", the activation gemma-2/gemma-3's FFN declares) —
// not currently reachable from any shipped architecture loader (no
// registry entry sets `ModelConfig::activation` to `Gelu` yet, see
// issue #16's audit), but the kernel/dispatch seam is landed and unit-
// tested ahead of an actual gemma-family checkpoint.
extern "C" __global__ void gelu_mul_f32(
    const float* gate, const float* up, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = gate[i];
        const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
        const float k1 = 0.044715f;
        float inner = k0 * (g + k1 * g * g * g);
        float gelu = 0.5f * g * (1.0f + tanhf(inner));
        out[i] = gelu * up[i];
    }
}

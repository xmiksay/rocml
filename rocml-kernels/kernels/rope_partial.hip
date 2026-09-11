// Partial-rotary NEOX rope: like rope_neox_f32, but only the first `rot_dim`
// of each `head_dim`-wide head is rotated (pairing i with i + rot_dim/2,
// inv_freq = theta_base^(-2i/rot_dim)); the remaining `head_dim - rot_dim`
// components are left untouched. Used by qwen35's full-attention layers,
// where `rot_dim` (`rope.dimension_count`) is a quarter of `head_dim`.
// `rot_dim` must be even. No block-size or shape constraint.
#include <hip/hip_runtime.h>

extern "C" __global__ void rope_neox_partial_f32(
    float* x, unsigned tokens, unsigned heads, unsigned head_dim, unsigned rot_dim,
    unsigned pos_base, float theta_base) {
    unsigned half_rot = rot_dim / 2;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = tokens * heads * half_rot;
    if (idx >= total) {
        return;
    }

    unsigned i = idx % half_rot;
    unsigned tmp = idx / half_rot;
    unsigned h = tmp % heads;
    unsigned t = tmp / heads;

    unsigned pos = pos_base + t;
    float inv_freq = powf(theta_base, -2.0f * (float)i / (float)rot_dim);
    float angle = (float)pos * inv_freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    size_t base = ((size_t)t * heads + h) * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + i + half_rot];
    x[base + i] = x0 * cos_a - x1 * sin_a;
    x[base + i + half_rot] = x0 * sin_a + x1 * cos_a;
}

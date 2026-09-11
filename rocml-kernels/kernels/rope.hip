// NEOX-style rotary position embedding, applied in place over a
// tokens x heads x head_dim contiguous buffer. Token t uses absolute
// position pos_base + t. NEOX pairing rotates element i against i +
// head_dim/2 (not the interleaved GPT-J adjacent-pair convention), with
// inv_freq = theta_base^(-2i/head_dim). head_dim must be even.
#include <hip/hip_runtime.h>

extern "C" __global__ void rope_neox_f32(
    float* x, unsigned tokens, unsigned heads, unsigned head_dim, unsigned pos_base,
    float theta_base) {
    unsigned half_dim = head_dim / 2;
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = tokens * heads * half_dim;
    if (idx >= total) {
        return;
    }

    unsigned i = idx % half_dim;
    unsigned tmp = idx / half_dim;
    unsigned h = tmp % heads;
    unsigned t = tmp / heads;

    unsigned pos = pos_base + t;
    float inv_freq = powf(theta_base, -2.0f * (float)i / (float)head_dim);
    float angle = (float)pos * inv_freq;
    float cos_a = cosf(angle);
    float sin_a = sinf(angle);

    size_t base = ((size_t)t * heads + h) * head_dim;
    float x0 = x[base + i];
    float x1 = x[base + i + half_dim];
    x[base + i] = x0 * cos_a - x1 * sin_a;
    x[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
}

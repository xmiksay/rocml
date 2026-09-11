// Fused dequant-GEMV for Q5_K: like Q4_K but with an extra high-bit plane
// (`qh`) giving 5-bit codes. W's rows are raw GGUF Q5_K blocks (176 bytes),
// never materialized to f32 in VRAM. One workgroup per output row; each lane
// owns a strided subset of the row's 256-element super-blocks. Must match
// rocml-core's `quant::q5_k::dequantize` exactly, including the `u1`/`u2`
// high-bit mask rotation across sub-block pairs.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

extern "C" __global__ void gemv_q5_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 176;
    unsigned blocks_per_row = n / qk;
    const unsigned char* row_ptr =
        (const unsigned char*)w + (size_t)row * blocks_per_row * block_bytes;

    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned blk = threadIdx.x; blk < blocks_per_row; blk += blockDim.x) {
        const unsigned char* bptr = row_ptr + (size_t)blk * block_bytes;
        __half d_h, dmin_h;
        std::memcpy(&d_h, bptr, sizeof(__half));
        std::memcpy(&dmin_h, bptr + 2, sizeof(__half));
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);
        const unsigned char* scales = bptr + 4;
        const unsigned char* qh = bptr + 16;
        const unsigned char* qs = bptr + 48;
        const float* xs = x + (size_t)blk * qk;

        // Mirrors dequantize_row_q5_K: qh stays fixed across the 4 sub-block
        // pairs while the tested bit (u1/u2) rotates left by 2 each time.
        unsigned is = 0;
        unsigned char u1 = 1, u2 = 2;
        float local = 0.0f;
        for (unsigned j = 0; j < qk; j += 64) {
            const unsigned char* ql = qs + j / 2;
            unsigned char sc, mn;
            get_scale_min_k4(is, scales, &sc, &mn);
            float d1 = d * sc;
            float m1 = dmin * mn;
            get_scale_min_k4(is + 1, scales, &sc, &mn);
            float d2 = d * sc;
            float m2 = dmin * mn;
            const float* x1 = xs + j;
            const float* x2 = xs + j + 32;

#pragma unroll
            for (unsigned l = 0; l < 32; ++l) {
                float hi1 = (qh[l] & u1) ? 16.0f : 0.0f;
                float hi2 = (qh[l] & u2) ? 16.0f : 0.0f;
                local += (d1 * ((float)(ql[l] & 0xF) + hi1) - m1) * x1[l];
                local += (d2 * ((float)(ql[l] >> 4) + hi2) - m2) * x2[l];
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
        sum += local;
    }

    partial[threadIdx.x] = sum;
    __syncthreads();
    // Requires a power-of-two block size so this halves evenly to index 0.
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = partial[0];
    }
}

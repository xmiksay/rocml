// Fused dequant-GEMV for Q4_K: y = W * x where W's rows are raw GGUF Q4_K
// blocks (144 bytes: f16 `d`/`dmin`, a packed 6-bit scale/min table per
// 32-element sub-block, 128 bytes of 4-bit codes), never materialized to f32
// in VRAM. One workgroup per output row; each lane owns a strided subset of
// the row's 256-element super-blocks. Must match rocml-core's
// `quant::q4_k::dequantize` exactly: `d * sc * nibble - dmin * m`, same
// (scale, min) unpack and sub-block traversal order.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

extern "C" __global__ void gemv_q4_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 144;
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
        const unsigned char* qs = bptr + 16;
        const float* xs = x + (size_t)blk * qk;

        // Mirrors dequantize_row_q4_K's 4 sub-block-pairs-of-64 traversal:
        // each iteration decodes two 32-wide (scale, min) pairs and unpacks
        // the low/high nibble halves of the same 32 code bytes.
        unsigned is = 0;
        float local = 0.0f;
        for (unsigned j = 0; j < qk; j += 64) {
            const unsigned char* q = qs + j / 2;
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
                local += (d1 * (float)(q[l] & 0xF) - m1) * x1[l];
                local += (d2 * (float)(q[l] >> 4) - m2) * x2[l];
            }
            is += 2;
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

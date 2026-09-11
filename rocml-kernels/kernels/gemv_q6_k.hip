// Fused dequant-GEMV for Q6_K: y = W * x where W's rows are raw GGUF Q6_K
// blocks (210 bytes: 128 bytes of 4-bit low codes, 64 bytes of 2-bit high
// codes, 16 signed 8-bit sub-block scales, one f16 `d`), never materialized
// to f32 in VRAM. One workgroup per output row; each lane owns a strided
// subset of the row's 256-element super-blocks. Must match rocml-core's
// `quant::q6_k::dequantize` exactly, including its two-half / four-way
// interleaved output ordering (see the half/l loop below) and — despite the
// `BlockQ6K.scales` field being typed `[i8; 16]` — its actual behavior of
// reading the sub-block scale bytes as *unsigned* (the dequant function
// slices the raw `&[u8]` block directly and never reinterprets through that
// struct, so `sc[is] as f32` there is an unsigned cast).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

extern "C" __global__ void gemv_q6_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 210;
    unsigned blocks_per_row = n / qk;
    const unsigned char* row_ptr =
        (const unsigned char*)w + (size_t)row * blocks_per_row * block_bytes;

    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned blk = threadIdx.x; blk < blocks_per_row; blk += blockDim.x) {
        const unsigned char* bptr = row_ptr + (size_t)blk * block_bytes;
        const unsigned char* ql_full = bptr;
        const unsigned char* qh_full = bptr + 128;
        const unsigned char* sc_full = bptr + 192;
        __half d_h;
        std::memcpy(&d_h, bptr + 208, sizeof(__half));
        float d = __half2float(d_h);
        const float* xs = x + (size_t)blk * qk;

        // Mirrors dequantize_row_q6_K: the super-block splits into two
        // 128-element halves, each producing 4 interleaved 32-wide groups
        // (y[l], y[l+32], y[l+64], y[l+96]) per `l` rather than a
        // sequential 0..128 walk — output element order must match this
        // exactly since it's what indexes into `x`/`xs`.
#pragma unroll
        for (unsigned half_idx = 0; half_idx < 2; ++half_idx) {
            const unsigned char* ql = ql_full + 64 * half_idx;
            const unsigned char* qh = qh_full + 32 * half_idx;
            const unsigned char* sc = sc_full + 8 * half_idx;
            const float* xh = xs + 128 * half_idx;

#pragma unroll
            for (unsigned l = 0; l < 32; ++l) {
                unsigned is = l / 16;
                int q1 = (int)((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * (float)sc[is] * (float)q1 * xh[l];
                sum += d * (float)sc[is + 2] * (float)q2 * xh[l + 32];
                sum += d * (float)sc[is + 4] * (float)q3 * xh[l + 64];
                sum += d * (float)sc[is + 6] * (float)q4 * xh[l + 96];
            }
        }
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

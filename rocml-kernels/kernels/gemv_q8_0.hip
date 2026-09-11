// Fused dequant-GEMV for Q8_0: y = W * x where W's rows are raw GGUF Q8_0
// blocks (34 bytes: f16 scale `d` + 32 signed 8-bit codes), never
// materialized to f32 in VRAM. One workgroup per output row; each lane owns
// a strided subset of the row's blocks and folds its 32-wide dot product
// into a block-wide dot-with-x sum, then the standard power-of-two
// shared-mem tree reduction. Must match rocml-core's
// `quant::q8_0::dequantize` exactly: `(qs[i] as i8) as f32 * d`.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

extern "C" __global__ void gemv_q8_0(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const unsigned qk = 32;
    const unsigned block_bytes = 34;
    unsigned blocks_per_row = n / qk;
    const unsigned char* row_ptr =
        (const unsigned char*)w + (size_t)row * blocks_per_row * block_bytes;

    extern __shared__ float partial[];
    float sum = 0.0f;
    for (unsigned blk = threadIdx.x; blk < blocks_per_row; blk += blockDim.x) {
        const unsigned char* bptr = row_ptr + (size_t)blk * block_bytes;
        // d is 2-byte-aligned (block_bytes is even) but not 4-byte-aligned in
        // general, so read it via memcpy rather than a typed pointer cast.
        __half d_h;
        std::memcpy(&d_h, bptr, sizeof(__half));
        float d = __half2float(d_h);
        const signed char* qs = (const signed char*)(bptr + 2);
        const float* xs = x + (size_t)blk * qk;

        // d factors out of the block's dot product (distributivity over the
        // sum), so it's applied once per block instead of per element —
        // same result as rocml-core's per-element `q as f32 * d`, fewer flops.
        float local = 0.0f;
#pragma unroll
        for (unsigned i = 0; i < qk; ++i) {
            local += (float)qs[i] * xs[i];
        }
        sum += d * local;
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

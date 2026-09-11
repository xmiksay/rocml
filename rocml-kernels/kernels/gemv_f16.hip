// Decode-path GEMV: y = W * x, W is row-major m x n f16, x/y are f32. Each
// block reduces one output row; threads stride across columns so consecutive
// threads read consecutive f16 elements (coalesced). Accumulation is f32
// throughout to keep long reductions accurate despite f16 storage.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_f16(
    const __half* w, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    extern __shared__ float partial[];
    const __half* wrow = w + (size_t)row * n;

    float sum = 0.0f;
    for (unsigned col = threadIdx.x; col < n; col += blockDim.x) {
        sum += __half2float(wrow[col]) * x[col];
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

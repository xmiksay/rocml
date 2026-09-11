// Naive row-per-block GEMV: y = mat * x, mat is row-major m x n. Correctness
// first, performance later (this is a milestone-1 smoke kernel, not the
// tuned gfx1101 GEMV). Each block reduces one output row with a
// shared-memory tree reduction; callers must launch with a power-of-two
// block size so the reduction halves evenly down to index 0.
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_f32(
    const float* mat, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    extern __shared__ float partial[];

    float sum = 0.0f;
    for (unsigned col = threadIdx.x; col < n; col += blockDim.x) {
        sum += mat[row * n + col] * x[col];
    }
    partial[threadIdx.x] = sum;
    __syncthreads();

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

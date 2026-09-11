// Prefill-path linear layer: out = x * W^T, where x is rows x n (f32) and W
// is m x n row-major f16 (out is rows x m, f32). Both operands are indexed
// [outer][k] so this is a standard shared-memory tiled matmul with the
// second operand's tile read by row instead of by column — no separate
// transpose step needed. TILE is fixed at 16 x 16; callers must launch with
// exactly that block shape.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define TILE 16

extern "C" __global__ void gemm_xwt_f16(
    const float* x, const __half* w, float* out, unsigned rows, unsigned m, unsigned n) {
    __shared__ float xs[TILE][TILE];
    __shared__ float ws[TILE][TILE];

    unsigned row = blockIdx.y * TILE + threadIdx.y; // index into rows / out rows
    unsigned col = blockIdx.x * TILE + threadIdx.x; // index into m / out cols

    float acc = 0.0f;
    unsigned num_tiles = (n + TILE - 1) / TILE;
    for (unsigned t = 0; t < num_tiles; ++t) {
        unsigned k_x = t * TILE + threadIdx.x;
        unsigned k_w = t * TILE + threadIdx.y;

        xs[threadIdx.y][threadIdx.x] =
            (row < rows && k_x < n) ? x[(size_t)row * n + k_x] : 0.0f;
        ws[threadIdx.y][threadIdx.x] =
            (col < m && k_w < n) ? __half2float(w[(size_t)col * n + k_w]) : 0.0f;
        __syncthreads();

        for (unsigned kk = 0; kk < TILE; ++kk) {
            acc += xs[threadIdx.y][kk] * ws[kk][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < rows && col < m) {
        out[(size_t)row * m + col] = acc;
    }
}

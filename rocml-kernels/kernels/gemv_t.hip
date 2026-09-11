// Transposed GEMV: y = A^T * x, A is row-major rows x n (e.g. a KV-cache
// value plane, one row per cached time step). Unlike gemv_f32/gemv_f16 (one
// block reduces one output row), there's no per-output reduction here — each
// output column accumulates independently over `rows` — so this assigns one
// thread per output column and loops that thread over all rows. Consecutive
// threads (consecutive columns) read consecutive addresses for the same row,
// so the row loop is fully coalesced despite the simple one-thread-per-column
// scheme. No block-size or shape constraint.
#include <hip/hip_runtime.h>

extern "C" __global__ void gemv_t_f32(
    const float* a, const float* x, float* y, unsigned rows, unsigned n) {
    unsigned col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) {
        return;
    }

    float sum = 0.0f;
    for (unsigned row = 0; row < rows; ++row) {
        sum += a[(size_t)row * n + col] * x[row];
    }
    y[col] = sum;
}

// Embedding table lookup with widening cast: out[i] = f32(table[ids[i]]).
// One block per token row; no reduction, so any block size works.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

extern "C" __global__ void embedding_f16_f32(
    const unsigned* ids, const __half* table, float* out, unsigned t, unsigned dim) {
    unsigned row = blockIdx.x;
    if (row >= t) {
        return;
    }

    const __half* src = table + (size_t)ids[row] * dim;
    float* dst = out + (size_t)row * dim;
    for (unsigned d = threadIdx.x; d < dim; d += blockDim.x) {
        dst[d] = __half2float(src[d]);
    }
}

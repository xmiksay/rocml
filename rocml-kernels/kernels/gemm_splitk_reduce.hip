// Second pass of the split-K WMMA GEMM (`kernels/gemm_xwt_wmma_splitk.hip`):
// sums the `num_splits` partial-sum buffers that kernel wrote into
// `[num_splits, rows, m]` down to the real `[rows, m]` output. One thread
// per output element, summing splits in ascending index order — a fixed,
// deterministic summation order for a given `(rows, m, num_splits)` shape,
// independent of which GPU wave happened to finish first (the whole point
// of the two-pass design over an atomicAdd-into-`out` accumulation: atomics
// would make summation order — and therefore the exact bit pattern of the
// f32 result — depend on scheduling, which the exact-greedy parity gates
// can't tolerate even though their logit tolerances easily could).
#include <hip/hip_runtime.h>

extern "C" __global__ void gemm_splitk_reduce_f32(
    const float* partial, float* out, unsigned rows, unsigned m, unsigned num_splits) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = rows * m;
    if (idx >= total) {
        return;
    }
    float sum = 0.0f;
    for (unsigned s = 0; s < num_splits; ++s) {
        sum += partial[(size_t)s * total + idx];
    }
    out[idx] = sum;
}

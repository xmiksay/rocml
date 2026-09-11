// Smoke-test kernel: exercises the module load / launch / copy-back path
// end to end with the simplest possible correct kernel.
#include <hip/hip_runtime.h>

extern "C" __global__ void vec_add_f32(const float* a, const float* b, float* out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}

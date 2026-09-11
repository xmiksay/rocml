// Per-row RMS norm: out = x / rms(x) * weight, rms(x) = sqrt(mean(x^2) +
// eps). One block per row; the sum-of-squares reduction requires a
// power-of-two block size (same convention as gemv_f32/gemv_f16) so it works
// for any n, including n not a multiple of the block size.
#include <hip/hip_runtime.h>

extern "C" __global__ void rmsnorm_f32(
    const float* x, const float* weight, float* out, unsigned rows, unsigned n, float eps) {
    unsigned row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float partial[];
    const float* xin = x + (size_t)row * n;
    float* xout = out + (size_t)row * n;

    float sum_sq = 0.0f;
    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xin[i];
        sum_sq += v * v;
    }
    partial[threadIdx.x] = sum_sq;
    __syncthreads();

    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    __shared__ float inv_rms;
    if (threadIdx.x == 0) {
        float mean_sq = partial[0] / (float)n;
        inv_rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();

    for (unsigned i = threadIdx.x; i < n; i += blockDim.x) {
        xout[i] = xin[i] * inv_rms * weight[i];
    }
}

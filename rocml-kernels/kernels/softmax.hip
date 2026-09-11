// Causal-attention building block: per row, scale then softmax over only
// the first valid_len[row] entries (numerically stable max-subtract), and
// zero the remaining columns. One block per row; both reductions require a
// power-of-two block size (same convention as gemv_f32/rmsnorm_f32).
// valid_len[row] == 0 zeroes the whole row without reading undefined state.
#include <hip/hip_runtime.h>

extern "C" __global__ void softmax_varlen_f32(
    float* x, const unsigned* valid_len, unsigned rows, unsigned cols, float scale) {
    unsigned row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    extern __shared__ float shared[];
    float* xrow = x + (size_t)row * cols;
    unsigned vlen = valid_len[row];
    if (vlen > cols) {
        vlen = cols;
    }

    float local_max = -INFINITY;
    for (unsigned i = threadIdx.x; i < vlen; i += blockDim.x) {
        local_max = fmaxf(local_max, xrow[i] * scale);
    }
    shared[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] = fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    __shared__ float row_max;
    if (threadIdx.x == 0) {
        row_max = shared[0];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (unsigned i = threadIdx.x; i < vlen; i += blockDim.x) {
        local_sum += expf(xrow[i] * scale - row_max);
    }
    shared[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    __shared__ float row_sum;
    if (threadIdx.x == 0) {
        row_sum = shared[0];
    }
    __syncthreads();

    for (unsigned i = threadIdx.x; i < cols; i += blockDim.x) {
        xrow[i] = (i < vlen) ? expf(xrow[i] * scale - row_max) / row_sum : 0.0f;
    }
}

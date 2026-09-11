// Fused dequant-GEMV for Q4_K: y = W * x where W's rows are raw GGUF Q4_K
// blocks (144 bytes: f16 `d`/`dmin`, a packed 6-bit scale/min table per
// 32-element sub-block, 128 bytes of 4-bit codes), never materialized to f32
// in VRAM. Must match rocml-core's `quant::q4_k::dequantize` exactly:
// `d * sc * nibble - dmin * m`, same (scale, min) unpack and sub-block
// traversal order (reduction ORDER may differ — see below).
//
// Launch contract (fixed by the host, `rocml::forward::kernels_quant`, not
// this crate): one workgroup per output row, `REDUCE_BLOCK` (128) threads,
// power-of-two block size, `blockDim.x * sizeof(float)` bytes of dynamic
// shared memory.
//
// Occupancy fix (issue #7): the original design gave one lane per 256-wide
// super-block, so a 4096-wide row (16 super-blocks) left 112 of 128 lanes
// idle for the whole kernel. Every super-block splits cleanly into 8
// independent 32-element chunks (4 sub-block-pairs x {low nibble, high
// nibble} half) — GROUP_SIZE=8 lanes now cooperate on one super-block, one
// chunk each, so a 128-thread block forms 16 groups and stays fully active
// whenever a row has >=16 super-blocks (every shape this crate serves does:
// n=4096 -> 16, n=12288 -> 48). Each lane's 8-chunk decode loop is exactly
// the original math's per-(sub-block,half) inner loop, just distributed
// across lanes instead of run sequentially by one — chunk `c` needs
// `get_scale_min_k4(c, ...)` directly (the original's call sequence over
// its 4 sub-block-pairs x 2 halves is exactly c=0..7 in order). Reduction
// combines lanes within a group via wavefront `__shfl_xor` (free — no LDS
// round trip within RDNA3's 32-wide wavefront, and GROUP_SIZE=8 divides the
// wavefront evenly into 4 independent butterfly groups) and combines the 16
// per-group sums via the original small shared-mem tree, so `y[row]`'s
// terms land in a different order than before: fine per this kernel's
// contract (bit-for-bit dequant math, reduction order may reorder f32
// accumulation).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

extern "C" __global__ void gemv_q4_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 144;
    unsigned blocks_per_row = n / qk;
    const unsigned char* row_ptr =
        (const unsigned char*)w + (size_t)row * blocks_per_row * block_bytes;

    // 8 lanes cooperate on one super-block: chunk `c` (0..7) is sub-block
    // pair `c/2` (byte offset `j = (c/2)*64` into the 128-byte code plane),
    // low nibble if `c` is even, high nibble if odd — see the file comment.
    const unsigned GROUP_SIZE = 8;
    const unsigned lane = threadIdx.x % GROUP_SIZE;
    const unsigned group = threadIdx.x / GROUP_SIZE;
    const unsigned num_groups = blockDim.x / GROUP_SIZE;

    float sum = 0.0f;
    for (unsigned blk = group; blk < blocks_per_row; blk += num_groups) {
        const unsigned char* bptr = row_ptr + (size_t)blk * block_bytes;
        __half d_h, dmin_h;
        std::memcpy(&d_h, bptr, sizeof(__half));
        std::memcpy(&dmin_h, bptr + 2, sizeof(__half));
        float d = __half2float(d_h);
        float dmin = __half2float(dmin_h);
        const unsigned char* scales = bptr + 4;
        const unsigned char* qs = bptr + 16;
        const float* xs = x + (size_t)blk * qk;

        unsigned j = (lane / 2) * 64;
        unsigned half = lane & 1;
        const unsigned char* q = qs + j / 2;
        unsigned char sc, mn;
        get_scale_min_k4(lane, scales, &sc, &mn);
        float dd = d * (float)sc;
        float mm = dmin * (float)mn;
        const float* xh = xs + j + half * 32;

        float local = 0.0f;
#pragma unroll
        for (unsigned l = 0; l < 32; ++l) {
            unsigned char nib = half == 0 ? (q[l] & 0xF) : (q[l] >> 4);
            local += (dd * (float)nib - mm) * xh[l];
        }
        sum += local;
    }

    // Wavefront butterfly reduction within each GROUP_SIZE-lane group —
    // every lane in the group ends up holding the group's total, so any one
    // of them (lane 0) can publish it.
#pragma unroll
    for (unsigned offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, GROUP_SIZE);
    }

    extern __shared__ float partial[];
    if (lane == 0) {
        partial[group] = sum;
    }
    __syncthreads();
    // num_groups is a power of two (blockDim.x and GROUP_SIZE both are).
    for (unsigned stride = num_groups / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        y[row] = partial[0];
    }
}

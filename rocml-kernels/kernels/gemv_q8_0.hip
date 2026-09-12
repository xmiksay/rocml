// Fused dequant-GEMV for Q8_0: y = W * x where W's rows are raw GGUF Q8_0
// blocks (34 bytes: f16 scale `d` + 32 signed 8-bit codes), never
// materialized to f32 in VRAM. Must match rocml-core's `quant::q8_0::dequantize`
// exactly: `(qs[i] as i8) as f32 * d`.
//
// Launch contract (fixed by the host, `rocml::forward::kernels_quant`, not
// this crate): one workgroup per `ROWS_PER_WG` (2) consecutive output rows,
// `REDUCE_BLOCK` (128) threads, power-of-two block size,
// `blockDim.x * ROWS_PER_WG * sizeof(float)` bytes of dynamic shared memory
// — see `gemv_q4_k.hip`'s file comment for the full multi-row rationale
// (identical here: `x`/scale-independent quantities computed once per `blk`
// iteration and reused across `ROWS_PER_WG` independent weight loads).
//
// Occupancy: Q8_0's 32-wide blocks already give one-lane-per-block full
// occupancy at n>=4096 (blocks_per_row=n/32>=128=blockDim.x) — the issue
// #7 pathology (16 of 128 lanes active) is specific to the K-quants' 256-wide
// super-blocks. Still, a smaller `n` (e.g. n=256..2048, blocks_per_row
// 8..64) leaves the same fraction of lanes idle Q4_K/Q5_K/Q6_K had at
// n=4096, so this kernel gets the same fix at a matching granularity:
// GROUP_SIZE=4 lanes split one 32-element block into 4 8-element chunks
// (instead of Q4_K/Q5_K/Q6_K's 8-lane/32-element split — Q8_0's block is 8x
// smaller, so an 8-lane group here would give each lane only 4 elements,
// too little work to amortize the shuffle-reduction cost). A 128-thread
// block forms 32 groups, so occupancy is now full for any n>=1024
// (blocks_per_row>=32) and improved (4x fewer idle lanes) below that.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

extern "C" __global__ void gemv_q8_0(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    const unsigned ROWS_PER_WG = 2;
    unsigned row_base = blockIdx.x * ROWS_PER_WG;
    if (row_base >= m) {
        return;
    }

    const unsigned qk = 32;
    const unsigned block_bytes = 34;
    const unsigned GROUP_SIZE = 4;
    const unsigned chunk = qk / GROUP_SIZE;
    unsigned blocks_per_row = n / qk;
    const unsigned char* w_bytes = (const unsigned char*)w;

    const unsigned lane = threadIdx.x % GROUP_SIZE;
    const unsigned group = threadIdx.x / GROUP_SIZE;
    const unsigned num_groups = blockDim.x / GROUP_SIZE;

    float sum[ROWS_PER_WG];
#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
        sum[r] = 0.0f;
    }

    for (unsigned blk = group; blk < blocks_per_row; blk += num_groups) {
        const float* xs = x + (size_t)blk * qk + lane * chunk;

#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
            unsigned row = min(row_base + r, m - 1);
            const unsigned char* bptr =
                w_bytes + (size_t)row * blocks_per_row * block_bytes + (size_t)blk * block_bytes;
            // d is 2-byte-aligned (block_bytes is even) but not 4-byte-aligned
            // in general, so read it via memcpy rather than a typed pointer
            // cast.
            __half d_h;
            std::memcpy(&d_h, bptr, sizeof(__half));
            float d = __half2float(d_h);
            const signed char* qs = (const signed char*)(bptr + 2) + lane * chunk;

            // d factors out of the block's dot product (distributivity over
            // the sum), so it's applied once per block instead of per
            // element — same result as rocml-core's per-element
            // `q as f32 * d`, fewer flops.
            float local = 0.0f;
#pragma unroll
            for (unsigned i = 0; i < chunk; ++i) {
                local += (float)qs[i] * xs[i];
            }
            sum[r] += d * local;
        }
    }

#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
#pragma unroll
        for (unsigned offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
            sum[r] += __shfl_xor(sum[r], offset, GROUP_SIZE);
        }
    }

    extern __shared__ float partial[];
    if (lane == 0) {
#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
            partial[r * blockDim.x + group] = sum[r];
        }
    }
    __syncthreads();
    for (unsigned stride = num_groups / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
                partial[r * blockDim.x + threadIdx.x] += partial[r * blockDim.x + threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
            if (row_base + r < m) {
                y[row_base + r] = partial[r * blockDim.x];
            }
        }
    }
}

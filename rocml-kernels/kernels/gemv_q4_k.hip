// Fused dequant-GEMV for Q4_K: y = W * x where W's rows are raw GGUF Q4_K
// blocks (144 bytes: f16 `d`/`dmin`, a packed 6-bit scale/min table per
// 32-element sub-block, 128 bytes of 4-bit codes), never materialized to f32
// in VRAM. Must match rocml-core's `quant::q4_k::dequantize` exactly:
// `d * sc * nibble - dmin * m`, same (scale, min) unpack and sub-block
// traversal order (reduction ORDER may differ — see below).
//
// Launch contract (fixed by the host, `rocml::forward::kernels_quant`, not
// this crate): one workgroup per `ROWS_PER_WG` (2) consecutive output rows,
// `REDUCE_BLOCK` (128) threads, power-of-two block size,
// `blockDim.x * ROWS_PER_WG * sizeof(float)` bytes of dynamic shared memory.
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
//
// Multi-row occupancy fix (issue #7 bandwidth follow-up): at the smallest
// served `m` (4096, e.g. qkv/attn-out projections), `blocks_per_row` (16)
// equals `num_groups` (16) exactly — every group runs its loop body exactly
// once, so a workgroup's whole lifetime is one iteration of memory latency
// with nothing else to overlap it with. `ROWS_PER_WG` independent output
// rows per workgroup turns that into `ROWS_PER_WG` independent weight loads
// per lane per iteration (the `x` operand and every scale/index/nibble-mask
// computation below is identical across rows, so it's computed once and
// reused) — same total launch count reduction as increasing grid-stride,
// but by giving the compiler/scheduler more independent outstanding loads
// per wave to hide global memory latency with instead of relying on having
// enough *other* resident workgroups. Measured via `gemv_quant_perf.rs`
// (see that crate's report for the ROWS_PER_WG=1/2/4/8 sweep, both isolated
// and against real end-to-end decode tok/s): 2 was the best overall
// trade-off — it wins decisively on the smallest/most latency-bound shapes
// while 4 and 8 start losing ground on those same shapes (extra live
// registers per lane outweighing the added memory-level parallelism once
// each workgroup's iteration count is already low) despite still winning on
// the largest, already-well-occupied shapes (lm-head). `row_base + r >= m`
// rows (only possible when `m` isn't a multiple of `ROWS_PER_WG`, e.g. this
// file's own non-power-of-two correctness test) clamp their read address to
// a real row (safe, redundant work) and simply skip the final write.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

extern "C" __global__ void gemv_q4_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    const unsigned ROWS_PER_WG = 2;
    unsigned row_base = blockIdx.x * ROWS_PER_WG;
    if (row_base >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 144;
    unsigned blocks_per_row = n / qk;
    const unsigned char* w_bytes = (const unsigned char*)w;

    // 8 lanes cooperate on one super-block: chunk `c` (0..7) is sub-block
    // pair `c/2` (byte offset `j = (c/2)*64` into the 128-byte code plane),
    // low nibble if `c` is even, high nibble if odd — see the file comment.
    const unsigned GROUP_SIZE = 8;
    const unsigned lane = threadIdx.x % GROUP_SIZE;
    const unsigned group = threadIdx.x / GROUP_SIZE;
    const unsigned num_groups = blockDim.x / GROUP_SIZE;

    float sum[ROWS_PER_WG];
#pragma unroll
    for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
        sum[r] = 0.0f;
    }

    for (unsigned blk = group; blk < blocks_per_row; blk += num_groups) {
        const float* xs = x + (size_t)blk * qk;

        unsigned j = (lane / 2) * 64;
        unsigned half = lane & 1;
        const float* xh = xs + j + half * 32;

#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
            // Redundant-but-safe clamp for the tail workgroup when `m` isn't
            // a multiple of `ROWS_PER_WG` — never read past the weight
            // buffer's real rows; the result is simply not written below.
            unsigned row = min(row_base + r, m - 1);
            const unsigned char* bptr =
                w_bytes + (size_t)row * blocks_per_row * block_bytes + (size_t)blk * block_bytes;
            __half d_h, dmin_h;
            std::memcpy(&d_h, bptr, sizeof(__half));
            std::memcpy(&dmin_h, bptr + 2, sizeof(__half));
            float d = __half2float(d_h);
            float dmin = __half2float(dmin_h);
            const unsigned char* scales = bptr + 4;
            const unsigned char* qs = bptr + 16;
            const unsigned char* q = qs + j / 2;
            unsigned char sc, mn;
            get_scale_min_k4(lane, scales, &sc, &mn);
            float dd = d * (float)sc;
            float mm = dmin * (float)mn;

            float local = 0.0f;
#pragma unroll
            for (unsigned l = 0; l < 32; ++l) {
                unsigned char nib = half == 0 ? (q[l] & 0xF) : (q[l] >> 4);
                local += (dd * (float)nib - mm) * xh[l];
            }
            sum[r] += local;
        }
    }

    // Wavefront butterfly reduction within each GROUP_SIZE-lane group —
    // every lane in the group ends up holding the group's total, so any one
    // of them (lane 0) can publish it.
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
    // num_groups is a power of two (blockDim.x and GROUP_SIZE both are).
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

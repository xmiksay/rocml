// Fused dequant-GEMV for Q5_K: like Q4_K but with an extra high-bit plane
// (`qh`) giving 5-bit codes. W's rows are raw GGUF Q5_K blocks (176 bytes),
// never materialized to f32 in VRAM. Must match rocml-core's
// `quant::q5_k::dequantize` exactly, including the `u1`/`u2` high-bit mask
// rotation across sub-block pairs (reduction ORDER may differ — see below).
//
// Launch contract and occupancy fix (both the original 8-lane-group split
// and the `ROWS_PER_WG` multi-row follow-up): identical to `gemv_q4_k.hip`
// — see its file comment for the full rationale, including why rows in the
// tail workgroup beyond `m` clamp their read address instead of skipping it.
// Chunk `c` (0..7, GROUP_SIZE=8 lanes per super-block) is sub-block pair
// `c/2`, low nibble/`u1` if `c` is even, high nibble/`u2` if odd;
// `get_scale_min_k4(c, ...)` and the `u1`/`u2` mask rotation both collapse to
// a function of `c` alone the same way the Q4_K scale/min index does.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

extern "C" __global__ void gemv_q5_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    const unsigned ROWS_PER_WG = 2;
    unsigned row_base = blockIdx.x * ROWS_PER_WG;
    if (row_base >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 176;
    unsigned blocks_per_row = n / qk;
    const unsigned char* w_bytes = (const unsigned char*)w;

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

        unsigned pair = lane / 2;
        unsigned half = lane & 1;
        unsigned j = pair * 64;
        const float* xh = xs + j + half * 32;
        // Mirrors dequantize_row_q5_K's u1/u2 rotation: sub-block pair
        // `pair`'s tested bit is `1 << (2*pair)` for the low-nibble half,
        // `2 << (2*pair)` for the high-nibble half.
        unsigned char u = (unsigned char)((half == 0 ? 1u : 2u) << (2 * pair));

#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
            unsigned row = min(row_base + r, m - 1);
            const unsigned char* bptr =
                w_bytes + (size_t)row * blocks_per_row * block_bytes + (size_t)blk * block_bytes;
            __half d_h, dmin_h;
            std::memcpy(&d_h, bptr, sizeof(__half));
            std::memcpy(&dmin_h, bptr + 2, sizeof(__half));
            float d = __half2float(d_h);
            float dmin = __half2float(dmin_h);
            const unsigned char* scales = bptr + 4;
            const unsigned char* qh = bptr + 16;
            const unsigned char* qs = bptr + 48;
            const unsigned char* ql = qs + j / 2;
            unsigned char sc, mn;
            get_scale_min_k4(lane, scales, &sc, &mn);
            float dd = d * (float)sc;
            float mm = dmin * (float)mn;

            float local = 0.0f;
#pragma unroll
            for (unsigned l = 0; l < 32; ++l) {
                unsigned char nib = half == 0 ? (ql[l] & 0xF) : (ql[l] >> 4);
                float hi = (qh[l] & u) ? 16.0f : 0.0f;
                local += (dd * ((float)nib + hi) - mm) * xh[l];
            }
            sum[r] += local;
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

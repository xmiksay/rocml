// Activation quantizer for the int8 MMQ-style prefill GEMM prototype
// (issue #6's WMMA-pipeline round, lever 2): turns an f32 chunked-prefill
// activation matrix into llama.cpp Q8_1-style per-32-element-block int8
// codes plus a per-block (scale, running-sum) pair, so
// `gemm_xwt_mmq_q*.hip` can feed the RDNA3 integer matrix unit
// (`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32`) directly against a
// GGUF-quantized weight's *native* stored codes — no weight dequant to f16
// at all, unlike the `gemm_xwt_wmma_q*` f16 path.
//
// Block width is fixed at 32 to match both Q8_0's native block size and
// every K-quant's native 32-element sub-block size (`get_scale_min_k4`'s
// granularity) — the GEMM kernel's K-reduction walks the same 32-wide
// stride on both operands so an activation block and a weight (sub)block
// always align exactly.
//
// One warp per (row, block): lane `e` owns element `blk*32 + e`, so the
// per-block absmax and sum reductions are a single native-width `__shfl_xor`
// butterfly with no partial/tail handling. `n` must be a multiple of 32
// (enforced by the host dispatch, same contract as `gemv_q8_0`'s block
// width).
//
// `block_sum` stores `d * sum_e(code[e])`, not the raw integer sum — this
// is exactly llama.cpp's Q8_1 `ds.y` field, precomputed once per activation
// block and reused by every weight row's asymmetric (min-shifted) k-quant
// dot product against it: `sum_e(a[e]) = d * sum_e(code[e])`, needed to
// cancel a k-quant sub-block's `-dmin*mn` offset term without re-reading
// every element (see `gemm_xwt_quant_mmq.hip`'s module doc for the full
// algebra).
#include <hip/hip_runtime.h>

namespace {

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, 32);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max_f32(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor(v, offset, 32));
    }
    return v;
}

} // namespace

extern "C" __global__ void quantize_act_q8_blk(
    const float* x, signed char* codes, float* scale, float* block_sum, unsigned rows,
    unsigned n) {
    constexpr unsigned ACT_BLOCK = 32;
    unsigned n_blocks = n / ACT_BLOCK;
    unsigned warp_id = blockIdx.x * blockDim.y + threadIdx.y;
    if (warp_id >= rows * n_blocks) {
        return;
    }
    unsigned row = warp_id / n_blocks;
    unsigned blk = warp_id % n_blocks;
    unsigned lane = threadIdx.x;

    const float* p = x + (size_t)row * n + (size_t)blk * ACT_BLOCK;
    float v = p[lane];

    float amax = warp_reduce_max_f32(fabsf(v));
    float d = amax > 0.0f ? amax / 127.0f : 1.0f;
    float q = roundf(v / d);
    q = fminf(fmaxf(q, -127.0f), 127.0f);
    signed char code = (signed char)q;

    codes[(size_t)row * n + (size_t)blk * ACT_BLOCK + lane] = code;

    float sum = warp_reduce_sum_f32((float)code);
    if (lane == 0) {
        unsigned blk_idx = row * n_blocks + blk;
        scale[blk_idx] = d;
        block_sum[blk_idx] = d * sum;
    }
}

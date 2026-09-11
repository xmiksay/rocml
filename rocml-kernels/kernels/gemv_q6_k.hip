// Fused dequant-GEMV for Q6_K: y = W * x where W's rows are raw GGUF Q6_K
// blocks (210 bytes: 128 bytes of 4-bit low codes, 64 bytes of 2-bit high
// codes, 16 signed 8-bit sub-block scales, one f16 `d`), never materialized
// to f32 in VRAM. Must match rocml-core's `quant::q6_k::dequantize` exactly,
// including its two-half / four-way interleaved output ordering (see the
// `out_group` decode below). The 16 per-sub-block scale bytes are *signed*
// (`BlockQ6K.scales: [i8; 16]`, per ggml/candle) — hence `signed char`
// below; a byte >= 0x80 is a negative scale, not a large positive one.
//
// Launch contract and occupancy fix: same rationale as `gemv_q4_k.hip` (see
// its file comment) — multiple lanes now cooperate on one super-block
// instead of one lane owning the whole thing. The original code's
// `half_idx in {0,1}` x `l in 0..32` loop already produces 4 interleaved
// outputs per `l` (offsets 0/32/64/96, ggml's q1..q4); `half_idx` x
// `out_group` (0..3, picks which of q1..q4) x a 4-way split of `l` gives
// GROUP_SIZE=32 independent chunks of 8 output elements each — a full
// RDNA3 wavefront per super-block. Q4_K/Q5_K settled on GROUP_SIZE=8;
// Q6_K's per-element decode carries more ALU work (an extra qh shift +
// scale multiply), so it stayed bandwidth-bound at a coarser split only
// once pushed all the way to a full wavefront — measured, not assumed, via
// gemv_quant_perf.rs's before/after numbers (see this crate's report for
// the GROUP_SIZE=8/16/32 comparison). No vectorized load here (unlike
// Q4_K/Q5_K): 210 is not a multiple of 4, so a super-block's byte offset
// within a row alternates 4-byte alignment class every other block — not
// safe to reinterpret as `uint4`/`uint32_t` without a runtime alignment
// check that would cost more than it saves at this block size.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

extern "C" __global__ void gemv_q6_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    unsigned row = blockIdx.x;
    if (row >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 210;
    unsigned blocks_per_row = n / qk;
    const unsigned char* row_ptr =
        (const unsigned char*)w + (size_t)row * blocks_per_row * block_bytes;

    // See the file comment for why Q6_K needs a full wavefront (32 lanes)
    // per super-block instead of Q4_K/Q5_K's 8.
    const unsigned GROUP_SIZE = 32;
    const unsigned lane = threadIdx.x % GROUP_SIZE;
    const unsigned group = threadIdx.x / GROUP_SIZE;
    const unsigned num_groups = blockDim.x / GROUP_SIZE;

    float sum = 0.0f;
    for (unsigned blk = group; blk < blocks_per_row; blk += num_groups) {
        const unsigned char* bptr = row_ptr + (size_t)blk * block_bytes;
        const unsigned char* ql_full = bptr;
        const unsigned char* qh_full = bptr + 128;
        const signed char* sc_full = (const signed char*)(bptr + 192);
        __half d_h;
        std::memcpy(&d_h, bptr + 208, sizeof(__half));
        float d = __half2float(d_h);
        const float* xs = x + (size_t)blk * qk;

        unsigned half_idx = lane / 16;
        unsigned out_group = (lane / 4) & 3;
        // `l_quarter` picks which contiguous 8-of-32 output slice this lane
        // owns; `l_half = l_quarter / 2` recovers `is = l/16` (0 or 1) as a
        // per-lane constant instead of a per-iteration branch.
        unsigned l_quarter = lane & 3;
        unsigned l_half = l_quarter / 2;
        const unsigned char* ql = ql_full + 64 * half_idx;
        const unsigned char* qh = qh_full + 32 * half_idx;
        const signed char* sc = sc_full + 8 * half_idx;
        const float* xh = xs + 128 * half_idx + 32 * out_group + 8 * l_quarter;

        // out_group in {0,1,2,3} = ggml's q1..q4: q{1,3} read ql[l]'s
        // low/high nibble, q{2,4} read ql[l+32]'s; the qh shift and scale
        // index both step by 2 per out_group. `scale` is constant across
        // this lane's loop, so `d * scale` is hoisted out of it.
        unsigned ql_off = (out_group & 1) * 32;
        bool high_nibble = out_group >= 2;
        unsigned qh_shift = out_group * 2;
        float dscale = d * (float)sc[l_half + out_group * 2];

        float local = 0.0f;
#pragma unroll
        for (unsigned l = 0; l < 8; ++l) {
            unsigned ql_idx = l_quarter * 8 + l;
            int nib = high_nibble ? (ql[ql_idx + ql_off] >> 4) : (ql[ql_idx + ql_off] & 0xF);
            int qv = (nib | (((qh[ql_idx] >> qh_shift) & 3) << 4)) - 32;
            local += dscale * (float)qv * xh[l];
        }
        sum += local;
    }

#pragma unroll
    for (unsigned offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset, GROUP_SIZE);
    }

    extern __shared__ float partial[];
    if (lane == 0) {
        partial[group] = sum;
    }
    __syncthreads();
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

// Fused dequant-GEMV for Q6_K: y = W * x where W's rows are raw GGUF Q6_K
// blocks (210 bytes: 128 bytes of 4-bit low codes, 64 bytes of 2-bit high
// codes, 16 signed 8-bit sub-block scales, one f16 `d`), never materialized
// to f32 in VRAM. Must match rocml-core's `quant::q6_k::dequantize` exactly,
// including its two-half / four-way interleaved output ordering (see the
// `out_group` decode below). The 16 per-sub-block scale bytes are *signed*
// (`BlockQ6K.scales: [i8; 16]`, per ggml/candle) — hence `signed char`
// below; a byte >= 0x80 is a negative scale, not a large positive one.
//
// Launch contract: same shape as `gemv_q4_k.hip` — one workgroup per
// `ROWS_PER_WG` (2) consecutive output rows, `REDUCE_BLOCK` (128) threads,
// `blockDim.x * ROWS_PER_WG * sizeof(float)` bytes of dynamic shared memory
// — see that file's comment for the full multi-row rationale (identical
// here).
//
// Occupancy fix: same rationale as `gemv_q4_k.hip`'s original 8-lane-group
// split — multiple lanes cooperate on one super-block instead of one lane
// owning the whole thing. The original code's `half_idx in {0,1}` x `l in
// 0..32` loop already produces 4 interleaved outputs per `l` (offsets
// 0/32/64/96, ggml's q1..q4); `half_idx` x `out_group` (0..3, picks which of
// q1..q4) x a 4-way split of `l` gives GROUP_SIZE=32 independent chunks of 8
// output elements each — a full RDNA3 wavefront per super-block. Q4_K/Q5_K
// settled on GROUP_SIZE=8; Q6_K's per-element decode carries more ALU work
// (an extra qh shift + scale multiply), so it stayed bandwidth-bound at a
// coarser split only once pushed all the way to a full wavefront — measured,
// not assumed, via gemv_quant_perf.rs's before/after numbers.
//
// ALU/load diet (issue #7 bandwidth follow-up): the original per-`l`
// iteration read `ql[ql_idx+ql_off]` and `qh[ql_idx]` as individual
// single-byte loads — 16 scalar `global_load_ubyte`s per lane per block,
// confirmed via `hipcc`'s generated ISA (`global_load_d16_u8`/`_hi_u8`
// dominating, vs. Q4_K/Q5_K's LLVM-auto-vectorized `global_load_b128`s for
// their own per-byte loops — Q6_K's extra branch on `high_nibble` and the
// 210-byte block's shifting 4-byte-alignment class defeated the same
// auto-vectorization there). For a *fixed* lane, `ql_idx = l_quarter*8 + l`
// is 8 consecutive byte offsets as `l` runs 0..7 (`l_quarter` is a per-lane
// constant), so both `ql[...+ql_off]` and `qh[...]` are read once each as an
// 8-byte (`uint64_t`) load — legal regardless of the block's shifting
// alignment class since it goes through `memcpy` (no strict-alignment typed
// read) — and each of the loop's 8 iterations then just shifts/masks a byte
// out of the two already-resident registers instead of issuing a new global
// load. This also improves cross-lane coalescing: lanes sharing an
// `out_group`/`half_idx` differ only in `l_quarter`, so 4 adjacent lanes'
// 8-byte reads tile one contiguous 32-byte span.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

extern "C" __global__ void gemv_q6_k(
    const void* w, const float* x, float* y, unsigned m, unsigned n) {
    const unsigned ROWS_PER_WG = 2;
    unsigned row_base = blockIdx.x * ROWS_PER_WG;
    if (row_base >= m) {
        return;
    }

    const unsigned qk = 256;
    const unsigned block_bytes = 210;
    unsigned blocks_per_row = n / qk;
    const unsigned char* w_bytes = (const unsigned char*)w;

    // See the file comment for why Q6_K needs a full wavefront (32 lanes)
    // per super-block instead of Q4_K/Q5_K's 8.
    const unsigned GROUP_SIZE = 32;
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

        unsigned half_idx = lane / 16;
        unsigned out_group = (lane / 4) & 3;
        // `l_quarter` picks which contiguous 8-of-32 output slice this lane
        // owns; `l_half = l_quarter / 2` recovers `is = l/16` (0 or 1) as a
        // per-lane constant instead of a per-iteration branch.
        unsigned l_quarter = lane & 3;
        unsigned l_half = l_quarter / 2;
        const float* xh = xs + 128 * half_idx + 32 * out_group + 8 * l_quarter;

        // out_group in {0,1,2,3} = ggml's q1..q4: q{1,3} read ql[l]'s
        // low/high nibble, q{2,4} read ql[l+32]'s; the qh shift and scale
        // index both step by 2 per out_group.
        unsigned ql_off = (out_group & 1) * 32;
        bool high_nibble = out_group >= 2;
        unsigned qh_shift = out_group * 2;
        unsigned sc_idx = l_half + out_group * 2;
        unsigned ql_idx0 = l_quarter * 8;

#pragma unroll
        for (unsigned r = 0; r < ROWS_PER_WG; ++r) {
            unsigned row = min(row_base + r, m - 1);
            const unsigned char* bptr = w_bytes +
                (size_t)row * blocks_per_row * block_bytes + (size_t)blk * block_bytes;
            const unsigned char* ql_full = bptr;
            const unsigned char* qh_full = bptr + 128;
            const signed char* sc_full = (const signed char*)(bptr + 192);
            __half d_h;
            std::memcpy(&d_h, bptr + 208, sizeof(__half));
            float d = __half2float(d_h);

            const unsigned char* ql = ql_full + 64 * half_idx;
            const unsigned char* qh = qh_full + 32 * half_idx;
            const signed char* sc = sc_full + 8 * half_idx;
            float dscale = d * (float)sc[sc_idx];

            // See the file comment: 8 consecutive bytes per lane, one wide
            // load each instead of 8 single-byte loads.
            unsigned long long ql_word, qh_word;
            std::memcpy(&ql_word, ql + ql_off + ql_idx0, sizeof(ql_word));
            std::memcpy(&qh_word, qh + ql_idx0, sizeof(qh_word));

            float local = 0.0f;
#pragma unroll
            for (unsigned l = 0; l < 8; ++l) {
                unsigned char qlb = (unsigned char)(ql_word >> (8 * l));
                unsigned char qhb = (unsigned char)(qh_word >> (8 * l));
                int nib = high_nibble ? (qlb >> 4) : (qlb & 0xF);
                int qv = (nib | (((qhb >> qh_shift) & 3) << 4)) - 32;
                local += dscale * (float)qv * xh[l];
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

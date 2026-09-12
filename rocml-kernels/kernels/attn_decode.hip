// Fused flash-decoding-style single-token causal attention. Replaces the old
// per-q-head `gemv_f32` (scores) + `softmax_varlen` + `gemv_t_f32`
// (weighted-V) composition — ~2*n_heads+1 launches/layer, most of them a
// single tiny workgroup per cached position (the score gemv) or a single
// workgroup total for the whole weighted-V pass — with two launches total.
// Online (running max/sum) softmax means scores are never materialized to
// global memory.
//
// Split-K over the sequence axis (`attn_decode_partial_f32` +
// `attn_decode_reduce_f32`): a decode step has only `n_kv_heads` independent
// units of work if each kv head gets one workgroup (e.g. 8 on Ornith-9B) —
// nowhere near enough to fill gfx1101's 60 CUs, which is exactly the
// "attn-out at 1.7% of roofline" failure this replaces. So the sequence is
// chopped into `n_splits` chunks (grid.y), each an independent workgroup
// that computes a partial (running-max, running-sum, unnormalized
// weighted-V) triple; `attn_decode_reduce_f32` then merges the (typically
// <=32) per-split partials with the standard log-sum-exp rule. At shallow
// `cur_len` the caller picks `n_splits == 1`, so the reduce kernel is a
// trivial pass-through — still one extra launch, but 2 total is nowhere
// near the old ~2*n_heads+1.
//
// GQA traffic sharing: a workgroup covers one kv head's *whole* q-head
// group (`group` warps, one per q head, block = `(32, group, 1)`) and
// cooperatively stages each K/V tile into LDS once per block before any
// warp reads it — so `group` q heads sharing a kv head pay for one global
// read of each K/V row instead of `group` reads. `load_kv_f32` is the one
// place a K/V element is read off the cache plane; a future quantized-KV
// cache (issue #2) only needs to change that function (its dequant cost is
// paid once per tile-load, amortized across the whole group, not once per
// (q head, timestep) as a naive per-head kernel would pay it).
//
// Occupancy math: LDS holds one `TILE_T`-row tile of K and one of V,
// `TILE_T * head_dim * 4 bytes` each. At `head_dim` == 256 (Ornith) that's
// 8*256*4 = 8KiB per tile, 16KiB for K+V — comfortably under gfx1101's 64KiB
// LDS/workgroup budget, leaving room for several resident workgroups per CU
// (LDS occupancy isn't the binding constraint). Per-lane registers: each of
// the 32 lanes in a warp owns up to `head_dim / 32 == 8` strided elements of
// `q` and of the running accumulator (`MAX_LANE_ELEMS`, sized for the
// task's head_dim <= 256 ceiling) — ~16-20 VGPRs/lane for this kernel's
// working set, far below gfx1101's 256-VGPR/lane budget, so register
// pressure doesn't cap occupancy either; the design is free to launch
// enough workgroups (`n_kv_heads * n_splits`, chosen by the Rust-side
// launcher) to hide the K/V global-memory latency behind other waves.
//
// Causal semantics: a decode step's query is the newest token, so it
// attends to every one of the `cur_len` cached positions — no masking
// beyond the loop bound `[0, cur_len)` (unlike prefill's varying valid_len
// per row, decode's "valid_len" is just `cur_len` for every head).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "attn_common.h"

// head_dim <= 256 / warp size 32 = 8 elements/lane at most (see the module
// doc's occupancy note). A head_dim above 256 would need a bigger array —
// not a shape this codebase uses today (dense Qwen3 128, qwen35 256).
#define MAX_LANE_ELEMS 8
// Rows of K/V staged into LDS at a time; kept in sync with the launcher's
// `shared_mem_bytes` computation (`ATTN_DECODE_TILE_T` in
// rocml/src/forward/kernels.rs) — see the module doc's LDS budget math.
#define TILE_T 8

// The single K/V read seam: issue #3's f16 KV cache and a future quantized
// KV cache (issue #2) both slot in here without touching the
// tiling/online-softmax logic around it — `T` is the cache's on-device
// element type (`float` for the f32 parity-test path, `__half` for the
// default f16 path), always widened to `float` for the online-softmax math.
// The tile-load loop below is what amortizes this per-element dequant cost
// across a whole GQA group's worth of reads.
template <typename T>
__device__ __forceinline__ float load_kv(const T* plane, size_t idx);

template <>
__device__ __forceinline__ float load_kv<float>(const float* plane, size_t idx) {
    return plane[idx];
}

template <>
__device__ __forceinline__ float load_kv<__half>(const __half* plane, size_t idx) {
    return __half2float(plane[idx]);
}

// One workgroup per (kv head, sequence split). Produces the split's
// unnormalized weighted-V accumulator plus its running (max, sum) for every
// q head in this kv head's GQA group; `attn_decode_reduce_f32` combines
// splits afterward (dtype-independent — it only ever reads the f32 partial
// buffers this kernel writes, never the cache itself, so there is exactly
// one reduce kernel regardless of KV storage dtype). `partial_out`/
// `partial_m`/`partial_l` are `[n_heads, n_splits, ...]`-shaped (a q head's
// own index `h`, not its kv head's, indexes them — see the Rust launcher),
// so this kernel and the reduce kernel agree on layout with no extra
// bookkeeping.
template <typename T>
__device__ __forceinline__ void attn_decode_partial_impl(
    const float* q, const T* k_layer, const T* v_layer, float* partial_out, float* partial_m,
    float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim, unsigned max_seq,
    unsigned cur_len, unsigned split_len, unsigned n_splits, float scale) {
    unsigned kvh = blockIdx.x;
    unsigned s = blockIdx.y;
    if (kvh >= n_kv_heads || s >= n_splits) {
        return;
    }

    unsigned lane = threadIdx.x;
    unsigned g = threadIdx.y;
    unsigned tid = g * 32 + lane;
    unsigned total_threads = blockDim.x * blockDim.y;
    unsigned h = kvh * group + g;

    unsigned start = s * split_len;
    // If a caller (e.g. a kernel test exercising the reduce path directly)
    // asks for more splits than `cur_len` has room for, `end < start` here
    // and the tile loop below simply never executes for this workgroup —
    // it contributes m=-inf/l=0/acc=0, a well-formed empty partial, with no
    // separate bounds check needed.
    unsigned end = min(start + split_len, cur_len);

    extern __shared__ float smem[];
    float* k_tile = smem;
    float* v_tile = smem + (size_t)TILE_T * head_dim;

    const float* q_h = q + (size_t)h * head_dim;
    const T* k_plane = k_layer + (size_t)kvh * max_seq * head_dim;
    const T* v_plane = v_layer + (size_t)kvh * max_seq * head_dim;

    float q_reg[MAX_LANE_ELEMS];
    unsigned n_elems = 0;
    for (unsigned d = lane; d < head_dim; d += 32) {
        q_reg[n_elems++] = q_h[d];
    }

    float acc[MAX_LANE_ELEMS];
    for (unsigned i = 0; i < n_elems; ++i) {
        acc[i] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (unsigned tile_base = start; tile_base < end; tile_base += TILE_T) {
        unsigned tile_len = min((unsigned)TILE_T, end - tile_base);

        for (unsigned idx = tid; idx < tile_len * head_dim; idx += total_threads) {
            unsigned row = idx / head_dim;
            unsigned col = idx % head_dim;
            size_t src = (size_t)(tile_base + row) * head_dim + col;
            k_tile[idx] = load_kv(k_plane, src);
            v_tile[idx] = load_kv(v_plane, src);
        }
        __syncthreads();

        for (unsigned t = 0; t < tile_len; ++t) {
            const float* k_row = k_tile + (size_t)t * head_dim;
            float partial = 0.0f;
            unsigned i = 0;
            for (unsigned d = lane; d < head_dim; d += 32, ++i) {
                partial += q_reg[i] * k_row[d];
            }
            float score = warp_reduce_sum(partial) * scale;

            float new_m = fmaxf(m, score);
            float correction = expf(m - new_m);
            float p = expf(score - new_m);
            l = l * correction + p;

            const float* v_row = v_tile + (size_t)t * head_dim;
            i = 0;
            for (unsigned d = lane; d < head_dim; d += 32, ++i) {
                acc[i] = acc[i] * correction + p * v_row[d];
            }
            m = new_m;
        }
        // Every thread in the block ran the same `tile_len` iterations
        // above (tile_len is block-uniform), so this barrier is reached
        // uniformly before the next iteration's load loop overwrites
        // k_tile/v_tile.
        __syncthreads();
    }

    if (lane == 0) {
        partial_m[h * n_splits + s] = m;
        partial_l[h * n_splits + s] = l;
    }
    unsigned i = 0;
    float* out_row = partial_out + (size_t)(h * n_splits + s) * head_dim;
    for (unsigned d = lane; d < head_dim; d += 32, ++i) {
        out_row[d] = acc[i];
    }
}

// f32 KV storage: the parity-test reference path (see
// `crate::model::LoadOptions`) — never selected by the CLI/server.
extern "C" __global__ void attn_decode_partial_f32(
    const float* q, const float* k_layer, const float* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned cur_len, unsigned split_len, unsigned n_splits, float scale) {
    attn_decode_partial_impl<float>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, cur_len, split_len, n_splits, scale);
}

// f16 KV storage: the default path (issue #3) — halves KV bandwidth/VRAM
// vs f32 with the dequant folded into this same tile-load loop, never
// materialized back to f32 in memory.
extern "C" __global__ void attn_decode_partial_f16(
    const float* q, const __half* k_layer, const __half* v_layer, float* partial_out,
    float* partial_m, float* partial_l, unsigned n_kv_heads, unsigned group, unsigned head_dim,
    unsigned max_seq, unsigned cur_len, unsigned split_len, unsigned n_splits, float scale) {
    attn_decode_partial_impl<__half>(
        q, k_layer, v_layer, partial_out, partial_m, partial_l, n_kv_heads, group, head_dim,
        max_seq, cur_len, split_len, n_splits, scale);
}

// Merges one q head's per-split partials with the log-sum-exp rule. One
// block per q head, `head_dim` threads (`head_dim <= 256` per this
// codebase's models), one output element per thread — `n_splits` is small
// (<=32 by construction, see the Rust launcher), so the serial loops below
// cost `O(head_dim * n_splits)` per head, negligible next to the partial
// kernel's `O(cur_len * head_dim)`.
extern "C" __global__ void attn_decode_reduce_f32(
    const float* partial_out, const float* partial_m, const float* partial_l, float* out,
    unsigned head_dim, unsigned n_splits) {
    unsigned h = blockIdx.x;
    unsigned d = threadIdx.x;

    __shared__ float global_m;
    __shared__ float global_l;
    if (d == 0) {
        float m = -INFINITY;
        for (unsigned s = 0; s < n_splits; ++s) {
            m = fmaxf(m, partial_m[h * n_splits + s]);
        }
        float l = 0.0f;
        for (unsigned s = 0; s < n_splits; ++s) {
            l += partial_l[h * n_splits + s] * expf(partial_m[h * n_splits + s] - m);
        }
        global_m = m;
        // l == 0 only if every split contributed zero valid positions,
        // which can't happen for a real decode step (cur_len >= 1 always
        // puts at least one position in split 0) — the fallback just avoids
        // a NaN from 0/0 if a test ever drives n_splits with no positions
        // at all.
        global_l = (l > 0.0f) ? l : 1.0f;
    }
    __syncthreads();

    if (d >= head_dim) {
        return;
    }
    float acc = 0.0f;
    for (unsigned s = 0; s < n_splits; ++s) {
        float w = expf(partial_m[h * n_splits + s] - global_m);
        acc += partial_out[(h * n_splits + s) * head_dim + d] * w;
    }
    out[h * head_dim + d] = acc / global_l;
}

// Small batched reshape/broadcast kernels the chunked-prefill forward pass
// needs that the per-token decode path didn't (a single token's per-head
// split/KV-append is cheap enough as host-issued device-to-device copies —
// see `qwen35::forward::attention::attention_step` and `AttnPlane::append`
// — but a T-token chunk needs it batched into one launch, not T*heads tiny
// copies per layer).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

// Extracts one `head_dim`-wide field (Q or the output gate) out of a fused
// per-head `[..., stride]` projection row into a compact `[tokens, heads,
// head_dim]` buffer: `dst[t,h,d] = src[t*heads*stride + h*stride +
// src_offset + d]`. `src_offset=0`/`stride=head_dim` (no output gate) makes
// this a plain copy; `src_offset=head_dim`/`stride=2*head_dim` pulls the
// gate half instead of the query half — see `attn_q_raw`'s per-head
// `[query|gate]` layout in `qwen35::forward::attention`.
extern "C" __global__ void extract_heads_f32(
    const float* src, float* dst, unsigned tokens, unsigned heads, unsigned head_dim,
    unsigned src_stride, unsigned src_offset) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = tokens * heads * head_dim;
    if (idx >= total) {
        return;
    }
    unsigned d = idx % head_dim;
    unsigned tmp = idx / head_dim;
    unsigned h = tmp % heads;
    unsigned t = tmp / heads;
    size_t src_idx = (size_t)t * heads * src_stride + (size_t)h * src_stride + src_offset + d;
    dst[idx] = src[src_idx];
}

// Batch-appends a chunk's `[chunk_len, n_kv_heads, head_dim]` K/V into the
// `[n_kv_heads, max_seq, head_dim]` cache planes at `[pos_base,
// pos_base+chunk_len)` — the chunked sibling of `AttnPlane::append`'s
// per-token host loop.
extern "C" __global__ void scatter_kv_chunk_f32(
    const float* k_src, const float* v_src, float* k_dst, float* v_dst, unsigned n_kv_heads,
    unsigned head_dim, unsigned max_seq, unsigned chunk_len, unsigned pos_base) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = chunk_len * n_kv_heads * head_dim;
    if (idx >= total) {
        return;
    }
    unsigned d = idx % head_dim;
    unsigned tmp = idx / head_dim;
    unsigned h = tmp % n_kv_heads;
    unsigned t = tmp / n_kv_heads;
    size_t src_idx = (size_t)t * n_kv_heads * head_dim + (size_t)h * head_dim + d;
    size_t dst_idx = (size_t)h * max_seq * head_dim + (size_t)(pos_base + t) * head_dim + d;
    k_dst[dst_idx] = k_src[src_idx];
    v_dst[dst_idx] = v_src[src_idx];
}

// f16-cache sibling of `scatter_kv_chunk_f32` (issue #3's default KV dtype):
// identical index math, casting each element on the way in instead of a
// same-dtype store.
extern "C" __global__ void scatter_kv_chunk_f16(
    const float* k_src, const float* v_src, __half* k_dst, __half* v_dst, unsigned n_kv_heads,
    unsigned head_dim, unsigned max_seq, unsigned chunk_len, unsigned pos_base) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned total = chunk_len * n_kv_heads * head_dim;
    if (idx >= total) {
        return;
    }
    unsigned d = idx % head_dim;
    unsigned tmp = idx / head_dim;
    unsigned h = tmp % n_kv_heads;
    unsigned t = tmp / n_kv_heads;
    size_t src_idx = (size_t)t * n_kv_heads * head_dim + (size_t)h * head_dim + d;
    size_t dst_idx = (size_t)h * max_seq * head_dim + (size_t)(pos_base + t) * head_dim + d;
    k_dst[dst_idx] = __float2half(k_src[src_idx]);
    v_dst[dst_idx] = __float2half(v_src[src_idx]);
}

// Batched sibling of `gdn_gate_f32` (see that kernel's doc comment for the
// math): `a_raw`/`b_raw`/`beta_out`/`g_out` are `[chunk_len, heads]`
// (`total = chunk_len*heads`), but `a_log`/`dt_bias` stay per-head weight
// vectors of length `heads` — broadcast across tokens via `i % heads`
// instead of indexing `i` directly.
extern "C" __global__ void gdn_gate_chunk_f32(
    const float* a_raw, const float* b_raw, const float* a_log, const float* dt_bias,
    float* beta_out, float* g_out, unsigned heads, unsigned total) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) {
        return;
    }
    unsigned head = i % heads;
    beta_out[i] = 1.0f / (1.0f + expf(-b_raw[i]));
    float softplus = logf(1.0f + expf(a_raw[i] + dt_bias[head]));
    g_out[i] = -expf(a_log[head]) * softplus;
}

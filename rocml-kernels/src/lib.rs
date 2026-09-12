//! Embedded HIP kernel code objects. `build.rs` compiles every `kernels/*.hip`
//! file to a `.hsaco` code object via `hipcc --genco`, and this crate pulls
//! each one in with `include_bytes!` so the compiled binary carries the
//! kernels directly — no runtime dependency on `hipcc` or the `kernels/`
//! sources being present on the machine that runs it.

/// `kernels/smoke.hip`: elementwise `out[i] = a[i] + b[i]`.
pub const VEC_ADD_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/smoke.hsaco"));
pub const VEC_ADD_F32_KERNEL: &str = "vec_add_f32";

/// `kernels/gemv.hip`: naive row-per-block `y = mat * x` (mat is row-major
/// m x n). Must be launched with a power-of-two block size.
pub const GEMV_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv.hsaco"));
pub const GEMV_F32_KERNEL: &str = "gemv_f32";

/// `kernels/gemv_f16.hip`: decode-path `y = W * x` (W is row-major m x n f16,
/// x/y f32, f32 accumulation). One block per output row; block size must be
/// a power of two (shared-memory tree reduction).
pub const GEMV_F16_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_f16.hsaco"));
pub const GEMV_F16_KERNEL: &str = "gemv_f16";

/// `kernels/gemm_f16.hip`: prefill-path linear layer `out = x * W^T` (x is
/// rows x n f32, W is row-major m x n f16, out is rows x m f32). Fixed
/// 16x16 shared-memory tile; must be launched with block = (16, 16, 1).
pub const GEMM_XWT_F16_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemm_f16.hsaco"));
pub const GEMM_XWT_F16_KERNEL: &str = "gemm_xwt_f16";

/// `kernels/rmsnorm.hip`: per-row `out = x / rms(x) * weight`. One block per
/// row; block size must be a power of two (shared-memory tree reduction).
pub const RMSNORM_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rmsnorm.hsaco"));
pub const RMSNORM_F32_KERNEL: &str = "rmsnorm_f32";

/// `kernels/rope.hip`: in-place NEOX-style rotary embedding over a
/// tokens x heads x head_dim buffer (element i pairs with i + head_dim/2).
/// `head_dim` must be even. No shared-memory or block-size constraint.
pub const ROPE_NEOX_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/rope.hsaco"));
pub const ROPE_NEOX_F32_KERNEL: &str = "rope_neox_f32";

/// `kernels/silu_mul.hip`: `out = silu(gate) * up`, elementwise.
pub const SILU_MUL_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/silu_mul.hsaco"));
pub const SILU_MUL_F32_KERNEL: &str = "silu_mul_f32";

/// `kernels/softmax.hip`: per-row scaled softmax over the first
/// `valid_len[row]` columns, zeroing the rest. One block per row; block size
/// must be a power of two (shared-memory tree reduction).
pub const SOFTMAX_VARLEN_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/softmax.hsaco"));
pub const SOFTMAX_VARLEN_F32_KERNEL: &str = "softmax_varlen_f32";

/// `kernels/embedding.hip`: `out[i] = f32(table[ids[i]])` row lookup (table
/// is row-major vocab x dim f16). One block per token row.
pub const EMBEDDING_F16_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/embedding.hsaco"));
pub const EMBEDDING_F16_F32_KERNEL: &str = "embedding_f16_f32";

/// `kernels/elementwise.hip`: in-place residual add, f16<->f32 casts, and the
/// sigmoid output gate. All four kernels share one code object.
pub const ELEMENTWISE_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/elementwise.hsaco"));
pub const ADD_INPLACE_F32_KERNEL: &str = "add_inplace_f32";
pub const CAST_F16_F32_KERNEL: &str = "cast_f16_f32";
pub const CAST_F32_F16_KERNEL: &str = "cast_f32_f16";
pub const SIGMOID_MUL_F32_KERNEL: &str = "sigmoid_mul_f32";

/// `kernels/rope_partial.hip`: NEOX rope over only the first `rot_dim` of
/// each head, used by qwen35's partial-rotary full-attention layers.
pub const ROPE_NEOX_PARTIAL_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/rope_partial.hsaco"));
pub const ROPE_NEOX_PARTIAL_F32_KERNEL: &str = "rope_neox_partial_f32";

/// `kernels/gdn_conv.hip`: Gated Delta Net causal depthwise conv1d + SiLU,
/// one decode step.
pub const GDN_CONV1D_DECODE_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_conv.hsaco"));
pub const GDN_CONV1D_DECODE_F32_KERNEL: &str = "causal_conv1d_decode_f32";

/// `kernels/gdn_gate.hip`: Gated Delta Net per-head beta/decay gate scalars,
/// one decode step.
pub const GDN_GATE_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gdn_gate.hsaco"));
pub const GDN_GATE_F32_KERNEL: &str = "gdn_gate_f32";

/// `kernels/gdn_recurrence.hip`: Gated Delta Net fused state update +
/// readout, one decode step. Launch block size must be
/// `max(head_k_dim, head_v_dim)`. Supports grouped query/key heads
/// (`num_k_heads` a divisor of `num_heads`, contiguous broadcast).
pub const GDN_RECURRENCE_DECODE_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_recurrence.hsaco"));
pub const GDN_RECURRENCE_DECODE_F32_KERNEL: &str = "gdn_recurrence_decode_f32";

/// `kernels/gdn_chunk.hip`: Gated Delta Net batched causal conv1d
/// (`causal_conv1d_chunk_f32`) over a whole prefill chunk in one launch,
/// reproducing `causal_conv1d_decode_f32`'s per-step math exactly. The
/// recurrence's chunk kernel used to share this file (`gdn_recurrence_chunk_f32`)
/// — see `kernels/gdn_chunkwise.hip`, which replaced it with the chunkwise
/// (blocked delta-rule) formulation.
pub const GDN_CONV1D_CHUNK_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_chunk.hsaco"));
pub const GDN_CONV1D_CHUNK_F32_KERNEL: &str = "causal_conv1d_chunk_f32";

/// `kernels/gemv_t.hip`: decode-attention building block `y = A^T * x`, A is
/// row-major rows x n (e.g. a cached-V plane, one row per time step). One
/// thread per output column, looping over rows; no block-size constraint.
pub const GEMV_T_F32_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_t.hsaco"));
pub const GEMV_T_F32_KERNEL: &str = "gemv_t_f32";

/// `kernels/gemv_q8_0.hip`: fused dequant-GEMV `y = W * x` where W's rows are
/// raw GGUF Q8_0 blocks (34 bytes: f16 `d` + 32 signed 8-bit codes) — weights
/// stay in ggml block format in VRAM, dequantized in-register. One workgroup
/// per output row; block size must be a power of two (shared-mem tree
/// reduction). `n` must be a multiple of 32.
pub const GEMV_Q8_0_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q8_0.hsaco"));
pub const GEMV_Q8_0_KERNEL: &str = "gemv_q8_0";

/// `kernels/gemv_q4_k.hip`: fused dequant-GEMV `y = W * x` where W's rows are
/// raw GGUF Q4_K blocks (144 bytes/256-element super-block). One workgroup
/// per output row; block size must be a power of two. `n` must be a multiple
/// of 256.
pub const GEMV_Q4_K_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q4_k.hsaco"));
pub const GEMV_Q4_K_KERNEL: &str = "gemv_q4_k";

/// `kernels/gemv_q5_k.hip`: fused dequant-GEMV `y = W * x` where W's rows are
/// raw GGUF Q5_K blocks (176 bytes/256-element super-block). One workgroup
/// per output row; block size must be a power of two. `n` must be a multiple
/// of 256.
pub const GEMV_Q5_K_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q5_k.hsaco"));
pub const GEMV_Q5_K_KERNEL: &str = "gemv_q5_k";

/// `kernels/gemv_q6_k.hip`: fused dequant-GEMV `y = W * x` where W's rows are
/// raw GGUF Q6_K blocks (210 bytes/256-element super-block). One workgroup
/// per output row; block size must be a power of two. `n` must be a multiple
/// of 256.
pub const GEMV_Q6_K_HSACO: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/gemv_q6_k.hsaco"));
pub const GEMV_Q6_K_KERNEL: &str = "gemv_q6_k";

/// `kernels/gemm_xwt_quant.hip`: batched prefill-path linear layer for
/// GGUF-quantized weights, `out[rows,m] = X[rows,n] * dequant(W)^T` — the
/// batched sibling of `gemv_q*` (which stays the fast path for rows==1
/// decode). One weight row shared by 8 output rows per workgroup; launch
/// with block = (32, 8, 1) and `256 * sizeof(f32)` bytes of dynamic shared
/// memory (see the kernel source's module doc for the tiling design). All
/// four dtypes share one code object.
pub const GEMM_XWT_Q8_0_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_xwt_quant.hsaco"));
pub const GEMM_XWT_Q8_0_KERNEL: &str = "gemm_xwt_q8_0";
pub const GEMM_XWT_Q4_K_HSACO: &[u8] = GEMM_XWT_Q8_0_HSACO;
pub const GEMM_XWT_Q4_K_KERNEL: &str = "gemm_xwt_q4_k";
pub const GEMM_XWT_Q5_K_HSACO: &[u8] = GEMM_XWT_Q8_0_HSACO;
pub const GEMM_XWT_Q5_K_KERNEL: &str = "gemm_xwt_q5_k";
pub const GEMM_XWT_Q6_K_HSACO: &[u8] = GEMM_XWT_Q8_0_HSACO;
pub const GEMM_XWT_Q6_K_KERNEL: &str = "gemm_xwt_q6_k";

/// `kernels/gemm_xwt_quant_wmma.hip`: WMMA matrix-core version of
/// `gemm_xwt_q*` (issue #6's follow-up) — same `out[rows,m] = X[rows,n] *
/// dequant(W)^T` contract, but feeds gfx11's wave32 matrix unit instead of
/// scalar FMA. Launch with block = (32, 8, 1) and `(TILE_ROWS + TILE_M) *
/// K_STAGE * sizeof(f16)` bytes of dynamic shared memory (see the kernel
/// source's module doc for the tile sizes and fragment layout); grid =
/// `(ceil(m/TILE_M), ceil(rows/TILE_ROWS), 1)`. Requires `m` and `n`
/// multiples of 16 and `rows >= 16` to be worth it — the scalar
/// `gemm_xwt_q*` kernels above stay the dispatch fallback otherwise. All
/// four dtypes share one code object.
pub const GEMM_XWT_WMMA_Q8_0_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gemm_xwt_quant_wmma.hsaco"));
pub const GEMM_XWT_WMMA_Q8_0_KERNEL: &str = "gemm_xwt_wmma_q8_0";
pub const GEMM_XWT_WMMA_Q4_K_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_Q4_K_KERNEL: &str = "gemm_xwt_wmma_q4_k";
pub const GEMM_XWT_WMMA_Q5_K_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_Q5_K_KERNEL: &str = "gemm_xwt_wmma_q5_k";
pub const GEMM_XWT_WMMA_Q6_K_HSACO: &[u8] = GEMM_XWT_WMMA_Q8_0_HSACO;
pub const GEMM_XWT_WMMA_Q6_K_KERNEL: &str = "gemm_xwt_wmma_q6_k";

/// `kernels/chunk_reshape.hip`: small batched reshape/broadcast kernels the
/// chunked-prefill forward pass needs (per-head extraction, batched KV-cache
/// append, the GDN gate's per-token broadcast). All three share one code
/// object.
pub const EXTRACT_HEADS_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/chunk_reshape.hsaco"));
pub const EXTRACT_HEADS_F32_KERNEL: &str = "extract_heads_f32";
pub const SCATTER_KV_CHUNK_F32_HSACO: &[u8] = EXTRACT_HEADS_F32_HSACO;
pub const SCATTER_KV_CHUNK_F32_KERNEL: &str = "scatter_kv_chunk_f32";
/// f16-cache sibling of `scatter_kv_chunk_f32` (issue #3's default KV
/// dtype) — same code object, different entry point.
pub const SCATTER_KV_CHUNK_F16_HSACO: &[u8] = EXTRACT_HEADS_F32_HSACO;
pub const SCATTER_KV_CHUNK_F16_KERNEL: &str = "scatter_kv_chunk_f16";
pub const GDN_GATE_CHUNK_F32_HSACO: &[u8] = EXTRACT_HEADS_F32_HSACO;
pub const GDN_GATE_CHUNK_F32_KERNEL: &str = "gdn_gate_chunk_f32";

/// `kernels/attn_decode.hip`: fused flash-decoding-style single-token causal
/// attention, split-K over the sequence axis. `attn_decode_partial_f32`
/// (grid = `[n_kv_heads, n_splits]`, block = `[32, group]`) produces one
/// per-(q head, split) partial online-softmax triple; `attn_decode_reduce_f32`
/// (grid = `[n_heads]`, block = `[head_dim]`) merges them. See the kernel
/// source's module doc for the full design and occupancy math.
pub const ATTN_DECODE_PARTIAL_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_decode.hsaco"));
pub const ATTN_DECODE_PARTIAL_F32_KERNEL: &str = "attn_decode_partial_f32";
pub const ATTN_DECODE_REDUCE_F32_HSACO: &[u8] = ATTN_DECODE_PARTIAL_F32_HSACO;
pub const ATTN_DECODE_REDUCE_F32_KERNEL: &str = "attn_decode_reduce_f32";
/// f16-KV-cache sibling of `attn_decode_partial_f32` (issue #3's default KV
/// dtype) — same code object, different entry point; the reduce kernel is
/// shared unchanged (it only ever reads the f32 partial buffers, never the
/// cache itself).
pub const ATTN_DECODE_PARTIAL_F16_HSACO: &[u8] = ATTN_DECODE_PARTIAL_F32_HSACO;
pub const ATTN_DECODE_PARTIAL_F16_KERNEL: &str = "attn_decode_partial_f16";

/// `kernels/attn_prefill.hip`: batched causal attention for a prefill chunk
/// of `chunk_len` new query rows against the KV cache (which already holds
/// this chunk's own appended K/V). One workgroup per (kv head, query row);
/// launch with block = `[32, group]` and `2 * TILE_T(8) * head_dim *
/// sizeof(f32)` bytes of dynamic shared memory (same K/V tile size as
/// `attn_decode`). No split-K/reduce pass needed — see the kernel source's
/// module doc.
pub const ATTN_PREFILL_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_prefill.hsaco"));
pub const ATTN_PREFILL_F32_KERNEL: &str = "attn_prefill_f32";
/// f16-KV-cache sibling of `attn_prefill_f32` (issue #3's default KV dtype).
pub const ATTN_PREFILL_F16_HSACO: &[u8] = ATTN_PREFILL_F32_HSACO;
pub const ATTN_PREFILL_F16_KERNEL: &str = "attn_prefill_f16";

/// `kernels/kv_quant.hip`: quantize-on-evict for the KIVI-style mixed KV
/// cache (issue #2) — one batch launch per evicted `WINDOW_LEN`-position
/// block. K is per-channel Q8 (`quantize_evict_k_f16_to_q8`, block =
/// `[head_dim, 1, 1]`, grid = `[n_kv_heads, 1, 1]`); V is per-token, either
/// Q8 or Q4-packed (`quantize_evict_v_f16_to_q{8,4}`, block = `[32, 1, 1]`,
/// grid = `[n_kv_heads, window_len, 1]`). All three share one code object.
pub const QUANTIZE_EVICT_K_Q8_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/kv_quant.hsaco"));
pub const QUANTIZE_EVICT_K_Q8_KERNEL: &str = "quantize_evict_k_f16_to_q8";
pub const QUANTIZE_EVICT_V_Q8_HSACO: &[u8] = QUANTIZE_EVICT_K_Q8_HSACO;
pub const QUANTIZE_EVICT_V_Q8_KERNEL: &str = "quantize_evict_v_f16_to_q8";
pub const QUANTIZE_EVICT_V_Q4_HSACO: &[u8] = QUANTIZE_EVICT_K_Q8_HSACO;
pub const QUANTIZE_EVICT_V_Q4_KERNEL: &str = "quantize_evict_v_f16_to_q4";

/// `kernels/attn_decode_mixed.hip`: fused decode attention for a KIVI-style
/// mixed KV layer (issue #2) — extends `attn_decode_partial_f16`'s
/// online-softmax split-K design with a per-position dequant-on-load that
/// reads from whichever of the sink/bulk/window regions a position
/// currently lives in (see the kernel source's module doc). `_q8`/`_q4`
/// differ only in V's bit width; K is always per-channel Q8. Same launch
/// shape as `attn_decode_partial_f16`; shares the same
/// `attn_decode_reduce_f32` reduce pass.
pub const ATTN_DECODE_PARTIAL_MIXED_Q8_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/attn_decode_mixed.hsaco"));
pub const ATTN_DECODE_PARTIAL_MIXED_Q8_KERNEL: &str = "attn_decode_partial_mixed_q8";
pub const ATTN_DECODE_PARTIAL_MIXED_Q4_HSACO: &[u8] = ATTN_DECODE_PARTIAL_MIXED_Q8_HSACO;
pub const ATTN_DECODE_PARTIAL_MIXED_Q4_KERNEL: &str = "attn_decode_partial_mixed_q4";

/// `kernels/gdn_chunkwise.hip`: the chunkwise (blocked delta-rule)
/// gated-delta-rule recurrence — replaces `gdn_recurrence_chunk_f32`'s
/// token-serial-inside-chunk loop with O(chunk^2) parallel matmul-shaped
/// kernels plus one short (`tile_len`-long) triangular-inverse dependency
/// chain. See the kernel source's module doc for the seven-stage pipeline
/// and the exact per-kernel launch contract (grid/block/shared-mem shapes).
/// All seven share one code object.
pub const GDN_CW_PREP_F32_HSACO: &[u8] =
    include_bytes!(concat!(env!("OUT_DIR"), "/gdn_chunkwise.hsaco"));
pub const GDN_CW_PREP_F32_KERNEL: &str = "gdn_chunkwise_prep_f32";
pub const GDN_CW_UT_BUILD_F32_HSACO: &[u8] = GDN_CW_PREP_F32_HSACO;
pub const GDN_CW_UT_BUILD_F32_KERNEL: &str = "gdn_chunkwise_ut_build_f32";
pub const GDN_CW_TINV_F32_HSACO: &[u8] = GDN_CW_PREP_F32_HSACO;
pub const GDN_CW_TINV_F32_KERNEL: &str = "gdn_chunkwise_tinv_f32";
pub const GDN_CW_UV_VNEW_F32_HSACO: &[u8] = GDN_CW_PREP_F32_HSACO;
pub const GDN_CW_UV_VNEW_F32_KERNEL: &str = "gdn_chunkwise_uv_vnew_f32";
pub const GDN_CW_OUTPUT_F32_HSACO: &[u8] = GDN_CW_PREP_F32_HSACO;
pub const GDN_CW_OUTPUT_F32_KERNEL: &str = "gdn_chunkwise_output_f32";
pub const GDN_CW_STATE_F32_HSACO: &[u8] = GDN_CW_PREP_F32_HSACO;
pub const GDN_CW_STATE_F32_KERNEL: &str = "gdn_chunkwise_state_f32";

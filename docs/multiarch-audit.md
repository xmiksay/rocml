# Multi-architecture readiness audit (issue #16)

Systematic sweep of shared code (kernels, dispatch, chunked prefill, KV
cache, sampler, chat rendering) for assumptions baked in from the two
architectures this engine has shipped so far (dense `qwen3`, hybrid
`qwen35`), with gemma-class dense models as the named example of "the next
family in the door." For each assumption: where it lives, whether it's
already config-driven/architecture-generic, whether something was fixed
this round, or — where fixing it would mean building an unvalidated kernel
against no real checkpoint — what the concrete gap is, cited against a real
GGUF header (not guessed).

**Method for the gemma gap list**: `general.architecture` metadata, KV
pairs, and `blk.0.*` tensor names/shapes were read directly off two real,
public GGUFs via HTTP range-GET (header bytes only, never downloading the
multi-GB weight payload) — `bartowski/gemma-2-9b-it-GGUF` (`gemma-2-9b-it-
Q4_K_M.gguf`) and `unsloth/gemma-3-4b-it-GGUF` (`gemma-3-4b-it-Q4_K_M.gguf`).
Every config key and tensor name cited below is copy-pasted from that dump,
not inferred from memory of the architecture.

## Audit table

| # | Assumption | Where | Disposition |
|---|---|---|---|
| 1 | FFN activation hardcoded to SiLU/SwiGLU | `forward/ffn.rs`, `forward/ffn_chunk.rs` (dense); `qwen35/forward/ffn.rs`, `qwen35/forward/gdn.rs` (hybrid) | **Fixed** (dense path) — see below |
| 2 | Head-dim derived from hidden/n_heads | *(none found)* | **Already generic** — `qwen3.attention.key_length` is a required, explicit GGUF key (`config.rs:70`); never computed as `hidden/head_count` anywhere |
| 3 | Partial-rope hardcoding | dense: full-dim only (`forward/attention.rs`, `forward/kernels.rs::rope`); hybrid: partial via `qwen35.rope.dimension_count` (`qwen35/config.rs:36`) | **Justified** — each arch's own config already drives its own rope width; not a shared-code leak. Gemma needs full-dim rope (matches dense), no partial-rope gap |
| 4 | Single global rope theta (no per-layer) | `config.rs:44` (`rope_freq_base: f32`, one value for the whole model); `qwen35/config.rs:37` (same) | **Gemma gap** — see below |
| 5 | `rope.freq_base` treated as a required GGUF key | `config.rs:71` (`gguf.get_f32("qwen3.rope.freq_base")?` — hard error if absent) | **Gemma gap** — see below |
| 6 | Attention/final logit softcapping absent | *(no softcap hook anywhere: `attn_decode.hip`, `attn_prefill*.hip`, `forward/mod.rs`'s lm-head scope)* | **Gemma gap (gemma-2 only)** — see below |
| 7 | Sliding-window attention absent | `attn_decode.hip`/`attn_prefill.hip`/flash variants: causal bound is always `[0, cur_len)`, no window parameter exists at the kernel level | **Gemma gap (both gemma-2 and gemma-3)** — see below |
| 8 | Q/K-norm assumed always present | `weights/layer.rs:51-52` (`attn_q_norm`/`attn_k_norm` loaded unconditionally, `gguf.tensor(name)?` errors if the tensor is missing) | **Gemma gap (gemma-2 only)** — see below |
| 9 | Tied embeddings (no separate `output.weight`) | `weights/mod.rs:48-62` | **Already fixed/generic** (prior round) — falls back to `token_embd.weight` when `output.weight` is absent. Verified: neither gemma-2 nor gemma-3's GGUF has an `output.weight` tensor, so this path is exactly what a gemma loader would also hit |
| 10 | Attention scale hardcoded to `1/sqrt(head_dim)` | `forward/attention.rs:160`, `forward/attention_chunk.rs` (2 call sites); `qwen35/forward/attention.rs`, `attention_chunk{,_mixed}.rs` (same pattern) | **Gemma gap, low-effort** — see below (kernel already takes `scale: f32` at runtime; only a config field is missing) |
| 11 | Only two norms per layer (pre-attn, pre-ffn) | `forward/attention.rs` (residual add right after `attn_output` projection, no hook), `forward/ffn.rs` (same after `ffn_down`) | **Gemma gap — structural, the largest one found** — see below |
| 12 | Embedding-scale-by-`sqrt(hidden)` absent | `forward/kernels.rs::embedding`, `forward/mod.rs`'s embed scope | **Gemma gap** — see below |
| 13 | Sampler assumes nothing architecture-specific | `sample.rs` (temperature/top-k/top-p/repeat-penalty, pure token-id/logit arithmetic) | **Already generic** — verified, no arch-specific code found anywhere in this file |
| 14 | Chat template hardcoded to one model's Jinja | `chat/render.rs:1` (module doc self-declares: "Hardcoded renderer for `chat_template.jinja` (Ornith-1.0-9B / qwen35)") | **Justified/known limitation** — already documented project-wide (`.claude/CLAUDE.md`'s `rocml-serve` section); gemma's own template (`<start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`, confirmed from both real GGUFs' `tokenizer.chat_template` key) would need its own renderer, same pattern this one already follows, whenever a gemma checkpoint is actually served |
| 15 | Mixed/quantized KV cache dense-arch support | `cache.rs` (`DenseAttnCache`), `qwen35::cache_mixed` (`MixedAttnPlane`) | **Done, verify-only per issue scope** — confirmed architecture-generic: every entry point takes `n_kv_heads`/`head_dim`/`max_seq`/`v_bits`/`sink_len`/`window_len` as runtime arguments, nothing qwen3/qwen35-specific. A gemma loader gets this for free once it produces a `ModelConfig`-shaped set of dimensions |
| 16 | Quant-tensor-name policy audit hardcoded to one arch | `quant_policy.rs` | **Already generic** (landed in a prior round, self-documented: "issue #16: a `COMMON_RULES` table plus a `GDN_RULES` table gated on `ArchFamily::Qwen35Hybrid`") — a gemma family adds its own `ArchFamily` variant + rule table, no existing code touched |
| 17 | Architecture dispatch is an exact-string match, no fallback/plugin registry | `model.rs:35` (`match arch.as_str() { "qwen3" => ..., "qwen35" => ..., other => Err(...) }`), `budget/mixed.rs:105` (identical match), `registry/catalog.rs`'s `ModelFamily` enum (already anticipates this: "a future family (e.g. a Gemma variant) just adds a variant here") | **By design, not a bug** — this is exactly the seam the registry recipe (below) extends; a `match` arm per architecture is the intended shape, not something to "genericize away" |
| 18 | Dense prefill was token-serial (no chunked path) | `forward/mod.rs` (module doc, pre-this-round) | **Fixed this round** — see "Dense chunked prefill" below |
| 19 | Orphaned dead code from a superseded design | `rocml/src/hybrid/config.rs` — not declared as a `mod` anywhere in `lib.rs`, so it isn't compiled; superseded by `qwen35/config.rs` | **Noted, not touched** — pre-existing, out of this round's scope (not an architecture assumption, just leftover dead code); flagged for a future cleanup pass |

## Fixed this round: config-driven FFN activation

`ModelConfig` (`rocml/src/config.rs`) gained an `Activation` enum
(`SiLu`/`Gelu`) and a `pub activation: Activation` field, set from the
loader's own `ARCH` constant (GGUF carries no generic "activation function"
metadata key — llama.cpp itself infers this from `general.architecture` in
its C++ source, not a per-model field, confirmed by its absence from both
real gemma GGUF headers this audit pulled) rather than a bare
`kernels.silu_mul(...)` call. `forward/ffn.rs`'s decode-step FFN and
`forward/ffn_chunk.rs`'s batched chunked-prefill FFN both `match
config.activation` between `Kernels::silu_mul` and the new
`ActivationKernels::gelu_mul` (`forward/kernels_act.rs`, a new file —
`kernels.rs` is already well past the 400-line cap, and the workspace's hard
rule against growing a file already over that cap rules out adding a field
there the way `kernels_flash.rs`'s own `flash` field once was; `ActivationKernels`
is instead a sibling field on `forward::Model` itself, passed into
`ffn_step`/`ffn_chunk_step` as its own `act_kernels` parameter).

The GELU-gated kernel itself (`gelu_mul_f32` in `kernels/silu_mul.hip`, same
compiled code object as `silu_mul_f32`) implements the tanh approximation
(HF's `gelu_pytorch_tanh`, the activation gemma-2/gemma-3's FFN actually
declares — not exact erf-based GELU), unit-tested against an f64 CPU
reference (`rocml-kernels/tests/gelu_mul.rs`, 3 cases: aligned/unaligned/
degenerate sizes) — all pass. **Not reachable from any shipped architecture
yet** (`ModelConfig::from_gguf`'s only arm sets `Activation::SiLu`): this is
a landed seam, not a claim that gemma loads today.

Deliberately **not** attempted this round (per the issue's own
"don't speculatively implement SWA/softcap kernels" instruction): softcap
elementwise ops, a windowed-attention kernel variant, per-layer rope theta
plumbing, the optional Q/K-norm loader change, and the sandwich-norm forward
pass shape — all of these need either a real checkpoint to validate against
or touch multiple files/kernels together, unlike the activation seam, which
was a clean, narrow, already-isolated dispatch point.

## Gemma-specific gap list

Every key below is copy-pasted from the real GGUF headers (see "Method"
above), not from memory of the HF `config.json` shape.

### 1. Sliding-window attention (gemma-2 and gemma-3)

- **Config keys** (both present, real values shown): `gemma2.attention.
  sliding_window = 4096`; `gemma3.attention.sliding_window = 1024`.
- **Pattern**: gemma-2 alternates local/global attention every other layer
  (1:1); gemma-3 uses a 5-local:1-global pattern. **Neither GGUF exposes the
  interleave pattern as a metadata key** — no `sliding_window_pattern` (or
  similarly named) key exists in either real header dumped for this audit.
  llama.cpp hardcodes the ratio per architecture in its own C++ source, not
  in the GGUF — a gemma loader in this codebase would need to hardcode the
  same ratio (or derive it from `general.architecture` == `"gemma2"` vs.
  `"gemma3"`), the same way llama.cpp does.
- **Where the hook goes**: `attn_decode.hip`/`attn_prefill*.hip`'s causal
  bound is currently always `[0, cur_len)` (see `attn_decode.hip`'s own
  module doc: "attends to every one of the `cur_len` cached positions — no
  masking beyond the loop bound"). A windowed variant needs an additional
  `window: u32` kernel parameter clamping the lower bound to
  `max(0, cur_len - window)`, threaded through `Kernels::attn_decode`/
  `attn_prefill`/the flash variants, and a per-layer "is this a local or
  global layer" bit in `ModelConfig` (mirroring `HybridConfig::
  is_full_attention`'s existing per-layer-kind pattern).

### 2. Attention/final logit softcapping (gemma-2 only)

- **Config keys** (gemma-2-9b real values): `gemma2.attn_logit_softcapping
  = 50.0`; `gemma2.final_logit_softcapping = 30.0`.
- **Confirmed absent from gemma-3**: neither key appears in the real
  gemma-3-4b header — gemma-3 dropped softcapping entirely (matches public
  documentation; verified here against an actual file rather than trusted
  on faith).
- **Where the hook goes**: two independent elementwise ops, `x <-
  softcap * tanh(x / softcap)` — one applied to raw attention logits before
  the causal softmax inside `attn_decode.hip`/`attn_prefill*.hip` (needs a
  `softcap: f32` kernel parameter, `0.0` meaning "disabled" so the qwen3/
  qwen35 kernels stay a no-op-equivalent zero-cost path), one applied to the
  final lm-head logits in `forward/mod.rs`'s/`forward/chunk_forward.rs`'s
  lm-head scope (a plain elementwise kernel over the `[vocab]` output, no
  new kernel infra needed — could reuse the `elementwise.hip` file).

### 3. Per-layer rope theta (gemma-3 only)

- **Config key** (gemma-3-4b real value): `gemma3.rope.freq_base =
  1000000.0`. Also present: `gemma3.rope.scaling.type = "linear"`,
  `gemma3.rope.scaling.factor = 8.0` — a rope-scaling scheme this codebase's
  `HybridConfig`/`ModelConfig` have no field for at all (a separate gap from
  per-layer theta, called out here since it's in the same rope-config
  neighborhood).
- **No local-layer theta key exists in the GGUF.** Gemma-3's local
  (sliding-window) layers use a different, much smaller rope theta (10000,
  per public documentation) than the global layers' `rope.freq_base`
  (1000000) above — llama.cpp hardcodes the local value in source rather
  than reading it from metadata, exactly like the SWA pattern in gap #1.
  Confirmed by its absence: no second `rope.freq_base`-shaped key, no
  `rope.local_freq_base` or similar, anywhere in the real header.
  gemma-2 has no such split (`gemma2` has no `rope.freq_base` key at
  all — see gap #4 below — and no local/global rope distinction).
- **Where the hook goes**: `ModelConfig`/a gemma config would need two theta
  fields (or one plus a hardcoded local constant) and the per-layer
  local/global bit from gap #1, threaded to `Kernels::rope`'s existing
  `theta_base: f32` parameter — that parameter is already a per-call
  runtime argument (`forward/kernels.rs::rope`), so this is a config/
  call-site change, not a kernel change. Rope scaling (linear factor 8.0)
  would need its own inv_freq formula change inside `rope.hip` itself
  (currently `theta_base^(-2i/head_dim)` with no scaling term) — a real
  kernel change, unlike the theta split.

### 4. `rope.freq_base` is a required key in this codebase — gemma-2 has none

- **Real finding**: gemma-2-9b's GGUF header has **no** `gemma2.rope.
  freq_base` key at all (confirmed: all 33 KV pairs enumerated, none
  matches). llama.cpp defaults to 10000.0 for gemma2 when the key is
  absent, per its own architecture-specific hparam defaults.
- **Where the hook goes**: `ModelConfig::from_gguf` (`config.rs:71`) calls
  `gguf.get_f32("qwen3.rope.freq_base")?` — a bare `?` that hard-errors on a
  missing key. `rocml_core::gguf::GgufFile` has no `get_f32_or(key, default)`
  variant at all (checked: every `get_*` method returns `Result`, none has
  a default-value sibling) — a gemma-2 loader mirroring this exact pattern
  would fail to load. The fix is small (an `unwrap_or`-style fallback per
  architecture-appropriate default) but is a real, concrete trap for anyone
  copying this loader's shape verbatim for a new family whose GGUF omits a
  key qwen3's always has.

### 5. Q/K-norm presence (gemma-2 only)

- **Real finding**: gemma-2-9b's `blk.0.*` tensor list has **no**
  `attn_q_norm.weight`/`attn_k_norm.weight` tensors — confirmed by dumping
  every `blk.0.*` tensor name/shape/dtype from the real header (11 tensors
  total, none matching). gemma-3-4b's `blk.0.*` **does** have both
  (`attn_q_norm.weight`/`attn_k_norm.weight`, shape `[256]` = its
  `head_dim` — the exact same tensor name and per-head-shared-vector shape
  convention `weights/layer.rs` already uses for qwen3).
- **Where the hook goes**: `weights/layer.rs:51-52` calls
  `load_vector_f32(gguf, "{p}.attn_q_norm.weight", head_dim)?` unconditionally
  — `gguf.tensor(name)?` inside that helper hard-errors if the tensor is
  absent. A gemma-family loader needs this to be optional (`Option<DeviceBuffer
  <f32>>`, `None` skipping the per-head rmsnorm call in `attention.rs`/
  `attention_chunk.rs` entirely) to load gemma-2 at all; gemma-3 would work
  with the tensor made present-but-optional (same tensor name, same shape
  convention already used).

### 6. Sandwich norm — two extra per-layer norms (both gemma-2 and gemma-3)

**The largest structural gap found in this audit**, not called out by name
in the issue's own suspect list but found by actually reading the tensor
table: both real GGUFs' `blk.0.*` tensors include `post_attention_norm.
weight` and `post_ffw_norm.weight` **in addition to** the `attn_norm.
weight`/`ffn_norm.weight` pair qwen3 has. This is gemma-2's "sandwich norm"
architecture (introduced in gemma-2, kept in gemma-3): each sub-layer is
`x + post_norm(sublayer(pre_norm(x)))`, not qwen3's `x + sublayer(pre_norm
(x))` — an extra RMSNorm applied to the sub-layer's *output*, before the
residual add, on both the attention and the FFN sub-layers.

- **Where the hook goes**: `forward/attention.rs`'s `AttnOut` scope currently
  goes straight from `layer.attn_output.matvec(...)` to `kernels.add_inplace
  (x, attn_out, hidden)` — no norm in between. Same shape in `forward/
  attention_chunk.rs` (`.matmul` variant) and both FFN files (`forward/
  ffn.rs`/`forward/ffn_chunk.rs`, `down`-projection scope). A gemma layer
  needs an extra optional `post_attn_norm`/`post_ffn_norm` pair of weight
  buffers (`LayerWeights`-shaped, `Option`) and one more `rmsnorm` call
  right before each `add_inplace`, on all four call sites (decode + chunked,
  attention + FFN).
- **Why this is called out separately from the "audit table" line item**:
  it isn't a hardcoded constant or a missing config field the way the other
  gaps are — it's a genuine difference in the forward pass's *data flow
  shape* (an extra op between projection and residual-add), the kind of
  change that can't be a config knob alone. Confirmed structural, not
  guessed: found by reading the real tensor table, not by recalling gemma's
  paper.

### 7. Embedding scale (both gemma-2 and gemma-3)

- Gemma multiplies the embedding lookup output by `sqrt(hidden_size)`
  before the first layer (a well-documented gemma quirk; llama.cpp applies
  it as a per-architecture hardcoded multiplier, not a GGUF metadata value —
  consistent with this audit's pattern of finding several gemma behaviors
  baked into llama.cpp's C++ source rather than exposed in the GGUF header
  at all). No corresponding key exists in either real header (checked;
  `embedding_length` is the raw hidden size, not a scale factor).
- **Where the hook goes**: `forward/kernels.rs::embedding`'s host wrapper
  (or a config-gated post-scale elementwise multiply right after the
  `Embed` `Profiler::scope` in `forward/mod.rs`/`chunk_forward.rs`) — a
  one-line change once a `ModelConfig::embedding_scale: Option<f32>`-shaped
  field exists, no new kernel needed (`Kernels` has no generic "scale a
  buffer by a scalar" op today, but it's a two-line addition to
  `elementwise.hip` if one doesn't already fit — cheapest gap on this list).

### 8. Attention scale beyond `1/sqrt(head_dim)`

- gemma-2-9b's real HF `config.json` (not the GGUF — this field isn't in
  the GGUF header at all, confirmed) sets `query_pre_attn_scalar = 224`,
  which is `hidden_size(3584) / num_attention_heads(16)`, **not**
  `head_dim(256)` — the two diverge for gemma-2-9b specifically (they
  coincide for most other gemma sizes and for qwen3, which is presumably
  why this was never noticed as a "hidden" assumption until checked
  directly against a 9B checkpoint). Since this value isn't in the GGUF at
  all, llama.cpp must hardcode gemma-2's scale per-architecture too, and
  this audit did not chase llama.cpp's own source to confirm the exact
  hardcoded formula it uses instead of `query_pre_attn_scalar` — flagged as
  an open question a real gemma-2 checkpoint-selection round needs to
  resolve (compare against llama.cpp's `llm_build_gemma2` scale computation
  directly), not asserted as solved or broken here.
- **Where the hook goes**: every attention kernel already takes `scale: f32`
  as a plain runtime argument (`forward/attention.rs:160` computes it
  in Rust, not in the kernel) — this is the cheapest gap on the list to
  actually wire up once the right formula is confirmed: add a `ModelConfig::
  attn_scale: f32` field (defaulting to `1/sqrt(head_dim)` for qwen3) and
  read it instead of recomputing `1.0f32/(head_dim as f32).sqrt()` at each
  of the four call sites.

## Registry recipe: adding a new dense family

See `.claude/CLAUDE.md`'s "Adding a new dense model family" section for the
exact steps (kept there, not duplicated here, since it's a living
maintenance doc future rounds should update in place). In short: parse a
new config struct from the GGUF (mirroring `config.rs`'s pattern — every
field read from the file, explicit errors on missing/malformed keys, no
silent defaults for anything qwen3-specific), pick a forward path (reuse
`forward::Model` if the new family fits its per-layer shape, e.g. any dense
all-full-attention transformer with SwiGLU/GeGLU FFN and no partial rope;
write a new one otherwise), add a `general.architecture` match arm in
`model.rs`/`budget/mixed.rs`, and add a `registry::catalog::ModelFamily`
variant plus a `ModelSpec` entry. Every seam this needs (`AttnLayerCache`/
`MixedAttnPlane` for KV cache, `quant_policy::ArchFamily`, `LinearWeight`
for weight loading, `Kernels`/`ChunkKernels` for compute) is already
architecture-generic per this audit's findings above — a new family's own
config/loader is genuinely the only new code needed for a model that fits
the dense forward path's shape.

## Dense chunked prefill (issue #16)

**Landed, not just planned.** The dense (`qwen3`) architecture's prompt
phase was token-serial-only before this round (`Model::forward_prompt`'s
`Dense` arm looped `forward_token_profiled` once per prompt token,
identical in shape to decode). It now goes through a batched chunked-prefill
path mirroring the qwen35 hybrid's own design (`.claude/CLAUDE.md`'s
"Chunked prefill" section), simplified for the dense architecture's
narrower shape:

- **New files**: `forward/chunk_scratch.rs` (per-chunk scratch buffers,
  `CHUNK_CAP`=512 tokens — no GDN fields, unlike the hybrid path's
  `ChunkScratch`), `forward/attention_chunk.rs` (batched QKV projection,
  per-head norm, full — not partial — rope, batched KV-cache append, causal
  attention with the same shallow/deep flash-vs-single-pass dispatch the
  hybrid path uses, `AttnLayerCache::{Dense,Mixed}` dispatch reused as-is
  from the decode step's own `attention.rs`), `forward/ffn_chunk.rs`
  (batched SwiGLU/GeGLU FFN), `forward/chunk_forward.rs`
  (`forward_chunk`/`forward_prompt_chunked`, `PREFILL_CHUNK_SIZE`=512).
- **Why this was a clean port, not a large one**: every kernel the hybrid
  path's chunked prefill uses (`gemm_xwt_*`/`LinearWeight::matmul`,
  `attn_prefill*`/the flash-tiled split-K design, `Kernels::rope`'s existing
  `tokens` batching parameter, `ChunkKernels::scatter_kv_chunk{,_f16}`,
  `MixedAttnPlane::append_chunk`, `FlashPrefillMixedKernels`) is already
  architecture-generic — confirmed by reading each one before writing any
  new code, not assumed. The dense architecture's own shape is *simpler*
  than the hybrid's chunked path in two ways that reduced the port further:
  no GDN layers (every layer takes the identical batched full-attention
  step, no per-layer-kind dispatch), and no fused Q+output-gate projection
  (a dense `attn_q` projection's output is already `[chunk_len, n_heads,
  head_dim]`, so no `extract_heads` call was needed at all, unlike the
  hybrid path's `attention_chunk.rs`). `ChunkKernels`/`MixedKernels`/
  `FlashPrefillMixedKernels` are reused directly from `qwen35::forward` (not
  duplicated) — they were already architecture-generic per this same
  audit's KV-cache finding above.
- **Wiring**: `crate::model::Model::forward_prompt`'s `Dense` arm now calls
  `forward::Model::forward_prompt_chunked` instead of looping
  `forward_token_profiled`. `LoadOptions::use_mmq` now genuinely reaches the
  dense architecture too (previously documented as "inert" there, since the
  dense path never called `matmul` before this round) — left at its
  existing default-`false` (never separately re-validated for the dense
  path's activation shapes, which lack the qwen35 hybrid's own documented
  MMQ failure mode source but also were never checked to be free of it).
- **New correctness gate**: `rocml/tests/dense_chunked_prefill_parity.rs`,
  two tests — `qwen3_0_6b_chunked_prefill_matches_token_serial` (fp32 KV,
  prompt lengths `{1,4,127,128,129,500,512,513,1024,2048}` spanning the
  512-token chunk boundary, element-wise final-logits closeness at `1e-2`
  relative tolerance plus exact-or-near-tie greedy continuation — mirrors
  `qwen35_chunked_prefill_parity.rs` exactly) and
  `qwen3_0_6b_chunked_prefill_matches_token_serial_mixed_kv` (both `Q8` and
  `Q4Mixed` KV cache modes, prompt lengths spanning the sink/window
  boundaries `{32,128,160,500,2048}`). The mixed-KV variant checks argmax-
  match-or-near-tie rather than raw logit magnitude, per
  `dense_mixed_kv_parity.rs`'s own already-established finding that
  Qwen3-0.6B's 26 consecutive mixed layers (vs. the qwen35 hybrid's 4)
  compound enough quantization noise that individual small-magnitude
  logits can show a large relative deviation despite the argmax decision
  itself being stable — confirmed by a first attempt at this test using a
  raw element-wise tolerance, which failed at `len=500` with a real 33%
  relative deviation on one logit even though the argmax/greedy-continuation
  check (below) passes cleanly. Both tests pass.
- **Measured** (Qwen3-0.6B-Q8_0, `bench --depth 2048`, median of 3 runs):
  prefill 113 -> 3216.6 tok/s (**~28.5x**), decode 106.8 tok/s (flat vs. the
  pre-existing baseline — the decode path is untouched by this port). The
  dense architecture's much larger jump than the qwen35 hybrid path's own
  chunked-prefill wins (a few-fold, not ~28x) reflects how *un*-optimized
  the token-serial baseline was to begin with (one kernel launch per token
  per op, no batching whatsoever) rather than the chunked path itself being
  unusually fast for its shape.
- **Not attempted this round**: the multi-round perf-tuning pass the qwen35
  hybrid path's chunked prefill received (WMMA pipelining sweeps, split-K,
  per-shape dispatch tuning, flash-prefill row-tile/split-K sweeps) — this
  port reuses those kernels' existing tuning as-is. A dedicated tuning round
  for the dense architecture's own shapes (Qwen3-family `hidden`/
  `feed_forward_length` ratios differ from Ornith/Qwen3.5's) is a real,
  separate follow-up if dense-architecture prefill throughput becomes a
  priority — flagged, not estimated in detail here since it would repeat
  the hybrid path's own multi-round measurement process rather than being a
  fixed-size task.

# Weight-quant sensitivity audit and load-time policy (issue #4)

Design-review concern: quantization error in the Gated Delta Net (GDN)
scan-recurrence path compounds over the whole sequence, so the tensors
feeding it deserve a higher precision floor than a generic per-tensor K-quant
scheme gives them. This audits what the registry's three flagship GGUFs
*actually* assign, reconciles that against issue #15's outcome-level eval,
and documents the load-time policy check (`rocml::quant_policy`) that turns
this from a one-time audit into an ongoing informational warning.

**Tooling**: `cargo run -p rocml-core --example inspect_gguf -- <path.gguf>
--full` dumps every tensor's name/shape/dtype (added this round; the plain
`inspect_gguf <path.gguf>` behavior from issue #8 is unchanged). All tables
below come straight from that dump against the checkpoints under
`$ROCML_CHECKPOINT_DIR` (`~/checkpoints` on this machine).

## The policy, restated precisely

The issue's own wording overlaps two different things under the names
"alpha"/"beta": the raw scan-recurrence scalars (decay/write-strength) and
the 2D linear layers that *produce* per-token alpha/beta logits from the
hidden state. In this GGUF layout (llama.cpp's `qwen35` tensor names) they
are genuinely different tensors:

| Issue's term | Actual GGUF tensor | Shape | Role |
|---|---|---|---|
| `A_log` | `blk.N.ssm_a` | 1D `[num_v_heads]` | raw scan decay scalar |
| `dt_bias` | `blk.N.ssm_dt.bias` | 1D `[num_v_heads]` | raw scan bias scalar |
| "alpha" | `blk.N.ssm_alpha.weight` | 2D `[hidden, num_v_heads]` | in-projection producing the per-token decay logit |
| "beta" | `blk.N.ssm_beta.weight` | 2D `[hidden, num_v_heads]` | in-projection producing the per-token write-strength logit |
| short-conv weight | `blk.N.ssm_conv1d.weight` | 2D `[conv_kernel, conv_dim]` | depthwise causal conv taps |
| gate projection | `blk.N.attn_gate.weight` | 2D `[hidden, value_dim]` | output SiLU-gate (`Z`) in-projection |
| fused QKV | `blk.N.attn_qkv.weight` | 2D `[hidden, conv_dim]` | feeds conv1d + K/V into the recurrence |
| GDN's `o_proj`/`down_proj` analog | `blk.N.ssm_out.weight` | 2D `[value_dim, hidden]` | shapes the residual stream |

This audit therefore checks two separate things against two separate
policy classes, rather than forcing everything under "must be float":

- **Must stay float** (`ssm_a`, `ssm_dt.bias`, `ssm_conv1d.weight`, and —
  llama.cpp's own blanket convention — every 1D norm weight): these are the
  literal scan state and the small per-channel conv taps, never worth the
  VRAM/bandwidth win of quantizing.
- **>= Q6 (GDN 2D projections, generic converter treatment)**: `ssm_alpha.weight`,
  `ssm_beta.weight`, `attn_qkv.weight`, `attn_gate.weight` — these produce
  the recurrence's per-token inputs but get no special handling from
  llama.cpp's GGUF converter, so they're quantized exactly like any other
  linear layer in whatever scheme the file uses.
- **One step higher than the rest** (`attn_output.weight` / `ffn_down.weight`
  / `ssm_out.weight` — everything shaping the residual stream): modeled as
  an absolute Q5_K floor in the load-time checker below (see that section
  for why absolute, not scheme-relative).
- **Lowest precision tolerable** (`ffn_gate.weight`/`ffn_up.weight`): no
  floor enforced.

## Per-model audit

Ornith-1.0-9B: `qwen35.block_count = 32`, `full_attention_interval = 4`
(full-attention layers at block indices 3/7/11/15/19/23/27/31, 8 of 32; the
rest are GDN), `num_v_heads = 32`, `num_k_heads = 16` (grouped GDN heads).
Qwen3.5-2B: `block_count = 24`, same `full_attention_interval = 4` (6 of 24
full-attention), `num_v_heads = num_k_heads = 16` (no grouping).

### `ornith-1.0-9b-Q4_K_M.gguf` (registry default `ornith-9b`)

| Category | Tensor(s) | Dtype found | Verdict |
|---|---|---|---|
| Must stay float | `ssm_a`, `ssm_dt.bias`, `ssm_conv1d.weight`, all `*_norm.weight` | **F32** (every layer) | **Compliant** |
| >= Q6, GDN 2D in-projections | `ssm_alpha.weight`, `ssm_beta.weight`, `attn_gate.weight` | **Q4_K** (all 24 GDN layers) | **Violation** |
| >= Q6, GDN 2D in-projections | `attn_qkv.weight` | mixed **Q4_K/Q6_K** per layer (12 Q4_K, 12 Q6_K — the scheme's own importance heuristic, not GDN-aware) | **Partial violation** |
| >= Q6, embeddings/lm_head | `token_embd.weight` | **Q4_K** | **Violation** |
| >= Q6, embeddings/lm_head | `output.weight` | **Q6_K** | Compliant |
| One step higher, `o_proj`/`down_proj` | `attn_output.weight` (full-attn layers) | **Q4_K** uniformly (all 8) | **Violation** (not bumped at all) |
| One step higher, `o_proj`/`down_proj` | `ssm_out.weight` (GDN layers) | **Q4_K** uniformly (all 24) | **Violation** (not bumped at all) |
| One step higher, `o_proj`/`down_proj` | `ffn_down.weight` | mixed **Q4_K/Q6_K** per layer (16 Q4_K, 16 Q6_K) | **Partial compliance** (this one the scheme does bump, just not GDN-aware about which layers) |
| Lowest tolerable | `ffn_gate.weight`, `ffn_up.weight` | **Q4_K** uniformly | Compliant (no floor expected) |

**Reconciliation with the issue-15 eval**: the eval (agentic tool-use score
19/20, PPL 1.0807) already promoted this exact file to the registry default
over Q6_K, on measured outcomes. Every violation above is real and is
exactly what a generic, GDN-unaware K-quant converter produces — but the
eval is outcome-level ground truth and it says this file works. **The eval
wins**: this is not "the checkpoint is broken", it's "the design-review
prior turned out to be more conservative than the model needs at this
scale". Kept as a documented, monitored tradeoff, not fixed by re-quanting.

### `ornith-1.0-9b-Q6_K.gguf` (registry `ornith-9b-q6`)

| Category | Dtype found | Verdict |
|---|---|---|
| Must stay float (`ssm_a`/`ssm_dt.bias`/`ssm_conv1d.weight`/norms) | **F32** | Compliant |
| Everything else (uniform scheme) | **Q6_K** | Compliant with every floor (>= Q6 and >= Q5 both trivially satisfied) |

Zero violations — a pure single-quant-level scheme has no "some tensors are
generic-treated worse than others" problem by construction. This is the
`make test-model`/`ornith_e2e` and issue-15's fp16-KV baseline checkpoint.

### `Qwen3.5-2B-Q8_0.gguf` (registry `qwen3.5-2b`)

| Category | Dtype found | Verdict |
|---|---|---|
| Must stay float | **F32** | Compliant |
| Everything else (uniform scheme) | **Q8_0** | Compliant with every floor |

Zero violations, same reasoning as the Q6_K file above — Q8_0 (~8.5
bits/weight) clears every floor this policy defines.

## Load-time policy check

`rocml::quant_policy` (`rocml/src/quant_policy.rs`) turns the table above
into a reusable, architecture-generic check, called from both weight
loaders' entry points (`rocml/src/weights/mod.rs::ModelWeights::load` and
`rocml/src/qwen35/weights/mod.rs::ModelWeights::load`) right before any
tensor is actually loaded:

- `PrecisionClass` (`FloatOnly`/`Q6OrBetter`/`Q5OrBetter`/`Any`) compared
  against a coarse `precision_rank(dtype)` ordering (F32 > F16/BF16 > Q8_0 >
  Q6_K > Q5_K > Q4_K > Q3_K > Q2_K, `Unsupported` ranked below everything) —
  this is what lets ">= Q6" also accept Q8_0/F16/F32 rather than an exact
  quant-type match.
- Two rule tables, matched by tensor-name suffix (or exact name for the two
  top-level tensors, `token_embd.weight`/`output.weight` — see the module's
  own doc comment for why suffix alone would misclassify `attn_output.weight`
  as the lm_head): `COMMON_RULES` (norms, embeddings/lm_head, `attn_output`/
  `ffn_down`/`ffn_gate`/`ffn_up` — every architecture) and `GDN_RULES`
  (`ssm_*`/`attn_qkv`/`attn_gate` — only consulted for
  `ArchFamily::Qwen35Hybrid`). A future architecture family adds its own
  table instead of touching either.
- `audit(gguf, family) -> PolicyReport` is pure and infallible (no I/O
  beyond the already-open `GgufFile`'s in-memory tensor table); `checked`
  counts tensors any rule matched, `violations` lists every tensor whose
  dtype fell below its rule's floor.
- `PolicyReport::warn_violations()` prints one `warning: quant-policy: ...`
  line **per distinct violated label**, not per tensor — a Q4_K_M
  checkpoint violates the same GDN-in-projection rule once per layer, and a
  wall of ~24 near-identical lines at every startup would bury the signal.
  Each line names the count and one example tensor, in the same
  `warning: ...` eprintln convention `registry::clamp_ctx`/`forward::Model::load`'s
  VRAM-budget warning already use. **Never fails the load** — this is
  informational, existing checkpoints (including the registry default,
  which does violate several rules) must keep loading unchanged.

Design choice worth calling out: "one step higher than the rest" is
naturally *scheme-relative* (llama.cpp's real Q4_K_M convention bumps
`ffn_down`/`attn_v` for a fraction of layers based on the file's own
baseline, not an absolute quant level). The checker instead uses an
absolute Q5_K floor for `Q5OrBetter` — simpler to check at load time
without knowing "what's the baseline this file chose", and it already
catches the case that matters here: `attn_output.weight`/`ssm_out.weight`
quantized all the way down to the scheme's own floor with no bump at all
(exactly what the Q4_K_M audit above found).

Real-checkpoint pin: `rocml/tests/quant_policy_audit.rs` asserts the exact
findings above against the live registry files (skip-if-checkpoint-missing,
no GPU needed — pure GGUF header/tensor-table reads) so a future re-quant or
re-upload that silently changes the mix gets caught here, not just in this
prose. Pattern-matching unit tests live inline in `quant_policy.rs`.

## Provenance

The registry (`rocml/src/registry/catalog.rs`) resolves:

| Registry name | `hf_repo` | `hf_file` | Local file | Size match | sha256 match |
|---|---|---|---|---|---|
| `ornith-9b` | `ornith-ai/Ornith-1.0-9B-GGUF` | `ornith-1.0-9b-Q4_K_M.gguf` | 5,629,108,704 B | Yes | Yes (`5720d1f6…6087b106`) |
| `ornith-9b-q6` | `ornith-ai/Ornith-1.0-9B-GGUF` | `ornith-1.0-9b-Q6_K.gguf` | 7,359,259,072 B | Yes | Yes (`33b6f6a3…026e8387`) |
| `qwen3.5-2b` | `unsloth/Qwen3.5-2B-GGUF` | `Qwen3.5-2B-Q8_0.gguf` | 2,012,012,800 B | Yes | Yes (`1b04acba…1021f2c1`) |

Verified via the HF API (metadata/tree endpoints only — no re-download of
any multi-GB file; sha256 was computed locally against the file already on
disk, not fetched over the network):

- **`deepreinforce-ai/Ornith-1.0-9B-GGUF` is not a second, unofficial repo**:
  `GET https://huggingface.co/deepreinforce-ai/Ornith-1.0-9B-GGUF` returns
  HTTP 307, redirecting to `ornith-ai/Ornith-1.0-9B-GGUF` — a genuine
  Hugging Face org rename (also visible in the live repo's own `cardData`,
  which still links its license at the old `deepreinforce-ai/...` path).
  There is exactly one canonical repo; the registry's own code comment
  ("the org was renamed, not an unofficial re-upload") is confirmed, not
  just asserted.
- **File identity**: `GET /api/models/ornith-ai/Ornith-1.0-9B-GGUF/tree/main?recursive=true`
  reports `ornith-1.0-9b-Q4_K_M.gguf` at 5,629,108,704 bytes (LFS oid
  `5720d1f671b4996481274fffe01868c3c36e87c135cc8538471cc7bd6087b106`) and
  `ornith-1.0-9b-Q6_K.gguf` at 7,359,259,072 bytes (LFS oid
  `33b6f6a3e3f05078438e12df8a4b55c8acf78ceadcc639d2af1cf35a026e8387`) — both
  byte-size and sha256 match the local files under
  `$ROCML_CHECKPOINT_DIR/Ornith-1.0-9B-GGUF/` exactly (`sha256sum` run
  locally against the already-downloaded files, no network transfer of the
  GGUF content itself).
- **`unsloth/Qwen3.5-2B-GGUF`**: repo exists, author `unsloth`, no rename
  redirect involved. `Qwen3.5-2B-Q8_0.gguf` is 2,012,012,800 bytes (LFS oid
  `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1`),
  matching the local file exactly. Confirms the registry's existing
  comment that no official Qwen-org GGUF exists for this size (`Qwen/Qwen3.5-2B`
  only ships safetensors) — `unsloth`'s conversion is the only GGUF source,
  same as before this round.
- Both repos are live, public, non-gated, with substantial download/like
  counts (Ornith GGUF repo: ~3.7M downloads, 667 likes at audit time) —
  nothing here reads as an obscure or suspicious mirror.

The registry pins no explicit revision (`hf download <repo> <file>` always
resolves the `main` branch's current commit — `ornith-ai/Ornith-1.0-9B-GGUF`
was at commit `3296bc7a404871a72ac3f1903f561459c09b5c17` at audit time). This
predates issue #4 and isn't changed by it; noted here only so a future
"why did this file change under us" question has the commit this audit was
run against.

## Bottom line

No prominent, unaddressed risk finding here: the one place the audit found
a scan-recurrence-adjacent tensor below the "must stay float" floor is
nowhere — `ssm_a`/`ssm_dt.bias`/`ssm_conv1d.weight` are F32 in all three
files, every quant level. The real finding is the documented Q4_K_M
GDN-in-projection/`ssm_out`/`attn_output`/`token_embd` gap against the
">= Q6"/"one step higher" design-review priors — already measured
outcome-neutral by issue #15's eval, and now visible at every load via
`quant_policy::PolicyReport::warn_violations()` instead of only in this
file. No registry or re-quant change is recommended by this round; that
decision (if ever revisited) belongs to whoever owns the tradeoff, informed
by this audit rather than blocked on it.

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
Ornith-1.5-9B: `qwen35.block_count = 33`/`nextn_predict_layers = 1` — one
trailing MTP draft block (`blk.32`) that `Qwen35Config::from_gguf` excludes
from `block_count`, so its 32 forward-pass layers have the identical
`full_attention_interval = 4`/`num_v_heads`/`num_k_heads` layout as 1.0's.

### `ornith-1.0-9b-Q4_K_M.gguf` (registry `ornith-1.0-9b`)

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
19/20, PPL 1.0807) already promoted this exact file to be Ornith-1.0-9B's
own default quant (`ornith-1.0-9b`, originally named `ornith-9b` before that
name was repurposed as an alias for Ornith-1.5-9B's Q6_K — see the registry
catalog's own comment) over Q6_K, on measured outcomes. Every violation
above is real and is
exactly what a generic, GDN-unaware K-quant converter produces — but the
eval is outcome-level ground truth and it says this file works. **The eval
wins**: this is not "the checkpoint is broken", it's "the design-review
prior turned out to be more conservative than the model needs at this
scale". Kept as a documented, monitored tradeoff, not fixed by re-quanting.

### `ornith-1.0-9b-Q6_K.gguf` (registry `ornith-1.0-9b-q6`)

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

### `Ornith-1.5-9B-Q4_K_M.gguf` / `Ornith-1.5-9B-Q6_K.gguf` (registry `ornith-1.5-9b` / `ornith-1.5-9b-q6`)

Per-category dtypes across the 32 forward-pass layers (`blk.0`-`blk.31`) are
identical to the matching 1.0 files above: Q4_K_M reproduces the exact same
GDN-in-projection/`ssm_out`/`token_embd` Q4_K findings (`ssm_alpha`/
`ssm_beta`/`attn_gate`/`ssm_out` 24 Q4_K each; `attn_qkv` 12 Q4_K/12 Q6_K;
`ffn_down` 16 Q4_K/16 Q6_K; `token_embd.weight` Q4_K; `output.weight`
Q6_K), and Q6_K is uniform Q6_K/F32 throughout with zero violations —
confirmed directly against both files' tensor tables
(`inspect_gguf --full`), not assumed from the file names.

`audit()` iterates every tensor in the GGUF (`gguf.tensors()`), not just the
forward pass's own `block_count` layers, so it also walks `blk.32` — the
trailing MTP draft block `Qwen35Config::from_gguf` excludes from
`block_count` (see "MTP speculative decoding" in `.claude/CLAUDE.md`) and
whose tensors are therefore never uploaded to the GPU. `blk.32` is itself an
ordinary full-attention+FFN layer (separate `attn_q`/`attn_k`/`attn_v`, no
`ssm_*`/`attn_qkv`/`attn_gate` at all, so a `GDN_RULES` match is
structurally impossible for it) that llama.cpp's own per-tensor importance
heuristic quantized the same way as any other layer: `attn_output`/
`attn_q`/`ffn_gate`/`ffn_up` Q4_K, `attn_v`/`ffn_down` Q6_K in the Q4_K_M
file; uniform Q6_K in the Q6_K file. That gives Q4_K_M exactly one extra
`COMMON_RULES` violation over 1.0's own count — `o_proj`
(residual-stream output projection) rises from 8 to **9** tensors
(`blk.32.attn_output.weight = Q4_K`, confirmed via the real
`warning: quant-policy: ...` line at load time) — every other violation
count unchanged (`ffn_down`'s own 16-tensor count doesn't move since
`blk.32.ffn_down.weight` is Q6_K, compliant). On Q6_K, `blk.32` is uniformly
Q6_K/F32, adding zero violations, so `report.is_clean()` still holds.
This is a correct, documented byproduct of `audit()` having no MTP-block
concept — not a bug, and not worth teaching it to skip `blk.32`: a
converter-quality question ("did anything in this file drop below the
floor") should cover every tensor the file actually ships, whether or not
this engine's own forward pass happens to upload it.

**Reconciliation with the issue-15 eval**: `make eval`'s Ornith-1.5-9B runs
(`bench/eval/results/ornith15-{q6k,q4km}-fp16kv.json`) score both
`ornith-1.5-9b-q6` and `ornith-1.5-9b` (Q4_K_M) at 19/20 — identical to
1.0's own pass count — with PPL 1.0918 (Q6_K) vs 1.1183 (Q4_K_M), a larger
Q6_K-to-Q4_K_M PPL gap than 1.0's own (1.0800 vs 1.0807) but still not
enough to move the agentic pass count at all. Both 1.5 configs fail the
identical single scenario 1.0 already fails (`two_city_weather`, "expected
exactly 1 tool call, got 2") — a pre-existing model-family quirk, not
something this round's extra `o_proj` violation introduced. See
`.claude/CLAUDE.md`'s "Agentic eval" section for the full four-checkpoint
comparison table. Same conclusion as 1.0's own reconciliation above: the
design-review prior is more conservative than the model needs at this
scale, and this file is kept as a documented, monitored tradeoff rather
than fixed by re-quanting.

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
  informational, existing checkpoints (including `ornith-1.0-9b`, which
  does violate several rules) must keep loading unchanged.

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
| `ornith-1.0-9b` | `ornith-ai/Ornith-1.0-9B-GGUF` | `ornith-1.0-9b-Q4_K_M.gguf` | 5,629,108,704 B | Yes | Yes (`5720d1f6…6087b106`) |
| `ornith-1.0-9b-q6` | `ornith-ai/Ornith-1.0-9B-GGUF` | `ornith-1.0-9b-Q6_K.gguf` | 7,359,259,072 B | Yes | Yes (`33b6f6a3…026e8387`) |
| `ornith-1.5-9b` | `ornith-ai/Ornith-1.5-9B-GGUF` | `Ornith-1.5-9B-Q4_K_M.gguf` | 5,780,090,816 B | Yes | Yes (`70c11219…07e8fab6`) |
| `ornith-1.5-9b-q6` | `ornith-ai/Ornith-1.5-9B-GGUF` | `Ornith-1.5-9B-Q6_K.gguf` | 7,558,901,696 B | Yes | Yes (`b6f76e74…81e4154a`) |
| `ornith-9b` (alias of `ornith-1.5-9b-q6`, the default model — see README's Model registry section) | `ornith-ai/Ornith-1.5-9B-GGUF` | `Ornith-1.5-9B-Q6_K.gguf` | 7,558,901,696 B | Yes | Yes (`b6f76e74…81e4154a`) |
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
- **`ornith-ai/Ornith-1.5-9B-GGUF` file identity**: `GET
  /api/models/ornith-ai/Ornith-1.5-9B-GGUF/tree/main?recursive=true` reports
  `Ornith-1.5-9B-Q4_K_M.gguf` at 5,780,090,816 bytes (LFS oid
  `70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6`) and
  `Ornith-1.5-9B-Q6_K.gguf` at 7,558,901,696 bytes (LFS oid
  `b6f76e74f86245b3caee014b797c10dca931c4dfdaabfb134eab655f81e4154a`) — both
  byte-size and sha256 match the local files under
  `$ROCML_CHECKPOINT_DIR/Ornith-1.5-9B-GGUF/` exactly (`sha256sum` run
  locally, no network transfer of the GGUF content itself). Repo is live,
  public, non-gated (5.1M downloads, 387 likes at audit time); no rename
  redirect involved for this one (unlike 1.0's `deepreinforce-ai` origin
  above) — `ornith-ai/Ornith-1.5-9B-GGUF` is the repo's only, original name.
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
run against. `ornith-ai/Ornith-1.5-9B-GGUF` was at commit
`abdd624b12ebf020b767fff532ff44fe552b28c3` (`GET
/api/models/ornith-ai/Ornith-1.5-9B-GGUF`'s `sha` field) when this section
was written, for the same reason.

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

## qwen35moe addendum (Ornith-1.5-35B-A3B, M1)

`ArchFamily::Qwen35Moe` reuses the identical `GDN_RULES` table (same hybrid
body) plus a new `MOE_RULES` table for the mixture-of-experts FFN: routed
experts' `ffn_gate_exps`/`ffn_up_exps` and the shared expert's
`ffn_{gate,up}_shexp` get `ffn_gate`/`ffn_up`'s "no floor" treatment;
`ffn_down_exps`/`ffn_down_shexp` get `ffn_down`'s `Q5OrBetter`
residual-stream-shaping floor; the router (`ffn_gate_inp`) and the shared
expert's own gate (`ffn_gate_inp_shexp`) get a `FloatOnly` floor — both are
F32 on the real Q4_K_M checkpoint (llama.cpp's own convention keeps routing
logits full-precision), so this documents an invariant the file already
holds rather than a finding. `rocml-cli generate --model ornith-1.5-35b`
prints the same violation warnings this audit's methodology predicts:
`token_embd`/GDN in-projections/`ssm_out` below their floors (identical
pattern to Ornith-1.0/1.5-9B's own Q4_K_M), plus `ffn_down_exps`/
`ffn_down_shexp` below `Q5OrBetter` on the layers llama.cpp's Q4_K_M
importance heuristic picked Q4_K over Q6_K for. Not re-audited against the
full per-model table above (no eval data for this checkpoint yet to
reconcile against, unlike the 9B entries) — see `.claude/CLAUDE.md`'s
"qwen35moe mixture-of-experts (M1)" section for the fuller writeup.

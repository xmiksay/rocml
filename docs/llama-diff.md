# llama.cpp per-layer reference diff (issue #10)

Issue #10 asked for the qwen35 per-layer diff harness (`qwen35::forward::
layer_capture`, built during the `mmq_precision` round — `.claude/CLAUDE.md`'s
MMQ section) to be pointed at an independent reference implementation
instead of just rocml's own MMQ-vs-WMMA comparison. This doc is that: a
reproducible pipeline that produces a llama.cpp CPU-forward-pass reference
dump, a converter that maps its ggml node names onto rocml's `LayerDump`
schema, `make llama-layer-diff`, and a recorded healthy-build baseline to
compare a future regression against.

llama.cpp is never vendored into this repo — it lives only in a scratch
clone. Everything below is written so that clone can be recreated from
scratch against the pinned commit.

## 1. Building `rocml-dump`

**Pinned llama.cpp commit**: `f3a33dff26f5d5ba8fbf47a26d4857c6edfe69a8`
(2026-09-12), the checkout this doc's mapping table and baseline were
verified against. `git log -1` in your scratch clone should match, or be
close enough that `src/models/qwen35.cpp`'s `cb(...)` call sites (Section 3)
haven't moved.

**qwen35moe update (M2)**: the `Ornith-1.5-35B-A3B-GGUF` reference dump
(`bench/eval/llama_ref/ornith-1.5-35b-ref.txt`) was produced from a newer
checkout, commit `0bec16e3880a148a7fc3887cdf71f81c547aff7a`, needed for that
build's qwen35moe (`build_moe_ffn`) graph support — later than the pinned
commit above, which predates qwen35moe entirely. The dense/hybrid mapping
table (Section 3) was unaffected (verified against this dump's own
GDN/full-attention node names, unchanged); Section 3's table gained six
MoE-only rows (`moe_shexp_swiglu`/`moe_shexp_down`/`moe_shared_gate`/
`moe_shared_gate_sigmoid`/`moe_shexp_gated`/`moe_routed_sum`), documented
inline in `convert.rs`'s `NODE_MAP`.

llama.cpp's own `llama-eval-callback` example (`examples/eval-callback`,
`common/debug.cpp`'s `common_debug_cb_eval`) prints per-node tensor stats,
but truncates every tensor to 3 elements per edge for human review
(`common_debug_print_tensor`'s hardcoded `n=3` — not a CLI flag) — useless
for an exact per-element diff. `rocml-dump` is a ~180-line sibling example,
built the same way, that dumps every element of every matched node in a
plain line format instead. It is not part of upstream llama.cpp; add it to
your scratch clone as `examples/rocml-dump/{rocml-dump.cpp,CMakeLists.txt}`
and register it in `examples/CMakeLists.txt` next to `add_subdirectory(
eval-callback)`.

`examples/rocml-dump/CMakeLists.txt`:

```cmake
set(TARGET rocml-dump)
add_executable(${TARGET} rocml-dump.cpp)
install(TARGETS ${TARGET} RUNTIME)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
```

`examples/rocml-dump/rocml-dump.cpp`:

```cpp
// rocml's llama.cpp reference-dump tool (issue #10's per-layer diff harness,
// see rocml's docs/llama-diff.md). Unlike llama-eval-callback's
// common_debug_cb_eval (which truncates each tensor to 3 elements per edge
// for human review, see common/debug.cpp), this dumps every element of
// every matched graph node in a simple line-oriented format so a Rust
// converter can do an exact per-element diff against rocml's own
// LayerCapture dumps.
//
// Not part of upstream llama.cpp - lives only in the scratchpad clone used
// to produce reference dumps; see docs/llama-diff.md for the exact commit
// this was built against and how to recreate this file.
//
// Env vars (kept out of the CLI arg surface, which common_params_parse
// already owns for -m/-p/-f/-ngl/--temp/etc.):
//   ROCML_DUMP_OUT     output file path (default: rocml_dump.txt)
//   ROCML_DUMP_FILTER  comma-separated list of regexes matched (anchored at
//                       the start, like common_debug_cb_user_data's own
//                       filter) against the ggml node's base name; empty
//                       (default) matches every node.
//
// Output format, one record per matched node:
//   #TENSOR <name> <ne0> <ne1> <ne2> <ne3>
//   <ne1*ne2*ne3 lines of ne0 space-separated floats, row-major innermost-first>

#include "arg.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

struct dump_ctx {
    std::ofstream           out;
    std::vector<std::regex> filters;
    std::vector<uint8_t>    staging;
};

static bool matches(const dump_ctx & ctx, const std::string & name) {
    if (ctx.filters.empty()) {
        return true;
    }
    for (const auto & re : ctx.filters) {
        if (std::regex_search(name, re)) {
            return true;
        }
    }
    return false;
}

static float get_float(const uint8_t * data, ggml_type type, size_t off) {
    if (type == GGML_TYPE_F32) {
        return *(const float *) (data + off);
    }
    if (type == GGML_TYPE_F16) {
        return ggml_fp16_to_fp32(*(const ggml_fp16_t *) (data + off));
    }
    if (type == GGML_TYPE_BF16) {
        return ggml_bf16_to_fp32(*(const ggml_bf16_t *) (data + off));
    }
    return 0.0f;
}

static bool rocml_dump_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * ctx = (dump_ctx *) user_data;

    if (ask) {
        return true;
    }
    if (ggml_is_quantized(t->type) || !matches(*ctx, t->name)) {
        return true;
    }

    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * data;
    if (is_host) {
        data = (const uint8_t *) t->data;
    } else {
        const size_t n_bytes = ggml_nbytes(t);
        ctx->staging.resize(n_bytes);
        ggml_backend_tensor_get(t, ctx->staging.data(), 0, n_bytes);
        data = ctx->staging.data();
    }

    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];

    ctx->out << "#TENSOR " << t->name << " " << ne0 << " " << ne1 << " " << ne2 << " " << ne3 << "\n";
    char buf[32];
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const size_t off = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                    const float  v   = get_float(data, t->type, off);
                    snprintf(buf, sizeof(buf), "%.9g", v);
                    ctx->out << buf;
                    if (i0 + 1 < ne0) {
                        ctx->out << ' ';
                    }
                }
                ctx->out << '\n';
            }
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    dump_ctx    ctx;
    const char *out_path = std::getenv("ROCML_DUMP_OUT");
    ctx.out.open(out_path ? out_path : "rocml_dump.txt");
    if (!ctx.out) {
        fprintf(stderr, "rocml-dump: failed to open output file '%s'\n", out_path ? out_path : "rocml_dump.txt");
        return 1;
    }
    if (const char * filt = std::getenv("ROCML_DUMP_FILTER")) {
        const std::string s(filt);
        size_t            pos = 0;
        while (pos <= s.size()) {
            const size_t      comma = s.find(',', pos);
            const std::string pat   = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
            if (!pat.empty()) {
                ctx.filters.emplace_back("^" + pat);
            }
            if (comma == std::string::npos) {
                break;
            }
            pos = comma + 1;
        }
    }

    params.cb_eval           = rocml_dump_cb;
    params.cb_eval_user_data = &ctx;
    params.warmup            = false;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    auto * model    = llama_init->model();
    auto * lctx     = llama_init->context();
    if (model == nullptr || lctx == nullptr) {
        fprintf(stderr, "rocml-dump: failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const bool           add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(lctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        fprintf(stderr, "rocml-dump: no input tokens (pass -p/-f)\n");
        return 1;
    }

    if (llama_decode(lctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        fprintf(stderr, "rocml-dump: llama_decode failed\n");
        return 1;
    }

    ctx.out.close();
    llama_backend_free();
    return 0;
}
```

Build (CPU backend — **preferred**: llama.cpp's GPU MMVQ activation
quantization path adds ~5e-3 noise of its own past ~layer 5, decorrelating
a deep-layer comparison chaotically; llama-CPU is the clean reference, per
the recipe issue #10's own history recorded before this harness existed):

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=OFF -DGGML_CUDA=OFF -DGGML_VULKAN=OFF \
  -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF
cmake --build build --target rocml-dump -j4
```

## 2. Producing a reference dump

The two sides must tokenize **exactly** the same prompt to the same token
IDs — rocml's `BpeTokenizer::encode` never prepends BOS, and Qwen-family
GGUFs carry no `tokenizer.ggml.add_bos_token` key, so llama.cpp's own
`add_bos` also comes out false; this was verified, not assumed (a probe run
confirmed `l_out-0`'s `ne1` == the prompt's raw token count, no off-by-one).

The pinned prompt: the first 64 tokens of `bench/eval/corpus.txt`, the same
corpus `mmq-layer-diff` uses (`rocml/tests/mmq_layer_diff.rs`'s
`PROMPT_LEN`=128 for its MMQ-vs-WMMA comparison; 64 here — see
`rocml/tests/llama_layer_diff.rs`'s module doc for why this harness needs a
smaller one: a llama.cpp *CPU* forward pass and the resulting
full-precision text dump both scale directly with it, and the filtered
node set below already produces a ~500MB-1.3GB dump at 64 tokens depending
on the checkpoint's hidden/intermediate/conv width).

Decode that exact token prefix back to text with rocml's own tokenizer
(round-trip verified: re-encoding the decoded text reproduces the identical
64 IDs) rather than guessing a word count, then feed that text to both
sides:

```sh
# rocml side: rocml/tests/llama_layer_diff.rs tokenizes
# bench/eval/corpus.txt itself and truncates to PROMPT_LEN=64 — no manual
# step needed there. This step is only to get the *same* text into
# llama.cpp's -f flag.

# llama.cpp side, from the scratch clone:
ROCML_DUMP_OUT=bench/eval/llama_ref/<checkpoint>-ref.txt \
ROCML_DUMP_FILTER="attn_residual,attn_post_norm,ffn_swiglu,ffn_out,l_out,attn_norm,linear_attn_qkv_mixed,final_output,linear_attn_out" \
./build/bin/rocml-dump -m <same .gguf rocml loads> -f <64-token prompt text> -ngl 0 -t $(nproc)
```

`ROCML_DUMP_FILTER`'s 9 patterns are exactly [`NODE_MAP`]/[`GDN_NODE_MAP`]'s
llama.cpp base names (Section 3) — this is what keeps the dump to "only the
nodes the converter can use" instead of every node in the graph (an
unfiltered dump on a real checkpoint runs into multiple GB and tens of
minutes; see Section 5's "friction" note).

`bench/eval/llama_ref/` is `.gitignore`d — dumps are diagnostic
intermediates, never committed; only Section 5's summary numbers are.

## 3. The node-name mapping table

Every entry below was confirmed against a real `rocml-dump` run on
Qwen3.5-2B-Q8_0 (layer 0, a GDN layer, and layer 3, a full-attention layer
— confirmed by the absence of `linear_attn_qkv_mixed-3`/`linear_attn_out-3`
in an unfiltered probe), not guessed from source alone. Implementation:
`rocml/tests/support/llama_ref/convert.rs`.

llama.cpp names every graph node via `cb(tensor, "name", il)`
(`src/models/qwen35.cpp`), which `llama_context::graph_get_cb()`
(`src/llama-context.cpp`) turns into the literal ggml tensor name
`"{name}-{il}"` for `il >= 0` (every layer-scoped node) or just `"{name}"`
for `il == -1` (model-level nodes like `result_output`).

**Applies to every layer** (`ffn_chunk_step` runs identically for GDN and
full-attention layers; llama.cpp's qwen35 graph builds the residual add and
the FFN the same way before branching back together):

| rocml `LayerCapture` key | llama.cpp node | why |
|---|---|---|
| `resid_pre_ffn` | `attn_residual` | `scratch.x` right after the attention/GDN residual add, before the post-attention norm |
| `ffn_xn` | `attn_post_norm` | `rmsnorm(x, post_attention_norm)` feeding the FFN gate/up projections |
| `ffn_gate_silu` | `ffn_swiglu` | `silu(gate) * up`, fused into one op by llama.cpp's `ggml_swiglu_split` (qwen35 uses a parallel, not sequential, SwiGLU gate) |
| `ffn_down_out` | `ffn_out` | ffn_down's raw projection output, before the FFN residual add — `llm_graph_context::build_ffn` only names this `"ffn_down"` when a down bias exists (qwen35 has none), so the caller's `"ffn_out"` name is the down-projection's raw output |
| `resid_post` | `l_out` | the full layer's output (post both residual adds); `l_out` is `post_ffn` run through `build_cvec`, a no-op without control vectors |

**GDN-layer-only** (`LayerKind::LinearAttention`, llama.cpp's
`build_layer_attn_linear`):

| rocml key | llama.cpp node | why |
|---|---|---|
| `gdn_xn` | `attn_norm` | `rmsnorm(x, attn_norm)` feeding the GDN QKVZ/alpha/beta projections — the *same* `attn_norm` node a full-attention layer's Q/K/V projections read from, since llama.cpp computes it once before branching on layer kind |
| `gdn_qkv_raw` | `linear_attn_qkv_mixed` | the raw QKV-mixed projection, before the causal conv1d |
| `gdn_y_silu` | `final_output` | the gated-RMSNorm output (`norm(o) * silu(z)`) feeding `ssm_out` |
| `gdn_ssm_out_raw` | `linear_attn_out` | `ssm_out`'s raw projection output, before the layer's residual add |

**Unmapped, not errors** (nodes real on one side only — `convert`'s
`UnmappedReport.unmapped_llama_nodes`, keyed per exact `"{name}-{il}"` not
consumed, not per base name, since e.g. `attn_norm-{il}` is consumed for a
GDN layer but not for a full-attention one): every full-attention-only
intermediate (`Qcur_full`, `Kcur`, `Vcur`, `gate_sigmoid`, `attn_pregate`,
`attn_gated`, `attn_output`, the RoPE/cache-view chain, ...) and every
GDN-internal intermediate rocml doesn't separately capture (`z`, `beta`,
`beta_sigmoid`, `alpha`, `a_softplus`, `gate`, `state_predelta`,
`conv_output_raw`, `conv_output_silu`, `q_conv`/`k_conv`/`v_conv` and their
`_predelta` variants, ...) — rocml's own harness only ever captured the 9
tensors above (built for the MMQ-vs-WMMA investigation, `.claude/
CLAUDE.md`'s MMQ section), so there was never a finer-grained capture point
on rocml's side to map these onto. `convert` also reports
`missing_rocml_keys` — a `"{layer}:{tensor}"` rocml expected but the dump
didn't contain, almost always meaning `ROCML_DUMP_FILTER` was too narrow
for that run.

## 4. Diff driver

```sh
make llama-layer-diff \
  LLAMA_DUMP=bench/eval/llama_ref/<checkpoint>-ref.txt
```

Internally: `cargo test --release -p rocml --test llama_layer_diff --
--ignored --nocapture`. The test tokenizes `bench/eval/corpus.txt` itself
(`PROMPT_LEN`=64), runs `Model::load` + `forward_prompt_chunked_captured`
against the checkpoint (`DEFAULT_GGUF_REL` = Ornith-1.0-9B-Q6_K; override
with `LLAMA_DIFF_GGUF=<testpaths-relative .gguf>` to point at a different
qwen35-arch checkpoint), parses `LLAMA_DUMP`, converts it via Section 3's
table, and prints every comparable tensor's max/mean relative and absolute
error sorted worst-`max_rel`-first — `layer_capture::diff_dumps`, the same
function `mmq-layer-diff` already used for its own MMQ-vs-WMMA comparison.
`#[ignore]`d (a diagnostic tool, not a correctness gate) and skips itself
if the checkpoint or `LLAMA_DUMP` is missing.

## 5. Healthy-build baseline (regression yardstick)

**Checkpoint**: Qwen3.5-2B-Q8_0, not the Ornith-1.0-9B flagship this
harness defaults to — see "Friction" below for why, and the exact commands
to redo this against Ornith-1.0-9B-Q6_K once VRAM allows. Both are the same
`qwen35` architecture and go through the identical mapping table; only the
absolute numbers below are checkpoint-specific.

**Prompt**: the pinned 64-token corpus prefix (Section 2). **Build**: this
worktree's `feat/llama-diff-adapter` branch (`main` @ `bf584d1`, forward
path unmodified by this issue — `qwen35_chunked_prefill_parity` still
passes, see the PR/commit history). **llama.cpp**: commit `f3a33dff2` (CPU
backend, `-t 16`).

Sanity floor (validates the pipeline end-to-end, not just the mapping):
layer 0's `gdn_xn` — the embedding + first RMSNorm, before any quantized
weight matmul — matches to **max_rel = 0.000492**, confirming token
order/position alignment between the two engines is exact; everything
below is real per-op accumulation-order/quantization noise starting at the
first linear projection, not a mapping artifact (spot-checked directly:
`linear_attn_qkv_mixed-0`'s raw values range up to ~6.3 in magnitude while
rocml's `gdn_qkv_raw` mean absolute error against it is ~0.003 — a ~0.05%
typical per-element error).

Per-tensor-kind summary (192 `(layer, tensor)` rows aggregated by tensor
kind; `n` = row count = layers of that kind x1):

| tensor | n | mean(max_rel) | max(max_rel) | mean(mean_rel) | mean(max_abs) | mean(mean_abs) |
|---|---:|---:|---:|---:|---:|---:|
| `gdn_xn` | 18 | 84.80 | 152.22 | 0.136 | 0.350 | 0.0210 |
| `ffn_xn` | 24 | 83.41 | 189.75 | 0.142 | 0.228 | 0.0196 |
| `gdn_qkv_raw` | 18 | 83.42 | 108.44 | 0.105 | 0.309 | 0.0171 |
| `resid_post` | 24 | 26.59 | 130.52 | 0.107 | 0.088 | 0.0047 |
| `resid_pre_ffn` | 24 | 19.87 | 63.45 | 0.106 | 0.071 | 0.0042 |
| `ffn_gate_silu` | 24 | 31.60 | 88.52 | 0.122 | 0.187 | 0.0026 |
| `ffn_down_out` | 24 | 13.74 | 37.75 | 0.113 | 0.060 | 0.0025 |
| `gdn_y_silu` | 18 | 21.09 | 59.44 | 0.094 | 0.180 | 0.0020 |
| `gdn_ssm_out_raw` | 18 | 9.70 | 47.65 | 0.098 | 0.039 | 0.0016 |

**Reading these numbers — `max_rel` alone is not the regression signal.**
`diff_dumps`'s relative error is `|a-b| / max(|a|, REL_EPS=1e-3)`: on a
6144-wide GDN QKV projection with real values spanning ~[-6, 1], several
elements per row sit near zero, where even a small absolute error (~0.05,
consistent with `mean_abs`'s scale) divides into a triple-digit "relative"
number that has nothing to do with a regression — it's an artifact of
comparing near-zero floats, called out in `diff_dumps`'s own doc comment.
**`mean_abs` is the practical yardstick**: it stays in a tight, sane
0.0016-0.021 band across every tensor kind here, shrinking toward the
residual stream (where errors partially cancel) and away from raw
projection outputs (where they don't). A future run showing `mean_abs` an
order of magnitude above the matching row here, for the same
checkpoint/prompt, localizes a real regression to that layer/tensor exactly
as issue #10 asked; `max_rel`'s ranking is still useful for *localizing
which tensor to look at first* (as it was for the original MMQ-vs-WMMA
round), just not as a pass/fail threshold on its own.

## 6. Test coverage without llama.cpp installed

`rocml/tests/llama_ref_convert.rs` (not `--ignored`, no real hardware, runs
under plain `cargo test`/`make test`) exercises the parser and Section 3's
mapping table against `rocml/tests/data/llama_ref_fixture.txt` — a small
fixture of real, trimmed `rocml-dump` output (Qwen3.5-2B-Q8_0, layer 0
GDN + layer 3 full-attention, values truncated to 16 columns, plus two
real unmapped nodes captured on purpose: `Qcur_full` and `beta`) — plus
`layer_capture::diff_dumps`'s own max/mean relative-error math on synthetic
data (that function had no prior unit tests).

## Friction

- **VRAM contention blocked the Ornith-1.0-9B-Q6_K baseline this session**:
  a live `rocml-serve --model ornith-9b --ctx 150000` process (from the
  main checkout, `/mnt/nvme/miksa/projects/personal/rocml` — out of this
  worktree's scope to touch) held ~10.3GB of the card's 16GB, leaving too
  little headroom for Ornith's ~7.5GB Q6_K weights alone. Section 5 used
  Qwen3.5-2B-Q8_0 instead (~2.5GB, comfortable headroom) — same
  architecture, same mapping table, same harness. To redo against the
  flagship once VRAM is free:
  ```sh
  ROCML_DUMP_OUT=bench/eval/llama_ref/ornith-1.0-9b-ref.txt \
  ROCML_DUMP_FILTER="attn_residual,attn_post_norm,ffn_swiglu,ffn_out,l_out,attn_norm,linear_attn_qkv_mixed,final_output,linear_attn_out" \
  ./build/bin/rocml-dump -m <checkpoint dir>/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf \
    -f <64-token prompt text, decoded via rocml's tokenizer per Section 2> -ngl 0 -t $(nproc)

  make llama-layer-diff LLAMA_DUMP=bench/eval/llama_ref/ornith-1.0-9b-ref.txt
  ```
- **An unfiltered `rocml-dump` run is impractical**: every non-quantized
  node in the graph (RoPE caches, per-op views/reshapes, the KV cache
  scratch tensors, ...) at full precision blew past several GB and didn't
  finish in a reasonable time even on a 2B model — always pass
  `ROCML_DUMP_FILTER` with Section 3's 9 patterns.
- `common_debug_cb_eval`'s pretty-printed nested-bracket format (with its
  hardcoded 3-element-per-edge truncation) was not reusable for an exact
  diff, hence `rocml-dump`'s own plain line format instead of trying to
  parse llama.cpp's existing debug output.

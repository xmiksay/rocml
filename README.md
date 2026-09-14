# rocml

rocml is a standalone Rust inference engine for Qwen3.5-hybrid/Ornith and dense Qwen3 models on AMD ROCm, built around hand-tuned HIP kernels for gfx1101 (RX 7800 XT), GGUF weight loading, and an OpenAI-compatible server.

## Crates

- `rocml` — the engine (GGUF loading, GPU forward passes, sampling, chat templating).
- `rocml-cli` — `chat` (interactive REPL), `bench` (throughput harness), `generate` (one-shot completion), `models` (list the model registry), `eval` (agentic quality-eval harness, issue #15).
- `rocml-serve` — `POST /v1/chat/completions` (streaming and non-streaming, tool calls) and `GET /v1/models`.

## Build requirements

- ROCm >= 7 installed (default path `/opt/rocm`, override with `ROCM_PATH`)
- `hipcc` on the ROCm install (used to compile kernels to code objects at build time)
- A gfx1101 GPU by default; override the target architecture with `ROCML_GFX_ARCH`

## Make targets

- `make build` — build the workspace
- `make test` — run all tests (unit + integration, requires a working GPU) plus `make test-model`
- `make test-unit` — run library unit tests only
- `make test-integration` — run integration tests only
- `make test-model` — release-mode tests against real GGUF checkpoints and independent references (each skips itself if its checkpoint isn't present): dense Qwen3-0.6B vs. candle's CPU implementation, the qwen3.5 hybrid Gated Delta Net architecture vs. Crane's candle-based implementation (a from-scratch CPU reference first, then the real GPU forward pass), the Ornith-1.0-9B well-formedness smoke test, and `rocml-serve`'s HTTP end-to-end test
- `make lint` — `cargo clippy --all-targets -- -D warnings` plus `cargo fmt --check`
- `make fmt` — apply `cargo fmt`
- `make clean` — `cargo clean`
- `make serve` — run `rocml-serve` against `QWEN_MODEL` (defaults to the Qwen3.5-2B dev/test checkpoint; override with `QWEN_MODEL=/path/to.gguf make serve`)
- `make bench` — run `rocml-cli bench` against `QWEN_MODEL`, JSON output
- `make bench-profile` — profiled `bench` run, human-readable roofline table on stdout (`MODEL`/`DEPTH` override which checkpoint/context depth, e.g. `make bench-profile MODEL=ornith-9b DEPTH=8192`)
- `make bench-profile-json` — same run, machine-readable roofline JSON written to `OUT` (default `bench/profile/$(DEPTH).json`) — see "Observability" below and `docs/prefill-gap-analysis.md`
- `make eval` — run the agentic quality-eval harness (issue #15) for both `ornith-9b-q6` (Q6_K) and `ornith-9b` (Q4_K_M) at ctx 16384, writing `bench/eval/results/*.json`

All cargo invocations are run with `CARGO_BUILD_JOBS=4` to avoid overloading the build machine.

## Model registry

`--model` on every binary below accepts either a path to a `.gguf` file (unchanged, no defaults applied) or one of these compiled-in names (`rocml/src/registry.rs`), which additionally supply a default context budget, sampling params, and thinking-mode default:

| Name | Family | File | Default sampling | Thinking |
|---|---|---|---|---|
| `ornith-9b` | qwen3.5-hybrid | `Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `ornith-9b-q6` | qwen3.5-hybrid | `Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3.5-2b` | qwen3.5-hybrid | `Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf` | temp 1.0, top_p 1.0, top_k 20 | off |
| `qwen3.5-0.8b` | qwen3.5-hybrid | `Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf` | temp 1.0, top_p 1.0, top_k 20 | off |
| `qwen3-0.6b` | qwen3-dense | `Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3-1.7b` | qwen3-dense | `Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3-4b` | qwen3-dense | `Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3-8b` | qwen3-dense | `Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |

Sampling and thinking defaults are each model family's own documented recommendation, verified against its Hugging Face model card (see `rocml/src/registry.rs`'s doc comments for the source of each). `qwen3.5-2b`/`qwen3.5-0.8b` default to reasoning **off** — their own model cards state that's their default — unlike `ornith-9b`/dense Qwen3, which default it on.

`rocml::registry::resolve(name_or_path, download)` turns a `--model` argument into a path: anything containing `/`, ending in `.gguf`, or that already exists as a file is treated as a path outright (no preset applied); anything else is looked up by name. A registry hit whose file is missing under the checkpoint dir (`$ROCML_CHECKPOINT_DIR`, else `$HOME/checkpoints` — see `rocml_core::testpaths`) is downloaded via the `hf` CLI unless `--no-download` is passed, in which case the error names the exact missing path, the source repo, and both remedies.

**Override precedence**: every flag a preset can supply (`--ctx`, `--kv-cache`, `-t/--temperature`, `--top-p`, `--top-k`, `--seed`, `--no-think`) is optional — an unset flag falls through to the resolved model's preset, then (for a path with no preset) the engine's pre-registry default; an explicit flag always wins. `--ctx` now sizes the KV cache directly rather than a hardcoded cap (issue #3): a preset's `default_ctx` (or an explicit `--ctx`) above this checkpoint's estimated VRAM budget is clamped down with a warning naming the budget breakdown and a suggested max ctx (`rocml::registry::clamp_ctx`); `Model::load` performs the authoritative post-weights-load check and errors with the same breakdown if reality doesn't fit. `--kv-cache fp16|q8|q4-mixed` (default `fp16`) picks the KV cache's storage/quantization policy — `fp16` halves the cache vs. the pre-issue-#3 f32-only cache at negligible quality cost; `q8`/`q4-mixed` (issue #2, KIVI-style: fp16 attention sinks + recent window, quantized bulk, boundary attention layers left fp16 — implemented for both the qwen35 hybrid and the dense qwen3 architectures) shrink it further and stay opt-in until the #15 agentic eval judges their accuracy tradeoff. `--kv-sink`/`--kv-window` (default 32/128, no effect at `--kv-cache fp16`) tune the always-fp16 sink length and recent-window/quantize-on-evict-batch size for a quantized mode; both must be at least 1, `--kv-window` at most 65535 (a HIP kernel-launch grid-dimension limit), and `--kv-sink + --kv-window` must be less than `--ctx` (otherwise no history ever reaches the quantized region) — an invalid combination errors at load time rather than silently misbehaving. `--mmq` (default off) routes the qwen35 hybrid architecture's chunked-prefill matmuls through an int8 MMQ GEMM instead of f16 WMMA where the shape is eligible — kept experimental: it measurably fails the chunked-prefill parity suite (up to ~15% max relative logit deviation, and a real exact-greedy divergence on the mixed-KV-cache path), so it stays off by default pending further work (see `.claude/CLAUDE.md`'s "Chunked prefill" section for the full validation writeup).

`rocml-cli models` lists the registry with each entry's on-disk presence under the resolved checkpoint dir.

## rocml-cli

```
rocml-cli chat --model <name-or-gguf> [--no-download] [--kv-cache fp16|q8|q4-mixed] [--kv-sink N] [--kv-window N] [--mmq] [--no-think] [-t/--temperature] [--top-p] [--top-k] [--seed] [--max-tokens] [--ctx]
rocml-cli bench --model <name-or-gguf> [--no-download] [--kv-cache fp16|q8|q4-mixed] [--kv-sink N] [--kv-window N] [--mmq] [--prompt-tokens N] [--decode-tokens N] [--runs N] [--depth N] [--ctx N] [--profile] [--json]
rocml-cli generate --model <name-or-gguf> --prompt <text> [--no-download] [--kv-cache fp16|q8|q4-mixed] [--kv-sink N] [--kv-window N] [--mmq] [--raw] [--no-think] [-n N] [--ctx N] [--profile]
rocml-cli models
```

`chat` re-renders and reprocesses the whole conversation from `Model::reset()` each turn (see `rocml-cli/src/cmd_chat.rs`'s doc comment for why) — fine for an interactive REPL, not meant as a throughput benchmark (use `bench` for that).

`bench --depth N` pre-fills N synthetic tokens of context through the normal prefill path, then measures decode throughput starting from that depth instead of from an empty cache — the head-to-head metric vs. llama.cpp's own "decode t/s at depth D" numbers, since decode cost (attention/GDN-recurrence bytes read) grows with how much context already exists. Both the prefill-to-depth rate and the post-depth decode rate are reported.

See [Observability](#observability) below for `--profile`.

## Agentic eval (issue #15)

```
rocml-cli eval --model <name-or-gguf> --label <label> --out <path.json> [--scenarios bench/eval/scenarios.json] [--corpus bench/eval/corpus.txt] [--ctx 16384] [--max-gen-tokens 2048] [--resume]
```

A small, deterministic (greedy argmax, fixed everything) quality harness measuring quantization loss on real tool-use tasks rather than wikitext PPL — the harness that decided Q4_K_M replaces Q6_K as the default `ornith-9b` registry entry (identical 19/20 agentic score, PPL 1.0807 vs 1.0800, ~2x decode throughput). Runs entirely in-process (no `rocml-serve`), loading the model the same way `chat`/`generate` do and driving it through the real chat protocol (`rocml::chat::render`/`parse_assistant_output`) with thinking mode on (Ornith's own default).

`bench/eval/scenarios.json` (checked in, hand-authored, 20 scenarios) has four kinds, each scored deterministically:

- `tool_choice` — a request that must trigger exactly one tool call with the expected function name and arguments.
- `no_tool` — tools are available but the request must not trigger a call (a plain factual question).
- `multi_turn` — turn 1 must produce an expected tool call; the harness feeds back a canned tool result, then turn 2 is graded against either another expected tool call or a final answer containing an expected substring.
- `long_context` — a retrieval needle buried in ~4000 or ~8000 tokens of deterministically-generated filler (`rocml-cli/src/eval/filler.rs` — only the needle, its position fraction, and the target length are stored, not the filler itself); passes if the final answer contains the needle's fact.

`bench/eval/corpus.txt` (a small public-domain text excerpt) feeds a secondary, informational signal: teacher-forced perplexity over the decode path (one forward pass per token). The pass/fail call is the agentic score, not PPL.

Every scored scenario is written to `--out` immediately, so a run interrupted partway through (a 30-60 minute eval killed by, e.g., a wrapping timeout) can be resumed with `--resume`, which skips any scenario id already present in that file (and skips recomputing PPL if it's already recorded). `make eval` runs both the `ornith-9b-q6` (Q6_K) baseline and the `ornith-9b` (Q4_K_M) default at ctx 16384, `--resume` always on.

## Observability (issue #5)

`bench --profile`/`--profile-json <path>` and `generate --profile` collect per-op, per-layer, per-phase (prefill vs. decode) instrumentation — bytes moved, FLOPs, wall time, and position relative to this card's roofline (~624 GB/s HBM, ~15-20 TFLOP/s fp16 practical scalar ceiling, 75 TFLOP/s f16 WMMA theoretical peak — all three in one place, `rocml/src/profile/roofline.rs`) — and print a human-readable report after the run (`--profile`) and/or write it as machine-readable JSON (`--profile-json <path>`; `--json` also embeds it under a `"profile"` key). Each row additionally carries an analytical min-time bound `t_min = max(bytes/BW, flops/practical_peak)`, `efficiency = t_min/actual`, and `wasted_ms = actual - t_min` — sort a `--profile-json` report's rows by `wasted_ms` to get a ranked optimization worklist (also printed as its own section in the human table).

```
$ rocml-cli bench --model qwen3.5-2b --profile
...
== decode phase: 842.31 ms profiled ==
-- by op-kind --
label                             n         ms     %time      GB/s   GFLOP/s %BWroof %FLroof   waste_ms
ffn-gate-up                     127     312.040     37.0%     238.4      59.1   38.2%    0.3%    294.821
qkv                             127     198.442     23.6%     181.9      45.3   29.2%    0.3%    186.203
...

== top bottlenecks ==
decode ffn-gate-up: 37.0% of decode time, 238.4 GB/s (38.2% of BW roofline), 59.1 GFLOP/s (0.3% of FLOP roofline)
...

== optimization worklist (by wasted time) ==
decode ffn-gate-up: 294.821 ms wasted (312.0 ms actual vs 17.2 ms t_min, 6% efficient)
...
```

Profiling is opt-in and costs nothing when off: every instrumented call site is `Profiler::scope(prof, ..., || { ... })`, a plain function call when `prof` is `None`. When on, per-op HIP events are recorded through the whole run and only synchronized once at the end (`Profiler::finish`), not inside the hot loop — see `rocml/src/profile/mod.rs`'s doc comment for the full design. The qwen35 hybrid architecture's prompt phase runs through a batched, chunked forward pass (issue #6) with full per-op granularity, one event set per chunk; the dense Qwen3 architecture's still-token-serial prefill instead times each layer as one coarse span (bounding event count at a long `--depth`). Decode keeps full per-op granularity on both architectures. Byte/FLOP counts are analytical (weight sizes, KV/state sizes, dtype-aware), not measured — see `rocml/src/profile/cost.rs`.

For kernel-level ground truth (real measured kernel time, occupancy, VGPR/SGPR usage) instead of the analytical view above, see `docs/profiling.md` for the `rocprofv3` workflow. `docs/prefill-gap-analysis.md` is this round's worked example of both together: ranking ornith-9b's real prefill wasted-time ops, categorizing each gap, and an Amdahl-arithmetic lever list with estimated tok/s upside.

## rocml-serve

```
rocml-serve --model <name-or-gguf> [--no-download] [--host 127.0.0.1] [--port 8080] [--ctx N] [--kv-cache fp16|q8|q4-mixed] [--kv-sink N] [--kv-window N] [--mmq] [--max-tokens-default N] [--no-think] [--debug-endpoints]
```

OpenAI-compatible `POST /v1/chat/completions` (streaming via SSE, tool calls via the `tools`/`tool_calls` fields, `reasoning_content` as the de-facto extension carrying stripped `<think>` content) and `GET /v1/models` (reports the resolved registry name, e.g. `qwen3.5-2b`, when `--model` was a registry hit; otherwise the GGUF file's stem). Requests are served by a single dedicated worker thread that owns the model's GPU state and processes one request at a time — no batching or concurrent decode in this version. See `rocml-serve/src/worker.rs` for why the model can't just live behind a `Mutex` on a thread pool instead (HIP state isn't treated as `Send` in this codebase).

The chat renderer is hardcoded to Ornith-1.0-9B's `chat_template.jinja` and used for every model served — their templates differ in a few places (assistant-turn reasoning wrapping in multi-turn history, and critically, the *default* `enable_thinking` value: Ornith defaults reasoning on, Qwen3.5's own templates default it off). A registry name (e.g. `--model qwen3.5-2b`) now applies that model's own thinking default automatically; `--no-think` remains available (and always wins) for a path-based `--model` or to force it off regardless of the preset. Server-side sampling defaults similarly come from the resolved model's preset — a request field the client omits falls back to it rather than a hardcoded value.

**Think-block history (issue #9).** The raw `chat_template.jinja` does *not* strip a history assistant turn's thinking on its own — an incoming `reasoning_content` field (or a `<think>...</think>` block already embedded in `content`) renders back verbatim for every past turn, not just the newest one (confirmed directly against the template — see `rocml/tests/chat_fixtures.rs`). But the Qwen3-family training recipe Ornith descends from removes prior-turn thinking from history at training time; feeding it back is out of distribution and was observed to make the model loop, re-reasoning about an already-resolved point forever. So `rocml-serve` (and `rocml-cli chat`) strip every history assistant turn's reasoning by default — an incoming request's `reasoning_content`/embedded `<think>` never leaks into the rendered prompt, and outgoing responses still expose the *current* turn's thinking only via `reasoning_content`, never inline in `content`. `rocml::chat::RenderOpts::keep_history_reasoning` opts back into the raw template's byte-for-byte behavior for a caller that specifically needs it.

**Tool-result hygiene (issue #9).** `role: "tool"` message bodies are transported verbatim — no trimming beyond the same whitespace-trim every turn gets, no re-encoding, no synthesized placeholder for an empty body. Multiple tool results answering multiple calls from one assistant turn render in transcript order inside a single wrapper, matching the template; `tool_call_id` is accepted but unused, since pairing is by transcript position, not id. **For harness authors:** always return a non-empty tool result body. An empty or status-only result gives the model no evidence the call actually succeeded, and it will typically retry the call — that's a harness-side gap, not something this server papers over.

**Debug endpoint (issue #9).** `--debug-endpoints` mounts `GET /debug/last_prompt`, returning `{"prompt": <string|null>, "prompt_tokens": <number|null>, "timestamp": <unix seconds|null>}` for the most recently accepted (or still in-flight) request — the exact rendered prompt string sent to the tokenizer, the same diagnostic role llama.cpp's `/slots` played in tracing issue #9's stuck-agent loop back to the serving stack. Off by default, and the route doesn't exist at all unless the flag is passed. **This exposes full conversation content (system prompt, prior turns, tool results) to anyone who can reach the port — do not enable it on a shared host.**

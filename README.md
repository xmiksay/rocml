# rocml

rocml is a standalone Rust inference engine for Qwen3.5-hybrid/Ornith and dense Qwen3 models on AMD ROCm, built around hand-tuned HIP kernels for gfx1101 (RX 7800 XT), GGUF weight loading, and an OpenAI-compatible server.

## Crates

- `rocml` — the engine (GGUF loading, GPU forward passes, sampling, chat templating).
- `rocml-cli` — `chat` (interactive REPL), `bench` (throughput harness), `generate` (one-shot completion), `models` (list the model registry).
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

All cargo invocations are run with `CARGO_BUILD_JOBS=4` to avoid overloading the build machine.

## Model registry

`--model` on every binary below accepts either a path to a `.gguf` file (unchanged, no defaults applied) or one of these compiled-in names (`rocml/src/registry.rs`), which additionally supply a default context budget, sampling params, and thinking-mode default:

| Name | Family | File | Default sampling | Thinking |
|---|---|---|---|---|
| `ornith-9b` | qwen3.5-hybrid | `Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3.5-2b` | qwen3.5-hybrid | `Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf` | temp 1.0, top_p 1.0, top_k 20 | off |
| `qwen3.5-0.8b` | qwen3.5-hybrid | `Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf` | temp 1.0, top_p 1.0, top_k 20 | off |
| `qwen3-0.6b` | qwen3-dense | `Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3-1.7b` | qwen3-dense | `Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3-4b` | qwen3-dense | `Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |
| `qwen3-8b` | qwen3-dense | `Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf` | temp 0.6, top_p 0.95, top_k 20 | on |

Sampling and thinking defaults are each model family's own documented recommendation, verified against its Hugging Face model card (see `rocml/src/registry.rs`'s doc comments for the source of each). `qwen3.5-2b`/`qwen3.5-0.8b` default to reasoning **off** — their own model cards state that's their default — unlike `ornith-9b`/dense Qwen3, which default it on.

`rocml::registry::resolve(name_or_path, download)` turns a `--model` argument into a path: anything containing `/`, ending in `.gguf`, or that already exists as a file is treated as a path outright (no preset applied); anything else is looked up by name. A registry hit whose file is missing under the checkpoint dir (`$ROCML_CHECKPOINT_DIR`, else `$HOME/checkpoints` — see `rocml_core::testpaths`) is downloaded via the `hf` CLI unless `--no-download` is passed, in which case the error names the exact missing path, the source repo, and both remedies.

**Override precedence**: every flag a preset can supply (`--ctx`, `-t/--temperature`, `--top-p`, `--top-k`, `--seed`, `--no-think`) is optional — an unset flag falls through to the resolved model's preset, then (for a path with no preset) the engine's pre-registry default; an explicit flag always wins. A preset's `default_ctx` above the engine's current KV-cache cap (4096 tokens today, see `rocml::cache::MAX_SEQ_CAP`) is clamped down with a warning rather than erroring (`ornith-9b`'s 8192 is expected to hit this until a larger cache lands).

`rocml-cli models` lists the registry with each entry's on-disk presence under the resolved checkpoint dir.

## rocml-cli

```
rocml-cli chat --model <name-or-gguf> [--no-download] [--no-think] [-t/--temperature] [--top-p] [--top-k] [--seed] [--max-tokens] [--ctx]
rocml-cli bench --model <name-or-gguf> [--no-download] [--prompt-tokens N] [--decode-tokens N] [--runs N] [--depth N] [--profile] [--json]
rocml-cli generate --model <name-or-gguf> --prompt <text> [--no-download] [--raw] [--no-think] [-n N] [--profile]
rocml-cli models
```

`chat` re-renders and reprocesses the whole conversation from `Model::reset()` each turn (see `rocml-cli/src/cmd_chat.rs`'s doc comment for why) — fine for an interactive REPL, not meant as a throughput benchmark (use `bench` for that).

`bench --depth N` pre-fills N synthetic tokens of context through the normal prefill path, then measures decode throughput starting from that depth instead of from an empty cache — the head-to-head metric vs. llama.cpp's own "decode t/s at depth D" numbers, since decode cost (attention/GDN-recurrence bytes read) grows with how much context already exists. Both the prefill-to-depth rate and the post-depth decode rate are reported.

See [Observability](#observability) below for `--profile`.

## Observability

`bench --profile` and `generate --profile` collect per-op, per-layer, per-phase (prefill vs. decode) instrumentation — bytes moved, FLOPs, wall time, and position relative to this card's roofline (~624 GB/s HBM, ~15-20 TFLOP/s fp16 on gfx1101) — and print a report after the run (`bench --profile --json` embeds the same data under a `"profile"` key instead).

```
$ rocml-cli bench --model qwen3.5-2b --profile
...
== decode phase: 842.31 ms profiled ==
-- by op-kind --
label                             n         ms     %time      GB/s   GFLOP/s %BWroof %FLroof
ffn-gate-up                     127     312.040     37.0%     238.4      59.1   38.2%    0.3%
qkv                             127     198.442     23.6%     181.9      45.3   29.2%    0.3%
...

== top bottlenecks ==
decode ffn-gate-up: 37.0% of decode time, 238.4 GB/s (38.2% of BW roofline), 59.1 GFLOP/s (0.3% of FLOP roofline)
...
```

Profiling is opt-in and costs nothing when off: every instrumented call site is `Profiler::scope(prof, ..., || { ... })`, a plain function call when `prof` is `None`. When on, per-op HIP events are recorded through the whole run and only synchronized once at the end (`Profiler::finish`), not inside the hot loop — see `rocml/src/profile/mod.rs`'s doc comment for the full design. The qwen35 hybrid architecture's prompt phase runs through a batched, chunked forward pass (issue #6) with full per-op granularity, one event set per chunk; the dense Qwen3 architecture's still-token-serial prefill instead times each layer as one coarse span (bounding event count at a long `--depth`). Decode keeps full per-op granularity on both architectures. Byte/FLOP counts are analytical (weight sizes, KV/state sizes, dtype-aware), not measured — see `rocml/src/profile/cost.rs`.

## rocml-serve

```
rocml-serve --model <name-or-gguf> [--no-download] [--host 127.0.0.1] [--port 8080] [--ctx N] [--max-tokens-default N] [--no-think]
```

OpenAI-compatible `POST /v1/chat/completions` (streaming via SSE, tool calls via the `tools`/`tool_calls` fields, `reasoning_content` as the de-facto extension carrying stripped `<think>` content) and `GET /v1/models` (reports the resolved registry name, e.g. `qwen3.5-2b`, when `--model` was a registry hit; otherwise the GGUF file's stem). Requests are served by a single dedicated worker thread that owns the model's GPU state and processes one request at a time — no batching or concurrent decode in this version. See `rocml-serve/src/worker.rs` for why the model can't just live behind a `Mutex` on a thread pool instead (HIP state isn't treated as `Send` in this codebase).

The chat renderer is hardcoded to Ornith-1.0-9B's `chat_template.jinja` and used for every model served — their templates differ in a few places (assistant-turn reasoning wrapping in multi-turn history, and critically, the *default* `enable_thinking` value: Ornith defaults reasoning on, Qwen3.5's own templates default it off). A registry name (e.g. `--model qwen3.5-2b`) now applies that model's own thinking default automatically; `--no-think` remains available (and always wins) for a path-based `--model` or to force it off regardless of the preset. Server-side sampling defaults similarly come from the resolved model's preset — a request field the client omits falls back to it rather than a hardcoded value.

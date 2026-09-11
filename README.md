# rocml

rocml is a standalone Rust inference engine for Qwen3.5-hybrid/Ornith and dense Qwen3 models on AMD ROCm, built around hand-tuned HIP kernels for gfx1101 (RX 7800 XT), GGUF weight loading, and an OpenAI-compatible server.

## Crates

- `rocml` — the engine (GGUF loading, GPU forward passes, sampling, chat templating).
- `rocml-cli` — `chat` (interactive REPL), `bench` (throughput harness), `generate` (one-shot completion).
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

## rocml-cli

```
rocml-cli chat --model <gguf> [--no-think] [-t/--temperature] [--top-p] [--top-k] [--seed] [--max-tokens] [--ctx]
rocml-cli bench --model <gguf> [--prompt-tokens N] [--decode-tokens N] [--runs N] [--json]
rocml-cli generate --model <gguf> --prompt <text> [--raw] [-n N]
```

`chat` re-renders and reprocesses the whole conversation from `Model::reset()` each turn (see `rocml-cli/src/cmd_chat.rs`'s doc comment for why) — fine for an interactive REPL, not meant as a throughput benchmark (use `bench` for that).

## rocml-serve

```
rocml-serve --model <gguf> [--host 127.0.0.1] [--port 8080] [--ctx N] [--max-tokens-default N] [--no-think]
```

OpenAI-compatible `POST /v1/chat/completions` (streaming via SSE, tool calls via the `tools`/`tool_calls` fields, `reasoning_content` as the de-facto extension carrying stripped `<think>` content) and `GET /v1/models`. Requests are served by a single dedicated worker thread that owns the model's GPU state and processes one request at a time — no batching or concurrent decode in this version. See `rocml-serve/src/worker.rs` for why the model can't just live behind a `Mutex` on a thread pool instead (HIP state isn't treated as `Send` in this codebase).

The chat renderer is hardcoded to Ornith-1.0-9B's `chat_template.jinja` and used for every model served, including Qwen3.5-2B — their templates differ in a few places (assistant-turn reasoning wrapping in multi-turn history, and critically, the *default* `enable_thinking` value: Ornith defaults reasoning on, Qwen3.5-2B's own template defaults it off). This only matters for the reasoning-default difference in practice; pass `--no-think` if you want Qwen3.5-2B's own default behavior.

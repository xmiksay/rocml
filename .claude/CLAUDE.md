# rocml — Project Brief

Standalone Rust LLM inference engine for Qwen3.5-hybrid/Ornith and dense Qwen3, with hand-written HIP kernels tuned for AMD gfx1101 (RX 7800 XT). GGUF weight loading, OpenAI-compatible server. No candle, no rocBLAS/hipBLASLt, no bindgen — kernels and the HIP runtime binding are hand-written and hand-tuned in this repo.

This brief wins over `../CLAUDE.md` for anything specific to this project.

## Crate map

- `rocml-hip` — minimal safe wrapper over the HIP runtime C API (hand-written FFI, no bindgen), linked against `amdhip64`. Device/stream/buffer/module/kernel-launch primitives only — no ML logic here.
- `rocml-kernels` — HIP C++ kernel sources under `kernels/`, compiled to code objects (`.hsaco`) by `hipcc` in `build.rs` and embedded into the binary via `include_bytes!`. Exposes kernel bytes + entry point names for `rocml-hip` to load and launch.
- `rocml-core` — hardware-agnostic CPU-side building blocks: mmap'd GGUF v2/v3 parsing, ggml quant (Q2_K..Q6_K, Q8_0) CPU dequantize, and a byte-level BPE tokenizer built from GGUF `tokenizer.ggml.*` metadata (pre-tokenizer regex selected per `tokenizer.ggml.pre` — qwen35 vs qwen2 families).
- `rocml` — the engine: GGUF config/weight loading onto the GPU, a per-layer-per-kv-head-plane KV cache, the single-token decode-style forward pass, sampling (`src/sample.rs`), and generation (`src/generate.rs`). `src/model.rs` dispatches on `general.architecture`: `"qwen3"` runs the dense forward pass (`src/forward/`); `"qwen35"` runs the hybrid Gated Delta Net + full-attention forward pass (`src/qwen35/`) — every `qwen35.full_attention_interval`-th layer is full (softmax) attention, the rest are linear-attention GDN layers with persistent per-layer conv/recurrence state (`src/qwen35/cache.rs`).
  - GDN supports grouped key/value heads (`qwen35.ssm.group_count` < `qwen35.ssm.time_step_rank`, e.g. Ornith-1.0-9B's 16 key / 32 value heads): `gdn_recurrence_decode_f32` broadcasts key head `h % num_k_heads` to value head `h` — a **tiled** pattern, not the naive `repeat_interleave` HF transformers' source uses on the raw checkpoint. llama.cpp's GGUF converter already permutes every value-head-indexed GDN tensor (V, Z/gate, beta, alpha, A_log, dt_bias, conv1d's V channels, `ssm_out`'s input columns) from HF's grouped order into this tiled order at conversion time (`_LinearAttentionVReorderBase._reorder_v_heads` in `convert_hf_to_gguf`'s `conversion/qwen.py`), specifically so a cheap modulo broadcast works at inference — using `/` instead of `%` here loads and runs without error but produces well-formed, deterministic, finite garbage (verified against Ornith: fixed a real regression this way — see the kernel's own doc comment for the full citation). When `num_k_heads == num_v_heads` both formulas coincide, which is why this was invisible on Qwen3.5-2B.
  - Linear-layer weights are loaded through `weights::LinearWeight` (`src/weights/linear.rs`), shared by both architectures: a tensor whose dtype is Q8_0/Q4_K/Q5_K/Q6_K and whose row length is a multiple of that fused kernel's block width (32/256/256/256) stays in its raw GGUF block format in VRAM and runs through `Kernels::gemv_quant` (dispatching to `rocml-kernels`' `gemv_q8_0`/`gemv_q4_k`/`gemv_q5_k`/`gemv_q6_k`); everything else (F32/F16/BF16, an unsupported quant, or a non-conforming row length) falls back to the original CPU-dequant-to-f32-then-f16-upload path. This is what lets a Q6_K model like Ornith-1.0-9B (~7.5GB) fit in 16GB VRAM at all — dequant-to-f16 would not. The token embedding table stays f16-dequantized always (the embedding-lookup kernel only reads f16); the output/lm-head projection goes through `LinearWeight` like every other linear layer.
  - `src/sample.rs`: host-side temperature/top-k/top-p sampling plus an optional off-by-default repeat penalty, driven by a deterministic inline xorshift64* PRNG (no `rand` dependency). `generate::generate` stays greedy-only (existing parity fixtures depend on exact argmax behavior); `generate::generate_sampled` takes a `SamplingParams`; `generate::generate_sampled_with_stop` additionally lets `on_text` request early termination, which is how `rocml-serve` enforces OpenAI `stop` strings against the growing decoded text.
- `rocml-cli` — `chat` (interactive REPL over `rocml::chat`, streamed via `StreamScanner`), `bench` (synthetic prompt/decode throughput harness, the llama.cpp-comparison baseline), `generate` (one-shot completion; replaces the old `rocml/examples/generate.rs` smoke test, now rendering through the real chat template and sampler instead of a hardcoded format string).
- `rocml-serve` — OpenAI-compatible HTTP server (`POST /v1/chat/completions`, `GET /v1/models`). `src/worker.rs` is the one place that owns the loaded `Model`: a dedicated OS thread (HIP state isn't treated as `Send` in this codebase), fed by a `std::sync::mpsc` job queue and streaming results back per-job over a `tokio::sync::mpsc` channel — strictly sequential, no batching (the documented v1 concurrency model). `src/mapping.rs` translates OpenAI wire shapes (`src/openai.rs`) to/from `rocml::chat` types; `src/routes.rs` builds both the full-JSON and SSE response shapes from the same worker event stream. The chat renderer is hardcoded to Ornith-1.0-9B's template and used for both models it serves — see that crate's section in `../README.md` for the one behavioral difference this causes (the `enable_thinking` default) and `--no-think` as the workaround.

## Build commands

Always use the Makefile, never ad-hoc `cargo` invocations:

- `make build` / `make test` / `make test-unit` / `make test-integration` / `make test-model`
- `make lint` (clippy `-D warnings` + `fmt --check`) / `make fmt`
- `make clean`
- `make serve` / `make bench` — run `rocml-serve` / `rocml-cli bench` against `QWEN_MODEL` (override with `QWEN_MODEL=/path/to.gguf make serve`)

**Always `CARGO_BUILD_JOBS=4`** — the Makefile exports it for every target already; if you run cargo directly for a one-off, set it yourself.

Real-GGUF tests and `make serve`/`make bench` resolve model checkpoints via `rocml_core::testpaths` (`rocml-core/src/testpaths.rs`): `$ROCML_CHECKPOINT_DIR` if set, else `$HOME/checkpoints` — expected to contain `Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`, `Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf`, `Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf`, `Qwen3.5-2B-tokenizer/tokenizer.json`, and `Ornith-1.0-9B/chat_template.jinja`. Each test skips itself (with a message naming the resolved path) if its checkpoint isn't present. `rocml-core/tests/tokenizer_cross_validation.rs` additionally resolves a couple of files from the standard Hugging Face hub cache (`$HF_HOME/hub` else `$HOME/.cache/huggingface/hub`) via the same module's `hf_hub_file`.

Tests in `rocml-hip`, the `rocml-kernels` integration tests, `rocml`'s forward pass, and `rocml-serve`'s end-to-end test require a real ROCm GPU (device init, actual kernel launches) — they are not mocked. Run on the machine with the GPU attached.

`make test-model` runs real-GGUF `--release` integration tests, each skipping itself if its checkpoint is absent (`make test` excludes all of them from the plain `cargo test --workspace` pass — they'd still build in debug, just run unbearably slowly — and runs them separately here):
- `greedy_parity`: dense Qwen3-0.6B (`<checkpoint dir>/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`) greedy-decoded and checked against candle's independent CPU f32 implementation token-for-token.
- `qwen35_cpu_reference`: the qwen3.5 hybrid architecture's from-scratch pure-Rust f32 CPU reference (`tests/support/qwen35_cpu/`, no GPU) checked against Crane's fixture for a handful of tokens — isolates "did we understand the architecture" from "is the GPU kernel right".
- `qwen35_greedy_parity`: Qwen3.5-2B (`<checkpoint dir>/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf`) greedy-decoded on the GPU hybrid forward pass and checked against Crane's independent candle GPU implementation token-for-token, from fixtures in `rocml/tests/data/qwen35_greedy_fixtures.json`.
- `ornith_e2e`: Ornith-1.0-9B (`<checkpoint dir>/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf`, qwen35 arch, Q6_K) — no independent reference for this checkpoint, so this asserts internal well-formedness instead: finite logits throughout a 24-token greedy decode, and the exact same generated token sequence from two independent model loads. Exists because this checkpoint only fits in 16GB VRAM through the quantized-weight path (see `LinearWeight` above).
- `rocml-serve`'s `server_e2e`: spawns the real router on an ephemeral port (Qwen3.5-2B) and drives it with a hand-rolled minimal HTTP/1.1 client (`tests/support/`) — non-streamed completion, streamed SSE, and a tools request. A companion `#[ignore]`d test in the same file (`ornith_tool_call_is_emitted`) asserts a real tool call against the Ornith GGUF; `make test-model` runs it as its own separate `cargo test` invocation (`cargo test --release -p rocml-serve --test server_e2e -- --ignored ornith_tool_call_is_emitted`) rather than alongside the default test, since the two would compete for VRAM. Both use a `JoinHandle`-based shutdown sequence (see `worker::spawn`'s doc comment) — skipping it crashes the test process on exit with "pure virtual method called" from a still-live worker thread racing the HIP driver's own teardown.

`rocml/src/weights/linear.rs` also carries its own `#[cfg(test)]` per-layer spot checks (`cargo test --lib`, real hardware + real checkpoints, skip-if-missing): `LinearWeight::load` + `matvec` against one dense-Qwen3 tensor and one qwen3.5 GDN-layer/full-attention-layer tensor each, compared to `rocml-core`'s CPU dequant + a CPU dot product — this is the loader/dispatch layer, complementing the fused kernels' own tests in `rocml-kernels/tests/gemv_q*.rs`.

## Kernel build pipeline

`rocml-kernels/build.rs` shells out to `hipcc` for every `*.hip` file under `kernels/`:

```
${ROCM_PATH:-/opt/rocm}/bin/hipcc --genco --offload-arch=${ROCML_GFX_ARCH:-gfx1101} -O3 -o $OUT_DIR/<name>.hsaco <src>
```

The resulting `.hsaco` code objects land in `$OUT_DIR` and are pulled into the crate via `include_bytes!`, so the compiled binary carries the kernels — no runtime dependency on `hipcc` or the `kernels/` sources being present. `rocml-hip::module::Module::load_from_bytes` takes those bytes straight to `hipModuleLoadData`.

- Override the target GPU architecture with `ROCML_GFX_ARCH` (default `gfx1101`). Building for a different card (e.g. gfx1030, gfx90a) is just re-running the build with that env var set — no source changes needed unless the kernel itself is arch-specific.
- Override the ROCm install root with `ROCM_PATH` (default `/opt/rocm`) — used both to find `hipcc` and to locate `libamdhip64.so` for linking (`rocml-hip/build.rs`).
- `build.rs` reruns on changes to `kernels/` and to `ROCM_PATH`/`ROCML_GFX_ARCH`; a `hipcc` failure fails the build with its full stderr.

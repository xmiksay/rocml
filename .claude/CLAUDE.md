# rocml — Project Brief

Standalone Rust LLM inference engine for Qwen3.5-hybrid/Ornith and dense Qwen3, with hand-written HIP kernels tuned for AMD gfx1101 (RX 7800 XT). GGUF weight loading, OpenAI-compatible server. No candle, no rocBLAS/hipBLASLt, no bindgen — kernels and the HIP runtime binding are hand-written and hand-tuned in this repo.

This brief wins over `../CLAUDE.md` for anything specific to this project.

## Crate map

- `rocml-hip` — minimal safe wrapper over the HIP runtime C API (hand-written FFI, no bindgen), linked against `amdhip64`. Device/stream/buffer/module/kernel-launch primitives only — no ML logic here.
- `rocml-kernels` — HIP C++ kernel sources under `kernels/`, compiled to code objects (`.hsaco`) by `hipcc` in `build.rs` and embedded into the binary via `include_bytes!`. Exposes kernel bytes + entry point names for `rocml-hip` to load and launch.
- `rocml-core` — hardware-agnostic CPU-side building blocks: mmap'd GGUF v2/v3 parsing, ggml quant (Q2_K..Q6_K, Q8_0) CPU dequantize, and a byte-level BPE tokenizer built from GGUF `tokenizer.ggml.*` metadata (pre-tokenizer regex selected per `tokenizer.ggml.pre` — qwen35 vs qwen2 families).
- `rocml` — the engine: GGUF config/weight loading onto the GPU, a per-layer-per-kv-head-plane KV cache, the single-token decode-style forward pass (`src/forward/`), and greedy generation (`src/generate.rs`). Dense Qwen3 only for now (`general.architecture = "qwen3"`); `examples/generate.rs` is the CLI smoke-test binary.

Later milestones add `rocml-cli`, `rocml-serve` — not present yet, do not add them speculatively.

## Build commands

Always use the Makefile, never ad-hoc `cargo` invocations:

- `make build` / `make test` / `make test-unit` / `make test-integration` / `make test-model`
- `make lint` (clippy `-D warnings` + `fmt --check`) / `make fmt`
- `make clean`

**Always `CARGO_BUILD_JOBS=4`** — the Makefile exports it for every target already; if you run cargo directly for a one-off, set it yourself.

Tests in `rocml-hip`, the `rocml-kernels` integration tests, and `rocml`'s forward pass require a real ROCm GPU (device init, actual kernel launches) — they are not mocked. Run on the machine with the GPU attached.

`make test-model` runs `rocml`'s `greedy_parity` integration test in `--release` (dequantizing the real ~600MB Qwen3-0.6B GGUF and running a 28-layer x 48-token decode loop is unbearably slow unoptimized): greedy-decodes fixed prompts and checks the decoded text matches candle's independent CPU f32 implementation token-for-token, from `/mnt/nvme/miksa/checkpoints/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` (skips itself if that file is absent). `make test` excludes it from the plain `cargo test --workspace` pass (it would still build fine in debug, just run unbearably slowly) and runs it separately via `make test-model`.

## Kernel build pipeline

`rocml-kernels/build.rs` shells out to `hipcc` for every `*.hip` file under `kernels/`:

```
${ROCM_PATH:-/opt/rocm}/bin/hipcc --genco --offload-arch=${ROCML_GFX_ARCH:-gfx1101} -O3 -o $OUT_DIR/<name>.hsaco <src>
```

The resulting `.hsaco` code objects land in `$OUT_DIR` and are pulled into the crate via `include_bytes!`, so the compiled binary carries the kernels — no runtime dependency on `hipcc` or the `kernels/` sources being present. `rocml-hip::module::Module::load_from_bytes` takes those bytes straight to `hipModuleLoadData`.

- Override the target GPU architecture with `ROCML_GFX_ARCH` (default `gfx1101`). Building for a different card (e.g. gfx1030, gfx90a) is just re-running the build with that env var set — no source changes needed unless the kernel itself is arch-specific.
- Override the ROCm install root with `ROCM_PATH` (default `/opt/rocm`) — used both to find `hipcc` and to locate `libamdhip64.so` for linking (`rocml-hip/build.rs`).
- `build.rs` reruns on changes to `kernels/` and to `ROCM_PATH`/`ROCML_GFX_ARCH`; a `hipcc` failure fails the build with its full stderr.

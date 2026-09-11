# rocml — Project Brief

Standalone Rust LLM inference engine for Qwen3.5-hybrid/Ornith and dense Qwen3, with hand-written HIP kernels tuned for AMD gfx1101 (RX 7800 XT). GGUF weight loading, OpenAI-compatible server. No candle, no rocBLAS/hipBLASLt, no bindgen — kernels and the HIP runtime binding are hand-written and hand-tuned in this repo.

This brief wins over `../CLAUDE.md` for anything specific to this project.

## Crate map

- `rocml-hip` — minimal safe wrapper over the HIP runtime C API (hand-written FFI, no bindgen), linked against `amdhip64`. Device/stream/buffer/module/kernel-launch primitives only — no ML logic here.
- `rocml-kernels` — HIP C++ kernel sources under `kernels/`, compiled to code objects (`.hsaco`) by `hipcc` in `build.rs` and embedded into the binary via `include_bytes!`. Exposes kernel bytes + entry point names for `rocml-hip` to load and launch.

Later milestones add `rocml-core` (tensor/model graph), `rocml` (engine), `rocml-cli`, `rocml-serve` — not present yet, do not add them speculatively.

## Build commands

Always use the Makefile, never ad-hoc `cargo` invocations:

- `make build` / `make test` / `make test-unit` / `make test-integration`
- `make lint` (clippy `-D warnings` + `fmt --check`) / `make fmt`
- `make clean`

**Always `CARGO_BUILD_JOBS=4`** — the Makefile exports it for every target already; if you run cargo directly for a one-off, set it yourself.

Tests in `rocml-hip` and the `rocml-kernels` integration test require a real ROCm GPU (device init, actual kernel launches) — they are not mocked. Run on the machine with the GPU attached.

## Kernel build pipeline

`rocml-kernels/build.rs` shells out to `hipcc` for every `*.hip` file under `kernels/`:

```
${ROCM_PATH:-/opt/rocm}/bin/hipcc --genco --offload-arch=${ROCML_GFX_ARCH:-gfx1101} -O3 -o $OUT_DIR/<name>.hsaco <src>
```

The resulting `.hsaco` code objects land in `$OUT_DIR` and are pulled into the crate via `include_bytes!`, so the compiled binary carries the kernels — no runtime dependency on `hipcc` or the `kernels/` sources being present. `rocml-hip::module::Module::load_from_bytes` takes those bytes straight to `hipModuleLoadData`.

- Override the target GPU architecture with `ROCML_GFX_ARCH` (default `gfx1101`). Building for a different card (e.g. gfx1030, gfx90a) is just re-running the build with that env var set — no source changes needed unless the kernel itself is arch-specific.
- Override the ROCm install root with `ROCM_PATH` (default `/opt/rocm`) — used both to find `hipcc` and to locate `libamdhip64.so` for linking (`rocml-hip/build.rs`).
- `build.rs` reruns on changes to `kernels/` and to `ROCM_PATH`/`ROCML_GFX_ARCH`; a `hipcc` failure fails the build with its full stderr.

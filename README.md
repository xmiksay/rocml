# rocml

rocml is a standalone Rust inference engine for Qwen3.5-hybrid/Ornith and dense Qwen3 models on AMD ROCm, built around hand-tuned HIP kernels for gfx1101 (RX 7800 XT), GGUF weight loading, and an OpenAI-compatible server.

## Build requirements

- ROCm >= 7 installed (default path `/opt/rocm`, override with `ROCM_PATH`)
- `hipcc` on the ROCm install (used to compile kernels to code objects at build time)
- A gfx1101 GPU by default; override the target architecture with `ROCML_GFX_ARCH`

## Make targets

- `make build` — build the workspace
- `make test` — run all tests (unit + integration, requires a working GPU)
- `make test-unit` — run library unit tests only
- `make test-integration` — run integration tests only
- `make lint` — `cargo clippy --all-targets -- -D warnings` plus `cargo fmt --check`
- `make fmt` — apply `cargo fmt`
- `make clean` — `cargo clean`

All cargo invocations are run with `CARGO_BUILD_JOBS=4` to avoid overloading the build machine.

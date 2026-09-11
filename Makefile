export CARGO_BUILD_JOBS := 4

.PHONY: build test test-unit test-integration test-model lint fmt clean

build:
	cargo build --workspace

# The dense-Qwen3 greedy-parity test is excluded here (see test-model): it
# dequantizes a real ~600MB GGUF and runs a 28-layer x N-token decode loop,
# which is unbearably slow without optimizations.
test:
	cargo test --workspace -- --skip dense_qwen3_0_6b_greedy_matches_candle_cpu_reference
	$(MAKE) test-model

test-unit:
	cargo test --workspace --lib

test-integration:
	cargo test --workspace --test '*'

# Real-hardware, real-GGUF parity test against the candle CPU reference;
# needs --release for the CPU-side dequant/forward-pass loop to run in
# reasonable time. Skips itself if the checkpoint isn't present.
test-model:
	cargo test --release -p rocml --test greedy_parity

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --all -- --check

fmt:
	cargo fmt --all

clean:
	cargo clean

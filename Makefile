export CARGO_BUILD_JOBS := 4

.PHONY: build test test-unit test-integration test-model lint fmt clean

build:
	cargo build --workspace

# The real-GGUF model tests are excluded here (see test-model): they
# dequantize real multi-GB GGUFs and run many-layer x N-token decode loops,
# which are unbearably slow without optimizations.
test:
	cargo test --workspace -- \
		--skip dense_qwen3_0_6b_greedy_matches_candle_cpu_reference \
		--skip qwen35_cpu_reference_matches_crane_for_a_handful_of_tokens \
		--skip qwen35_2b_hybrid_greedy_matches_crane_gpu_reference \
		--skip ornith_9b_greedy_decode_is_well_formed_and_deterministic
	$(MAKE) test-model

test-unit:
	cargo test --workspace --lib

test-integration:
	cargo test --workspace --test '*'

# Real-hardware, real-GGUF tests: parity against independent candle/Crane
# references plus the Ornith-1.0-9B end-to-end smoke test; needs --release
# for the forward-pass decode loops to run in reasonable time. Each skips
# itself if its checkpoint isn't present.
test-model:
	cargo test --release -p rocml --test greedy_parity
	cargo test --release -p rocml --test qwen35_cpu_reference
	cargo test --release -p rocml --test qwen35_greedy_parity
	cargo test --release -p rocml --test ornith_e2e

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --all -- --check

fmt:
	cargo fmt --all

clean:
	cargo clean

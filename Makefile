export CARGO_BUILD_JOBS := 4

# Root directory for model checkpoints (see rocml_core::testpaths). Override
# if your checkpoints live elsewhere.
ROCML_CHECKPOINT_DIR ?= $(HOME)/checkpoints

# Dev/test model (fast); override to point at a different checkpoint.
QWEN_MODEL ?= $(ROCML_CHECKPOINT_DIR)/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf

.PHONY: build test test-unit test-integration test-model lint fmt clean serve bench eval

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
		--skip ornith_9b_greedy_decode_is_well_formed_and_deterministic \
		--skip qwen35_2b_chunked_prefill_matches_token_serial \
		--skip fp16_kv_matches_f32_kv_logits_and_greedy_decode \
		--skip q8_mixed_kv_vs_fp16_logits_and_greedy_stability \
		--skip q4_mixed_kv_vs_fp16_logits_and_greedy_stability \
		--skip q4_mixed_kv_greedy_divergence_is_a_near_tie_when_it_happens \
		--skip chat_completions_end_to_end \
		--skip two_turn_conversation_matches_output_with_snapshots_disabled \
		--skip qwen35_snapshot_equivalence
	$(MAKE) test-model

test-unit:
	cargo test --workspace --lib

test-integration:
	cargo test --workspace --test '*'

# Real-hardware, real-GGUF tests: parity against independent candle/Crane
# references, the Ornith-1.0-9B end-to-end smoke test, and the rocml-serve
# HTTP end-to-end test; needs --release for the forward-pass decode loops to
# run in reasonable time. Each skips itself if its checkpoint isn't present.
# `ornith_tool_call_is_emitted` in server_e2e is `#[ignore]`d and run as its
# own `cargo test` invocation (see the test's own doc comment) — it and the
# plain server_e2e test both load a model onto the same GPU and would
# compete for VRAM if run concurrently in the same test binary.
test-model:
	cargo test --release -p rocml --test greedy_parity
	cargo test --release -p rocml --test qwen35_cpu_reference
	cargo test --release -p rocml --test qwen35_greedy_parity
	cargo test --release -p rocml --test qwen35_chunked_prefill_parity
	cargo test --release -p rocml --test kv_dtype_parity
	cargo test --release -p rocml --test mixed_kv_parity
	cargo test --release -p rocml --test snapshot_equivalence
	cargo test --release -p rocml --test ornith_e2e -- --test-threads=1
	cargo test --release -p rocml-serve --test server_e2e
	cargo test --release -p rocml-serve --test server_e2e -- --ignored ornith_tool_call_is_emitted

lint:
	cargo clippy --workspace --all-targets -- -D warnings
	cargo fmt --all -- --check

fmt:
	cargo fmt --all

clean:
	cargo clean

# Run the OpenAI-compatible server against the dev/test model.
serve:
	cargo run --release -p rocml-serve -- --model $(QWEN_MODEL)

# Synthetic prompt/decode throughput benchmark against the dev/test model.
bench:
	cargo run --release -p rocml-cli -- bench --model $(QWEN_MODEL) --json

# Issue #15's agentic quality-eval harness: establishes the Q6_K fp16-KV
# baseline and the Q4_K_M candidate, both at ctx 16384 (long-context
# scenarios need ~8K tokens of filler plus thinking/answer headroom).
# `--resume` is always passed so a run interrupted partway through (a long
# eval, possibly split across several invocations) picks back up instead of
# re-scoring already-graded scenarios.
eval:
	cargo run --release -p rocml-cli -- eval \
		--model ornith-9b-q6 --ctx 16384 \
		--label ornith-q6k-fp16kv \
		--out bench/eval/results/ornith-q6k-fp16kv.json \
		--resume
	cargo run --release -p rocml-cli -- eval \
		--model ornith-9b --ctx 16384 \
		--label ornith-q4km-fp16kv \
		--out bench/eval/results/ornith-q4km-fp16kv.json \
		--resume

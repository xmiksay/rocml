export CARGO_BUILD_JOBS := 4

# Root directory for model checkpoints (see rocml_core::testpaths). Override
# if your checkpoints live elsewhere.
ROCML_CHECKPOINT_DIR ?= $(HOME)/checkpoints

# Dev/test model (fast); override to point at a different checkpoint.
QWEN_MODEL ?= $(ROCML_CHECKPOINT_DIR)/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf

.PHONY: build test test-unit test-integration test-model lint fmt clean serve bench bench-profile bench-profile-json eval mmq-layer-diff mmq-endtoend-measure mmq-calibrate mmq-smoothquant-measure kv-head-error-measure gdn-wmma-lds-perf gdn-uvvnew-perf rotational-kv-calibrate rotational-kv-measure rotational-kv-sim-parity eval-rotational-v3

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
		--skip qwen3_0_6b_chunked_prefill_matches_token_serial \
		--skip qwen3_0_6b_chunked_prefill_matches_token_serial_mixed_kv \
		--skip fp16_kv_matches_f32_kv_logits_and_greedy_decode \
		--skip q8_mixed_kv_vs_fp16_logits_and_greedy_stability \
		--skip q4_mixed_kv_vs_fp16_logits_and_greedy_stability \
		--skip q4_mixed_kv_greedy_divergence_is_a_near_tie_when_it_happens \
		--skip q4_mixed_kv_vs_fp16_at_non_default_sink_window \
		--skip q8_mixed_chunked_prefill_matches_token_serial \
		--skip q4_mixed_chunked_prefill_matches_token_serial \
		--skip dense_fp16_kv_matches_f32_kv_logits_and_greedy_decode \
		--skip dense_q8_mixed_kv_vs_fp16_logits_and_greedy_stability \
		--skip dense_q4_mixed_kv_vs_fp16_logits_and_greedy_stability \
		--skip chat_completions_end_to_end \
		--skip two_turn_conversation_matches_output_with_snapshots_disabled \
		--skip qwen35_snapshot_equivalence \
		--skip profile_json_is_valid_and_well_shaped
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
# mixed_kv_chunked_prefill_parity's three tests each load two full
# Qwen3.5-2B models (serial + chunked) — --test-threads=1 avoids up to six
# concurrent model loads exceeding a 16GB card's VRAM under cargo's default
# parallel test harness.
test-model:
	cargo test --release -p rocml --test greedy_parity
	cargo test --release -p rocml --test qwen35_cpu_reference
	cargo test --release -p rocml --test qwen35_greedy_parity
	cargo test --release -p rocml --test qwen35_chunked_prefill_parity
	cargo test --release -p rocml --test kv_dtype_parity
	cargo test --release -p rocml --test mixed_kv_parity
	cargo test --release -p rocml --test mixed_kv_chunked_prefill_parity -- --test-threads=1
	cargo test --release -p rocml --test dense_mixed_kv_parity
	cargo test --release -p rocml --test dense_chunked_prefill_parity
	cargo test --release -p rocml --test snapshot_equivalence
	cargo test --release -p rocml --test ornith_e2e -- --test-threads=1
	cargo test --release -p rocml-serve --test server_e2e
	cargo test --release -p rocml-serve --test server_e2e -- --ignored ornith_tool_call_is_emitted
	cargo test --release -p rocml-cli --test profile_json_smoke

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

# Issue #5's roofline observability: profiled bench run, human-readable
# table printed to stdout. Override MODEL/DEPTH to point at a different
# checkpoint/context depth (e.g. `make bench-profile MODEL=ornith-9b DEPTH=8192`).
MODEL ?= $(QWEN_MODEL)
DEPTH ?= 2048
bench-profile:
	cargo run --release -p rocml-cli -- bench --model $(MODEL) --depth $(DEPTH) --profile

# Same run, but the roofline report (per-phase/per-op/per-layer rows plus
# bandwidth/FLOP-roofline efficiency and wasted-time) is written as JSON to
# OUT instead of printed — the input `docs/prefill-gap-analysis.md`'s
# lever-ranking was built from. Override MODEL/DEPTH/OUT as needed.
OUT ?= bench/profile/$(DEPTH).json
bench-profile-json:
	mkdir -p $(dir $(OUT))
	cargo run --release -p rocml-cli -- bench --model $(MODEL) --depth $(DEPTH) --profile-json $(OUT)

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

# Issue #10's per-layer diff harness (`rocml/tests/mmq_layer_diff.rs`):
# runs Qwen3.5-2B's chunked-prefill path with MMQ off vs on over the same
# real-text prompt and reports per-layer max/mean relative error, worst
# first — the localization step for the int8-MMQ precision investigation.
# Diagnostic tool, not a correctness gate, hence `--ignored`.
mmq-layer-diff:
	cargo test --release -p rocml --test mmq_layer_diff -- --ignored --nocapture

# End-to-end companion to mmq-layer-diff (`rocml/tests/mmq_endtoend_measure.rs`):
# measures (never asserts — the real gate is qwen35_chunked_prefill_parity,
# left untouched) chunked-vs-token-serial final-logits max relative error
# and out-of-tolerance fraction with MMQ off and on, at the gate's own
# prompt lengths — how the int8-MMQ-integration round's 12.4%-15.1% figures
# and this round's `mmq_eligible_by_name` exclusion's real-world effect on
# them were reproduced.
mmq-endtoend-measure:
	cargo test --release -p rocml --test mmq_endtoend_measure -- --ignored --nocapture

# Issue #17's calibration harness (`rocml/tests/mmq_calibrate.rs`): captures
# per-input-channel amax for every MMQ-eligible matmul input over a real-text
# calibration run on Ornith-1.0-9B-Q4_K_M, persisted as a JSON sidecar under
# the OS temp dir for `mmq-smoothquant-measure` to read back.
mmq-calibrate:
	cargo test --release -p rocml --test mmq_calibrate -- --ignored --nocapture

# Issue #17's standalone SmoothQuant-style measurement
# (`rocml/tests/mmq_smoothquant_measure.rs`): CPU-only, reads the calibration
# sidecar `mmq-calibrate` produced plus real GGUF weight bytes, measures
# activation flatness / weight requant error / single-matmul output error at
# alpha in {0.5, 0.65, 0.8} for ssm_out and ffn_down, before deciding whether
# smoothing is worth integrating. Requires `mmq-calibrate` to have run first.
mmq-smoothquant-measure:
	cargo test --release -p rocml --test mmq_smoothquant_measure -- --ignored --nocapture

# Issue #2's per-head boundary-skip measurement (`rocml/tests/kv_head_error_measure.rs`):
# runs Ornith-1.0-9B's fp16-KV decode path over a real prompt, captures the
# exact K/V vectors via the snapshot layer, and measures the per-(layer,head)
# K/V quantization error the production mixed cache would introduce (against
# the same CPU reference the kernels are checked against) — the "does a
# small subset of heads dominate quantization error" measurement the issue
# asked for before implementing a per-head fp16-skip bitmask. Diagnostic
# tool, not a correctness gate, hence `--ignored`.
kv-head-error-measure:
	cargo test --release -p rocml --test kv_head_error_measure -- --ignored --nocapture

# gdn-wmma-lds round (issue #6): same-process interleaved scalar/naive-WMMA/
# LDS-staged-WMMA comparison for GDN chunkwise stages B (ut_build) and F
# (output) at Ornith-1.0-9B's real shape. Informational, not a gate (see
# gdn_chunkwise_wmma_lds.rs for the correctness gate).
gdn-wmma-lds-perf:
	cargo test --release -p rocml-kernels --test gdn_chunkwise_wmma_lds_perf -- --ignored --nocapture

# gdn-uvvnew round (issue #6): same-process interleaved scalar/LDS-staged-WMMA
# comparison for GDN chunkwise stage D+E (uv_vnew) at Ornith-1.0-9B's real
# shape. Informational, not a gate (see gdn_chunkwise_wmma_lds.rs for the
# correctness gate, including the 64-tile long-chain compounding check).
gdn-uvvnew-perf:
	cargo test --release -p rocml-kernels --test gdn_chunkwise_uv_vnew_wmma_lds_perf -- --ignored --nocapture

# Issue #14 phase 2's calibration harness (`rocml/tests/rotational_kv_calibrate.rs`):
# builds and overwrites the checked-in rotational-KV calibration sidecar
# (`rocml/data/rotational_kv_calibration.json`) from real K/V vectors
# captured off Ornith-1.0-9B-Q4_K_M's fp16-KV decode path — pairing scheme,
# per-pair Givens angles, and 2/3/4 bpw Lloyd-Max codebooks, searched and
# picked by measured held-out round-trip error. Only re-run intentionally.
rotational-kv-calibrate:
	cargo test --release -p rocml --test rotational_kv_calibrate -- --ignored --nocapture

# Issue #14 phase 2 step 2 (`rocml/tests/rotational_kv_measure.rs`): the
# tensor-level quality table — rotational (2/3/4 bpw) vs the current scalar
# K(q8)/V(q8,q4) encoding's vector RMSE and attention-score/output
# perturbation, on real held-out Ornith K/V. Requires the calibration
# sidecar above to already exist (checked in; re-run `rotational-kv-calibrate`
# to refresh it).
rotational-kv-measure:
	cargo test --release -p rocml --test rotational_kv_measure -- --ignored --nocapture

# Issue #14 phase 2 step 3(a) (`rocml/tests/rotational_kv_sim_parity.rs`):
# the mixed-KV parity gates' own greedy-check methodology, run under the
# debug rotational simulation (`--kv-rot-sim`) instead of asserting a bound
# — informational, records divergence at 2/3/4 bpw for the decision-gate
# report.
rotational-kv-sim-parity:
	cargo test --release -p rocml --test rotational_kv_sim_parity -- --ignored --nocapture

# Issue #14 phase 2 step 3(b) — THE DECISION GATE: the agentic eval
# (issue #15's `make eval` harness) on Ornith-1.0-9B-Q4_K_M with K left at
# the real production Q8 encoding and V round-tripped through rotational
# quantization at 3 bpw before every window eviction (`--kv-rot-sim 3`).
# Compare this run's overall pass rate + PPL against `make eval`'s own
# fp16-KV/q4-mixed-KV baselines recorded in `.claude/CLAUDE.md` — near-zero
# loss is issue #14's GO signal for phase 3 (fast kernels); visible
# degradation is STOP.
eval-rotational-v3:
	cargo run --release -p rocml-cli -- eval \
		--model ornith-9b --ctx 16384 \
		--kv-cache q8 --kv-rot-sim 3 \
		--label ornith-q4km-rotv3bpw \
		--out bench/eval/results/ornith-q4km-rotv3bpw.json \
		--resume

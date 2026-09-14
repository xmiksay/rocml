//! Issue #1's central correctness gate: capturing a snapshot at position N
//! and restoring it into a state-scrubbed model, then prefilling only the
//! remaining suffix, must produce the same result as prefilling the whole
//! prompt in one call — across several split boundaries (including ones
//! that don't land on the hybrid path's 128-token chunk boundary, and one
//! that straddles the 4096-token auto-snapshot boundary), and for both the
//! dense fp16 KV cache and the KIVI-style mixed-quantized cache (issue #2).
//!
//! Restored bytes are bit-identical to the captured ones (`copy_range_to_host`/
//! `copy_range_from_host` are plain memcpys, no re-quantization), but the
//! *logits* aren't asserted bitwise-equal: the hybrid path's chunked kernels
//! re-chunk each `forward_prompt` call's own slice from its own index 0, so
//! a split that isn't 128-token-aligned changes which tokens share a batched
//! GEMM/attention/GDN-chunk launch, which can reorder floating-point
//! summation — the same effect `qwen35_chunked_prefill_parity.rs` documents
//! for chunked-vs-serial prefill. Near-exact logits (relative tolerance) and
//! exact greedy-token continuations (with that same near-tie escape hatch
//! for a rare flip) is the honest bound here, not bitwise equality.
//!
//! Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`).

use rocml::snapshot::turn::run_turn;
use rocml::snapshot::{KvConfigStamp, ModelStamp, SnapshotStore};
use rocml::{KvCacheMode, LoadOptions, Model, SamplingParams};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const CONTINUATION_LEN: usize = 8;
/// Relative tolerance on the final logits — see the module doc for why this
/// is near-exact rather than bitwise (reduction-order sensitivity from
/// re-chunking at a non-128-aligned split), and issue #1's own text for why
/// this bound (not bitwise) is the honest one to assert.
///
/// Recalibrated from 1e-6 for the WMMA prefill GEMM: a restore+suffix
/// prefill re-chunks at a different offset than the full prefill, so
/// different rows go through the f16-operand WMMA kernel vs the f32 scalar
/// fallback — a precision-mode difference (same mechanism and magnitude as
/// `qwen35_chunked_prefill_parity`'s recalibration, measured <=0.55%), not
/// snapshot corruption. Exact greedy continuation (near-tie escape below)
/// remains the behavioral gate.
const LOGITS_REL_TOL: f32 = 1e-2;
/// Same near-tie escape hatch as `qwen35_chunked_prefill_parity.rs`: a
/// greedy-continuation disagreement only passes if the loser's own top-1/
/// top-2 gap was already this small (a real divergence has a much larger
/// gap and fails the test).
const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;

fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 5) % vocab_size).collect()
}

fn top2_gap(logits: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in logits {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    (best - second) / best.abs().max(1.0)
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("logits must be non-empty")
}

fn assert_logits_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        let tol = LOGITS_REL_TOL * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "{label}[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

fn greedy_continue(model: &mut Model, mut logits: Vec<f32>, n: usize) -> Vec<(u32, f32)> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let gap = top2_gap(&logits);
        let next = argmax(&logits);
        out.push((next, gap));
        logits = model.forward_token(next).expect("forward_token failed");
    }
    out
}

fn assert_continuations_match(label: &str, full: &[(u32, f32)], restored: &[(u32, f32)]) {
    for (step, ((full_tok, full_gap), (restored_tok, restored_gap))) in
        full.iter().zip(restored).enumerate()
    {
        if full_tok == restored_tok {
            continue;
        }
        assert!(
            full_gap.abs() <= NEAR_TIE_RELATIVE_GAP || restored_gap.abs() <= NEAR_TIE_RELATIVE_GAP,
            "{label} continuation step {step}: full-prefill picked {full_tok} (gap {full_gap}), \
             restored picked {restored_tok} (gap {restored_gap}) — gap too large to be a \
             documented near-tie"
        );
        eprintln!(
            "{label} continuation step {step}: documented near-tie flip, full={full_tok} (gap \
             {full_gap}) vs restored={restored_tok} (gap {restored_gap})"
        );
    }
}

/// Runs the equivalence check for one `(prompt, split)` pair: `model_full`
/// does the reference full prefill; `model_restored` is deliberately dirtied
/// with a *different* prompt of the same prefix length before the restore,
/// so a pass here proves the restore genuinely overwrites whatever state was
/// resident, not just that it happened to already match.
fn run_split_case(
    model_full: &mut Model,
    model_restored: &mut Model,
    full_prompt: &[u32],
    split: usize,
) {
    let label = format!("len={} split={split}", full_prompt.len());
    let vocab_size = model_full.vocab_size();
    let prefix = &full_prompt[..split];
    let suffix = &full_prompt[split..];

    model_full.reset().expect("reset failed");
    let full_logits = model_full
        .forward_prompt(full_prompt, None)
        .expect("forward_prompt failed");
    let full_continuation = greedy_continue(model_full, full_logits.clone(), CONTINUATION_LEN);

    // Capture at `split` from a model that processed exactly `prefix`.
    model_restored.reset().expect("reset failed");
    model_restored
        .forward_prompt(prefix, None)
        .expect("forward_prompt failed");
    let snap = model_restored
        .as_hybrid()
        .expect("qwen35 hybrid model")
        .capture_snapshot(prefix.to_vec())
        .expect("capture_snapshot failed");

    // Dirty every buffer the snapshot covers with an unrelated prompt of the
    // same length before restoring, so a stale-leftover match can't fake a
    // pass.
    model_restored.reset().expect("reset failed");
    let dirty_prompt: Vec<u32> = prefix.iter().map(|&t| (t + 1) % vocab_size).collect();
    model_restored
        .forward_prompt(&dirty_prompt, None)
        .expect("forward_prompt failed");

    model_restored.reset().expect("reset failed");
    model_restored
        .as_hybrid_mut()
        .expect("qwen35 hybrid model")
        .restore_snapshot(&snap)
        .expect("restore_snapshot failed");
    assert_eq!(model_restored.position(), split as u32);
    let restored_logits = model_restored
        .forward_prompt(suffix, None)
        .expect("forward_prompt failed");
    let restored_continuation =
        greedy_continue(model_restored, restored_logits.clone(), CONTINUATION_LEN);

    assert_logits_close(
        &restored_logits,
        &full_logits,
        &format!("{label} final logits"),
    );
    assert_continuations_match(&label, &full_continuation, &restored_continuation);
}

/// Long enough to cross the 4096-token auto-snapshot boundary.
fn check_fp16_split_equivalence(path: &std::path::Path) {
    const PROMPT_LEN: usize = 4200;
    const CTX: usize = 5000;
    let opts = LoadOptions::new(CTX).with_kv_cache(KvCacheMode::Fp16);
    let mut model_full = Model::load(path, opts).expect("Model::load failed");
    let mut model_restored = Model::load(path, opts).expect("Model::load failed");
    let vocab_size = model_full.vocab_size();
    let prompt = synthetic_prompt(PROMPT_LEN, vocab_size);

    for &split in &[
        1usize,
        127,
        128,
        500,
        4095,
        4096,
        4097,
        4150,
        PROMPT_LEN - 1,
    ] {
        run_split_case(&mut model_full, &mut model_restored, &prompt, split);
    }
}

/// Long enough to have real sink (32) + window (128) + at least one evicted
/// bulk block (see `kv_quant::layout`'s `SINK_LEN`/`WINDOW_LEN`).
fn check_mixed_q8_split_equivalence(path: &std::path::Path) {
    const PROMPT_LEN: usize = 400;
    const CTX: usize = 1024;
    let opts = LoadOptions::new(CTX).with_kv_cache(KvCacheMode::Q8);
    let mut model_full = Model::load(path, opts).expect("Model::load failed");
    let mut model_restored = Model::load(path, opts).expect("Model::load failed");
    let vocab_size = model_full.vocab_size();
    let prompt = synthetic_prompt(PROMPT_LEN, vocab_size);

    for &split in &[1usize, 33, 160, 200, PROMPT_LEN - 1] {
        run_split_case(&mut model_full, &mut model_restored, &prompt, split);
    }
}

/// Immutability (issue #1's correctness gate): restoring the same snapshot
/// twice, each time into a freshly-dirtied model, produces bit-identical
/// forward-pass output both times — proving `restore_snapshot` never
/// mutates the stored `SnapshotData` and that the restore itself is
/// deterministic.
fn check_double_restore_is_deterministic(path: &std::path::Path) {
    let opts = LoadOptions::new(2048).with_kv_cache(KvCacheMode::Fp16);
    let mut model = Model::load(path, opts).expect("Model::load failed");
    let vocab_size = model.vocab_size();
    let prompt = synthetic_prompt(300, vocab_size);
    let split = 200;
    let (prefix, suffix) = prompt.split_at(split);

    model.reset().expect("reset failed");
    model
        .forward_prompt(prefix, None)
        .expect("forward_prompt failed");
    let snap = model
        .as_hybrid()
        .expect("qwen35 hybrid model")
        .capture_snapshot(prefix.to_vec())
        .expect("capture_snapshot failed");

    let mut run_once = || -> Vec<f32> {
        model.reset().expect("reset failed");
        model
            .as_hybrid_mut()
            .expect("qwen35 hybrid model")
            .restore_snapshot(&snap)
            .expect("restore_snapshot failed");
        model
            .forward_prompt(suffix, None)
            .expect("forward_prompt failed")
    };

    let first = run_once();
    let second = run_once();
    assert_eq!(first, second, "double restore must be bit-identical");
}

/// End-to-end proof that `SnapshotStore`/`run_turn` (the exact mechanism
/// `rocml-serve`'s worker and `rocml-cli chat` drive) produces a real hit on
/// a second turn — the companion to `rocml-serve/tests/server_e2e.rs`'s
/// output-equivalence check, which can't observe hit/miss through the HTTP
/// API alone.
fn check_run_turn_reuses_a_prior_end_of_turn_snapshot(path: &std::path::Path) {
    let ctx = 4096;
    let opts = LoadOptions::new(ctx).with_kv_cache(KvCacheMode::Fp16);
    let mut model = Model::load(path, opts).expect("Model::load failed");
    let tokenizer = rocml_core::tokenizer::BpeTokenizer::from_gguf(
        &rocml_core::gguf::GgufFile::open(path).expect("gguf open failed"),
    )
    .expect("tokenizer load failed");
    let vocab_size = model.vocab_size();
    let model_stamp = ModelStamp::from_path(path).expect("stamp failed");
    let kv_config = KvConfigStamp {
        mode: KvCacheMode::Fp16,
        ctx,
    };
    let mut store = SnapshotStore::new(64 * 1024 * 1024, None, 0).expect("store failed");
    let sampling = SamplingParams::greedy();

    let turn1_prompt = synthetic_prompt(64, vocab_size);
    let outcome1 = run_turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &model_stamp,
        &kv_config,
        &turn1_prompt,
        16,
        false,
        &sampling,
        &[],
        None,
        |_| {},
    )
    .expect("turn 1 failed");
    assert_eq!(outcome1.reused_prefix, 0, "turn 1 must be a cold miss");

    let mut turn2_prompt = turn1_prompt.clone();
    turn2_prompt.extend_from_slice(&outcome1.stats.generated_ids);
    let extra: Vec<u32> = synthetic_prompt(20, vocab_size)
        .into_iter()
        .map(|t| (t + 1) % vocab_size)
        .collect();
    turn2_prompt.extend_from_slice(&extra);

    let outcome2 = run_turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &model_stamp,
        &kv_config,
        &turn2_prompt,
        16,
        false,
        &sampling,
        &[],
        None,
        |_| {},
    )
    .expect("turn 2 failed");

    assert!(
        outcome2.reused_prefix > 0,
        "turn 2 should have reused turn 1's end-of-turn snapshot"
    );
    assert_eq!(
        outcome2.reused_prefix as usize,
        turn1_prompt.len() + outcome1.stats.generated_ids.len(),
        "turn 2 should reuse exactly turn 1's whole final conversation length"
    );
}

/// One `#[test]` running every scenario above in sequence — deliberately not
/// four separate `#[test]` functions: `cargo test`'s default parallelism
/// would then load up to 6 concurrent `Model` instances on one GPU (two per
/// split-equivalence scenario, one each for the other two), which reliably
/// exhausts this project's 16GB dev GPU's VRAM. Running sequentially in one
/// test also means each scenario's `Model`s are dropped (freeing VRAM) before
/// the next scenario loads its own.
#[test]
fn qwen35_snapshot_equivalence() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    check_fp16_split_equivalence(&path);
    check_mixed_q8_split_equivalence(&path);
    check_double_restore_is_deterministic(&path);
    check_run_turn_reuses_a_prior_end_of_turn_snapshot(&path);
}

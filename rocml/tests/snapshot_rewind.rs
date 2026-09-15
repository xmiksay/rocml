//! Gate for the snapshot layer's GPU tier (`qwen35::forward::rewind`): a
//! rewind point saved at position N, followed by an arbitrary amount of
//! *other* decode work past N (which advances the GDN state, recycles and
//! evicts the mixed cache's recent window, and overwrites every dense-plane
//! row from N up), then a rewind and a prefill of the real suffix, must
//! land in *exactly* the state a `reset()` + host `restore_snapshot` of the
//! same point does (bitwise-equal logits and greedy continuation — both
//! paths re-chunk the suffix identically, so nothing else may differ), and,
//! for fp16/Q8, must also match a from-scratch full prefill within the same
//! near-exact bound `snapshot_equivalence.rs` documents for the host tier.
//!
//! Also pins the validity invariant `qwen35::cache::rewind` states: a slot
//! goes stale once positions below it are rewritten (a rewind to a lower
//! slot followed by new tokens, or `reset()`). `run_turn`'s tier policy on
//! top of this (GPU first, host fallback, store-off means all off) is
//! `snapshot_rewind_turns.rs`, split out for the 400-line cap.
//!
//! Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`).

use rocml::qwen35::forward::RewindSlot;
use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

mod support;

use support::snapshot_check::{
    assert_continuations_match, assert_logits_close, greedy_continue, synthetic_prompt,
    CONTINUATION_LEN,
};

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
/// Same bounds, same rationale as `snapshot_equivalence.rs`.
const LOGITS_REL_TOL: f32 = 1e-2;
const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;
/// Extra tokens processed past the rewind point before rewinding — long
/// enough to wrap and evict the mixed cache's 128-token window at least
/// twice, so the window restore is genuinely exercised.
const DIRTY_EXTRA: usize = 300;

/// One `(prompt, split)` case on a single model. The primary assertion is
/// the GPU tier's exact claim: after a rewind, the suffix prefill's logits
/// and greedy continuation are *bitwise identical* to those after a
/// `reset()` + host `restore_snapshot` of a capture taken at the same
/// point — both paths then run the identical re-chunked suffix, so any
/// difference would be a state the rewind failed to restore. With
/// `check_vs_full`, the host tier's near-exact bound against a monolithic
/// full prefill is asserted too (fp16/Q8 only: under Q4Mixed the split
/// point itself changes which fp32 values reach the 4-bit eviction
/// quantizer, and the resulting drift — measured 2%-10% max relative here,
/// identically for the host restore — is `mixed_kv_chunked_prefill_parity`'s
/// already-documented territory, not something this tier adds).
fn run_rewind_case(model: &mut Model, full_prompt: &[u32], split: usize, check_vs_full: bool) {
    let label = format!("len={} split={split}", full_prompt.len());
    let vocab_size = model.vocab_size();
    let (prefix, suffix) = full_prompt.split_at(split);

    let full = check_vs_full.then(|| {
        model.reset().expect("reset failed");
        let logits = model
            .forward_prompt(full_prompt, None)
            .expect("forward_prompt failed");
        let continuation = greedy_continue(model, logits.clone(), CONTINUATION_LEN);
        (logits, continuation)
    });

    model.reset().expect("reset failed");
    model
        .forward_prompt(prefix, None)
        .expect("forward_prompt failed");
    let hybrid = model.as_hybrid_mut().expect("qwen35 hybrid model");
    hybrid
        .save_rewind_point(RewindSlot::StableBoundary, prefix.to_vec())
        .expect("save_rewind_point failed");
    let snap = hybrid
        .capture_snapshot(prefix.to_vec())
        .expect("capture_snapshot failed");
    assert_eq!(
        hybrid.snapshot_byte_size(),
        snap.byte_size(),
        "{label}: snapshot_byte_size must match a real capture's byte_size"
    );

    // Everything from `split` on gets rewritten by unrelated tokens, and the
    // sequence runs well past the prompt's own end.
    let dirty: Vec<u32> = (0..suffix.len() + DIRTY_EXTRA)
        .map(|i| (full_prompt[(split + i) % full_prompt.len()] + 1) % vocab_size)
        .collect();
    model
        .forward_prompt(&dirty, None)
        .expect("forward_prompt failed");

    let rewound = model
        .as_hybrid_mut()
        .expect("qwen35 hybrid model")
        .rewind_to_prefix(full_prompt)
        .expect("rewind_to_prefix failed");
    assert_eq!(rewound, Some(split as u32), "{label}: rewind position");
    assert_eq!(model.position(), split as u32);
    let rewind_logits = model
        .forward_prompt(suffix, None)
        .expect("forward_prompt failed");
    let rewind_continuation = greedy_continue(model, rewind_logits.clone(), CONTINUATION_LEN);

    model.reset().expect("reset failed");
    model
        .as_hybrid_mut()
        .expect("qwen35 hybrid model")
        .restore_snapshot(&snap)
        .expect("restore_snapshot failed");
    let host_logits = model
        .forward_prompt(suffix, None)
        .expect("forward_prompt failed");
    let host_continuation = greedy_continue(model, host_logits.clone(), CONTINUATION_LEN);

    assert_eq!(
        rewind_logits, host_logits,
        "{label}: a GPU rewind must land in exactly the host restore's state"
    );
    let tokens = |c: &[(u32, f32)]| c.iter().map(|(t, _)| *t).collect::<Vec<_>>();
    assert_eq!(
        tokens(&rewind_continuation),
        tokens(&host_continuation),
        "{label}"
    );

    if let Some((full_logits, full_continuation)) = full {
        assert_logits_close(
            &rewind_logits,
            &full_logits,
            LOGITS_REL_TOL,
            &format!("{label} final logits"),
        );
        assert_continuations_match(
            &label,
            &full_continuation,
            &rewind_continuation,
            NEAR_TIE_RELATIVE_GAP,
        );
    }
}

/// Crosses the 4096-token prefill-capture grid, like the host-tier gate.
fn check_fp16_rewind_equivalence(path: &std::path::Path) {
    const PROMPT_LEN: usize = 4200;
    let opts = LoadOptions::new(PROMPT_LEN + DIRTY_EXTRA + 64).with_kv_cache(KvCacheMode::Fp16);
    let mut model = Model::load(path, opts).expect("Model::load failed");
    let prompt = synthetic_prompt(PROMPT_LEN, model.vocab_size());
    for &split in &[1usize, 127, 128, 500, 4095, 4097, 4150, PROMPT_LEN - 1] {
        run_rewind_case(&mut model, &prompt, split, true);
    }
}

/// Splits inside the sink (33 > SINK_LEN 32), exactly at the first eviction
/// boundary (160 = 32 + 128), and mid-window — every region of the mixed
/// layout the rewind has to get right.
fn check_mixed_rewind_equivalence(path: &std::path::Path, mode: KvCacheMode) {
    const PROMPT_LEN: usize = 400;
    let opts = LoadOptions::new(1024).with_kv_cache(mode);
    let mut model = Model::load(path, opts).expect("Model::load failed");
    let prompt = synthetic_prompt(PROMPT_LEN, model.vocab_size());
    for &split in &[1usize, 33, 160, 200, PROMPT_LEN - 1] {
        run_rewind_case(&mut model, &prompt, split, mode != KvCacheMode::Q4Mixed);
    }
}

/// The validity invariant, at the model API: a rewind to a lower slot
/// followed by new tokens stales the higher slot; `reset()` stales both.
fn check_rewind_invalidation(path: &std::path::Path) {
    let opts = LoadOptions::new(2048).with_kv_cache(KvCacheMode::Fp16);
    let mut model = Model::load(path, opts).expect("Model::load failed");
    let vocab_size = model.vocab_size();
    let a = synthetic_prompt(300, vocab_size);
    let extra: Vec<u32> = synthetic_prompt(20, vocab_size)
        .into_iter()
        .map(|t| (t + 7) % vocab_size)
        .collect();
    let mut a_plus = a.clone();
    a_plus.extend_from_slice(&extra);

    model.reset().expect("reset failed");
    model
        .forward_prompt(&a[..200], None)
        .expect("prefill failed");
    model
        .as_hybrid_mut()
        .unwrap()
        .save_rewind_point(RewindSlot::StableBoundary, a[..200].to_vec())
        .expect("save failed");
    model
        .forward_prompt(&a[200..], None)
        .expect("prefill failed");
    model
        .as_hybrid_mut()
        .unwrap()
        .save_rewind_point(RewindSlot::EndOfTurn, a.clone())
        .expect("save failed");

    // Both slots valid: the longer one wins.
    assert_eq!(
        model
            .as_hybrid_mut()
            .unwrap()
            .rewind_to_prefix(&a_plus)
            .unwrap(),
        Some(300)
    );
    // Rewinding to 300 stales nothing below it, so the stable slot is still
    // there for a prompt that diverges after position 200...
    let mut b = a[..200].to_vec();
    b.extend(a[200..].iter().map(|t| (t + 3) % vocab_size));
    b.extend_from_slice(&extra);
    assert_eq!(
        model.as_hybrid_mut().unwrap().rewind_to_prefix(&b).unwrap(),
        Some(200)
    );
    // ...and prefilling b's tail rewrites positions 200.., so the end-of-turn
    // slot (saved at 300 over a's tail) must now be stale: a_plus falls back
    // to the stable slot, never to a 300-token match over b's rows.
    model
        .forward_prompt(&b[200..300], None)
        .expect("prefill failed");
    assert_eq!(
        model
            .as_hybrid_mut()
            .unwrap()
            .rewind_to_prefix(&a_plus)
            .unwrap(),
        Some(200),
        "a slot above a rewind point must be invalidated once new tokens are written past it"
    );
    // A fresh sequence stales everything.
    model.reset().expect("reset failed");
    assert_eq!(
        model
            .as_hybrid_mut()
            .unwrap()
            .rewind_to_prefix(&a_plus)
            .unwrap(),
        None
    );
}

/// One `#[test]` running every scenario in sequence, for the same VRAM
/// reason `snapshot_equivalence.rs` gives.
#[test]
fn qwen35_gpu_rewind_points() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    check_fp16_rewind_equivalence(&path);
    check_mixed_rewind_equivalence(&path, KvCacheMode::Q8);
    check_mixed_rewind_equivalence(&path, KvCacheMode::Q4Mixed);
    check_rewind_invalidation(&path);
}

//! One turn's worth of orchestration: reset, snapshot lookup + restore,
//! resumed prefill (with periodic mid-prefill capture), decode, then an
//! end-of-turn capture — the single place `rocml-serve`'s worker and
//! `rocml-cli`'s `chat`/`bench --turns` all drive through, so the snapshot
//! wiring (issue #1) exists exactly once. Deliberately outside
//! `crate::generate`'s hot loop: `SnapshotStore::lookup`/`insert` and the
//! GPU D2H/H2D capture/restore calls happen only at prefill-segment
//! boundaries and turn end, never per decode token.
//!
//! Dense `qwen3` models take the plain "always full prefill" path
//! automatically here — `Model::as_hybrid` returns `None` for them, so every
//! snapshot lookup/capture is skipped and generation runs exactly as it did
//! before this module existed.

use std::time::Instant;

use rocml_core::tokenizer::BpeTokenizer;

use super::{KvConfigStamp, ModelStamp, SnapshotStore};
use crate::error::RocmlError;
use crate::generate::{generate_sampled_with_stop_resumed, GenerateStats};
use crate::model::Model;
use crate::sample::SamplingParams;

pub struct TurnOutcome {
    pub stats: GenerateStats,
    /// Tokens reused from a restored snapshot (0 means no hit, or a
    /// dense/non-hybrid model where snapshots don't apply).
    pub reused_prefix: u32,
    /// Wall time spent H2D-restoring a snapshot (0 on a miss).
    pub restore_seconds: f64,
    /// Wall time spent D2H-capturing snapshots this turn — usually one
    /// (end-of-turn), more for a long prefill that crossed a 4096-token
    /// boundary (`rocml-cli bench --turns` reports both separately so
    /// per-turn overhead is visible).
    pub capture_seconds: f64,
}

/// Runs one full request/turn: `model.reset()`, an optional snapshot
/// restore, prefill of whatever's left (auto-capturing every 4096 tokens for
/// a long prefill — see `crate::generate`'s `PREFILL_CAPTURE_INTERVAL`),
/// sampled decode honoring `stop_strings` (pass `&[]` for a caller that
/// doesn't need OpenAI-style stop strings, e.g. the CLI REPL — an empty list
/// never matches, so this reduces to plain unconditional decode), and a
/// final end-of-turn capture of the whole conversation (prompt + generated).
///
/// `store: None` (or a `SnapshotStore` with both tiers disabled) makes this
/// behave exactly like calling `generate_sampled_with_stop` directly — every
/// lookup/capture becomes a no-op.
#[allow(clippy::too_many_arguments)]
pub fn run_turn(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    mut store: Option<&mut SnapshotStore>,
    model_stamp: &ModelStamp,
    kv_config: &KvConfigStamp,
    full_prompt_ids: &[u32],
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    stop_strings: &[String],
    mut on_text: impl FnMut(&str),
) -> Result<TurnOutcome, RocmlError> {
    model.reset()?;
    let is_hybrid = model.as_hybrid().is_some();

    let mut reused_prefix = 0u32;
    let mut restore_seconds = 0.0;
    if is_hybrid {
        let hit = store
            .as_deref_mut()
            .and_then(|s| s.lookup(model_stamp, kv_config, full_prompt_ids));
        if let Some(snap) = hit {
            let start = Instant::now();
            model
                .as_hybrid_mut()
                .expect("is_hybrid checked above")
                .restore_snapshot(&snap)?;
            restore_seconds = start.elapsed().as_secs_f64();
            reused_prefix = snap.position;
        }
    }

    let mut capture_seconds = 0.0;
    let prefill_boundary = |m: &mut Model, processed: usize| {
        let (Some(s), Some(hybrid)) = (store.as_deref_mut(), m.as_hybrid()) else {
            return;
        };
        let start = Instant::now();
        match hybrid.capture_snapshot(full_prompt_ids[..processed].to_vec()) {
            Ok(snap) => {
                capture_seconds += start.elapsed().as_secs_f64();
                s.insert(model_stamp.clone(), *kv_config, snap);
            }
            Err(e) => eprintln!("snapshot: prefill-boundary capture failed: {e}"),
        }
    };

    let mut decoded_so_far = String::new();
    let stats = generate_sampled_with_stop_resumed(
        model,
        tokenizer,
        full_prompt_ids,
        reused_prefix as usize,
        max_new_tokens,
        stop_on_eos,
        params,
        prefill_boundary,
        |chunk| {
            on_text(chunk);
            decoded_so_far.push_str(chunk);
            stop_matched(&decoded_so_far, stop_strings)
        },
    )?;

    if let (Some(s), Some(hybrid)) = (store, model.as_hybrid()) {
        let mut full_tokens = full_prompt_ids.to_vec();
        full_tokens.extend_from_slice(&stats.generated_ids);
        let start = Instant::now();
        match hybrid.capture_snapshot(full_tokens) {
            Ok(snap) => {
                capture_seconds += start.elapsed().as_secs_f64();
                s.insert(model_stamp.clone(), *kv_config, snap);
            }
            Err(e) => eprintln!("snapshot: end-of-turn capture failed: {e}"),
        }
    }

    Ok(TurnOutcome {
        stats,
        reused_prefix,
        restore_seconds,
        capture_seconds,
    })
}

/// An empty stop string never matches (it would trivially match any text and
/// halt generation on the first chunk) — `contains` alone can't distinguish
/// "not yet seen" from "seen the whole prompt so far", so an empty string
/// must be filtered explicitly rather than relying on it. Moved here from
/// `rocml-serve`'s worker (issue #1's `run_turn` now owns the whole decode
/// loop, not just the model-serving binary).
fn stop_matched(decoded_so_far: &str, stop_strings: &[String]) -> bool {
    stop_strings
        .iter()
        .any(|s| !s.is_empty() && decoded_so_far.contains(s.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_stop_strings_never_matches() {
        assert!(!stop_matched("hello world", &[]));
    }

    #[test]
    fn matches_substring_anywhere_in_accumulated_text() {
        let stops = vec!["STOP".to_string()];
        assert!(!stop_matched("hello wor", &stops));
        assert!(stop_matched("hello world STOP here", &stops));
    }

    #[test]
    fn empty_stop_string_is_ignored() {
        let stops = vec![String::new(), "END".to_string()];
        assert!(!stop_matched("anything at all", &stops));
        assert!(stop_matched("reached the END now", &stops));
    }

    #[test]
    fn matches_first_of_several_stop_strings() {
        let stops = vec!["```".to_string(), "\n\n".to_string()];
        assert!(stop_matched("some code```", &stops));
        assert!(stop_matched("line one\n\nline two", &stops));
        assert!(!stop_matched("no stop here", &stops));
    }
}

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

use super::{KvConfigStamp, ModelStamp, SnapshotStore, SnapshotTier};
use crate::error::RocmlError;
use crate::generate::{generate_sampled_with_stop_resumed, GenerateStats};
use crate::model::Model;
use crate::qwen35::forward::RewindSlot;
use crate::sample::SamplingParams;

/// Which tier a turn's prefix reuse came from — see `crate::snapshot`'s
/// module doc for the three tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HitSource {
    Gpu,
    Ram,
    Disk,
}

impl HitSource {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Gpu => "gpu",
            Self::Ram => "ram",
            Self::Disk => "disk",
        }
    }
}

pub struct TurnOutcome {
    pub stats: GenerateStats,
    /// Tokens reused from a restored snapshot (0 means no hit, or a
    /// dense/non-hybrid model where snapshots don't apply).
    pub reused_prefix: u32,
    /// Where `reused_prefix` came from; `None` on a miss.
    pub hit_source: Option<HitSource>,
    /// Wall time spent restoring (a GPU rewind, or an H2D snapshot copy);
    /// 0 on a miss.
    pub restore_seconds: f64,
    /// Wall time spent saving state this turn: the GPU rewind points (cheap)
    /// plus whatever host captures the policy in [`run_turn`] took
    /// (`rocml-cli bench --turns` reports both separately so per-turn
    /// overhead is visible).
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
///
/// `stable_boundary` (issue #12, the thinking-enabled multi-turn snapshot
/// miss): an absolute token index into `full_prompt_ids` where the caller's
/// renderer knows the prompt is render-stable — see
/// `crate::chat::render_with_boundary` and [`stable_boundary_tokens`] for
/// how to derive it. When `Some`, an *extra* mid-prefill capture happens
/// there (on top of the existing prefill-boundary and end-of-turn captures),
/// so the next turn's re-rendered prompt has a prefix it can always hit even
/// when the prompt's own volatile tail (generation prompt, `<think>` block,
/// reply) isn't reproduced byte-for-byte by that next render. `None` is a
/// no-op — only the two pre-existing capture points fire, matching the
/// pre-issue-#12 behavior (still useful for thinking-off models and plain
/// regeneration, which don't need this).
///
/// **Tier policy.** With a `Some` store, the hybrid model's GPU rewind
/// points (`qwen35::forward::rewind`) are consulted first — a growing
/// single-session conversation hits there every turn with no host traffic —
/// and the RAM/disk store only on a GPU miss (a second conversation, an
/// edited history, a restarted server with `--snapshot-dir`). Saves per
/// turn: the stable boundary goes to the GPU `StableBoundary` slot *and*
/// the store as its pinned entry; the end of turn goes to the GPU
/// `EndOfTurn` slot (and to the store, pinned, only when the caller passed
/// no `stable_boundary`, so the store always holds exactly the one snapshot
/// the next turn is expected to hit); the periodic 4096-token mid-prefill
/// capture is taken only when the RAM budget can hold it beside the pinned
/// entry (it is crash/abort insurance for a long prefill, not something
/// worth a multi-GB D2H copy that would be evicted before it's ever used).
/// The prompt-end capture the pre-GPU-tier design took is gone: it's
/// covered by the end-of-turn point in every case a next turn can match.
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
    stable_boundary: Option<u32>,
    mut on_text: impl FnMut(&str),
) -> Result<TurnOutcome, RocmlError> {
    let is_hybrid = model.as_hybrid().is_some();
    let snapshots_on = is_hybrid && store.is_some();

    let mut reused_prefix = 0u32;
    let mut restore_seconds = 0.0;
    let mut hit_source = None;
    if snapshots_on {
        let start = Instant::now();
        let hybrid = model.as_hybrid_mut().expect("is_hybrid checked above");
        if let Some(pos) = hybrid.rewind_to_prefix(full_prompt_ids)? {
            restore_seconds = start.elapsed().as_secs_f64();
            reused_prefix = pos;
            hit_source = Some(HitSource::Gpu);
        }
    }
    if hit_source.is_none() {
        model.reset()?;
        let hit = if snapshots_on {
            store
                .as_deref_mut()
                .and_then(|s| s.lookup_with_tier(model_stamp, kv_config, full_prompt_ids))
        } else {
            None
        };
        if let Some((snap, tier)) = hit {
            let start = Instant::now();
            model
                .as_hybrid_mut()
                .expect("is_hybrid checked above")
                .restore_snapshot(&snap)?;
            restore_seconds = start.elapsed().as_secs_f64();
            reused_prefix = snap.position;
            hit_source = Some(match tier {
                SnapshotTier::Ram => HitSource::Ram,
                SnapshotTier::Disk => HitSource::Disk,
            });
        }
    }

    let mut capture_seconds = 0.0;
    let prefill_boundary = |m: &mut Model, processed: usize| {
        let (Some(s), Some(hybrid)) = (store.as_deref_mut(), m.as_hybrid_mut()) else {
            return;
        };
        let start = Instant::now();
        let prefix = &full_prompt_ids[..processed];
        if stable_boundary.is_some_and(|b| b as usize == processed) {
            if let Err(e) = hybrid.save_rewind_point(RewindSlot::StableBoundary, prefix.to_vec()) {
                eprintln!("snapshot: stable-boundary GPU rewind save failed: {e}");
            }
            if s.would_keep_pinned(hybrid.snapshot_byte_size()) {
                match hybrid.capture_snapshot(prefix.to_vec()) {
                    Ok(snap) => s.insert_pinned(model_stamp.clone(), *kv_config, snap),
                    Err(e) => eprintln!("snapshot: stable-boundary capture failed: {e}"),
                }
            }
        } else if processed < full_prompt_ids.len() && s.would_keep(hybrid.snapshot_byte_size()) {
            match hybrid.capture_snapshot(prefix.to_vec()) {
                Ok(snap) => s.insert(model_stamp.clone(), *kv_config, snap),
                Err(e) => eprintln!("snapshot: prefill-boundary capture failed: {e}"),
            }
        }
        capture_seconds += start.elapsed().as_secs_f64();
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
        stable_boundary.map(|b| b as usize),
        prefill_boundary,
        |chunk| {
            on_text(chunk);
            decoded_so_far.push_str(chunk);
            stop_matched(&decoded_so_far, stop_strings)
        },
    )?;

    if let (Some(s), Some(hybrid)) = (store, model.as_hybrid_mut()) {
        let mut full_tokens = full_prompt_ids.to_vec();
        full_tokens.extend_from_slice(&stats.generated_ids);
        // A stop-string ending pushes its last token without running it
        // through the model, so the cache is one token short of
        // prompt + generated; save the prefix it actually holds.
        full_tokens.truncate(hybrid.position() as usize);
        let start = Instant::now();
        if let Err(e) = hybrid.save_rewind_point(RewindSlot::EndOfTurn, full_tokens.clone()) {
            eprintln!("snapshot: end-of-turn GPU rewind save failed: {e}");
        }
        if stable_boundary.is_none() && s.would_keep_pinned(hybrid.snapshot_byte_size()) {
            match hybrid.capture_snapshot(full_tokens) {
                Ok(snap) => s.insert_pinned(model_stamp.clone(), *kv_config, snap),
                Err(e) => eprintln!("snapshot: end-of-turn capture failed: {e}"),
            }
        }
        capture_seconds += start.elapsed().as_secs_f64();
    }

    Ok(TurnOutcome {
        stats,
        reused_prefix,
        hit_source,
        restore_seconds,
        capture_seconds,
    })
}

/// Converts a `crate::chat::render_with_boundary` byte offset into the
/// token-index `run_turn`'s `stable_boundary` expects, verifying the
/// invariant the whole scheme rests on: `prompt[..boundary_byte]` must
/// tokenize to a genuine prefix of `prompt_ids` (the render's history
/// portion always ends on the `<|im_end|>` special-token boundary, which
/// can't BPE-merge with anything — see `render_with_boundary`'s doc comment
/// — so this holds for any well-formed render). Returns `None` rather than a
/// wrong boundary if it doesn't (a future template change breaking the
/// invariant, say): that only costs this turn's stable-boundary capture, not
/// correctness.
pub fn stable_boundary_tokens(
    tokenizer: &BpeTokenizer,
    prompt_text: &str,
    boundary_byte: usize,
    prompt_ids: &[u32],
) -> Option<u32> {
    let prefix_ids = tokenizer.encode(&prompt_text[..boundary_byte]);
    prompt_ids
        .starts_with(prefix_ids.as_slice())
        .then_some(prefix_ids.len() as u32)
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

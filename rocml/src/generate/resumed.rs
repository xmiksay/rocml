//! Snapshot-aware generation entry points (issue #1) plus `generate_core`,
//! the shared engine every function in `super` and this file ultimately
//! calls — split out of `generate.rs` purely for the 400-line file cap, see
//! that module's doc comment.

use std::time::Instant;

use rocml_core::tokenizer::BpeTokenizer;

use super::GenerateStats;
use crate::error::RocmlError;
use crate::model::Model;
use crate::profile::{Phase, Profiler};
use crate::sample::{self, Rng, SamplingParams};

/// Like [`super::generate_sampled`], but the model's cache already holds the
/// first `already_processed` tokens of `prompt_ids` (e.g. restored from a
/// snapshot, see `crate::snapshot::turn::run_turn`) — only
/// `prompt_ids[already_processed..]` is actually run through a forward
/// pass. `prompt_ids` as a whole (not just the suffix) still seeds sampling
/// history (repeat-penalty correctness).
///
/// `prefill_boundary` fires once after the initial prefill (whether or not
/// there was anything left to prefill) and, for a long suffix, again every
/// 4096 *absolute* tokens processed — see issue #1's "every 4096 processed
/// tokens during long prefills" auto-snapshot trigger. It receives the
/// model (so it can call `Model::as_hybrid_mut().capture_snapshot(..)`) and
/// the absolute position reached so far.
#[allow(clippy::too_many_arguments)]
pub fn generate_sampled_resumed(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    already_processed: usize,
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    prefill_boundary: impl FnMut(&mut Model, usize),
    mut on_text: impl FnMut(&str),
) -> Result<GenerateStats, RocmlError> {
    generate_core(
        model,
        tokenizer,
        prompt_ids,
        already_processed,
        max_new_tokens,
        stop_on_eos,
        params,
        None,
        None,
        prefill_boundary,
        |s| {
            on_text(s);
            false
        },
    )
}

/// Like [`super::generate_sampled_with_stop`], with the same
/// resumed-from-a-snapshot semantics as [`generate_sampled_resumed`] — this
/// is the variant `rocml-serve`'s worker uses (it needs OpenAI `stop` string
/// support).
///
/// `stable_boundary` (issue #12): an absolute index into `prompt_ids` where
/// the caller's renderer knows the prompt is render-stable (see
/// `crate::chat::render_with_boundary`) — forces an extra chunked-prefill
/// split there (in addition to the regular 4096-token grid) so
/// `prefill_boundary` fires at that exact position too, letting
/// `crate::snapshot::turn::run_turn` capture a snapshot the *next* turn's
/// re-rendered prompt can always hit, even when thinking is on and the
/// prompt's own tail (generation prompt, `<think>` block, reply) isn't
/// reproduced by the next render. `None` (or an index outside
/// `already_processed..prompt_ids.len()`, which needs no extra split)
/// reduces to the plain 4096-grid-only behavior.
#[allow(clippy::too_many_arguments)]
pub fn generate_sampled_with_stop_resumed(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    already_processed: usize,
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    stable_boundary: Option<usize>,
    prefill_boundary: impl FnMut(&mut Model, usize),
    on_text: impl FnMut(&str) -> bool,
) -> Result<GenerateStats, RocmlError> {
    generate_core(
        model,
        tokenizer,
        prompt_ids,
        already_processed,
        max_new_tokens,
        stop_on_eos,
        params,
        None,
        stable_boundary,
        prefill_boundary,
        on_text,
    )
}

/// Like [`super::generate_sampled_profiled`], with the resumed-from-a-snapshot
/// semantics of [`generate_sampled_resumed`] — `rocml-cli bench --turns`
/// uses this to measure per-turn prefill latency with snapshots on.
#[allow(clippy::too_many_arguments)]
pub fn generate_sampled_profiled_resumed(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    already_processed: usize,
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    prof: Option<&Profiler>,
    prefill_boundary: impl FnMut(&mut Model, usize),
    mut on_text: impl FnMut(&str),
) -> Result<GenerateStats, RocmlError> {
    generate_core(
        model,
        tokenizer,
        prompt_ids,
        already_processed,
        max_new_tokens,
        stop_on_eos,
        params,
        prof,
        None,
        prefill_boundary,
        |s| {
            on_text(s);
            false
        },
    )
}

/// Tokens per auto-snapshot boundary during a long prefill (issue #1): a
/// suffix longer than this gets forward-passed in multiple segments so
/// `prefill_boundary` can fire mid-prefill, not just once at the end —
/// making a partial prefix reusable even if generation is later aborted.
///
/// Splitting a `Model::forward_prompt` call at an arbitrary token boundary
/// preserves *what* gets computed (both architectures' forward passes are
/// pure sequential-position advancement) but not necessarily the exact
/// floating-point bits: the hybrid path's chunked kernels re-chunk each
/// call's own slice starting at its own index 0, so a split that doesn't
/// land on a 128-token chunk boundary changes which tokens land in the same
/// batched GEMM/attention/GDN-chunk launch together, which can reorder
/// summations — the same reduction-order sensitivity
/// `qwen35_chunked_prefill_parity`'s own near-tie escape hatch documents for
/// chunked-vs-serial. `rocml/tests/snapshot_equivalence.rs` is the dedicated
/// gate for this: near-exact logits (relative tolerance) and exact greedy
/// continuations, with that same near-tie escape hatch for a rare flip.
pub(super) const PREFILL_CAPTURE_INTERVAL: usize = 4096;

#[allow(clippy::too_many_arguments)]
pub(super) fn generate_core(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    already_processed: usize,
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    prof: Option<&Profiler>,
    stable_boundary: Option<usize>,
    mut prefill_boundary: impl FnMut(&mut Model, usize),
    mut on_text: impl FnMut(&str) -> bool,
) -> Result<GenerateStats, RocmlError> {
    let eos = tokenizer.eos_token_id;
    let mut pending = Vec::new();
    let mut generated_tokens = 0usize;
    let mut generated_ids = Vec::new();
    let mut history: Vec<u32> = prompt_ids.to_vec();
    let mut rng = Rng::new(params.seed);
    let mut stop_requested = false;

    let already = already_processed.min(prompt_ids.len());
    let suffix = &prompt_ids[already..];
    // Only a boundary strictly ahead of what's already processed and
    // strictly before the prompt's end needs forcing — anything else either
    // coincides with a restore that already covers it or with the natural
    // final chunk (see `generate_sampled_with_stop_resumed`'s doc comment).
    let stable_boundary = stable_boundary.filter(|&b| b > already && b < prompt_ids.len());

    if let Some(p) = prof {
        p.set_phase(Phase::Prefill);
    }
    let prompt_start = Instant::now();
    let mut logits;
    let mut processed = already;
    let mut i = 0usize;
    loop {
        let to_next_boundary = PREFILL_CAPTURE_INTERVAL - (processed % PREFILL_CAPTURE_INTERVAL);
        let mut end = (i + to_next_boundary).min(suffix.len());
        if let Some(b) = stable_boundary {
            if b > processed && b < processed + (end - i) {
                end = i + (b - processed);
            }
        }
        logits = model.forward_prompt(&suffix[i..end], prof)?;
        processed += end - i;
        i = end;
        prefill_boundary(model, processed);
        if i >= suffix.len() {
            break;
        }
    }
    let prompt_seconds = prompt_start.elapsed().as_secs_f64();

    if let Some(p) = prof {
        p.set_phase(Phase::Decode);
    }
    let decode_start = Instant::now();
    while !stop_requested && generated_tokens < max_new_tokens && !logits.is_empty() {
        let next_id = sample::sample(&logits, &history, params, &mut rng);
        if stop_on_eos && Some(next_id) == eos {
            break;
        }
        push_token_text(tokenizer, next_id, &mut pending, &mut |s| {
            if on_text(s) {
                stop_requested = true;
            }
        });
        history.push(next_id);
        generated_ids.push(next_id);
        generated_tokens += 1;
        if stop_requested {
            break;
        }
        logits = model.forward_token_profiled(next_id, prof)?;
    }
    flush_pending(&mut pending, &mut |s| {
        on_text(s);
    });
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    Ok(GenerateStats {
        prompt_tokens: suffix.len(),
        generated_tokens,
        prompt_seconds,
        decode_seconds,
        generated_ids,
    })
}

fn push_token_text(
    tokenizer: &BpeTokenizer,
    id: u32,
    pending: &mut Vec<u8>,
    on_text: &mut impl FnMut(&str),
) {
    if let Some(bytes) = tokenizer.token_bytes(id) {
        pending.extend_from_slice(bytes);
    }
    flush_pending(pending, on_text);
}

/// Emits every complete UTF-8 prefix of `pending`, leaving a trailing
/// incomplete multi-byte sequence (if any) buffered for the next token.
/// A genuinely malformed byte sequence (not just incomplete — shouldn't
/// happen for well-formed byte-level BPE output, but this must never hang)
/// emits U+FFFD and drops the offending bytes rather than buffering forever.
fn flush_pending(pending: &mut Vec<u8>, on_text: &mut impl FnMut(&str)) {
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                if !s.is_empty() {
                    on_text(s);
                }
                pending.clear();
                return;
            }
            Err(e) => {
                let valid_len = e.valid_up_to();
                if valid_len > 0 {
                    // SAFETY-equivalent: `valid_up_to()` guarantees
                    // `pending[..valid_len]` is valid UTF-8.
                    let s = std::str::from_utf8(&pending[..valid_len])
                        .expect("valid_up_to() prefix is valid UTF-8 by definition");
                    on_text(s);
                    pending.drain(..valid_len);
                }
                match e.error_len() {
                    Some(bad_len) => {
                        on_text("\u{FFFD}");
                        pending.drain(..bad_len.max(1));
                        // loop again: more of `pending` may now be decodable.
                    }
                    None => return, // trailing incomplete sequence; wait for more bytes.
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flush_pending_buffers_incomplete_utf8() {
        let mut pending = vec![0xE2, 0x9C]; // incomplete 3-byte sequence (✓ is E2 9C 93)
        let mut out = String::new();
        flush_pending(&mut pending, &mut |s| out.push_str(s));
        assert_eq!(out, "");
        pending.push(0x93);
        flush_pending(&mut pending, &mut |s| out.push_str(s));
        assert_eq!(out, "\u{2713}");
        assert!(pending.is_empty());
    }

    #[test]
    fn flush_pending_emits_ascii_immediately() {
        let mut pending = b"hello".to_vec();
        let mut out = String::new();
        flush_pending(&mut pending, &mut |s| out.push_str(s));
        assert_eq!(out, "hello");
    }
}

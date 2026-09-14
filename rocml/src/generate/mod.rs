//! Greedy generation loop: feed prompt tokens through the model one at a
//! time (prefill == decode here, see `forward` module docs), then greedily
//! sample (host-side argmax over the copied-back logits) until `eos` or
//! `max_new_tokens`, streaming decoded text as tokens complete.
//!
//! Split into this file (the four original, always-full-prefill entry
//! points) and `resumed` (the snapshot-aware variants issue #1 added, plus
//! the shared `generate_core` engine both sets call into) purely for the
//! workspace's 400-line file cap — there is no layering distinction beyond
//! that; `resumed`'s functions are a strict superset (`already_processed:
//! 0` reduces to exactly this file's behavior).

mod resumed;

use rocml_core::tokenizer::BpeTokenizer;

use crate::error::RocmlError;
use crate::model::Model;
use crate::profile::Profiler;
use crate::sample::SamplingParams;
use resumed::generate_core;
pub use resumed::{
    generate_sampled_profiled_resumed, generate_sampled_resumed, generate_sampled_with_stop_resumed,
};

#[derive(Debug, Clone)]
pub struct GenerateStats {
    /// Tokens actually run through a forward pass this call — for a
    /// snapshot-resumed turn (see `crate::snapshot::turn`) this is just the
    /// *new* suffix, not the whole conversation, which is the point: it's
    /// what `prompt_tokens_per_sec` should be measured against.
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_seconds: f64,
    pub decode_seconds: f64,
    /// Token ids sampled during this call, in order — the snapshot layer
    /// appends these to the prompt's token ids to know the exact token
    /// sequence an end-of-turn capture represents (see
    /// `crate::snapshot::turn::run_turn`).
    pub generated_ids: Vec<u32>,
}

impl GenerateStats {
    pub fn prompt_tokens_per_sec(&self) -> f64 {
        checked_rate(self.prompt_tokens, self.prompt_seconds)
    }

    pub fn decode_tokens_per_sec(&self) -> f64 {
        checked_rate(self.generated_tokens, self.decode_seconds)
    }
}

fn checked_rate(count: usize, seconds: f64) -> f64 {
    if seconds > 0.0 {
        count as f64 / seconds
    } else {
        0.0
    }
}

/// Greedy-generates up to `max_new_tokens` tokens continuing `prompt_ids`
/// from the model's current cache position (call `Model::reset` first for a
/// fresh sequence). `on_text` receives each newly complete UTF-8 chunk as
/// tokens decode; a token whose bytes don't yet complete a UTF-8 sequence is
/// buffered instead of erroring, per `BpeTokenizer::token_bytes`'s contract.
///
/// `stop_on_eos`: if true, generation stops (without emitting the eos token)
/// the moment eos is sampled. The greedy-parity fixtures were captured by a
/// reference implementation that always emits exactly `n` tokens regardless
/// of eos, so the parity test passes `false` to compare like for like;
/// real usage (the `generate` example) wants `true`.
pub fn generate(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    stop_on_eos: bool,
    on_text: impl FnMut(&str),
) -> Result<GenerateStats, RocmlError> {
    generate_sampled(
        model,
        tokenizer,
        prompt_ids,
        max_new_tokens,
        stop_on_eos,
        &SamplingParams::greedy(),
        on_text,
    )
}

/// Like [`generate`], but draws each next token via [`crate::sample::sample`]
/// against `params` instead of always taking the argmax — `params.is_greedy`
/// (temperature `<= 0`) reduces to exactly the same argmax path `generate`
/// always used, so this is a strict superset, not a behavior change for
/// existing greedy callers (the parity fixture tests call `Model` directly,
/// not this function, so they're unaffected either way).
pub fn generate_sampled(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    mut on_text: impl FnMut(&str),
) -> Result<GenerateStats, RocmlError> {
    generate_core(
        model,
        tokenizer,
        prompt_ids,
        0,
        max_new_tokens,
        stop_on_eos,
        params,
        None,
        None,
        |_, _| {},
        |s| {
            on_text(s);
            false
        },
    )
}

/// Like [`generate_sampled`], but `on_text` returns `true` to request
/// generation stop immediately after the chunk it was just given (the
/// server uses this to enforce OpenAI-style `stop` strings, checked against
/// the growing decoded text one chunk at a time — this function has no
/// opinion on what "should stop" means, it just reacts to the answer).
pub fn generate_sampled_with_stop(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    on_text: impl FnMut(&str) -> bool,
) -> Result<GenerateStats, RocmlError> {
    generate_core(
        model,
        tokenizer,
        prompt_ids,
        0,
        max_new_tokens,
        stop_on_eos,
        params,
        None,
        None,
        |_, _| {},
        on_text,
    )
}

/// Like [`generate_sampled`], but records per-op timing/bytes/FLOPs through
/// `prof` (issue #5's observability instrumentation) — `prof.set_phase` is
/// flipped from `Prefill` to `Decode` at the same boundary
/// [`GenerateStats`]'s own prompt/decode split uses, so a `Profiler::finish()`
/// report and this call's returned tok/s numbers describe the same two
/// windows.
#[allow(clippy::too_many_arguments)]
pub fn generate_sampled_profiled(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    prompt_ids: &[u32],
    max_new_tokens: usize,
    stop_on_eos: bool,
    params: &SamplingParams,
    prof: Option<&Profiler>,
    mut on_text: impl FnMut(&str),
) -> Result<GenerateStats, RocmlError> {
    generate_core(
        model,
        tokenizer,
        prompt_ids,
        0,
        max_new_tokens,
        stop_on_eos,
        params,
        prof,
        None,
        |_, _| {},
        |s| {
            on_text(s);
            false
        },
    )
}

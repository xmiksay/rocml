//! Greedy generation loop: feed prompt tokens through the model one at a
//! time (prefill == decode here, see `forward` module docs), then greedily
//! sample (host-side argmax over the copied-back logits) until `eos` or
//! `max_new_tokens`, streaming decoded text as tokens complete.

use std::time::Instant;

use rocml_core::tokenizer::BpeTokenizer;

use crate::error::RocmlError;
use crate::model::Model;

#[derive(Debug, Clone, Copy)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prompt_seconds: f64,
    pub decode_seconds: f64,
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
    mut on_text: impl FnMut(&str),
) -> Result<GenerateStats, RocmlError> {
    let eos = tokenizer.eos_token_id;
    let mut pending = Vec::new();
    let mut generated_tokens = 0usize;

    let prompt_start = Instant::now();
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id)?;
    }
    let prompt_seconds = prompt_start.elapsed().as_secs_f64();

    let decode_start = Instant::now();
    while generated_tokens < max_new_tokens && !logits.is_empty() {
        let next_id = argmax(&logits);
        if stop_on_eos && Some(next_id) == eos {
            break;
        }
        push_token_text(tokenizer, next_id, &mut pending, &mut on_text);
        generated_tokens += 1;
        logits = model.forward_token(next_id)?;
    }
    flush_pending(&mut pending, &mut on_text);
    let decode_seconds = decode_start.elapsed().as_secs_f64();

    Ok(GenerateStats {
        prompt_tokens: prompt_ids.len(),
        generated_tokens,
        prompt_seconds,
        decode_seconds,
    })
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as u32;
        }
    }
    best_idx
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

    #[test]
    fn argmax_picks_highest_logit() {
        assert_eq!(argmax(&[0.1, 5.0, -3.0, 4.9]), 1);
        assert_eq!(argmax(&[0.0]), 0);
    }
}

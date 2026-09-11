//! Byte-level BPE tokenizer built from GGUF metadata (gpt2 vocab/merges +
//! the qwen3.5-family pre-tokenization regex, verified against the real HF
//! `tokenizer.json` for this model family).
//!
//! Pipeline for `encode`: NFC-normalize the input -> split off
//! special/control tokens (they must never be touched by BPE) -> regex
//! pre-tokenize each remaining text run into "words" -> remap each word's
//! UTF-8 bytes into GPT-2's byte-level char alphabet -> greedy lowest-rank
//! BPE merge -> vocab lookup.

mod bpe;
mod byte_level;
mod error;
mod special;

pub use error::TokenizerError;

use std::collections::HashMap;

use fancy_regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::gguf::GgufFile;

/// The pre-tokenization regex the real qwen3.5 tokenizer uses (verified
/// against the `Split` pretokenizer pattern embedded in
/// `Qwen3.5-2B-tokenizer/tokenizer.json`, the actual HF tokenizer for the
/// same 248k-vocab family as both GGUF files this crate targets).
///
/// This MUST stay byte-identical to that `tokenizer.json` Split pattern —
/// do not "clean up" or approximate it. In particular: `[\p{L}\p{M}]+`
/// (letters plus combining marks, not just `\p{L}+`) so that e.g. "é" typed
/// as e + combining acute stays fused to its base letter, the leading
/// ` ?` before the punctuation-run alternative, and `\p{M}` excluded from
/// that same punctuation class. An earlier revision of this pattern was
/// wrong on all three points. The negative lookahead in `\s+(?!\S)` is why
/// this needs `fancy_regex` rather than the plain `regex` crate.
///
/// Public so `tests/tokenizer_cross_validation.rs` can configure an
/// independent reference tokenizer (built with the `tokenizers` crate) with
/// this exact same pattern for an apples-to-apples comparison.
pub const PRETOKENIZE_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Control/special token type id in `tokenizer.ggml.token_type`.
const TOKEN_TYPE_CONTROL: i32 = 3;

pub struct BpeTokenizer {
    vocab: HashMap<String, u32>,
    /// Raw bytes each token id decodes to, precomputed at construction so
    /// `decode`/`token_bytes` are just an index (needed for streaming).
    id_to_bytes: Vec<Vec<u8>>,
    merge_ranks: bpe::MergeRanks,
    byte_to_char: [char; 256],
    /// Sorted longest-first so a special token that prefixes another still
    /// resolves to the longer, more specific match.
    specials: Vec<(String, u32)>,
    pretokenize: Regex,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl BpeTokenizer {
    /// Builds a tokenizer from a GGUF file's `tokenizer.ggml.*` metadata.
    /// Only the gpt2 byte-level BPE model (used by the whole
    /// qwen2/qwen3/qwen3.5 family) is supported.
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self, TokenizerError> {
        let model = gguf.get_str("tokenizer.ggml.model")?;
        if model != "gpt2" {
            return Err(TokenizerError::UnsupportedModel {
                found: model.to_string(),
            });
        }
        let tokens = gguf.get_str_arr("tokenizer.ggml.tokens")?;
        if tokens.is_empty() {
            return Err(TokenizerError::EmptyVocab);
        }
        let merges = gguf.get_str_arr("tokenizer.ggml.merges")?;
        let token_type = gguf
            .get_i32_arr("tokenizer.ggml.token_type")
            .unwrap_or_else(|_| vec![1; tokens.len()]);
        let bos_token_id = gguf.get_u32("tokenizer.ggml.bos_token_id").ok();
        let eos_token_id = gguf.get_u32("tokenizer.ggml.eos_token_id").ok();

        Self::from_parts(tokens, merges, token_type, bos_token_id, eos_token_id)
    }

    fn from_parts(
        tokens: Vec<String>,
        merges: Vec<String>,
        token_type: Vec<i32>,
        bos_token_id: Option<u32>,
        eos_token_id: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        let (byte_to_char, char_to_byte) = byte_level::tables();
        let merge_ranks = bpe::build_ranks(&merges)?;

        let mut vocab = HashMap::with_capacity(tokens.len());
        let mut id_to_bytes = Vec::with_capacity(tokens.len());
        let mut specials = Vec::new();
        for (id, token) in tokens.iter().enumerate() {
            vocab.insert(token.clone(), id as u32);
            id_to_bytes.push(decode_token_bytes(token, &char_to_byte));
            if token_type.get(id) == Some(&TOKEN_TYPE_CONTROL) {
                specials.push((token.clone(), id as u32));
            }
        }
        specials.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

        // The pattern is a fixed constant validated by this crate's own
        // tests, so a compile failure here would be a bug in rocml, not
        // something a GGUF file's content could trigger.
        let pretokenize = Regex::new(PRETOKENIZE_PATTERN)
            .expect("PRETOKENIZE_PATTERN is a valid static regex, checked by tests");

        Ok(Self {
            vocab,
            id_to_bytes,
            merge_ranks,
            byte_to_char,
            specials,
            pretokenize,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Encodes `text` into token ids. Never panics: a pathological input
    /// that defeats the pre-tokenizer regex (extremely unlikely — it has no
    /// catastrophic-backtracking shape) degrades to per-byte fallback
    /// tokens for the unmatched remainder instead of erroring out.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // Mirrors the qwen35 tokenizer.json's NFC `normalizer` stage: the
        // vocab's merges were trained on NFC-normalized text, so e.g. "e" +
        // combining acute accent (U+0301) must be canonically composed to
        // "é" *before* special-token splitting and BPE, or it falls back to
        // several smaller pieces instead of the single token the vocab has
        // for the composed form.
        let normalized: String = text.nfc().collect();
        let mut ids = Vec::new();
        for segment in special::split(&normalized, &self.specials) {
            match segment {
                special::Segment::Special(id) => ids.push(id),
                special::Segment::Text(chunk) => self.encode_normal_text(chunk, &mut ids),
            }
        }
        ids
    }

    fn encode_normal_text(&self, text: &str, ids: &mut Vec<u32>) {
        let mut pos = 0usize;
        let mut matches = self.pretokenize.find_iter(text);
        loop {
            match matches.next() {
                Some(Ok(m)) => {
                    pos = m.end();
                    self.encode_word(m.as_str(), ids);
                }
                Some(Err(_)) => {
                    // See the doc comment on `encode`: fall back to raw
                    // bytes for whatever the regex engine couldn't match
                    // rather than losing or panicking on the input.
                    self.encode_bytes_fallback(&text[pos..], ids);
                    return;
                }
                None => return,
            }
        }
    }

    fn encode_word(&self, word: &str, ids: &mut Vec<u32>) {
        let mut symbols: Vec<String> = word
            .bytes()
            .map(|b| self.byte_to_char[b as usize].to_string())
            .collect();
        bpe::merge(&mut symbols, &self.merge_ranks);
        for symbol in symbols {
            match self.vocab.get(&symbol) {
                Some(&id) => ids.push(id),
                // Shouldn't happen for a well-formed byte-level vocab (all
                // 256 single-byte symbols are always present), but don't
                // drop input silently if it does.
                None => self.encode_bytes_fallback(&symbol, ids),
            }
        }
    }

    fn encode_bytes_fallback(&self, text: &str, ids: &mut Vec<u32>) {
        for b in text.bytes() {
            let symbol = self.byte_to_char[b as usize].to_string();
            if let Some(&id) = self.vocab.get(&symbol) {
                ids.push(id);
            }
        }
    }

    /// Decodes token ids back to a string. Invalid UTF-8 (e.g. an id list
    /// that splits a multi-byte character mid-token) is replaced, not
    /// panicked on.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(b) = self.token_bytes(id) {
                bytes.extend_from_slice(b);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Raw bytes token `id` decodes to, or `None` if `id` is out of range.
    /// Exposed for streaming decoders that need to buffer partial UTF-8
    /// across tokens.
    pub fn token_bytes(&self, id: u32) -> Option<&[u8]> {
        self.id_to_bytes.get(id as usize).map(Vec::as_slice)
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_bytes.len()
    }
}

/// Reverses the byte-level char remapping for one vocab entry, producing
/// the raw bytes it represents. Characters outside the remap table (which
/// shouldn't occur in a well-formed gpt2 vocab) are skipped rather than
/// treated as fatal, since this only affects display/decoding, never parsing.
fn decode_token_bytes(token: &str, char_to_byte: &HashMap<char, u8>) -> Vec<u8> {
    token
        .chars()
        .filter_map(|c| char_to_byte.get(&c).copied())
        .collect()
}

#[cfg(test)]
mod tests;

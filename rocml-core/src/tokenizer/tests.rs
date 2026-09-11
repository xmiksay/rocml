//! Unit tests against a small synthetic vocab (real-vocab cross-validation
//! against the `tokenizers` crate lives in `tests/tokenizer_cross_validation.rs`).

use super::*;

/// Builds a tokenizer whose base vocab is exactly the 256 byte-level
/// symbols (so `token id == byte value` for every plain ASCII byte, which
/// makes assertions readable), plus whatever extra `extra_tokens` are
/// appended and whatever `merges` are configured. `extra_token_types[i]`
/// applies to token id `256 + i`.
fn build(merges: &[&str], extra_tokens: &[(&str, i32)]) -> BpeTokenizer {
    let (byte_to_char, _) = byte_level::tables();
    let mut tokens: Vec<String> = (0u32..256)
        .map(|b| byte_to_char[b as usize].to_string())
        .collect();
    let mut token_type = vec![1i32; 256];
    for (tok, ty) in extra_tokens {
        tokens.push(tok.to_string());
        token_type.push(*ty);
    }
    let merges: Vec<String> = merges.iter().map(|s| s.to_string()).collect();
    BpeTokenizer::from_parts(tokens, merges, token_type, None, None).unwrap()
}

#[test]
fn merges_adjacent_bytes_per_rank() {
    // Real vocabs always list a merge's resulting piece as its own token
    // (tokens/merges are independent parallel arrays in GGUF metadata);
    // mirror that here so the merged "ab" symbol resolves to a real id.
    let tok = build(&["a b"], &[("ab", 1)]);
    let ids = tok.encode("ab");
    assert_eq!(ids, vec![256]);
    assert_eq!(tok.decode(&ids), "ab");
}

#[test]
fn plain_ascii_roundtrips_without_merges() {
    let tok = build(&[], &[]);
    let ids = tok.encode("Hello!");
    assert_eq!(tok.decode(&ids), "Hello!");
}

#[test]
fn special_token_is_never_split_by_a_matching_merge() {
    // If "<|s|>" were run through BPE, "< |" could plausibly merge; since
    // it's marked control (type 3), it must come out as a single id
    // untouched by any merge rule.
    let tok = build(&["< |"], &[("<|s|>", 3)]);
    let ids = tok.encode("x<|s|>y");
    assert_eq!(ids, vec![b'x' as u32, 256, b'y' as u32]);
    assert_eq!(tok.decode(&ids), "x<|s|>y");
}

#[test]
fn decode_skips_out_of_range_ids_instead_of_panicking() {
    let tok = build(&[], &[]);
    assert_eq!(tok.decode(&[u32::MAX]), "");
}

#[test]
fn empty_input_encodes_to_no_tokens() {
    let tok = build(&[], &[]);
    assert!(tok.encode("").is_empty());
}

#[test]
fn vocab_size_and_token_bytes() {
    let tok = build(&[], &[("<|s|>", 3)]);
    assert_eq!(tok.vocab_size(), 257);
    assert_eq!(tok.token_bytes(b'A' as u32), Some(&b"A"[..]));
    assert_eq!(tok.token_bytes(256), Some(&b"<|s|>"[..]));
    assert_eq!(tok.token_bytes(9999), None);
}

#[test]
fn bos_eos_are_none_when_absent() {
    let tok = build(&[], &[]);
    assert_eq!(tok.bos_token_id, None);
    assert_eq!(tok.eos_token_id, None);
}

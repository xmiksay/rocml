//! Cross-validates `BpeTokenizer` against the `tokenizers` crate (HF's own,
//! battle-tested BPE implementation).
//!
//! The spec for this milestone asks to cross-check against the HF
//! `tokenizer.json` cached for a matching model. In practice, on this
//! machine, none matches: the two real GGUF files
//! (`Qwen3.5-2B-Q8_0.gguf`, `ornith-1.0-9b-Q6_K.gguf`) both use a 248320-entry
//! `qwen35`-family vocab, while the only fully cached HF tokenizer
//! (`Qwen/Qwen3-0.6B`) has an unrelated ~151.6k vocab from an older,
//! differently-trained tokenizer — comparing token ids across those two
//! would be comparing apples to oranges by construction, not a real
//! correctness check. `Qwen/Qwen3.5-2B`'s own HF cache entry exists but its
//! snapshot was never fully downloaded on this machine (no `tokenizer.json`
//! present), so that path isn't available either. See
//! `vocab_size_confirms_no_cached_tokenizer_json_matches` below for the
//! concrete numbers.
//!
//! So instead we cross-validate what we *can*: an independent `tokenizers`
//! crate `Tokenizer` built from the exact same vocab/merges/regex/special
//! tokens this crate reads out of the GGUF file. That isolates and checks
//! the part that's actually ours to get right — byte-level remapping,
//! merge-rank BPE, pre-tokenization regex application, and special-token
//! splitting — against a well-tested reference implementation, using real
//! production vocab data rather than a toy one.

use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::{BpeTokenizer, PRETOKENIZE_PATTERN};
use tokenizers::models::bpe::{Merges, Vocab, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer};

const QWEN35_PATH: &str = "/mnt/nvme/miksa/checkpoints/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const HF_QWEN3_0_6B_TOKENIZER_GLOB: &str =
    "/home/miksa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json";
const TOKEN_TYPE_CONTROL: i32 = 3;

/// Resolves a path containing exactly one `*` path component (e.g. the HF
/// hub's `snapshots/<commit-hash>/...` layout) by listing that directory.
/// No `glob` crate in the allowed dependency list; a single wildcard
/// component is all this needs.
fn glob_one(pattern: &str) -> Option<std::path::PathBuf> {
    let parts: Vec<&str> = pattern.split('/').collect();
    let star_idx = parts.iter().position(|&p| p == "*")?;
    let base_dir: std::path::PathBuf = parts[..star_idx].join("/").into();
    let rest = &parts[star_idx + 1..];
    for entry in std::fs::read_dir(&base_dir).ok()?.flatten() {
        let candidate = rest.iter().fold(entry.path(), |acc, seg| acc.join(seg));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Builds an independent HF `tokenizers::Tokenizer` from the exact same
/// vocab/merges/specials this crate's `BpeTokenizer::from_gguf` reads,
/// configured with the same pre-tokenization regex.
fn build_reference(gguf: &GgufFile) -> Tokenizer {
    let tokens = gguf.get_str_arr("tokenizer.ggml.tokens").unwrap();
    let merges_raw = gguf.get_str_arr("tokenizer.ggml.merges").unwrap();
    let token_type = gguf.get_i32_arr("tokenizer.ggml.token_type").unwrap();

    let vocab: Vocab = tokens
        .iter()
        .enumerate()
        .map(|(id, t)| (t.clone(), id as u32))
        .collect();
    let merges: Merges = merges_raw
        .iter()
        .map(|m| {
            let (a, b) = m.split_once(' ').expect("well-formed merge entry");
            (a.to_string(), b.to_string())
        })
        .collect();
    let bpe = BPE::builder()
        .vocab_and_merges(vocab, merges)
        .build()
        .expect("build reference BPE model");

    let mut tok = Tokenizer::new(bpe);
    tok.with_pre_tokenizer(Some(PreTokenizerWrapper::Sequence(Sequence::new(vec![
        PreTokenizerWrapper::Split(
            Split::new(
                SplitPattern::Regex(PRETOKENIZE_PATTERN.to_string()),
                SplitDelimiterBehavior::Isolated,
                false,
            )
            .expect("valid regex"),
        ),
        PreTokenizerWrapper::ByteLevel(ByteLevel::new(false, false, false)),
    ]))));

    let specials: Vec<AddedToken> = tokens
        .iter()
        .zip(token_type.iter())
        .filter(|(_, &ty)| ty == TOKEN_TYPE_CONTROL)
        .map(|(t, _)| AddedToken::from(t.clone(), true))
        .collect();
    tok.add_special_tokens(specials)
        .expect("register special tokens");
    tok
}

const TEST_INPUTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Příliš žluťoučký kůň úpěl ďábelské ódy",
    "Hello 👋 world 🌍 emoji test 🚀🚀🚀",
    "こんにちは世界、これは日本語のテストです。中文测试也在这里。",
    "fn main() {\n\tlet x = {a: 1, b: 2};\n\tprintln!(\"{x:?}\");\n}",
    "   leading, trailing, and   multiple   spaces   ",
    "12345",
    "",
    "this text contains <|im_start|> literally in the middle",
];

#[test]
fn matches_reference_tokenizer_on_a_battery_of_tricky_inputs() {
    if !std::path::Path::new(QWEN35_PATH).exists() {
        eprintln!("skipping: {QWEN35_PATH} not present on this machine");
        return;
    }
    let gguf = GgufFile::open(QWEN35_PATH).expect("open GGUF");
    let ours = BpeTokenizer::from_gguf(&gguf).expect("build BpeTokenizer");
    let reference = build_reference(&gguf);

    for &text in TEST_INPUTS {
        let got = ours.encode(text);
        let want = reference
            .encode(text, false)
            .unwrap_or_else(|e| panic!("reference encode failed for {text:?}: {e}"))
            .get_ids()
            .to_vec();
        assert_eq!(got, want, "token id mismatch for input {text:?}");

        // Round-trip through our own decoder too, independent of the
        // reference: byte-level BPE is lossless over its own vocab.
        assert_eq!(ours.decode(&got), text, "decode roundtrip for {text:?}");
    }
}

#[test]
fn vocab_size_confirms_no_cached_tokenizer_json_matches() {
    if !std::path::Path::new(QWEN35_PATH).exists() {
        eprintln!("skipping: {QWEN35_PATH} not present on this machine");
        return;
    }
    let gguf = GgufFile::open(QWEN35_PATH).unwrap();
    let gguf_vocab_len = gguf.get_str_arr("tokenizer.ggml.tokens").unwrap().len();
    assert_eq!(gguf_vocab_len, 248_320);

    match glob_one(HF_QWEN3_0_6B_TOKENIZER_GLOB) {
        Some(path) => {
            let hf_tok = Tokenizer::from_file(&path).expect("load cached Qwen3-0.6B tokenizer");
            let hf_vocab_len = hf_tok.get_vocab_size(true);
            // Documents the mismatch this file's doc comment explains: two
            // different tokenizer generations, not interchangeable.
            assert_ne!(
                gguf_vocab_len, hf_vocab_len,
                "expected the cached Qwen3-0.6B tokenizer's vocab to differ from \
                 the qwen35 GGUF vocab; if this ever matches, switch the main \
                 cross-validation test to compare ids directly against it"
            );
        }
        None => {
            eprintln!(
                "no cached tokenizer.json found at {HF_QWEN3_0_6B_TOKENIZER_GLOB}, skipping the size check"
            );
        }
    }
}

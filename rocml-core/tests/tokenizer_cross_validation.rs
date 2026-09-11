//! Cross-validates `BpeTokenizer` against the `tokenizers` crate (HF's own,
//! battle-tested BPE implementation).
//!
//! The primary check (`matches_real_qwen35_tokenizer_json_on_tricky_inputs`)
//! loads the *real* HF tokenizer for this model family directly —
//! `/mnt/nvme/miksa/checkpoints/Qwen3.5-2B-tokenizer/tokenizer.json` — and
//! compares token ids one-for-one against `BpeTokenizer` built from
//! `Qwen3.5-2B-Q8_0.gguf`'s own metadata. Both share the same 248320-entry
//! `qwen35`-family vocab, so this is a genuine ground-truth comparison, not
//! an approximation.
//!
//! (Earlier this test could only cross-check against `Qwen/Qwen3-0.6B`'s
//! cached tokenizer, an unrelated ~151.6k-vocab, differently-trained
//! tokenizer from the same lab — comparing ids across those would have
//! been apples to oranges. `vocab_size_confirms_qwen3_0_6b_is_unrelated`
//! below keeps that finding on record.)
//!
//! `matches_independently_built_reference_from_same_gguf_vocab` is a second,
//! self-contained check: an independent `tokenizers::Tokenizer` built from
//! the exact same vocab/merges/regex/specials `BpeTokenizer::from_gguf`
//! reads out of the GGUF, with no external file dependency beyond the GGUF
//! itself. It isolates the same thing (byte-level remapping, merge-rank
//! BPE, regex pre-tokenization, special-token splitting) against a
//! well-tested reference implementation.

use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::{BpeTokenizer, PRETOKENIZE_PATTERN_QWEN35};
use tokenizers::models::bpe::{Merges, Vocab, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::pre_tokenizers::sequence::Sequence;
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{AddedToken, PreTokenizerWrapper, SplitDelimiterBehavior, Tokenizer};

const QWEN35_GGUF_PATH: &str = "/mnt/nvme/miksa/checkpoints/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const QWEN35_HF_TOKENIZER_PATH: &str =
    "/mnt/nvme/miksa/checkpoints/Qwen3.5-2B-tokenizer/tokenizer.json";
const HF_QWEN3_0_6B_TOKENIZER_GLOB: &str =
    "/home/miksa/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json";
const QWEN3_0_6B_GGUF_PATH: &str =
    "/mnt/nvme/miksa/checkpoints/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
/// Mirrors `BpeTokenizer`'s own `SPECIAL_TOKEN_TYPES` (ggml's CONTROL and
/// USER_DEFINED token-type ids) so this file's independent reference
/// tokenizer special-cases exactly the same tokens `ours` does — see that
/// constant's doc comment for why both types matter (issue #11).
const SPECIAL_TOKEN_TYPES: [i32; 2] = [3, 4];

fn skip_if_missing(path: &str) -> bool {
    if !std::path::Path::new(path).exists() {
        eprintln!("skipping: {path} not present on this machine");
        return true;
    }
    false
}

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

/// Battery for the id-for-id comparisons. Deliberately excludes any control
/// token substring (e.g. `<|im_start|>`): the `tokenizers` crate's
/// added-vocabulary extraction runs unconditionally regardless of
/// `encode`'s `add_special_tokens` flag (that flag only gates
/// post-processor template tokens), so a string containing one would still
/// get special-cased by the real tokenizer.json's `added_tokens` — fine in
/// itself, but it would no longer be testing plain BPE/regex behavior,
/// which is the point of this battery. Special-token handling is checked
/// separately in `rocml_special_token_is_a_single_id`, against rocml alone.
///
/// NFD-decomposed combining-mark text (e.g. "e" + U+0301 instead of
/// precomposed "é") is exercised separately in
/// `matches_reference_on_nfd_decomposed_combining_marks` below, alongside
/// its precomposed counterpart, since the interesting assertion there is
/// that both spellings now tokenize identically.
const TRICKY_INPUTS: &[&str] = &[
    "The quick brown fox jumps over the lazy dog.",
    "Příliš žluťoučký kůň úpěl ďábelské ódy",
    "café", // precomposed é (single code point) — exercises [\p{L}\p{M}]+
    "Hello 👋 world 🌍 emoji test 🚀🚀🚀",
    "family: 👨‍👩‍👧‍👦 flag: 🇨🇿🇯🇵", // multi-codepoint ZWJ/regional-indicator emoji sequences
    "こんにちは世界、これは日本語のテストです。中文测试也在这里。",
    "fn main() {\n\tlet x = {a: 1, b: 2};\n\tprintln!(\"{x:?}\");\n}",
    "   leading, trailing, and   multiple   spaces   ",
    "hello , world !!",
    "a !! b",
    "trailing spaces at the end   ",
    "12345",
    "",
];

#[test]
fn matches_real_qwen35_tokenizer_json_on_tricky_inputs() {
    if skip_if_missing(QWEN35_GGUF_PATH) || skip_if_missing(QWEN35_HF_TOKENIZER_PATH) {
        return;
    }
    let gguf = GgufFile::open(QWEN35_GGUF_PATH).expect("open GGUF");
    let ours = BpeTokenizer::from_gguf(&gguf).expect("build BpeTokenizer");
    let reference =
        Tokenizer::from_file(QWEN35_HF_TOKENIZER_PATH).expect("load real qwen3.5 tokenizer.json");

    for &text in TRICKY_INPUTS {
        let got = ours.encode(text);
        let want = reference
            .encode(text, false)
            .unwrap_or_else(|e| panic!("reference encode failed for {text:?}: {e}"))
            .get_ids()
            .to_vec();
        assert_eq!(got, want, "token id mismatch for input {text:?}");
        assert_eq!(ours.decode(&got), text, "decode roundtrip for {text:?}");
    }
}

/// (decomposed, precomposed) pairs: the left spelling uses a base letter
/// followed by a combining mark, the right spelling is the same text with
/// that pair canonically composed into one code point. NFC normalization
/// (mirroring the qwen35 tokenizer.json's `normalizer` stage — see
/// `BpeTokenizer::encode`) must make both spellings tokenize identically.
const NFD_DECOMPOSED_INPUTS: &[(&str, &str)] = &[
    // "e" + combining acute accent (U+0301) -> "é".
    ("e\u{0301}cole", "école"),
    // "z" + combining caron (U+030C) -> "ž" ("život" = Czech for "life").
    ("z\u{030C}ivot", "život"),
];

/// Was `known_gap_no_nfc_normalization` before `BpeTokenizer::encode` grew
/// NFC normalization: the real `tokenizer.json` configures an NFC
/// `normalizer` (composing e.g. "e" + combining acute accent U+0301 into a
/// single precomposed "é") because its vocab's merges were trained on
/// NFC-normalized text — a decomposed spelling that skips this step falls
/// back to several smaller pieces instead of the single token the vocab
/// has for the composed form. rocml's `[\p{L}\p{M}]+` regex correctly
/// *groups* a decomposed base letter with its combining marks into one
/// pre-token, but grouping alone isn't composing; both are required.
#[test]
fn matches_reference_on_nfd_decomposed_combining_marks() {
    if skip_if_missing(QWEN35_GGUF_PATH) || skip_if_missing(QWEN35_HF_TOKENIZER_PATH) {
        return;
    }
    let gguf = GgufFile::open(QWEN35_GGUF_PATH).unwrap();
    let ours = BpeTokenizer::from_gguf(&gguf).unwrap();
    let reference = Tokenizer::from_file(QWEN35_HF_TOKENIZER_PATH).unwrap();

    for &(decomposed, composed) in NFD_DECOMPOSED_INPUTS {
        let got = ours.encode(decomposed);
        let want = reference
            .encode(decomposed, false)
            .unwrap_or_else(|e| panic!("reference encode failed for {decomposed:?}: {e}"))
            .get_ids()
            .to_vec();
        assert_eq!(got, want, "token id mismatch for NFD input {decomposed:?}");

        // `encode` NFC-normalizes internally, so decoding yields the
        // composed spelling, not the original decomposed bytes — expected,
        // and it matches the reference tokenizer's own (lossy) behavior.
        assert_eq!(
            ours.decode(&got),
            composed,
            "decode roundtrip for {decomposed:?}"
        );

        // The decomposed and precomposed spellings must now tokenize
        // identically, proving NFC normalization actually ran rather than
        // the regex's `[\p{L}\p{M}]+` grouping happening to line up.
        assert_eq!(got, ours.encode(composed));
    }
}

#[test]
fn rocml_special_token_is_a_single_id() {
    // rocml-only: the real tokenizer.json's `add_special_tokens=false`
    // doesn't skip added-vocabulary matching (see the doc comment on
    // TRICKY_INPUTS), so this is checked against rocml alone rather than
    // folded into the id-for-id battery above.
    if skip_if_missing(QWEN35_GGUF_PATH) {
        return;
    }
    let gguf = GgufFile::open(QWEN35_GGUF_PATH).unwrap();
    let ours = BpeTokenizer::from_gguf(&gguf).unwrap();

    let tokens = gguf.get_str_arr("tokenizer.ggml.tokens").unwrap();
    let im_start_id = tokens
        .iter()
        .position(|t| t == "<|im_start|>")
        .expect("<|im_start|> present in vocab") as u32;

    let ids = ours.encode("this text contains <|im_start|> literally in the middle");
    assert!(
        ids.contains(&im_start_id),
        "expected {im_start_id} (<|im_start|>) in {ids:?}"
    );
    assert_eq!(ours.encode("<|im_start|>"), vec![im_start_id]);
    assert_eq!(ours.decode(&[im_start_id]), "<|im_start|>");
}

/// Regression test for issue #11: `<tool_call>` is ggml token_type 4
/// (USER_DEFINED), not 3 (CONTROL), in this vocab — a real GGUF check that
/// `SPECIAL_TOKEN_TYPES` actually covers both, not just a synthetic-vocab
/// unit test. An earlier revision only special-cased type 3, so this
/// string BPE-split into 4 ordinary sub-word tokens instead of the single
/// id the Ornith-1.0-9B checkpoint (which shares this vocab) was trained
/// on, feeding it out-of-distribution input in the tools system block.
#[test]
fn tool_call_marker_is_a_single_id() {
    if skip_if_missing(QWEN35_GGUF_PATH) {
        return;
    }
    let gguf = GgufFile::open(QWEN35_GGUF_PATH).unwrap();
    let ours = BpeTokenizer::from_gguf(&gguf).unwrap();

    let tokens = gguf.get_str_arr("tokenizer.ggml.tokens").unwrap();
    let token_type = gguf.get_i32_arr("tokenizer.ggml.token_type").unwrap();
    let tool_call_id = tokens
        .iter()
        .position(|t| t == "<tool_call>")
        .expect("<tool_call> present in vocab") as u32;
    assert_eq!(
        token_type[tool_call_id as usize], 4,
        "test assumption broken: <tool_call> is no longer USER_DEFINED in this GGUF"
    );

    assert_eq!(ours.encode("<tool_call>"), vec![tool_call_id]);
    assert_eq!(ours.decode(&[tool_call_id]), "<tool_call>");
}

/// Builds an independent HF `tokenizers::Tokenizer` from the exact same
/// vocab/merges/specials this crate's `BpeTokenizer::from_gguf` reads,
/// configured with the same pre-tokenization regex.
fn build_reference_from_gguf(gguf: &GgufFile) -> Tokenizer {
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
                SplitPattern::Regex(PRETOKENIZE_PATTERN_QWEN35.to_string()),
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
        .filter(|(_, &ty)| SPECIAL_TOKEN_TYPES.contains(&ty))
        .map(|(t, _)| AddedToken::from(t.clone(), true))
        .collect();
    tok.add_special_tokens(specials)
        .expect("register special tokens");
    tok
}

#[test]
fn matches_independently_built_reference_from_same_gguf_vocab() {
    if skip_if_missing(QWEN35_GGUF_PATH) {
        return;
    }
    let gguf = GgufFile::open(QWEN35_GGUF_PATH).expect("open GGUF");
    let ours = BpeTokenizer::from_gguf(&gguf).expect("build BpeTokenizer");
    let reference = build_reference_from_gguf(&gguf);

    let mut inputs: Vec<&str> = TRICKY_INPUTS.to_vec();
    // Unlike the real-tokenizer.json battery, this reference has the exact
    // same special-token registration as `ours`, so it's safe to also
    // exercise the literal-control-token case here.
    inputs.push("this text contains <|im_start|> literally in the middle");

    for &text in &inputs {
        let got = ours.encode(text);
        let want = reference
            .encode(text, false)
            .unwrap_or_else(|e| panic!("reference encode failed for {text:?}: {e}"))
            .get_ids()
            .to_vec();
        assert_eq!(got, want, "token id mismatch for input {text:?}");
        assert_eq!(ours.decode(&got), text, "decode roundtrip for {text:?}");
    }
}

#[test]
fn vocab_size_confirms_qwen3_0_6b_is_unrelated() {
    if skip_if_missing(QWEN35_GGUF_PATH) {
        return;
    }
    let gguf = GgufFile::open(QWEN35_GGUF_PATH).unwrap();
    let gguf_vocab_len = gguf.get_str_arr("tokenizer.ggml.tokens").unwrap().len();
    assert_eq!(gguf_vocab_len, 248_320);

    match glob_one(HF_QWEN3_0_6B_TOKENIZER_GLOB) {
        Some(path) => {
            let hf_tok = Tokenizer::from_file(&path).expect("load cached Qwen3-0.6B tokenizer");
            let hf_vocab_len = hf_tok.get_vocab_size(true);
            assert_ne!(
                gguf_vocab_len, hf_vocab_len,
                "expected the cached Qwen3-0.6B tokenizer's vocab to differ from the \
                 qwen35 GGUF vocab (documenting why it can't be used for id-for-id \
                 cross-validation; see the real tokenizer.json test instead)"
            );
        }
        None => {
            eprintln!(
                "no cached tokenizer.json found at {HF_QWEN3_0_6B_TOKENIZER_GLOB}, skipping the size check"
            );
        }
    }
}

/// The dense Qwen3-0.6B GGUF reports `tokenizer.ggml.pre = "qwen2"`, a
/// distinct pre-tokenizer family from qwen35 (see `pretokenize_pattern_for`
/// in `rocml-core/src/tokenizer/mod.rs`). This is the id-for-id ground-truth
/// check for that family, mirroring
/// `matches_real_qwen35_tokenizer_json_on_tricky_inputs` above but against
/// the real cached `Qwen/Qwen3-0.6B` `tokenizer.json`.
#[test]
fn matches_real_qwen2_tokenizer_json_on_tricky_inputs() {
    if skip_if_missing(QWEN3_0_6B_GGUF_PATH) {
        return;
    }
    let Some(hf_tokenizer_path) = glob_one(HF_QWEN3_0_6B_TOKENIZER_GLOB) else {
        eprintln!("no cached tokenizer.json found at {HF_QWEN3_0_6B_TOKENIZER_GLOB}, skipping");
        return;
    };

    let gguf = GgufFile::open(QWEN3_0_6B_GGUF_PATH).expect("open GGUF");
    assert_eq!(
        gguf.get_str("tokenizer.ggml.pre").ok(),
        Some("qwen2"),
        "this test's premise is that the dense Qwen3-0.6B GGUF is qwen2-family"
    );
    let ours = BpeTokenizer::from_gguf(&gguf).expect("build BpeTokenizer");
    let reference =
        Tokenizer::from_file(&hf_tokenizer_path).expect("load real qwen2-family tokenizer.json");

    for &text in TRICKY_INPUTS {
        let got = ours.encode(text);
        let want = reference
            .encode(text, false)
            .unwrap_or_else(|e| panic!("reference encode failed for {text:?}: {e}"))
            .get_ids()
            .to_vec();
        assert_eq!(got, want, "token id mismatch for input {text:?}");
        assert_eq!(ours.decode(&got), text, "decode roundtrip for {text:?}");
    }
}

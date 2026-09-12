//! Architecture-level parity: dense Qwen3-0.6B greedy-decoded through
//! rocml's HIP forward pass must match candle's independent CPU f32
//! implementation token-for-token, from the same GGUF. Fixtures in
//! `tests/data/qwen3_greedy_fixtures.json` were captured by candle's
//! quantized-qwen3 example (see `_meta` in that file).
//!
//! Run via `make test-model` (`cargo test --release -p rocml --test
//! greedy_parity`) — release mode because this dequantizes a real ~600MB
//! GGUF and runs a 28-layer x 47-token decode loop, unbearably slow
//! unoptimized. Skips itself if the checkpoint isn't present on this
//! machine.

use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;
use serde::Deserialize;

const GGUF_REL: &str = "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
const FIXTURES_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/data/qwen3_greedy_fixtures.json"
);
// `_meta.note` says "the first ~47 greedy tokens", but the recorded
// `continuation` strings are actually candle's full n=48 (`_meta.generator`)
// run decoded verbatim: generating only 47 here left every case's decoded
// text a byte-for-byte prefix of the fixture, missing exactly the final
// word. 48 gives an exact match on all three cases.
const NUM_TOKENS: usize = 48;
/// Below this relative top-2 logit gap at the first divergent token, treat
/// the mismatch as a plausible f16-vs-f32 near-tie rather than a bug (see
/// the module doc comment's mismatch policy).
const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;
const MIN_EXACT_PREFIX: usize = 32;

#[derive(Deserialize)]
struct Fixtures {
    #[serde(rename = "_meta")]
    meta: Meta,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Meta {
    template: String,
}

#[derive(Deserialize)]
struct Case {
    prompt: String,
    continuation: String,
}

/// Greedily decodes `n` tokens after `prompt_ids`, returning the generated
/// ids plus each step's (best, second-best) logit value — needed for the
/// near-tie policy on a mismatch.
fn generate_with_logit_gaps(
    model: &mut Model,
    prompt_ids: &[u32],
    n: usize,
) -> (Vec<u32>, Vec<(f32, f32)>) {
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id).expect("forward_token failed");
    }
    let mut ids = Vec::with_capacity(n);
    let mut gaps = Vec::with_capacity(n);
    for _ in 0..n {
        let (best_idx, best_val, second_val) = top2(&logits);
        ids.push(best_idx);
        gaps.push((best_val, second_val));
        logits = model.forward_token(best_idx).expect("forward_token failed");
    }
    (ids, gaps)
}

fn top2(logits: &[f32]) -> (u32, f32, f32) {
    let (mut best_idx, mut best_val, mut second_val) = (0u32, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            second_val = best_val;
            best_val = v;
            best_idx = i as u32;
        } else if v > second_val {
            second_val = v;
        }
    }
    (best_idx, best_val, second_val)
}

/// Concatenates each generated token's raw bytes (mirrors
/// `BpeTokenizer::decode`), also returning the cumulative byte length after
/// each token so a byte offset can be mapped back to "which token produced
/// this".
fn token_bytes_with_boundaries(tokenizer: &BpeTokenizer, ids: &[u32]) -> (Vec<u8>, Vec<usize>) {
    let mut all_bytes = Vec::new();
    let mut boundaries = Vec::with_capacity(ids.len());
    for &id in ids {
        if let Some(b) = tokenizer.token_bytes(id) {
            all_bytes.extend_from_slice(b);
        }
        boundaries.push(all_bytes.len());
    }
    (all_bytes, boundaries)
}

#[test]
fn dense_qwen3_0_6b_greedy_matches_candle_cpu_reference() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };
    let fixtures: Fixtures =
        serde_json::from_str(&std::fs::read_to_string(FIXTURES_PATH).expect("read fixtures json"))
            .expect("parse fixtures json");

    let gguf = GgufFile::open(&gguf_path).expect("open GGUF");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("build tokenizer");
    drop(gguf); // Model::load mmaps its own handle; no need to hold two.

    // f32 KV explicitly (not the default f16) so this suite pins the exact
    // pre-issue-#3 reference numerics candle's CPU f32 implementation was
    // compared against — see `rocml::LoadOptions`'s doc comment.
    let mut model = Model::load(
        &gguf_path,
        LoadOptions::new(4096).with_kv_cache(KvCacheMode::F32),
    )
    .expect("load model");

    for case in &fixtures.cases {
        model.reset().expect("reset failed");
        let prompt_text = fixtures.meta.template.replace("{prompt}", &case.prompt);
        let prompt_ids = tokenizer.encode(&prompt_text);

        let (ids, gaps) = generate_with_logit_gaps(&mut model, &prompt_ids, NUM_TOKENS);
        let got_text = tokenizer.decode(&ids);

        if got_text == case.continuation {
            continue;
        }

        // Find the first divergent *byte* directly against our own
        // generated byte stream (not by re-tokenizing the reference text:
        // BPE is not injective, so e.g. "<think>" mid-string can re-encode
        // to a different id sequence than the one that actually produced
        // that exact text, which would misattribute the divergence point).
        let (got_bytes, token_end) = token_bytes_with_boundaries(&tokenizer, &ids);
        let want_bytes = case.continuation.as_bytes();
        let diff_byte = got_bytes
            .iter()
            .zip(want_bytes)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| got_bytes.len().min(want_bytes.len()));
        let i = token_end
            .iter()
            .position(|&end| end > diff_byte)
            .unwrap_or(ids.len().saturating_sub(1));

        let (best, second) = gaps[i];
        let rel_gap = (best - second).abs() / best.abs().max(1.0);
        if rel_gap < NEAR_TIE_RELATIVE_GAP && i >= MIN_EXACT_PREFIX {
            eprintln!(
                "NOTE: prompt {:?} diverges from the candle CPU reference at token {i} \
                 (relative top-2 logit gap {rel_gap:.6} -- a plausible f16-vs-f32 near-tie); \
                 asserting an exact match up to token {i} instead of the full {NUM_TOKENS}. \
                 got {got_text:?}, want {:?}",
                case.prompt, case.continuation
            );
            let prefix_end = if i == 0 { 0 } else { token_end[i - 1] };
            assert_eq!(
                &got_bytes[..prefix_end],
                &want_bytes[..prefix_end.min(want_bytes.len())],
                "prompt {:?}: prefix mismatch before the documented near-tie at token {i}",
                case.prompt
            );
        } else {
            panic!(
                "prompt {:?}: greedy decode diverges from the candle CPU reference at token {i} \
                 (byte offset {diff_byte}), top-2 logits {best} vs {second} (relative gap \
                 {rel_gap:.6}) -- not a near-tie, likely a forward-pass bug (check rope \
                 convention, qk-norm, GQA head mapping, cache layout). got text {got_text:?}, \
                 want {:?}",
                case.prompt, case.continuation
            );
        }
    }
}

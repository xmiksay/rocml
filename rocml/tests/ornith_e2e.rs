//! End-to-end smoke test for Ornith-1.0-9B (qwen35 hybrid arch, Q6_K,
//! ~7.5GB): the checkpoint milestone 5b's quantized-weight loader exists to
//! make possible at all — dequant-to-f16 would need ~15GB just for weights,
//! which doesn't fit this card's 16GB VRAM alongside the KV/GDN cache and
//! scratch buffers. There is no independent CPU/GPU reference for this
//! checkpoint (unlike the candle/Crane parity suites), so this test checks
//! internal well-formedness instead: finite logits throughout, and greedy
//! decoding produces the exact same token sequence from two independent
//! model loads.
//!
//! Run via `make test-model` (release: a real ~7.4GB GGUF and a 32-layer x
//! 24-token decode loop, twice, is unbearably slow unoptimized). Skips
//! itself if the checkpoint isn't present on this machine.

use rocml::{LoadOptions, Model};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;

const GGUF_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf";
/// A raw continuation prompt (no chat template) — this test cares about
/// forward-pass well-formedness, not chat behavior.
const PROMPT: &str = "The capital of France is";
const NUM_TOKENS: usize = 24;

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

/// Greedily decodes `n` tokens after `prompt_ids`, asserting every logits
/// vector along the way is entirely finite (a NaN/inf would silently corrupt
/// argmax rather than error, so this is checked explicitly rather than left
/// to show up as a garbled decode).
fn greedy_generate(model: &mut Model, prompt_ids: &[u32], n: usize) -> Vec<u32> {
    let mut logits = Vec::new();
    for &id in prompt_ids {
        logits = model.forward_token(id).expect("forward_token failed");
    }
    let mut ids = Vec::with_capacity(n);
    for _ in 0..n {
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "non-finite logit encountered during decode"
        );
        let best = argmax(&logits);
        ids.push(best);
        logits = model.forward_token(best).expect("forward_token failed");
    }
    ids
}

#[test]
fn ornith_9b_greedy_decode_is_well_formed_and_deterministic() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };

    let gguf = GgufFile::open(&gguf_path).expect("open GGUF");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("build tokenizer");
    drop(gguf); // Model::load mmaps its own handle; no need to hold two.
    let prompt_ids = tokenizer.encode(PROMPT);
    assert!(!prompt_ids.is_empty(), "prompt tokenized to zero ids");

    // Production default (f16 KV, issue #3) — this test has no independent
    // reference to pin against, only internal well-formedness/determinism.
    let opts = LoadOptions::new(4096);
    let mut model_a = Model::load(&gguf_path, opts).expect("load model (first run)");
    let ids_a = greedy_generate(&mut model_a, &prompt_ids, NUM_TOKENS);
    drop(model_a); // free VRAM before the second load

    let mut model_b = Model::load(&gguf_path, opts).expect("load model (second run)");
    let ids_b = greedy_generate(&mut model_b, &prompt_ids, NUM_TOKENS);

    assert_eq!(
        ids_a, ids_b,
        "greedy decode is not deterministic across two independent loads"
    );

    let text = tokenizer.decode(&ids_a);
    assert!(!text.is_empty(), "decoded text is empty");
}

/// Regression gate for the GDN decay-gate bug (`ssm_a` is pre-baked
/// `-exp(A_log)` in GGUF; re-exponentiating it wipes the recurrent state
/// every step): greedy decode of a real tool-bearing chat prompt must
/// reproduce the continuation that llama.cpp and HF transformers (same
/// GGUF, independent implementations) both produce token-for-token. With
/// the bug present this prompt degenerates into protocol babble instead.
#[test]
fn ornith_9b_tooled_prompt_greedy_matches_external_references() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };

    let gguf = GgufFile::open(&gguf_path).expect("open GGUF");
    let tokenizer = BpeTokenizer::from_gguf(&gguf).expect("build tokenizer");
    drop(gguf);

    let messages = vec![
        rocml::chat::Message::system(
            "You are a helpful assistant with access to tools. Use a tool whenever the \
             user's request requires current or external information."
                .to_string(),
        ),
        rocml::chat::Message::user("What's the weather like in Prague right now?".to_string()),
    ];
    let tools = vec![rocml::chat::Tool {
        name: "get_weather".into(),
        description: "Get the current weather for a location.".into(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }),
    }];
    let prompt = rocml::chat::render(
        &messages,
        &tools,
        rocml::chat::RenderOpts {
            add_generation_prompt: true,
            enable_thinking: None,
            keep_history_reasoning: false,
        },
    )
    .expect("render");
    let prompt_ids = tokenizer.encode(&prompt);

    let mut model = Model::load(&gguf_path, LoadOptions::new(4096)).expect("load model");
    let mut logits = Vec::new();
    for &id in &prompt_ids {
        logits = model.forward_token(id).expect("forward_token failed");
    }
    let mut out_ids = Vec::with_capacity(96);
    for _ in 0..96 {
        let best = argmax(&logits);
        out_ids.push(best);
        logits = model.forward_token(best).expect("forward_token failed");
    }
    let text = tokenizer.decode(&out_ids);

    // llama.cpp (`llama-completion --temp 0`, chunked AND -ub 1 sequential)
    // and HF transformers (same GGUF via `gguf_file=`) both open with
    // exactly this thinking sentence. The sentence tail is deliberately not
    // pinned: greedy near-ties can flip a word mid-sentence (same policy as
    // the parity suites' near-tie escape hatch) without invalidating the
    // gate this test exists for.
    assert!(
        text.trim_start()
            .starts_with("The user is asking about the current weather in Prague."),
        "greedy continuation diverges from llama.cpp/HF reference: {text:?}"
    );
    // ...and emit exactly this tool call.
    let expected_call = "<tool_call>\n<function=get_weather>\n<parameter=location>\nPrague\n\
                         </parameter>\n</function>\n</tool_call>";
    assert!(
        text.contains("</think>"),
        "thinking block never closed: {text:?}"
    );
    assert!(
        text.contains(expected_call),
        "expected exact tool-call block missing: {text:?}"
    );
}

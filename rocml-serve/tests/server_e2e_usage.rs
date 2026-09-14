//! End-to-end HTTP tests for OpenAI-compatible usage reporting: the
//! `usage.prompt_tokens_details.cached_tokens` field and streaming's
//! `stream_options: {"include_usage": true}`. Split out of `server_e2e.rs`
//! purely for the 400-line file cap — both binaries share `tests/support`'s
//! spawn helpers and hand-rolled HTTP client. Real GGUF, real GPU forward
//! passes, run via `make test-model` (`cargo test --release -p rocml-serve
//! --test server_e2e_usage`); skips itself if the checkpoint isn't present.

use rocml_core::testpaths::checkpoint;
use serde_json::{json, Value};

mod support;

use support::{spawn_test_server, spawn_test_server_with_snapshots, GGUF_REL};

/// Usage reporting: `usage.prompt_tokens_details.cached_tokens` is `0` (not
/// omitted) on a fresh conversation, and on a follow-up turn that extends it
/// reports exactly the number of prompt tokens `run_turn` restored from the
/// end-of-turn snapshot instead of re-prefilling. `SnapshotStore::lookup`'s
/// "longest exact prefix match" contract (`rocml/src/snapshot/mod.rs`) makes
/// this an exact equality — turn 1's whole conversation
/// (`prompt_tokens + completion_tokens`) is exactly what got captured and is
/// exactly the prefix turn 2's re-rendered transcript shares with it — not
/// just a `<=` bound.
#[tokio::test]
async fn usage_cached_tokens_reports_snapshot_hit_across_two_turns() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };
    let server = spawn_test_server_with_snapshots(gguf_path, 512).await;
    let addr = server.addr;

    let body1 = json!({
        "messages": [{"role": "user", "content": "In one short sentence, name the capital of France."}],
        "max_tokens": 24,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let resp1 = support::post_json(addr, "/v1/chat/completions", &body1).await;
    assert_eq!(resp1.status, 200, "turn 1 body: {}", resp1.body);
    let parsed1: Value = serde_json::from_str(&resp1.body).expect("valid JSON response");
    let cached1 = parsed1["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("prompt_tokens_details.cached_tokens is present and a number");
    assert_eq!(
        cached1, 0,
        "fresh conversation: usage was {}",
        parsed1["usage"]
    );
    let prompt_tokens1 = parsed1["usage"]["prompt_tokens"].as_u64().unwrap();
    let completion_tokens1 = parsed1["usage"]["completion_tokens"].as_u64().unwrap();
    let content1 = parsed1["choices"][0]["message"]["content"]
        .as_str()
        .expect("turn 1 content is a string")
        .to_string();

    let body2 = json!({
        "messages": [
            {"role": "user", "content": "In one short sentence, name the capital of France."},
            {"role": "assistant", "content": content1},
            {"role": "user", "content": "And what is a famous landmark there?"},
        ],
        "max_tokens": 32,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let resp2 = support::post_json(addr, "/v1/chat/completions", &body2).await;
    assert_eq!(resp2.status, 200, "turn 2 body: {}", resp2.body);
    let parsed2: Value = serde_json::from_str(&resp2.body).expect("valid JSON response");
    let cached2 = parsed2["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("prompt_tokens_details.cached_tokens is present and a number");
    let prompt_tokens2 = parsed2["usage"]["prompt_tokens"].as_u64().unwrap();

    assert!(cached2 > 0, "turn 2 usage: {}", parsed2["usage"]);
    assert!(
        cached2 <= prompt_tokens2,
        "cached_tokens ({cached2}) must not exceed prompt_tokens ({prompt_tokens2})"
    );
    assert_eq!(
        cached2,
        prompt_tokens1 + completion_tokens1,
        "turn 2's cached prefix should be exactly turn 1's whole conversation \
         (prompt_tokens {prompt_tokens1} + completion_tokens {completion_tokens1}); \
         turn 2 usage: {}",
        parsed2["usage"]
    );

    server.shutdown().await;
}

/// Streaming without `stream_options`: no chunk carries a `usage` key at
/// all — the field is fully omitted, matching pre-feature behavior exactly.
#[tokio::test]
async fn streaming_without_stream_options_has_no_usage_keys() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };
    let server = spawn_test_server(gguf_path).await;
    let body = json!({
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": true,
    })
    .to_string();
    let resp = support::post_json(server.addr, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    for chunk in sse_chunks(&resp.body) {
        assert!(
            chunk.get("usage").is_none(),
            "usage key should be absent without stream_options: {chunk}"
        );
    }
    server.shutdown().await;
}

/// `stream_options: {"include_usage": true}`: every chunk but the last
/// carries `"usage": null`, and one extra chunk after the `finish_reason`
/// chunk (empty `choices`, before `[DONE]`) carries the full usage object
/// including `prompt_tokens_details.cached_tokens`.
#[tokio::test]
async fn streaming_include_usage_reports_usage_on_final_chunk_only() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };
    let server = spawn_test_server(gguf_path).await;
    let body = json!({
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": true,
        "stream_options": {"include_usage": true},
    })
    .to_string();
    let resp = support::post_json(server.addr, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let chunks = sse_chunks(&resp.body);
    assert!(chunks.len() >= 2, "expected several chunks: {}", resp.body);

    let (last, rest) = chunks.split_last().expect("at least one chunk");
    for chunk in rest {
        assert!(
            chunk.get("usage").is_some_and(Value::is_null),
            "non-final chunk should have usage: null: {chunk}"
        );
    }
    assert_eq!(
        last["choices"].as_array().map(Vec::len),
        Some(0),
        "final usage chunk should have empty choices: {last}"
    );
    let cached = last["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("final chunk usage.prompt_tokens_details.cached_tokens is a number");
    assert_eq!(cached, 0, "fresh conversation: {last}");
    assert!(last["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
    assert!(last["usage"]["completion_tokens"].as_u64().unwrap() > 0);

    server.shutdown().await;
}

/// Parses an SSE body's `data: {...}` lines into JSON values, skipping the
/// literal `data: [DONE]` terminator.
fn sse_chunks(body: &str) -> Vec<Value> {
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).expect("chunk is valid JSON"))
        .collect()
}

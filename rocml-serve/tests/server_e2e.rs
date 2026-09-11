//! End-to-end HTTP tests against a real `rocml-serve` router bound to an
//! ephemeral port, driving it with a hand-rolled minimal HTTP client (see
//! `tests/support`). Real GGUF, real GPU forward passes — run via
//! `make test-model` (`cargo test --release -p rocml-serve --test
//! server_e2e`); debug mode makes even a handful of decode tokens
//! uncomfortably slow. Skips itself if the checkpoint isn't present.
//!
//! `--no-think` is used for the test server: this codebase's chat renderer
//! is hardcoded to Ornith-1.0-9B's template, whose *default* (`enable_thinking`
//! left unset) opens reasoning, while Qwen3.5-2B's own real template
//! defaults it closed (see the milestone report for the full byte-diff).
//! Forcing `--no-think` here sidesteps that mismatch and keeps the
//! streaming assertions independent of how many tokens the 2B model would
//! otherwise spend thinking before any visible content appears.

use std::path::Path;
use std::time::Duration;

use rocml_serve::{build, ServerConfig};
use serde_json::{json, Value};

mod support;

const GGUF_PATH: &str = "/mnt/nvme/miksa/checkpoints/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const ORNITH_GGUF_PATH: &str =
    "/mnt/nvme/miksa/checkpoints/Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf";

fn skip_if_missing(path: &str) -> bool {
    if !Path::new(path).exists() {
        eprintln!("skipping: {path} not present on this machine");
        return true;
    }
    false
}

struct TestServer {
    addr: std::net::SocketAddr,
    serve_task: tokio::task::JoinHandle<()>,
    worker_thread: Option<std::thread::JoinHandle<()>>,
}

impl TestServer {
    /// Aborts the serve task (dropping its `Router`/`Arc<AppState>`, and
    /// with it the job sender) and waits for the worker thread to notice
    /// and exit. Must run before the test function returns — see
    /// `worker::spawn`'s doc comment for why a still-live worker thread
    /// racing this process's own exit crashes with "pure virtual method
    /// called" instead of a clean exit.
    async fn shutdown(mut self) {
        self.serve_task.abort();
        if let Some(handle) = self.worker_thread.take() {
            let _ = tokio::task::spawn_blocking(move || handle.join()).await;
        }
    }
}

async fn spawn_test_server(model_path: &str) -> TestServer {
    let (app, worker_thread) = build(ServerConfig {
        model_path: model_path.into(),
        ctx: 4096,
        max_tokens_default: 128,
        no_think: true,
    })
    .expect("server failed to build (model load / tokenizer)");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    let serve_task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    TestServer {
        addr,
        serve_task,
        worker_thread: Some(worker_thread),
    }
}

const WEATHER_TOOL: &str = r#"{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"]
        }
    }
}"#;

#[tokio::test]
async fn chat_completions_end_to_end() {
    if skip_if_missing(GGUF_PATH) {
        return;
    }
    let server = spawn_test_server(GGUF_PATH).await;
    let addr = server.addr;

    // (1) Non-streamed completion: 200, non-empty content, sane usage.
    let body = json!({
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let resp = support::post_json(addr, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let parsed: Value = serde_json::from_str(&resp.body).expect("valid JSON response");
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .expect("choices[0].message.content is a string");
    assert!(!content.trim().is_empty(), "content was empty: {parsed}");
    let prompt_tokens = parsed["usage"]["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = parsed["usage"]["completion_tokens"].as_u64().unwrap_or(0);
    let total_tokens = parsed["usage"]["total_tokens"].as_u64().unwrap_or(0);
    assert!(prompt_tokens > 0, "usage: {}", parsed["usage"]);
    assert!(completion_tokens > 0, "usage: {}", parsed["usage"]);
    assert_eq!(total_tokens, prompt_tokens + completion_tokens);

    // (2) Streamed: role chunk, content deltas, terminated by [DONE].
    let body = json!({
        "messages": [{"role": "user", "content": "Say hello in one short sentence."}],
        "max_tokens": 32,
        "temperature": 0,
        "stream": true,
    })
    .to_string();
    let resp = support::post_json(addr, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    assert!(
        resp.body.contains("\"role\":\"assistant\""),
        "missing role chunk: {}",
        resp.body
    );
    assert!(
        resp.body.contains("\"content\":"),
        "missing content delta: {}",
        resp.body
    );
    assert!(
        resp.body.trim_end().ends_with("data: [DONE]"),
        "stream didn't terminate with [DONE]: {}",
        resp.body
    );

    // (3) Tools present + a user message crafted to trigger a call. The 2B
    // model isn't agentic and may or may not actually emit a tool_call —
    // either finish_reason is acceptable here, this only asserts the tools
    // path doesn't error. The Ornith GGUF variant below asserts a real call.
    let body = json!({
        "messages": [{"role": "user", "content": "What is the weather in Prague? Use the get_weather tool."}],
        "tools": [serde_json::from_str::<Value>(WEATHER_TOOL).unwrap()],
        "max_tokens": 200,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let resp = support::post_json(addr, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let parsed: Value = serde_json::from_str(&resp.body).expect("valid JSON response");
    let finish_reason = parsed["choices"][0]["finish_reason"]
        .as_str()
        .expect("finish_reason is a string");
    assert!(
        finish_reason == "stop" || finish_reason == "tool_calls" || finish_reason == "length",
        "unexpected finish_reason: {finish_reason}"
    );
    if finish_reason == "tool_calls" {
        let calls = parsed["choices"][0]["message"]["tool_calls"]
            .as_array()
            .expect("tool_calls is an array when finish_reason is tool_calls");
        assert!(!calls.is_empty());
        assert!(calls[0]["function"]["name"].as_str().is_some());
        let args = calls[0]["function"]["arguments"]
            .as_str()
            .expect("arguments is a JSON-encoded string");
        assert!(serde_json::from_str::<Value>(args).is_ok());
    }

    server.shutdown().await;
}

/// Companion to the tools scenario above: Ornith-1.0-9B is the agentic
/// flagship model, so this asserts the tool call actually happens rather
/// than merely tolerating either outcome. `#[ignore]`d (not because it's
/// skipped — `make test-model` runs it explicitly as its own `cargo test`
/// invocation, see the Makefile) but because it needs the ~7.4GB Ornith
/// checkpoint and shouldn't load alongside the 2B test's model in the same
/// test binary (two model loads competing for one GPU's VRAM). Run
/// directly: `cargo test --release -p rocml-serve --test server_e2e --
/// --ignored ornith_tool_call_is_emitted`.
#[tokio::test]
#[ignore]
async fn ornith_tool_call_is_emitted() {
    if skip_if_missing(ORNITH_GGUF_PATH) {
        return;
    }
    let server = spawn_test_server(ORNITH_GGUF_PATH).await;
    let addr = server.addr;
    // Ornith's own template defaults reasoning on; give it more headroom
    // than the 2B scenario so a thinking preamble doesn't eat the whole
    // token budget before the tool call itself appears.
    let body = json!({
        "messages": [{"role": "user", "content": "What is the weather in Prague? Use the get_weather tool."}],
        "tools": [serde_json::from_str::<Value>(WEATHER_TOOL).unwrap()],
        "max_tokens": 512,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let resp = tokio::time::timeout(
        Duration::from_secs(120),
        support::post_json(addr, "/v1/chat/completions", &body),
    )
    .await
    .expect("request timed out");
    assert_eq!(resp.status, 200, "body: {}", resp.body);
    let parsed: Value = serde_json::from_str(&resp.body).expect("valid JSON response");
    assert_eq!(parsed["choices"][0]["finish_reason"], "tool_calls");
    let calls = parsed["choices"][0]["message"]["tool_calls"]
        .as_array()
        .expect("tool_calls array");
    assert!(!calls.is_empty());
    assert_eq!(calls[0]["function"]["name"], "get_weather");

    server.shutdown().await;
}

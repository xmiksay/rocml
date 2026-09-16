//! A deliberately minimal hand-rolled HTTP/1.1 client for the end-to-end
//! test: the workspace's dependency allow-list for `rocml-serve` doesn't
//! include an HTTP client crate (`reqwest`, `hyper` as a client, etc.), and
//! adding one just for tests would be exactly the "extras" the brief asks
//! to avoid. We only need to POST JSON and read back either a
//! `Content-Length` body or a chunked (SSE) one, both well-formed because
//! we control the server on the other end — so a ~60-line client is enough,
//! and it exercises the real TCP/HTTP path `axum::serve` runs in production.
//!
//! Also carries the shared `TestServer`/`spawn_test_server*` router-spawning
//! harness both `server_e2e.rs` and `server_e2e_usage.rs` use. `mod
//! support;` compiles this whole module into each test *binary*
//! independently (unlike a real lib crate, a `tests/` integration binary has
//! no external callers to prove `pub` items reachable), so whichever helper
//! one binary doesn't call would otherwise be flagged dead code by the
//! other — hence the blanket allow.
#![allow(dead_code)]

use std::path::PathBuf;

use rocml_serve::{build, ServerConfig};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Dev/test checkpoints shared by every `server_e2e*` test binary.
pub const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
pub const ORNITH_GGUF_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf";

pub const WEATHER_TOOL: &str = r#"{
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

pub struct TestServer {
    pub addr: std::net::SocketAddr,
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
    pub async fn shutdown(mut self) {
        self.serve_task.abort();
        if let Some(handle) = self.worker_thread.take() {
            let _ = tokio::task::spawn_blocking(move || handle.join()).await;
        }
    }
}

pub async fn spawn_test_server(model_path: PathBuf) -> TestServer {
    spawn_test_server_with_snapshots(model_path, 0).await
}

/// Like [`spawn_test_server`], with the conversation-state snapshot RAM
/// budget (issue #1) as an explicit parameter — `0` (what every test that
/// doesn't care about snapshots uses) disables the snapshot layer entirely,
/// keeping them exactly as they behaved before that feature existed.
pub async fn spawn_test_server_with_snapshots(
    model_path: PathBuf,
    snapshot_ram_mb: usize,
) -> TestServer {
    spawn_test_server_with_options(model_path, snapshot_ram_mb, false).await
}

/// Like [`spawn_test_server`], with issue #9's `--debug-endpoints` flag on
/// (mounting `GET /debug/last_prompt`) — every other caller leaves it off,
/// matching the flag's off-by-default posture.
pub async fn spawn_test_server_with_debug_endpoints(model_path: PathBuf) -> TestServer {
    spawn_test_server_with_options(model_path, 0, true).await
}

pub async fn spawn_test_server_with_options(
    model_path: PathBuf,
    snapshot_ram_mb: usize,
    debug_endpoints: bool,
) -> TestServer {
    spawn_test_server_full(
        model_path,
        snapshot_ram_mb,
        debug_endpoints,
        true,
        rocml::KvCacheMode::Fp16,
    )
    .await
}

/// Like [`spawn_test_server_with_options`], with `--no-think` and
/// `--kv-cache` also controllable — every other helper hardcodes
/// `no_think: true` (see `chat_completions_end_to_end`'s doc comment on why:
/// qwen3.5-2b's own template defaults reasoning off, and this codebase's
/// renderer is hardcoded to Ornith's template, whose default is reasoning
/// *on*) and `KvCacheMode::Fp16`. Issue #12's server-path snapshot-miss
/// regression needs both non-default knobs turned on to reproduce: thinking
/// on (every prior server-level test ran with `no_think: true`) *and* a
/// quantized KV cache (`MixedAttnPlane::capture`'s bulk region — see its
/// doc comment — used to size every snapshot to the server's whole `ctx`
/// regardless of actual conversation length, which a bounded RAM budget
/// evicts before the next turn can look it up; `Fp16`'s dense `AttnPlane`
/// never had this problem since it always captured only the filled
/// prefix).
pub async fn spawn_test_server_full(
    model_path: PathBuf,
    snapshot_ram_mb: usize,
    debug_endpoints: bool,
    no_think: bool,
    kv_cache: rocml::KvCacheMode,
) -> TestServer {
    let (app, worker_thread) = build(ServerConfig {
        model_path,
        ctx: 4096,
        kv_cache,
        use_mmq: false,
        kv_sink: rocml::kv_quant::SINK_LEN,
        kv_window: rocml::kv_quant::WINDOW_LEN,
        moe_cache_slots: None,
        moe_decode_overlap: false,
        max_tokens_default: 128,
        no_think,
        default_sampling: rocml::SamplingParams::default(),
        model_id_override: None,
        snapshot_ram_mb,
        snapshot_dir: None,
        snapshot_disk_mb: 0,
        debug_endpoints,
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

pub struct HttpResponse {
    pub status: u16,
    pub body: String,
}

/// Sends `POST {path}` with a JSON body to `addr` and reads the response to
/// completion. Always sends `Connection: close` so the server closes the
/// socket once it's done, which is what lets `read_to_end` return instead of
/// blocking forever waiting for more bytes on a keep-alive connection.
pub async fn post_json(addr: std::net::SocketAddr, path: &str, body: &str) -> HttpResponse {
    let mut stream = TcpStream::connect(addr)
        .await
        .expect("connect to test server");
    let request = format!(
        "POST {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Content-Type: application/json\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    parse_response(&raw)
}

/// Sends `GET {path}` to `addr` and reads the response to completion — the
/// `POST`-only counterpart above, for issue #9's `GET /debug/last_prompt`.
pub async fn get(addr: std::net::SocketAddr, path: &str) -> HttpResponse {
    let mut stream = TcpStream::connect(addr)
        .await
        .expect("connect to test server");
    let request = format!(
        "GET {path} HTTP/1.1\r\n\
         Host: localhost\r\n\
         Connection: close\r\n\
         \r\n"
    );
    stream
        .write_all(request.as_bytes())
        .await
        .expect("write request");

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).await.expect("read response");
    parse_response(&raw)
}

fn parse_response(raw: &[u8]) -> HttpResponse {
    let header_end = find(raw, b"\r\n\r\n").expect("response has no header/body separator") + 4;
    let header_text = String::from_utf8_lossy(&raw[..header_end]);
    let status = header_text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|code| code.parse().ok())
        .unwrap_or(0);

    let raw_body = &raw[header_end..];
    let is_chunked = header_text
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked");
    let body_bytes = if is_chunked {
        dechunk(raw_body)
    } else {
        raw_body.to_vec()
    };
    HttpResponse {
        status,
        body: String::from_utf8_lossy(&body_bytes).into_owned(),
    }
}

/// Undoes HTTP/1.1 chunked transfer encoding: each chunk is a hex length
/// line, CRLF, that many body bytes, CRLF, repeated until a zero-length
/// chunk. Stops (rather than erroring) on anything unexpected, since a test
/// helper has no one to report a parse error to but the assertion that
/// follows it.
fn dechunk(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let Some(line_len) = find(&data[pos..], b"\r\n") else {
            break;
        };
        let size_line = String::from_utf8_lossy(&data[pos..pos + line_len]);
        let Ok(size) = usize::from_str_radix(size_line.trim(), 16) else {
            break;
        };
        pos += line_len + 2;
        if size == 0 || pos + size > data.len() {
            break;
        }
        out.extend_from_slice(&data[pos..pos + size]);
        pos += size + 2; // skip the chunk's trailing CRLF
    }
    out
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

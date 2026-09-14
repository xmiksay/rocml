//! Issue #12's server-path regression gate: a thinking-enabled multi-turn
//! conversation under a *quantized* KV cache, driven through the real
//! HTTP/routes/worker stack (not `run_turn` called directly, which
//! `rocml/tests/snapshot_thinking_e2e.rs` already covers), must get a real
//! snapshot hit on turn 2.
//!
//! This is exactly the coverage gap that let the bug ship:
//! `rocml-serve/tests/server_e2e.rs`'s two-turn correctness test always ran
//! with `--no-think` and the default `--kv-cache fp16`, so neither half of
//! the real bug was ever exercised together. The actual root cause wasn't
//! the boundary/tokenizer math (that was already verified correct against
//! both the qwen3.5 and Ornith tokenizers) — it was
//! `MixedAttnPlane::capture` (`rocml/src/qwen35/cache_mixed/snapshot.rs`)
//! sizing every quantized-KV snapshot to the server's whole configured
//! `ctx`, not to the conversation's actual length. A turn's own later,
//! larger captures (the natural prefill-boundary and end-of-turn snapshots)
//! would evict its own earlier, smaller stable-boundary capture from a
//! size-bounded RAM store before the *next* turn ever got a chance to look
//! it up — worse (and in production, fatal) the larger `ctx` is. `fp16`'s
//! dense `AttnPlane` never had this problem: its capture was already
//! filled-prefix-only, so a small snapshot in that mode really does cost
//! little RAM regardless of `ctx`.
//!
//! Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`).

use rocml::chat::{render_with_boundary, Message as ChatMessage, RenderOpts};
use rocml::snapshot::turn::stable_boundary_tokens;
use rocml::KvCacheMode;
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;
use serde_json::{json, Value};

mod support;

use support::{spawn_test_server_full, GGUF_REL};

const SYSTEM_PROMPT: &str = "You are a helpful assistant.";
const TURN1_USER: &str = "What is 2+2? Answer with just the final number.";
const TURN2_USER: &str = "Now what is 3+3? Answer with just the final number.";

#[tokio::test]
async fn thinking_enabled_second_turn_hits_the_server_path_snapshot() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        return;
    };

    // `no_think: false` -> `routes.rs` renders with `enable_thinking: None`
    // (thinking on); `KvCacheMode::Q4Mixed` -> the quantized attention path
    // whose oversized bulk-region capture is issue #12's actual root cause.
    // Both are needed together to reproduce.
    //
    // The two servers run sequentially (fully torn down before the next
    // loads), not concurrently: two live Qwen3.5-2B loads plus whatever
    // else already resides on a dev GPU can exceed its free VRAM, and nothing
    // about this gate needs them alive at the same time.
    let with_snapshots =
        spawn_test_server_full(gguf_path.clone(), 64, false, false, KvCacheMode::Q4Mixed).await;

    let (turn1, cached1) = run_turn1(with_snapshots.addr).await;
    assert_eq!(cached1, 0, "turn 1 must be a cold miss");

    let (turn2_with, cached2) = run_turn2(with_snapshots.addr, &turn1).await;
    assert!(
        cached2 > 0,
        "turn 2 must hit the server-path stable-boundary snapshot despite thinking \
         being on — this is issue #12's whole point"
    );

    // The expected boundary: independently recomputed via the exact same
    // `render_with_boundary` + `stable_boundary_tokens` sequence
    // `routes.rs` runs for turn 1's own request, so this assertion pins the
    // hit to *that specific* snapshot, not merely "something" hit.
    let expected_boundary = turn1_stable_boundary(&gguf_path);
    assert_eq!(
        cached2, expected_boundary as usize,
        "turn 2's cached_tokens should equal turn 1's render-stable boundary"
    );

    with_snapshots.shutdown().await;

    // Correctness of the restore+resume path: the same turn 2 conversation
    // against a fresh server with the snapshot layer fully disabled must
    // produce byte-identical output.
    let without_snapshots =
        spawn_test_server_full(gguf_path.clone(), 0, false, false, KvCacheMode::Q4Mixed).await;
    let (turn2_without, cached2_without) = run_turn2(without_snapshots.addr, &turn1).await;
    assert_eq!(
        cached2_without, 0,
        "the no-snapshot server must never cache"
    );
    assert_eq!(
        turn2_with, turn2_without,
        "a snapshot-resumed turn 2 must match a from-scratch turn 2 byte-for-byte"
    );

    without_snapshots.shutdown().await;
}

struct Turn1Output {
    content: String,
    reasoning_content: Option<String>,
}

/// Runs turn 1 ([system, user]) and returns its output plus
/// `usage.prompt_tokens_details.cached_tokens`.
async fn run_turn1(addr: std::net::SocketAddr) -> (Turn1Output, usize) {
    let body = json!({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TURN1_USER},
        ],
        "max_tokens": 96,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let resp = support::post_json(addr, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status, 200, "turn 1 body: {}", resp.body);
    let parsed: Value = serde_json::from_str(&resp.body).expect("valid JSON response");
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .expect("turn 1 content is a string")
        .to_string();
    let reasoning_content = parsed["choices"][0]["message"]["reasoning_content"]
        .as_str()
        .map(str::to_string);
    let cached = parsed["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("cached_tokens present") as usize;
    (
        Turn1Output {
            content,
            reasoning_content,
        },
        cached,
    )
}

/// Runs turn 2 ([system, user, assistant, user2]) against `addr` and returns
/// its content plus `usage.prompt_tokens_details.cached_tokens`.
async fn run_turn2(addr: std::net::SocketAddr, turn1: &Turn1Output) -> (String, usize) {
    let mut assistant_msg = json!({"role": "assistant", "content": turn1.content});
    if let Some(reasoning) = &turn1.reasoning_content {
        assistant_msg["reasoning_content"] = json!(reasoning);
    }
    let body = json!({
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": TURN1_USER},
            assistant_msg,
            {"role": "user", "content": TURN2_USER},
        ],
        "max_tokens": 64,
        "temperature": 0,
        "stream": false,
    })
    .to_string();
    let resp = support::post_json(addr, "/v1/chat/completions", &body).await;
    assert_eq!(resp.status, 200, "turn 2 body: {}", resp.body);
    let parsed: Value = serde_json::from_str(&resp.body).expect("valid JSON response");
    let content = parsed["choices"][0]["message"]["content"]
        .as_str()
        .expect("turn 2 content is a string")
        .to_string();
    let cached = parsed["usage"]["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .expect("cached_tokens present") as usize;
    (content, cached)
}

/// Recomputes turn 1's render-stable boundary exactly the way
/// `routes.rs`'s `handle_request` does for its own turn-1 request, so the
/// test can assert turn 2's hit lands on *that* snapshot specifically.
fn turn1_stable_boundary(gguf_path: &std::path::Path) -> u32 {
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(gguf_path).expect("gguf open failed"))
        .expect("tokenizer load failed");
    let messages = vec![
        ChatMessage::system(SYSTEM_PROMPT),
        ChatMessage::user(TURN1_USER),
    ];
    // Mirrors `routes.rs`: `no_think: false` -> `enable_thinking: None`.
    let opts = RenderOpts {
        add_generation_prompt: true,
        enable_thinking: None,
        keep_history_reasoning: false,
    };
    let (prompt_text, boundary_byte) =
        render_with_boundary(&messages, &[], opts).expect("turn 1 render failed");
    let prompt_ids = tokenizer.encode(&prompt_text);
    stable_boundary_tokens(&tokenizer, &prompt_text, boundary_byte, &prompt_ids)
        .expect("turn 1's stable boundary must be a genuine token prefix")
}

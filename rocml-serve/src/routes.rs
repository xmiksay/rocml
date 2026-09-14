//! Axum routes: `POST /v1/chat/completions` (both the full-JSON and SSE
//! shapes, the latter in `sse.rs`) and `GET /v1/models`. Streaming and
//! non-streaming share the same request mapping and worker dispatch
//! (`handle_request`); they only differ in how they drain the worker's
//! per-job event channel. The optional `GET /debug/last_prompt` route
//! (issue #9, `--debug-endpoints`) is mounted separately by
//! `rocml_serve::build` — see `debug` module.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::State;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rocml::chat::{self, RenderOpts, StreamScanner};
use serde_json::json;
use tokio::sync::mpsc;

use crate::error::ApiError;
use crate::mapping::{self, AssistantAccumulator};
use crate::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessageOut, ChoiceOut, FunctionCallOut,
    PromptTokensDetailsOut, ToolCallOut, UsageOut,
};
use crate::sse::stream_response;
use crate::state::AppState;
use crate::worker::{Job, WorkerEvent};

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .with_state(state)
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": [{"id": state.model_id, "object": "model", "created": 0, "owned_by": "rocml"}],
    }))
}

/// Deserializes the body ourselves (rather than via axum's `Json`
/// extractor) so a malformed request produces our OpenAI-shaped error JSON
/// instead of axum's default rejection body.
async fn chat_completions(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let request: ChatCompletionRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => return ApiError::bad_request(format!("invalid JSON body: {e}")).into_response(),
    };
    match handle_request(state, request).await {
        Ok(resp) => resp,
        Err(e) => e.into_response(),
    }
}

async fn handle_request(
    state: Arc<AppState>,
    request: ChatCompletionRequest,
) -> Result<Response, ApiError> {
    if let Some(model) = request.model.as_deref() {
        if !model.is_empty() && model != state.model_id {
            return Err(ApiError::not_found(format!("model {model:?} not found")));
        }
    }
    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .is_some_and(|o| o.include_usage);
    let messages = mapping::map_messages(&request.messages)?;
    let tools = mapping::map_tools(&request.tools);
    let render_opts = RenderOpts {
        add_generation_prompt: true,
        enable_thinking: state.no_think.then_some(false),
        // Issue #9: an incoming assistant history message's
        // `reasoning_content` (or a `<think>` block embedded in its
        // `content`) must never leak back into the rendered prompt —
        // matches the Qwen3-family training convention, see
        // `RenderOpts::keep_history_reasoning`'s doc comment.
        keep_history_reasoning: false,
    };
    // The generation-prompt tail opens `<think>\n` unless thinking is
    // explicitly off, so the response-side scanner must start already in
    // thinking mode to match — see `StreamScanner::new_primed_for_thinking`.
    let thinking_primed = render_opts.enable_thinking != Some(false);
    let prompt_text = chat::render(&messages, &tools, render_opts)
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let prompt_ids = state.tokenizer.encode(&prompt_text);
    if prompt_ids.len() > state.ctx {
        return Err(ApiError::bad_request(format!(
            "prompt has {} tokens, exceeding this server's context budget of {} tokens",
            prompt_ids.len(),
            state.ctx
        )));
    }
    let requested_max = request
        .max_tokens
        .unwrap_or(state.max_tokens_default)
        .max(1);
    // Clamp rather than reject when only the *combination* overflows —
    // an over-budget prompt alone is the hard error above.
    let max_new_tokens = requested_max.min(state.ctx - prompt_ids.len()).max(1);
    let sampling = mapping::map_sampling(&request, &state.default_sampling);
    let stop_strings = mapping::stop_strings(&request);
    let prompt_tokens = prompt_ids.len();

    // Issue #9's debug endpoint: recorded at accept time (before dispatch)
    // so an in-flight request's prompt is visible immediately, not only
    // after it completes — mirrors llama.cpp's `/slots`. `None` when
    // `--debug-endpoints` wasn't passed, so this is a no-op by default.
    if let Some(debug) = &state.debug {
        debug.record(prompt_text.clone(), prompt_tokens);
    }

    let (tx, rx) = mpsc::unbounded_channel();
    let job = Job {
        prompt_ids,
        max_new_tokens,
        sampling,
        stop_strings,
        respond_to: tx,
    };
    state
        .job_tx
        .send(job)
        .map_err(|_| ApiError::internal("model worker is not running"))?;

    let started = Instant::now();
    tracing::info!(
        route = "/v1/chat/completions",
        prompt_tokens,
        max_new_tokens,
        stream,
        "request accepted"
    );

    if stream {
        Ok(stream_response(
            state.model_id.clone(),
            prompt_tokens,
            max_new_tokens,
            rx,
            started,
            thinking_primed,
            include_usage,
        ))
    } else {
        let response = collect_response(
            &state.model_id,
            prompt_tokens,
            max_new_tokens,
            rx,
            thinking_primed,
        )
        .await?;
        tracing::info!(
            route = "/v1/chat/completions",
            duration_ms = started.elapsed().as_millis() as u64,
            "request completed"
        );
        Ok(response)
    }
}

async fn collect_response(
    model_id: &str,
    prompt_tokens: usize,
    max_new_tokens: usize,
    mut rx: mpsc::UnboundedReceiver<WorkerEvent>,
    thinking_primed: bool,
) -> Result<Response, ApiError> {
    let mut scanner = new_scanner(thinking_primed);
    let mut acc = AssistantAccumulator::default();
    loop {
        match rx.recv().await {
            Some(WorkerEvent::Chunk(text)) => {
                if let Ok(events) = scanner.feed(&text) {
                    for ev in events {
                        acc.apply(ev);
                    }
                }
            }
            Some(WorkerEvent::Done(turn)) => {
                for ev in scanner.finish() {
                    acc.apply(ev);
                }
                let finish_reason = acc.finish_reason(turn.stats.generated_tokens, max_new_tokens);
                let response = ChatCompletionResponse {
                    id: completion_id(),
                    object: "chat.completion",
                    created: now_unix(),
                    model: model_id.to_string(),
                    choices: vec![ChoiceOut {
                        index: 0,
                        message: message_out(&acc),
                        finish_reason,
                    }],
                    usage: UsageOut {
                        prompt_tokens,
                        completion_tokens: turn.stats.generated_tokens,
                        total_tokens: prompt_tokens + turn.stats.generated_tokens,
                        prompt_tokens_details: PromptTokensDetailsOut {
                            cached_tokens: turn.cached_tokens,
                        },
                    },
                };
                return Ok(Json(response).into_response());
            }
            Some(WorkerEvent::Error(e)) => return Err(ApiError::internal(e)),
            None => return Err(ApiError::internal("model worker closed unexpectedly")),
        }
    }
}

fn message_out(acc: &AssistantAccumulator) -> ChatMessageOut {
    ChatMessageOut {
        role: "assistant",
        content: if acc.content.is_empty() && !acc.tool_calls.is_empty() {
            None
        } else {
            Some(acc.content.clone())
        },
        reasoning_content: if acc.thinking.is_empty() {
            None
        } else {
            Some(acc.thinking.clone())
        },
        tool_calls: acc
            .tool_calls
            .iter()
            .enumerate()
            .map(|(i, tc)| ToolCallOut {
                id: format!("call_{i}"),
                kind: "function",
                function: FunctionCallOut {
                    name: tc.name.clone(),
                    arguments: serde_json::to_string(&tc.arguments).unwrap_or_default(),
                },
            })
            .collect(),
    }
}

/// Picks the scanner mode matching how the prompt's generation-prompt tail
/// left the `<think>` block — see `StreamScanner::new_primed_for_thinking`.
/// `pub(crate)`: shared with `sse::pump_stream`.
pub(crate) fn new_scanner(thinking_primed: bool) -> StreamScanner {
    if thinking_primed {
        StreamScanner::new_primed_for_thinking()
    } else {
        StreamScanner::new()
    }
}

pub(crate) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

pub(crate) fn completion_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-{}-{n}", now_unix())
}

//! Axum routes: `POST /v1/chat/completions` (both the full-JSON and SSE
//! shapes) and `GET /v1/models`. Streaming and non-streaming share the same
//! request mapping and worker dispatch (`handle_request`); they only differ
//! in how they drain the worker's per-job event channel.

use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use rocml::chat::{self, RenderOpts, ScanEvent, StreamScanner};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::error::ApiError;
use crate::mapping::{self, AssistantAccumulator};
use crate::openai::{
    self, ChatCompletionRequest, ChatCompletionResponse, ChatMessageOut, ChoiceOut, DeltaOut,
    FunctionCallOut, ToolCallDeltaOut, ToolCallOut, UsageOut,
};
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
    let messages = mapping::map_messages(&request.messages)?;
    let tools = mapping::map_tools(&request.tools);
    let render_opts = RenderOpts {
        add_generation_prompt: true,
        enable_thinking: state.no_think.then_some(false),
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
    let sampling = mapping::map_sampling(&request);
    let stop_strings = mapping::stop_strings(&request);
    let prompt_tokens = prompt_ids.len();

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
            Some(WorkerEvent::Done(stats)) => {
                for ev in scanner.finish() {
                    acc.apply(ev);
                }
                let finish_reason = acc.finish_reason(stats.generated_tokens, max_new_tokens);
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
                        completion_tokens: stats.generated_tokens,
                        total_tokens: prompt_tokens + stats.generated_tokens,
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
fn new_scanner(thinking_primed: bool) -> StreamScanner {
    if thinking_primed {
        StreamScanner::new_primed_for_thinking()
    } else {
        StreamScanner::new()
    }
}

fn stream_response(
    model_id: String,
    prompt_tokens: usize,
    max_new_tokens: usize,
    rx: mpsc::UnboundedReceiver<WorkerEvent>,
    started: Instant,
    thinking_primed: bool,
) -> Response {
    let (sse_tx, sse_rx) = mpsc::unbounded_channel::<Result<Event, Infallible>>();
    tokio::spawn(pump_stream(
        model_id,
        prompt_tokens,
        max_new_tokens,
        rx,
        sse_tx,
        started,
        thinking_primed,
    ));
    Sse::new(UnboundedReceiverStream::new(sse_rx))
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Drains the worker's event channel, translating each `ScanEvent` into an
/// OpenAI delta chunk as it arrives, then closes with a final chunk carrying
/// `finish_reason` and a literal `[DONE]` — never leaving the connection
/// hanging even if generation errors out mid-stream.
async fn pump_stream(
    model_id: String,
    prompt_tokens: usize,
    max_new_tokens: usize,
    mut rx: mpsc::UnboundedReceiver<WorkerEvent>,
    tx: mpsc::UnboundedSender<Result<Event, Infallible>>,
    started: Instant,
    thinking_primed: bool,
) {
    let id = completion_id();
    let created = now_unix();
    let _ = send_chunk(
        &tx,
        openai::chunk(
            &id,
            created,
            &model_id,
            DeltaOut {
                role: Some("assistant"),
                ..Default::default()
            },
            None,
        ),
    );

    let mut scanner = new_scanner(thinking_primed);
    let mut acc = AssistantAccumulator::default();
    let outcome = loop {
        match rx.recv().await {
            Some(WorkerEvent::Chunk(text)) => {
                if let Ok(events) = scanner.feed(&text) {
                    for ev in events {
                        emit_event(&tx, &id, created, &model_id, ev, &mut acc);
                    }
                }
            }
            Some(WorkerEvent::Done(stats)) => break Ok(stats),
            Some(WorkerEvent::Error(e)) => break Err(e),
            None => break Err("model worker closed unexpectedly".to_string()),
        }
    };

    let (finish_reason, completion_tokens) = match outcome {
        Ok(stats) => {
            for ev in scanner.finish() {
                emit_event(&tx, &id, created, &model_id, ev, &mut acc);
            }
            (
                acc.finish_reason(stats.generated_tokens, max_new_tokens),
                stats.generated_tokens,
            )
        }
        Err(e) => {
            tracing::error!(error = %e, "generation failed mid-stream");
            ("stop", 0)
        }
    };
    let _ = send_chunk(
        &tx,
        openai::chunk(
            &id,
            created,
            &model_id,
            DeltaOut::default(),
            Some(finish_reason),
        ),
    );
    let _ = tx.send(Ok(Event::default().data("[DONE]")));
    tracing::info!(
        route = "/v1/chat/completions",
        prompt_tokens,
        completion_tokens,
        duration_ms = started.elapsed().as_millis() as u64,
        "request completed"
    );
}

fn emit_event(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    id: &str,
    created: u64,
    model_id: &str,
    event: ScanEvent,
    acc: &mut AssistantAccumulator,
) {
    match event {
        ScanEvent::TextDelta(s) => {
            let delta = DeltaOut {
                content: Some(s.clone()),
                ..Default::default()
            };
            acc.apply(ScanEvent::TextDelta(s));
            let _ = send_chunk(tx, openai::chunk(id, created, model_id, delta, None));
        }
        ScanEvent::ThinkingDelta(s) => {
            let delta = DeltaOut {
                reasoning_content: Some(s.clone()),
                ..Default::default()
            };
            acc.apply(ScanEvent::ThinkingDelta(s));
            let _ = send_chunk(tx, openai::chunk(id, created, model_id, delta, None));
        }
        ScanEvent::ToolCallStarted => {}
        ScanEvent::ToolCallComplete(call) => {
            let index = acc.tool_calls.len();
            let delta = DeltaOut {
                tool_calls: vec![ToolCallDeltaOut {
                    index,
                    id: format!("call_{id}_{index}"),
                    kind: "function",
                    function: FunctionCallOut {
                        name: call.name.clone(),
                        arguments: serde_json::to_string(&call.arguments).unwrap_or_default(),
                    },
                }],
                ..Default::default()
            };
            acc.apply(ScanEvent::ToolCallComplete(call));
            let _ = send_chunk(tx, openai::chunk(id, created, model_id, delta, None));
        }
    }
}

fn send_chunk(
    tx: &mpsc::UnboundedSender<Result<Event, Infallible>>,
    chunk: openai::ChatCompletionChunk,
) -> Result<(), ()> {
    let data = serde_json::to_string(&chunk).map_err(|_| ())?;
    tx.send(Ok(Event::default().data(data))).map_err(|_| ())
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn completion_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-{}-{n}", now_unix())
}

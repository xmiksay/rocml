//! The SSE half of `POST /v1/chat/completions`: split out of `routes.rs`
//! purely for the 400-line cap (streaming vs. non-streaming response
//! building is the natural seam — `routes::handle_request` calls into
//! [`stream_response`] exactly the way it calls the non-streaming
//! `collect_response`, sharing `routes::new_scanner`/`now_unix`/
//! `completion_id`).

use std::convert::Infallible;
use std::time::Instant;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use rocml::chat::ScanEvent;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::mapping::AssistantAccumulator;
use crate::openai::{self, DeltaOut, FunctionCallOut, ToolCallDeltaOut};
use crate::routes::{completion_id, new_scanner, now_unix};
use crate::worker::WorkerEvent;

pub(crate) fn stream_response(
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

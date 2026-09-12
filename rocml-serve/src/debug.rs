//! Issue #9's diagnostic endpoint: `GET /debug/last_prompt`, the direct
//! analogue of llama.cpp's `/slots` (returning a slot's fully formatted
//! prompt), which was the decisive tool in tracing a stuck-agent loop back
//! to the serving stack rather than the model. Off by default — gated by
//! `rocml-serve --debug-endpoints` — because it exposes the exact rendered
//! prompt text of the last (or in-flight) request, i.e. full conversation
//! content including system prompts and tool results. **Do not enable this
//! on a shared host.**
//!
//! `routes::handle_request` calls [`LastPromptState::record`] at
//! request-accept time, before the job is dispatched to the worker, so an
//! in-flight request's prompt is visible immediately rather than only once
//! it completes.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct LastPromptResponse {
    pub prompt: Option<String>,
    pub prompt_tokens: Option<usize>,
    pub timestamp: Option<u64>,
}

#[derive(Default)]
pub struct LastPromptState(Mutex<LastPromptResponse>);

impl LastPromptState {
    /// Overwrites the recorded prompt. A poisoned lock (only possible if a
    /// prior holder panicked mid-write) falls back to the poisoned guard's
    /// data rather than propagating — a stale/lost debug read is never
    /// worth crashing a real request over.
    pub fn record(&self, prompt: String, prompt_tokens: usize) {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let mut guard = self.0.lock().unwrap_or_else(|e| e.into_inner());
        *guard = LastPromptResponse {
            prompt: Some(prompt),
            prompt_tokens: Some(prompt_tokens),
            timestamp: Some(timestamp),
        };
    }

    fn snapshot(&self) -> LastPromptResponse {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

pub fn router(state: Arc<LastPromptState>) -> Router {
    Router::new()
        .route("/debug/last_prompt", get(last_prompt))
        .with_state(state)
}

async fn last_prompt(State(state): State<Arc<LastPromptState>>) -> Json<LastPromptResponse> {
    Json(state.snapshot())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let state = LastPromptState::default();
        let snap = state.snapshot();
        assert!(snap.prompt.is_none());
        assert!(snap.prompt_tokens.is_none());
        assert!(snap.timestamp.is_none());
    }

    #[test]
    fn record_overwrites_the_previous_snapshot() {
        let state = LastPromptState::default();
        state.record("first prompt".to_string(), 3);
        state.record("second prompt".to_string(), 7);
        let snap = state.snapshot();
        assert_eq!(snap.prompt.as_deref(), Some("second prompt"));
        assert_eq!(snap.prompt_tokens, Some(7));
        assert!(snap.timestamp.is_some());
    }
}

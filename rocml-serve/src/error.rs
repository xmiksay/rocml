//! OpenAI-shaped error responses (`{"error": {"message", "type", "code"}}`)
//! for every failure path a client request can hit — malformed JSON, an
//! unknown model id, an over-budget prompt, or a generation failure — so a
//! client never sees a bare status code or an axum default rejection body.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug)]
pub enum ApiError {
    /// Client-caused: malformed JSON, invalid role, bad tool-call
    /// arguments, an over-budget prompt. Maps to 400.
    BadRequest(String),
    /// An unrecognized `model` field. Maps to 404, matching the real
    /// OpenAI API's `model_not_found`.
    NotFound(String),
    /// Something failed on the server's side (worker channel gone, a
    /// generation error surfaced from the model). Maps to 500 — never a
    /// panic, per the worker's own catch-unwind guard (see `worker`).
    Internal(String),
}

impl ApiError {
    pub fn bad_request(msg: impl Into<String>) -> Self {
        Self::BadRequest(msg.into())
    }

    pub fn not_found(msg: impl Into<String>) -> Self {
        Self::NotFound(msg.into())
    }

    pub fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, kind, message) = match self {
            ApiError::BadRequest(m) => (StatusCode::BAD_REQUEST, "invalid_request_error", m),
            ApiError::NotFound(m) => (StatusCode::NOT_FOUND, "invalid_request_error", m),
            ApiError::Internal(m) => (StatusCode::INTERNAL_SERVER_ERROR, "server_error", m),
        };
        let body = json!({
            "error": {
                "message": message,
                "type": kind,
                "code": serde_json::Value::Null,
            }
        });
        (status, Json(body)).into_response()
    }
}

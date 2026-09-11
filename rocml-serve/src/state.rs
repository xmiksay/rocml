//! Shared server state: the tokenizer (plain CPU data, safe to share) and a
//! handle to submit jobs to the model-owning worker thread. The `Model`
//! itself never appears here — see `worker` module docs for why.

use std::sync::mpsc::Sender;
use std::sync::Arc;

use rocml::SamplingParams;
use rocml_core::tokenizer::BpeTokenizer;

use crate::worker::Job;

pub struct AppState {
    pub job_tx: Sender<Job>,
    pub tokenizer: Arc<BpeTokenizer>,
    /// Served as both `GET /v1/models`' single entry and the value
    /// `POST /v1/chat/completions` compares an incoming `model` field
    /// against. The resolved registry name when `--model` was a registry
    /// hit, else the GGUF file's stem.
    pub model_id: String,
    /// Sampling defaults applied to any request field the client leaves
    /// unset — the resolved model's registry preset, or the engine's own
    /// greedy default for a path-based `--model`.
    pub default_sampling: SamplingParams,
    /// Soft context budget: requests whose prompt alone exceeds this are
    /// rejected; requests that would exceed it once `max_tokens` is added
    /// have `max_tokens` clamped down instead. The model's own KV cache
    /// capacity (fixed at load time from the GGUF's `context_length`) is a
    /// separate, usually larger, hard limit enforced independently by
    /// `Model::forward_token`.
    pub ctx: usize,
    pub max_tokens_default: usize,
    /// `--no-think`: pre-closes the `<think>` block on every request unless
    /// the request overrides it (not currently exposed per-request — the
    /// spec only asks for the server-wide flag).
    pub no_think: bool,
}

//! OpenAI-compatible HTTP server for rocml. Split into a library so
//! integration tests can build the router directly (binding an ephemeral
//! port themselves) without going through the `rocml-serve` binary.

pub mod error;
pub mod mapping;
pub mod openai;
pub mod routes;
pub mod state;
pub mod worker;

use std::path::PathBuf;
use std::sync::Arc;

use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::BpeTokenizer;

pub struct ServerConfig {
    pub model_path: PathBuf,
    pub ctx: usize,
    pub max_tokens_default: usize,
    pub no_think: bool,
}

/// Loads the tokenizer, spawns the model-owning worker thread (see
/// `worker` module docs for why the `Model` itself never leaves that
/// thread), and builds the axum router. Binding a socket and serving it is
/// the caller's job — this split is what lets tests drive the router
/// against an ephemeral port in-process.
///
/// The returned `JoinHandle` is for a clean-shutdown caller (see
/// `worker::spawn`'s doc comment); `rocml-serve`'s own `main` just drops it,
/// since the server process runs until killed rather than shutting down
/// through ordinary Rust control flow.
pub fn build(config: ServerConfig) -> Result<(axum::Router, std::thread::JoinHandle<()>), String> {
    let gguf = GgufFile::open(&config.model_path).map_err(|e| e.to_string())?;
    let tokenizer = Arc::new(BpeTokenizer::from_gguf(&gguf).map_err(|e| e.to_string())?);
    drop(gguf);

    let model_id = config
        .model_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string());
    let (job_tx, worker_handle) = worker::spawn(config.model_path.clone(), tokenizer.clone())?;

    let state = Arc::new(state::AppState {
        job_tx,
        tokenizer,
        model_id,
        ctx: config.ctx,
        max_tokens_default: config.max_tokens_default,
        no_think: config.no_think,
    });
    Ok((routes::router(state), worker_handle))
}

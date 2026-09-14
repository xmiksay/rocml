//! OpenAI-compatible HTTP server for rocml. Split into a library so
//! integration tests can build the router directly (binding an ephemeral
//! port themselves) without going through the `rocml-serve` binary.

pub mod debug;
pub mod error;
pub mod mapping;
pub mod openai;
pub mod routes;
pub mod sse;
pub mod state;
pub mod worker;

use std::path::PathBuf;
use std::sync::Arc;

use rocml::{KvCacheMode, SamplingParams};
use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::BpeTokenizer;

pub struct ServerConfig {
    pub model_path: PathBuf,
    pub ctx: usize,
    /// KV cache storage/quantization policy (issue #2/#3) — see
    /// `rocml::KvCacheMode`.
    pub kv_cache: KvCacheMode,
    /// Routes qwen35 hybrid chunked-prefill matmuls through the int8 MMQ
    /// GEMM instead of f16 WMMA where eligible — see
    /// `rocml::LoadOptions::use_mmq`'s doc comment. Off by default.
    pub use_mmq: bool,
    /// Attention-sink/recent-window lengths for a quantized `kv_cache` mode
    /// (issue #2 leftovers) — see `rocml::LoadOptions::kv_sink`/`kv_window`.
    pub kv_sink: u32,
    pub kv_window: u32,
    pub max_tokens_default: usize,
    pub no_think: bool,
    /// Sampling defaults for requests that omit a field — from the resolved
    /// model's registry preset, or `SamplingParams::default()` (greedy) for
    /// a path-based `--model`. See `rocml::registry`.
    pub default_sampling: SamplingParams,
    /// Overrides the served model id (normally the GGUF file's stem) with
    /// the registry name, when `--model` resolved through the registry.
    pub model_id_override: Option<String>,
    /// Conversation-state snapshot layer (issue #1), qwen35-hybrid-only —
    /// see `rocml::snapshot`. `snapshot_ram_mb == 0` disables the RAM tier
    /// entirely; `snapshot_dir: None` leaves the optional NVMe tier off.
    pub snapshot_ram_mb: usize,
    pub snapshot_dir: Option<PathBuf>,
    pub snapshot_disk_mb: u64,
    /// Issue #9: mounts `GET /debug/last_prompt` when `true`. Off by
    /// default — it exposes the exact rendered prompt text of the last (or
    /// in-flight) request, i.e. full conversation content. Do not enable
    /// on a shared host. See `debug` module docs.
    pub debug_endpoints: bool,
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

    let model_id = config.model_id_override.unwrap_or_else(|| {
        config
            .model_path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".to_string())
    });
    let load_opts = rocml::LoadOptions {
        ctx: config.ctx,
        kv_cache: config.kv_cache,
        use_mmq: config.use_mmq,
        kv_sink: config.kv_sink,
        kv_window: config.kv_window,
        // Issue #14 phase 2's debug quality-simulation flag is
        // research-only (driven through `rocml-cli`'s `eval`/`bench`), not
        // exposed by the server.
        kv_rot_sim: None,
        kv_rot_sim_k: false,
    };
    let snapshot_config = worker::SnapshotConfig {
        ram_mb: config.snapshot_ram_mb,
        dir: config.snapshot_dir,
        disk_mb: config.snapshot_disk_mb,
    };
    let (job_tx, worker_handle) = worker::spawn(
        config.model_path.clone(),
        load_opts,
        tokenizer.clone(),
        snapshot_config,
    )?;

    let debug_state = config
        .debug_endpoints
        .then(|| Arc::new(debug::LastPromptState::default()));

    let state = Arc::new(state::AppState {
        job_tx,
        tokenizer,
        model_id,
        default_sampling: config.default_sampling,
        ctx: config.ctx,
        max_tokens_default: config.max_tokens_default,
        no_think: config.no_think,
        debug: debug_state.clone(),
    });
    let mut app = routes::router(state);
    if let Some(debug_state) = debug_state {
        app = app.merge(debug::router(debug_state));
    }
    Ok((app, worker_handle))
}

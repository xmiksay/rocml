//! `rocml-serve --model <name-or-gguf> [--host] [--port] [--ctx] [--kv-cache] [--max-tokens-default] [--no-think] [--no-download] [--debug-endpoints]`

use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use rocml::KvCacheMode;

/// Fallback soft context budget when neither `--ctx` nor a resolved
/// registry preset supplies one (i.e. a path-based `--model` with no
/// preset) — the server's pre-registry default.
const DEFAULT_CTX: usize = 8192;

/// `--kv-cache` flag spelling, mapped to `rocml::KvCacheMode` — mirrors
/// `rocml-cli`'s `common::KvCacheArg` (duplicated rather than shared: it's
/// a two-variant-mapping enum, not worth a cross-crate dependency for).
#[derive(Debug, Clone, Copy, ValueEnum)]
enum KvCacheArg {
    Fp16,
    Q8,
    #[value(name = "q4-mixed")]
    Q4Mixed,
}

impl From<KvCacheArg> for KvCacheMode {
    fn from(arg: KvCacheArg) -> Self {
        match arg {
            KvCacheArg::Fp16 => KvCacheMode::Fp16,
            KvCacheArg::Q8 => KvCacheMode::Q8,
            KvCacheArg::Q4Mixed => KvCacheMode::Q4Mixed,
        }
    }
}

#[derive(Parser)]
#[command(
    name = "rocml-serve",
    about = "OpenAI-compatible HTTP server for rocml"
)]
struct Args {
    /// Registry name (e.g. `qwen3.5-2b`) or a path to a `.gguf` file.
    #[arg(long)]
    model: String,
    /// A registry hit whose file is missing errors out instead of
    /// downloading it via `hf` (the default).
    #[arg(long)]
    no_download: bool,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Soft context budget in tokens — see `rocml_serve::state::AppState::ctx`.
    /// Unset falls back to the resolved model's registry preset, then
    /// [`DEFAULT_CTX`]; either way it's clamped to this checkpoint's
    /// estimated VRAM budget (`rocml::registry::clamp_ctx`, issue #3) and
    /// is what the KV cache is actually allocated for.
    #[arg(long)]
    ctx: Option<usize>,
    /// KV cache storage/quantization policy (issue #2/#3) — see
    /// `rocml::KvCacheMode`. Default `fp16`; quantized modes are opt-in.
    #[arg(long, value_enum, default_value = "fp16")]
    kv_cache: KvCacheArg,
    /// Routes qwen35 hybrid chunked-prefill matmuls through the int8 MMQ
    /// GEMM instead of f16 WMMA where eligible — see
    /// `rocml::LoadOptions::use_mmq`'s doc comment. Off by default.
    #[arg(long)]
    mmq: bool,
    #[arg(long = "max-tokens-default", default_value_t = 512)]
    max_tokens_default: usize,
    /// Pre-close the `<think>` block on every request (reasoning off).
    /// Unset falls back to the resolved model's registry preset (e.g.
    /// Qwen3.5-2B defaults reasoning off; Ornith-1.0-9B defaults it on).
    #[arg(long)]
    no_think: bool,
    /// Conversation-state snapshot RAM budget in MiB (issue #1,
    /// qwen35-hybrid-only). `0` disables the snapshot layer's RAM tier
    /// entirely (a snapshot is only ever kept in RAM, never spilled to disk,
    /// unless `--snapshot-dir` is also set).
    #[arg(long = "snapshot-ram-mb", default_value_t = 4096)]
    snapshot_ram_mb: usize,
    /// Optional NVMe persistence tier for snapshots — content-addressed
    /// files under this directory, size-budgeted by `--snapshot-disk-mb`.
    /// Off by default.
    #[arg(long = "snapshot-dir")]
    snapshot_dir: Option<std::path::PathBuf>,
    #[arg(long = "snapshot-disk-mb", default_value_t = 20_000)]
    snapshot_disk_mb: u64,
    /// Issue #9: mount `GET /debug/last_prompt`, returning the exact
    /// rendered prompt string (plus token count and timestamp) of the
    /// last-accepted or in-flight request. Off by default — this exposes
    /// full conversation content (system prompt, prior turns, tool
    /// results). Do not enable on a shared host.
    #[arg(long = "debug-endpoints")]
    debug_endpoints: bool,
}

#[tokio::main]
async fn main() -> ExitCode {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();

    match run(Args::parse()).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run(args: Args) -> Result<(), String> {
    let host = args.host.clone();
    let port = args.port;
    let resolved = rocml::resolve(&args.model, !args.no_download).map_err(|e| e.to_string())?;
    let spec = resolved.spec;

    let kv_cache: KvCacheMode = args.kv_cache.into();
    // Explicit flags always win; an unset one falls through to the
    // resolved model's registry preset, then this binary's own
    // pre-registry default — then clamped to this checkpoint's estimated
    // VRAM budget (issue #3).
    let requested_ctx = args
        .ctx
        .unwrap_or_else(|| spec.map_or(DEFAULT_CTX, |s| s.default_ctx));
    let ctx = rocml::registry::clamp_ctx(requested_ctx, &resolved.path, kv_cache)
        .map_err(|e| e.to_string())?;
    // `--no-think` can only force reasoning off, not force it on over a
    // preset that defaults it off — matching the CLI's existing one-way
    // switch (there's no `--think` counterpart today).
    let no_think = args.no_think || spec.is_some_and(|s| !s.thinking_default);
    let default_sampling = spec.map(|s| s.sampling).unwrap_or_default();
    let model_id_override = spec.map(|s| s.name.to_string());

    let config = rocml_serve::ServerConfig {
        model_path: resolved.path,
        ctx,
        kv_cache,
        use_mmq: args.mmq,
        max_tokens_default: args.max_tokens_default,
        no_think,
        default_sampling,
        model_id_override,
        snapshot_ram_mb: args.snapshot_ram_mb,
        snapshot_dir: args.snapshot_dir,
        snapshot_disk_mb: args.snapshot_disk_mb,
        debug_endpoints: args.debug_endpoints,
    };
    // The worker-thread join handle is for callers that shut down cleanly
    // (see `rocml_serve::build`'s doc comment) — this server runs until
    // killed, so it's dropped rather than joined.
    let (app, _worker_handle) = rocml_serve::build(config)?;

    let listener = tokio::net::TcpListener::bind((host.as_str(), port))
        .await
        .map_err(|e| format!("failed to bind {host}:{port}: {e}"))?;
    let local_addr = listener
        .local_addr()
        .map_err(|e| format!("failed to read local address: {e}"))?;
    tracing::info!("rocml-serve listening on http://{local_addr}");

    axum::serve(listener, app)
        .await
        .map_err(|e| format!("server error: {e}"))
}

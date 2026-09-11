//! `rocml-serve --model <name-or-gguf> [--host] [--port] [--ctx] [--max-tokens-default] [--no-think] [--no-download]`

use std::process::ExitCode;

use clap::Parser;

/// Fallback soft context budget when neither `--ctx` nor a resolved
/// registry preset supplies one (i.e. a path-based `--model` with no
/// preset) — the server's pre-registry default.
const DEFAULT_CTX: usize = 8192;

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
    /// [`DEFAULT_CTX`]; either way it's clamped to the engine's current
    /// cache cap (`rocml::registry::clamp_ctx`).
    #[arg(long)]
    ctx: Option<usize>,
    #[arg(long = "max-tokens-default", default_value_t = 512)]
    max_tokens_default: usize,
    /// Pre-close the `<think>` block on every request (reasoning off).
    /// Unset falls back to the resolved model's registry preset (e.g.
    /// Qwen3.5-2B defaults reasoning off; Ornith-1.0-9B defaults it on).
    #[arg(long)]
    no_think: bool,
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

    // Explicit flags always win; an unset one falls through to the
    // resolved model's registry preset, then this binary's own
    // pre-registry default.
    let ctx = rocml::registry::clamp_ctx(
        args.ctx
            .unwrap_or_else(|| spec.map_or(DEFAULT_CTX, |s| s.default_ctx)),
    );
    // `--no-think` can only force reasoning off, not force it on over a
    // preset that defaults it off — matching the CLI's existing one-way
    // switch (there's no `--think` counterpart today).
    let no_think = args.no_think || spec.is_some_and(|s| !s.thinking_default);
    let default_sampling = spec.map(|s| s.sampling).unwrap_or_default();
    let model_id_override = spec.map(|s| s.name.to_string());

    let config = rocml_serve::ServerConfig {
        model_path: resolved.path,
        ctx,
        max_tokens_default: args.max_tokens_default,
        no_think,
        default_sampling,
        model_id_override,
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

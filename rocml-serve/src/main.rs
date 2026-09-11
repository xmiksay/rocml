//! `rocml-serve --model <gguf> [--host] [--port] [--ctx] [--max-tokens-default] [--no-think]`

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

#[derive(Parser)]
#[command(
    name = "rocml-serve",
    about = "OpenAI-compatible HTTP server for rocml"
)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8080)]
    port: u16,
    /// Soft context budget in tokens — see `rocml_serve::state::AppState::ctx`.
    #[arg(long, default_value_t = 8192)]
    ctx: usize,
    #[arg(long = "max-tokens-default", default_value_t = 512)]
    max_tokens_default: usize,
    /// Pre-close the `<think>` block on every request (reasoning off).
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
    let config = rocml_serve::ServerConfig {
        model_path: args.model,
        ctx: args.ctx,
        max_tokens_default: args.max_tokens_default,
        no_think: args.no_think,
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

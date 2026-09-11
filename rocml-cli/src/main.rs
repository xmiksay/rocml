//! `rocml-cli`: interactive chat REPL, a synthetic throughput benchmark, and
//! a one-shot prompt-completion command, all driving the `rocml` engine.

mod cmd_bench;
mod cmd_chat;
mod cmd_generate;
mod common;

use std::process::ExitCode;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "rocml-cli", about = "Chat, bench, and generate for rocml")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive chat REPL.
    Chat(cmd_chat::ChatArgs),
    /// Synthetic prompt/decode throughput benchmark.
    Bench(cmd_bench::BenchArgs),
    /// One-shot prompt completion.
    Generate(cmd_generate::GenerateArgs),
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match &cli.command {
        Command::Chat(args) => cmd_chat::run(args),
        Command::Bench(args) => cmd_bench::run(args),
        Command::Generate(args) => cmd_generate::run(args),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

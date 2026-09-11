//! `cargo run --release -p rocml --example generate -- --model <path> \
//!     --prompt "..." -n 64`
//!
//! Applies the Qwen chat template (a temporary hardcode — the real template
//! engine is milestone 5), greedy-generates, streams decoded text to stdout
//! as it's produced, and reports prompt/decode tokens-per-second.

use std::env;
use std::io::Write;
use std::process::ExitCode;

use rocml::generate::generate;
use rocml::Model;
use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::BpeTokenizer;

struct Args {
    model: String,
    prompt: String,
    n: usize,
    /// Skips the chat template and feeds the prompt text as-is — matches
    /// how the qwen3.5 parity fixtures were captured (raw continuation, no
    /// special tokens), and useful for probing raw completion behavior on
    /// any model.
    raw: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model = None;
    let mut prompt = None;
    let mut n = 64usize;
    let mut raw = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => model = Some(args.next().ok_or("--model needs a value")?),
            "--prompt" => prompt = Some(args.next().ok_or("--prompt needs a value")?),
            "-n" => {
                let v = args.next().ok_or("-n needs a value")?;
                n = v
                    .parse()
                    .map_err(|_| format!("-n: invalid integer {v:?}"))?;
            }
            "--raw" => raw = true,
            other => return Err(format!("unrecognized argument {other:?}")),
        }
    }

    Ok(Args {
        model: model.ok_or("--model <path> is required")?,
        prompt: prompt.ok_or("--prompt <text> is required")?,
        n,
        raw,
    })
}

/// Temporary hardcode; milestone 5 adds a real chat-template engine driven
/// by the GGUF's own `tokenizer.chat_template` metadata.
fn apply_chat_template(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!("usage: generate --model <path.gguf> --prompt <text> [-n <count>] [--raw]");
            return ExitCode::FAILURE;
        }
    };

    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), rocml::RocmlError> {
    eprintln!("loading {}...", args.model);
    let gguf = GgufFile::open(&args.model)?;
    let tokenizer = BpeTokenizer::from_gguf(&gguf)?;
    drop(gguf); // Model::load mmaps its own handle; no need to hold two.

    let mut model = Model::load(&args.model)?;
    let mem = model.memory_info()?;
    eprintln!(
        "loaded: {} layers, hidden={}, vocab={}; VRAM free {:.0} MiB / total {:.0} MiB",
        model.block_count(),
        model.embedding_length(),
        model.vocab_size(),
        mem.free as f64 / (1024.0 * 1024.0),
        mem.total as f64 / (1024.0 * 1024.0),
    );

    let prompt_text = if args.raw {
        args.prompt.clone()
    } else {
        apply_chat_template(&args.prompt)
    };
    let prompt_ids = tokenizer.encode(&prompt_text);

    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let stats = generate(&mut model, &tokenizer, &prompt_ids, args.n, true, |chunk| {
        let _ = lock.write_all(chunk.as_bytes());
        let _ = lock.flush();
    })?;
    println!();

    eprintln!(
        "prompt: {} tokens in {:.3}s ({:.1} tok/s); decode: {} tokens in {:.3}s ({:.1} tok/s)",
        stats.prompt_tokens,
        stats.prompt_seconds,
        stats.prompt_tokens_per_sec(),
        stats.generated_tokens,
        stats.decode_seconds,
        stats.decode_tokens_per_sec(),
    );

    Ok(())
}

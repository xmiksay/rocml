//! `rocml-cli generate`: one-shot prompt completion. Port of the old
//! `rocml/examples/generate.rs` smoke test, now driving the real chat
//! template (`rocml::chat::render`) instead of a hardcoded `<|im_start|>`
//! format string, and the real sampler instead of always-greedy.

use std::io::Write;

use clap::Args;
use rocml::chat::{Message, RenderOpts};
use rocml::generate::generate_sampled;
use rocml::RocmlError;

use crate::common::{self, ModelArgs, SamplingArgs};

#[derive(Args, Debug)]
pub struct GenerateArgs {
    #[command(flatten)]
    model_args: ModelArgs,
    #[arg(long)]
    prompt: String,
    /// Skip the chat template and feed `prompt` as a raw continuation —
    /// matches how the qwen3.5 parity fixtures were captured, and useful
    /// for probing raw completion behavior on any model. Ignores
    /// `--no-think` (there's no template to prime a `<think>` block in).
    #[arg(long)]
    raw: bool,
    /// Close the `<think>` block immediately (reasoning off) instead of the
    /// resolved model's registry preset (or the template's own default for
    /// a path-based `--model`).
    #[arg(long)]
    no_think: bool,
    #[arg(short = 'n', long = "max-tokens", default_value_t = 64)]
    n: usize,
    #[command(flatten)]
    sampling: SamplingArgs,
}

pub fn run(args: &GenerateArgs) -> Result<(), RocmlError> {
    let resolved = args.model_args.resolve()?;
    let spec = resolved.spec;
    eprintln!("loading {}...", args.model_args.model);
    let mut loaded = common::load(&resolved.path)?;
    let mem = loaded.model.memory_info()?;
    eprintln!(
        "loaded: {} layers, hidden={}, vocab={}; VRAM free {:.0} MiB / total {:.0} MiB",
        loaded.model.block_count(),
        loaded.model.embedding_length(),
        loaded.model.vocab_size(),
        mem.free as f64 / (1024.0 * 1024.0),
        mem.total as f64 / (1024.0 * 1024.0),
    );

    let prompt_text = if args.raw {
        args.prompt.clone()
    } else {
        let messages = [Message::user(&args.prompt)];
        let enable_thinking = if args.no_think {
            Some(false)
        } else {
            spec.map(|s| s.thinking_default)
        };
        rocml::chat::render(
            &messages,
            &[],
            RenderOpts {
                add_generation_prompt: true,
                enable_thinking,
            },
        )
        .map_err(|e| RocmlError::Config(e.to_string()))?
    };
    let prompt_ids = loaded.tokenizer.encode(&prompt_text);

    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let sampling = args.sampling.to_sampling_params(spec.map(|s| &s.sampling));
    let stats = generate_sampled(
        &mut loaded.model,
        &loaded.tokenizer,
        &prompt_ids,
        args.n,
        true,
        &sampling,
        |chunk| {
            let _ = lock.write_all(chunk.as_bytes());
            let _ = lock.flush();
        },
    )?;
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

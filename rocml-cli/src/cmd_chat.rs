//! `rocml-cli chat`: interactive REPL over `rocml::chat`.
//!
//! Deliberately re-renders and re-processes the *entire* conversation from
//! `Model::reset()` on every turn rather than trying to reuse the KV cache
//! incrementally: incremental reuse would need the freshly re-tokenized
//! transcript prefix to exactly match what's already resident in the cache
//! turn after turn, and BPE re-tokenization of model-decoded text isn't
//! guaranteed to round-trip byte-for-byte forever. A REPL session is short
//! enough that full reprocessing is cheap, so KISS wins over that fragility.

use std::io::{self, BufRead, Write};

use clap::Args;
use rocml::chat::{Message, RenderOpts, ScanEvent, StreamScanner, ToolCall};
use rocml::generate::generate_sampled;
use rocml::RocmlError;

use crate::common::{self, ModelArgs, SamplingArgs};

/// Fallback soft context budget when neither `--ctx` nor a resolved
/// registry preset supplies one (i.e. a path-based `--model` with no
/// preset) — this command's pre-registry default.
const DEFAULT_CTX: usize = 4096;

#[derive(Args, Debug)]
pub struct ChatArgs {
    #[command(flatten)]
    model_args: ModelArgs,
    /// Close the `<think>` block immediately (reasoning off) instead of the
    /// resolved model's registry preset (or leaving it open for a
    /// path-based `--model` with no preset).
    #[arg(long)]
    no_think: bool,
    #[arg(long = "max-tokens", default_value_t = 512)]
    max_tokens: usize,
    /// Soft context budget: a turn whose prompt plus `max-tokens` would
    /// exceed this is rejected up front with a friendly error instead of
    /// running into the model's own (harder to interpret) cache-capacity
    /// error partway through decoding. Unset falls back to the resolved
    /// model's registry preset, then [`DEFAULT_CTX`]; either way it's
    /// clamped to the engine's current cache cap
    /// (`rocml::registry::clamp_ctx`).
    #[arg(long)]
    ctx: Option<usize>,
    #[command(flatten)]
    sampling: SamplingArgs,
}

pub fn run(args: &ChatArgs) -> Result<(), RocmlError> {
    let resolved = args.model_args.resolve()?;
    let spec = resolved.spec;
    eprintln!("loading {}...", args.model_args.model);
    let mut loaded = common::load(&resolved.path)?;
    eprintln!(
        "ready: {} layers, hidden={}, vocab={}. Type a message and press enter \
         (Ctrl+D or /exit to quit).",
        loaded.model.block_count(),
        loaded.model.embedding_length(),
        loaded.model.vocab_size(),
    );

    let render_opts = RenderOpts {
        add_generation_prompt: true,
        enable_thinking: if args.no_think {
            Some(false)
        } else {
            spec.map(|s| s.thinking_default)
        },
    };
    let sampling = args.sampling.to_sampling_params(spec.map(|s| &s.sampling));
    let ctx = rocml::registry::clamp_ctx(
        args.ctx
            .unwrap_or_else(|| spec.map_or(DEFAULT_CTX, |s| s.default_ctx)),
    );
    let mut messages: Vec<Message> = Vec::new();
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("\n> ");
        io::stdout().flush().ok();
        line.clear();
        if stdin.lock().read_line(&mut line).unwrap_or(0) == 0 {
            break; // EOF
        }
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if text == "/exit" || text == "/quit" {
            break;
        }

        messages.push(Message::user(text));
        let prompt = match rocml::chat::render(&messages, &[], render_opts) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("error rendering chat template: {e}");
                messages.pop();
                continue;
            }
        };
        let prompt_ids = loaded.tokenizer.encode(&prompt);
        if prompt_ids.len() + args.max_tokens > ctx {
            eprintln!(
                "error: this turn needs ~{} tokens of context, over the --ctx budget of {}",
                prompt_ids.len() + args.max_tokens,
                ctx
            );
            messages.pop();
            continue;
        }

        loaded.model.reset()?;
        // The primed generation-prompt tail already opened `<think>\n`
        // unless `--no-think` closed it, so the scanner must start
        // already-in-thinking to match (see `StreamScanner::
        // new_primed_for_thinking`'s doc comment).
        let mut scanner = if render_opts.enable_thinking == Some(false) {
            StreamScanner::new()
        } else {
            StreamScanner::new_primed_for_thinking()
        };
        let mut content = String::new();
        let mut thinking = String::new();
        let mut tool_calls = Vec::new();
        let stats = generate_sampled(
            &mut loaded.model,
            &loaded.tokenizer,
            &prompt_ids,
            args.max_tokens,
            true,
            &sampling,
            |chunk| {
                feed_scanner(
                    &mut scanner,
                    chunk,
                    &mut content,
                    &mut thinking,
                    &mut tool_calls,
                )
            },
        )?;
        for event in scanner.finish() {
            handle_event(event, &mut content, &mut thinking, &mut tool_calls);
        }
        println!();
        eprintln!(
            "[prompt {:.1} tok/s, decode {:.1} tok/s]",
            stats.prompt_tokens_per_sec(),
            stats.decode_tokens_per_sec(),
        );

        let mut assistant = Message::assistant(content).with_reasoning(thinking);
        if !tool_calls.is_empty() {
            assistant = assistant.with_tool_calls(tool_calls);
        }
        messages.push(assistant);
    }

    Ok(())
}

fn feed_scanner(
    scanner: &mut StreamScanner,
    chunk: &str,
    content: &mut String,
    thinking: &mut String,
    tool_calls: &mut Vec<ToolCall>,
) {
    match scanner.feed(chunk) {
        Ok(events) => {
            for event in events {
                handle_event(event, content, thinking, tool_calls);
            }
        }
        // A malformed <tool_call> block from the model is a generation
        // artifact, not a REPL bug — surface it and keep the turn going.
        Err(e) => eprintln!("\n[warning: {e}]"),
    }
}

fn handle_event(
    event: ScanEvent,
    content: &mut String,
    thinking: &mut String,
    tool_calls: &mut Vec<ToolCall>,
) {
    match event {
        ScanEvent::TextDelta(s) => {
            print!("{s}");
            content.push_str(&s);
        }
        ScanEvent::ThinkingDelta(s) => {
            // Dim (ANSI SGR 2) so reasoning is visually distinct from the
            // final answer without needing a full terminal-styling crate.
            print!("\x1b[2m{s}\x1b[0m");
            thinking.push_str(&s);
        }
        ScanEvent::ToolCallStarted => {
            print!("\n\x1b[1m[tool_call]\x1b[0m ");
        }
        ScanEvent::ToolCallComplete(call) => {
            println!("{}({})", call.name, call.arguments);
            tool_calls.push(call);
        }
    }
    io::stdout().flush().ok();
}

//! `rocml-cli chat`: interactive REPL over `rocml::chat`.
//!
//! Every turn still re-renders the *entire* conversation transcript through
//! the chat template (BPE re-tokenization of model-decoded text isn't
//! guaranteed to round-trip byte-for-byte forever, so there's no cheaper way
//! to get an authoritative prompt-token sequence) and calls
//! `rocml::snapshot::turn::run_turn`, which does the actual KV/GDN-state
//! reuse (issue #1): a snapshot from the previous turn is restored whenever
//! its token ids are an exact prefix of the freshly re-rendered prompt, so
//! only the new suffix is re-prefilled — see `run_turn`'s doc comment. For a
//! dense `qwen3` model (no hybrid snapshot support) this reduces to exactly
//! the old always-full-reprocess behavior.

use std::io::{self, BufRead, Write};

use clap::Args;
use rocml::chat::{Message, RenderOpts, ScanEvent, StreamScanner, ToolCall};
use rocml::snapshot::turn::{run_turn, stable_boundary_tokens};
use rocml::snapshot::KvConfigStamp;
use rocml::{LoadOptions, RocmlError};

use crate::common::{self, ModelArgs, SamplingArgs, SnapshotArgs};

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
    /// clamped to this checkpoint's estimated VRAM budget
    /// (`rocml::registry::clamp_ctx`) and is what `Model::load` actually
    /// allocates the KV cache for — not just a soft check.
    #[arg(long)]
    ctx: Option<usize>,
    #[command(flatten)]
    sampling: SamplingArgs,
    #[command(flatten)]
    snapshots: SnapshotArgs,
}

pub fn run(args: &ChatArgs) -> Result<(), RocmlError> {
    let resolved = args.model_args.resolve()?;
    let spec = resolved.spec;
    let kv_cache: rocml::KvCacheMode = args.model_args.kv_cache.into();
    let ctx = common::resolve_ctx(
        args.ctx,
        spec,
        DEFAULT_CTX,
        &resolved.path,
        kv_cache,
        args.model_args.kv_sink,
        args.model_args.kv_window,
    )?;
    eprintln!("loading {}... (ctx {ctx})", args.model_args.model);
    let mut loaded = common::load(
        &resolved.path,
        LoadOptions {
            ctx,
            kv_cache,
            use_mmq: args.model_args.mmq,
            kv_sink: args.model_args.kv_sink,
            kv_window: args.model_args.kv_window,
            kv_rot_sim: args.model_args.kv_rot_sim,
            kv_rot_sim_k: args.model_args.kv_rot_sim_k,
        },
    )?;
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
        // Issue #9: every prior turn's stored `.with_reasoning(thinking)`
        // (below) is dropped when it's re-rendered as history — only the
        // turn currently being generated ever sees its own thinking.
        keep_history_reasoning: false,
    };
    let sampling = args.sampling.to_sampling_params(spec.map(|s| &s.sampling));
    let mut messages: Vec<Message> = Vec::new();
    let stdin = io::stdin();
    let mut line = String::new();

    // Multi-turn REPL reuse (issue #1, qwen35-hybrid-only — a no-op for a
    // dense `qwen3` model since `Model::as_hybrid` is `None` there).
    let mut snapshot_store = args.snapshots.build_store()?;
    let model_stamp = common::model_stamp(&resolved.path)?;
    let kv_config = KvConfigStamp {
        mode: kv_cache,
        ctx,
    };

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
        let (prompt, boundary_byte) =
            match rocml::chat::render_with_boundary(&messages, &[], render_opts) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("error rendering chat template: {e}");
                    messages.pop();
                    continue;
                }
            };
        let prompt_ids = loaded.tokenizer.encode(&prompt);
        // Issue #12: lets a thinking-enabled turn's mid-prefill snapshot
        // survive into the next turn even though the reply/think block this
        // turn produces won't be reproduced verbatim by that next render —
        // see `run_turn`'s `stable_boundary` doc comment.
        let stable_boundary =
            stable_boundary_tokens(&loaded.tokenizer, &prompt, boundary_byte, &prompt_ids);
        if prompt_ids.len() + args.max_tokens > ctx {
            eprintln!(
                "error: this turn needs ~{} tokens of context, over the --ctx budget of {}",
                prompt_ids.len() + args.max_tokens,
                ctx
            );
            messages.pop();
            continue;
        }

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
        let outcome = run_turn(
            &mut loaded.model,
            &loaded.tokenizer,
            snapshot_store.as_mut(),
            &model_stamp,
            &kv_config,
            &prompt_ids,
            args.max_tokens,
            true,
            &sampling,
            &[], // no OpenAI stop strings in the REPL
            stable_boundary,
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
        let stats = outcome.stats;
        for event in scanner.finish() {
            handle_event(event, &mut content, &mut thinking, &mut tool_calls);
        }
        println!();
        if outcome.reused_prefix > 0 {
            eprintln!(
                "[snapshot hit: reused {} of {} prompt tokens, restore {:.1}ms, capture {:.1}ms]",
                outcome.reused_prefix,
                prompt_ids.len(),
                outcome.restore_seconds * 1000.0,
                outcome.capture_seconds * 1000.0,
            );
        } else {
            eprintln!(
                "[snapshot miss, capture {:.1}ms]",
                outcome.capture_seconds * 1000.0
            );
        }
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

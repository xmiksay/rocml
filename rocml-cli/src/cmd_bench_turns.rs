//! `rocml-cli bench --turns N`: issue #1's acceptance benchmark. Simulates a
//! growing multi-turn conversation (each turn appends a fixed synthetic user
//! message, generates `--turn-decode-tokens`, then appends that output too)
//! and reports per-turn prefill latency — run once normally and once with
//! `--no-snapshots` to see the snapshot layer's effect on later turns'
//! prefill cost.

use rocml::snapshot::turn::run_turn;
use rocml::snapshot::KvConfigStamp;
use rocml::{LoadOptions, RocmlError, SamplingParams};
use serde_json::json;

use crate::cmd_bench::BenchArgs;
use crate::common;

/// Fixed per-turn synthetic "user message" length — deliberately small next
/// to `--turn-decode-tokens` (300 default), matching the issue's own
/// "~+300 tokens" per-turn conversation-growth description.
const FIXED_USER_MESSAGE_TOKENS: usize = 64;

pub fn run(args: &BenchArgs, turns: usize) -> Result<(), RocmlError> {
    if turns == 0 {
        return Err(RocmlError::Config("--turns must be at least 1".to_string()));
    }
    let resolved = args.model_args.resolve()?;
    let kv_cache: rocml::KvCacheMode = args.model_args.kv_cache.into();
    let needed = turns * (FIXED_USER_MESSAGE_TOKENS + args.turn_decode_tokens) + 512;
    let ctx_request = args.ctx.unwrap_or(needed).max(needed).max(1);
    let ctx = rocml::registry::clamp_ctx(ctx_request, &resolved.path, kv_cache)?;
    eprintln!("loading {}... (ctx {ctx})", args.model_args.model);
    let mut loaded = common::load(&resolved.path, LoadOptions { ctx, kv_cache })?;

    let model_stamp = common::model_stamp(&resolved.path)?;
    let kv_config = KvConfigStamp {
        mode: kv_cache,
        ctx,
    };
    let mut store = if args.no_snapshots {
        None
    } else {
        args.snapshots.build_store()?
    };

    let fixed_user_message = synthetic_tokens(&loaded, FIXED_USER_MESSAGE_TOKENS);
    let sampling = SamplingParams::greedy();
    let mut conversation: Vec<u32> = Vec::new();
    let mut rows = Vec::new();

    eprintln!(
        "simulating {turns} turns ({} decode tokens/turn, snapshots {})",
        args.turn_decode_tokens,
        if args.no_snapshots { "off" } else { "on" },
    );

    for turn in 1..=turns {
        conversation.extend_from_slice(&fixed_user_message);
        // `stop_on_eos: false` forces exactly `turn_decode_tokens` tokens
        // every turn, same reasoning as the plain throughput mode: every
        // turn should measure the same amount of decode work regardless of
        // what the (synthetic, meaningless) filler happens to continue with.
        let outcome = run_turn(
            &mut loaded.model,
            &loaded.tokenizer,
            store.as_mut(),
            &model_stamp,
            &kv_config,
            &conversation,
            args.turn_decode_tokens,
            false,
            &sampling,
            &[],
            |_| {},
        )?;
        conversation.extend_from_slice(&outcome.stats.generated_ids);
        eprintln!(
            "  turn {turn}/{turns}: conv_len {}, reused {} tok, prefill {} tok in {:.1}ms \
             ({:.1} tok/s), decode {:.1} tok/s, restore {:.2}ms, capture {:.2}ms",
            conversation.len(),
            outcome.reused_prefix,
            outcome.stats.prompt_tokens,
            outcome.stats.prompt_seconds * 1000.0,
            outcome.stats.prompt_tokens_per_sec(),
            outcome.stats.decode_tokens_per_sec(),
            outcome.restore_seconds * 1000.0,
            outcome.capture_seconds * 1000.0,
        );
        rows.push(json!({
            "turn": turn,
            "conversation_tokens": conversation.len(),
            "reused_prefix_tokens": outcome.reused_prefix,
            "prefill_tokens": outcome.stats.prompt_tokens,
            "prefill_ms": outcome.stats.prompt_seconds * 1000.0,
            "prefill_tok_s": outcome.stats.prompt_tokens_per_sec(),
            "decode_tok_s": outcome.stats.decode_tokens_per_sec(),
            "restore_ms": outcome.restore_seconds * 1000.0,
            "capture_ms": outcome.capture_seconds * 1000.0,
        }));
    }

    let (ram_used_mb, disk_used_mb) = store
        .as_ref()
        .map(|s| {
            (
                s.ram_used_bytes() as f64 / (1024.0 * 1024.0),
                s.disk_used_bytes() as f64 / (1024.0 * 1024.0),
            )
        })
        .unwrap_or((0.0, 0.0));
    eprintln!("snapshot store usage: {ram_used_mb:.1} MiB RAM, {disk_used_mb:.1} MiB disk");

    if args.json {
        let out = json!({
            "model": common::model_id(&resolved),
            "turns": turns,
            "snapshots_enabled": !args.no_snapshots,
            "ram_used_mb": ram_used_mb,
            "disk_used_mb": disk_used_mb,
            "rows": rows,
        });
        println!("{out}");
    }

    Ok(())
}

/// A fixed, deterministic synthetic "user message" of exactly `target_len`
/// tokens — same technique as `cmd_bench::synthetic_prompt`, duplicated
/// rather than shared since that one is private to this crate's other
/// module and the two have slightly different growth semantics (this one is
/// appended repeatedly across turns, not generated once).
fn synthetic_tokens(loaded: &common::Loaded, target_len: usize) -> Vec<u32> {
    const FILLER: &str = "Please review the update above and suggest the next concrete step. ";
    let mut text = String::with_capacity(FILLER.len() * (target_len / 4 + 4));
    let mut ids = Vec::new();
    while ids.len() < target_len {
        text.push_str(FILLER);
        ids = loaded.tokenizer.encode(&text);
    }
    ids.truncate(target_len.max(1));
    ids
}

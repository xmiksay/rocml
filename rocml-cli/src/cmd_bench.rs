//! `rocml-cli bench`: synthetic prompt/decode throughput harness — the
//! llama.cpp-comparison baseline. Always greedy (`SamplingParams::greedy`):
//! this measures raw forward-pass throughput, not sampling behavior, and
//! greedy keeps every run's token count identical for a clean median.

use clap::Args;
use rocml::generate::generate_sampled_profiled;
use rocml::{LoadOptions, Profiler, RocmlError, SamplingParams};
use serde_json::json;

use crate::common::{self, ModelArgs};

#[derive(Args, Debug)]
pub struct BenchArgs {
    #[command(flatten)]
    model_args: ModelArgs,
    #[arg(long = "prompt-tokens", default_value_t = 64)]
    prompt_tokens: usize,
    #[arg(long = "decode-tokens", default_value_t = 128)]
    decode_tokens: usize,
    #[arg(long, default_value_t = 3)]
    runs: usize,
    /// Pre-fill this many synthetic tokens of existing context (through the
    /// normal prefill path) before measuring decode throughput, instead of
    /// `--prompt-tokens` — the head-to-head metric vs llama.cpp at a given
    /// context depth (e.g. `--depth 2048`). Both the prefill-to-depth and
    /// the subsequent decode rate are reported.
    #[arg(long)]
    depth: Option<usize>,
    /// KV cache context length to allocate. Unset defaults to whatever
    /// `--depth`/`--decode-tokens` needs (so `bench --depth 16384` just
    /// works without also specifying `--ctx`), clamped to this
    /// checkpoint's estimated VRAM budget (issue #3) same as every other
    /// command.
    #[arg(long)]
    ctx: Option<usize>,
    #[arg(long)]
    json: bool,
    /// Collect per-op/per-layer roofline instrumentation and print a report
    /// after the run (see `rocml::profile`). Only the final run is
    /// profiled — the others still contribute to the tok/s median, but
    /// profiling every run would mix `--runs` independent sessions' spans
    /// into one report for no benefit.
    #[arg(long)]
    profile: bool,
}

pub fn run(args: &BenchArgs) -> Result<(), RocmlError> {
    if args.runs == 0 {
        return Err(RocmlError::Config("--runs must be at least 1".to_string()));
    }
    let resolved = args.model_args.resolve()?;
    let kv_cache: rocml::KvCacheMode = args.model_args.kv_cache.into();
    // Bench's own ctx default (unlike chat/generate's registry-preset
    // fallback): big enough for whatever --depth/--decode-tokens asks for,
    // since a throughput probe shouldn't require a separate --ctx just to
    // reach the depth it was asked to measure at.
    let needed = args.depth.unwrap_or(0) + args.decode_tokens;
    let ctx_request = args.ctx.unwrap_or(needed).max(needed).max(1);
    let ctx = rocml::registry::clamp_ctx(ctx_request, &resolved.path, kv_cache.dense_dtype())?;
    eprintln!("loading {}... (ctx {ctx})", args.model_args.model);
    let mut loaded = common::load(&resolved.path, LoadOptions { ctx, kv_cache })?;
    let prompt_len = args.depth.unwrap_or(args.prompt_tokens);
    let prompt_ids = synthetic_prompt(&loaded, prompt_len);
    eprintln!(
        "loaded: {} layers; benchmarking {} prompt tokens / {} decode tokens x {} run(s){}",
        loaded.model.block_count(),
        prompt_ids.len(),
        args.decode_tokens,
        args.runs,
        args.depth
            .map(|d| format!(" (decode measured at depth {d})"))
            .unwrap_or_default(),
    );

    let mut prompt_tps = Vec::with_capacity(args.runs);
    let mut decode_tps = Vec::with_capacity(args.runs);
    let sampling = SamplingParams::greedy();
    let mut report = None;
    for run_idx in 0..args.runs {
        loaded.model.reset()?;
        let profiler = (args.profile && run_idx + 1 == args.runs).then(Profiler::new);
        // `stop_on_eos: false` forces exactly `decode_tokens` tokens every
        // run regardless of what the (synthetic, meaningless) filler prompt
        // happens to continue with, so every run measures the same amount
        // of decode work.
        let stats = generate_sampled_profiled(
            &mut loaded.model,
            &loaded.tokenizer,
            &prompt_ids,
            args.decode_tokens,
            false,
            &sampling,
            profiler.as_ref(),
            |_| {},
        )?;
        if let Some(p) = &profiler {
            report = Some(p.finish()?);
        }
        eprintln!(
            "  run {}/{}: prompt {:.1} tok/s, decode {:.1} tok/s",
            run_idx + 1,
            args.runs,
            stats.prompt_tokens_per_sec(),
            stats.decode_tokens_per_sec(),
        );
        prompt_tps.push(stats.prompt_tokens_per_sec());
        decode_tps.push(stats.decode_tokens_per_sec());
    }

    let prompt_median = median(&mut prompt_tps);
    let decode_median = median(&mut decode_tps);

    if args.json {
        let mut out = json!({
            "model": common::model_id(&resolved),
            "prompt_tokens": prompt_ids.len(),
            "decode_tokens": args.decode_tokens,
            "depth": args.depth,
            "runs": args.runs,
            "prompt_tokens_per_sec_median": prompt_median,
            "decode_tokens_per_sec_median": decode_median,
        });
        if let Some(report) = &report {
            out["profile"] =
                serde_json::to_value(report).unwrap_or_else(|e| json!({"error": e.to_string()}));
        }
        println!("{out}");
    } else {
        let depth_note = args
            .depth
            .map(|d| format!(" @ depth {d}"))
            .unwrap_or_default();
        println!(
            "prompt: {:.1} tok/s (median of {}); decode{depth_note}: {:.1} tok/s (median of {})",
            prompt_median, args.runs, decode_median, args.runs
        );
        if let Some(report) = &report {
            println!("{}", report.to_human());
        }
    }

    Ok(())
}

/// A prompt of exactly `target_len` tokens built from a repeated filler
/// phrase (content is irrelevant to a throughput benchmark) fed raw, i.e.
/// without the chat template, so the token count isn't skewed by template
/// overhead.
fn synthetic_prompt(loaded: &common::Loaded, target_len: usize) -> Vec<u32> {
    const FILLER: &str = "The quick brown fox jumps over the lazy dog. ";
    let mut text = String::with_capacity(FILLER.len() * (target_len / 4 + 4));
    let mut ids = Vec::new();
    while ids.len() < target_len {
        text.push_str(FILLER);
        ids = loaded.tokenizer.encode(&text);
    }
    ids.truncate(target_len.max(1));
    ids
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

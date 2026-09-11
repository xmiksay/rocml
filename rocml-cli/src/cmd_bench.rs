//! `rocml-cli bench`: synthetic prompt/decode throughput harness — the
//! llama.cpp-comparison baseline. Always greedy (`SamplingParams::greedy`):
//! this measures raw forward-pass throughput, not sampling behavior, and
//! greedy keeps every run's token count identical for a clean median.

use clap::Args;
use rocml::generate::generate_sampled;
use rocml::{RocmlError, SamplingParams};
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
    #[arg(long)]
    json: bool,
}

pub fn run(args: &BenchArgs) -> Result<(), RocmlError> {
    if args.runs == 0 {
        return Err(RocmlError::Config("--runs must be at least 1".to_string()));
    }
    let resolved = args.model_args.resolve()?;
    eprintln!("loading {}...", args.model_args.model);
    let mut loaded = common::load(&resolved.path)?;
    let prompt_ids = synthetic_prompt(&loaded, args.prompt_tokens);
    eprintln!(
        "loaded: {} layers; benchmarking {} prompt tokens / {} decode tokens x {} run(s)",
        loaded.model.block_count(),
        prompt_ids.len(),
        args.decode_tokens,
        args.runs,
    );

    let mut prompt_tps = Vec::with_capacity(args.runs);
    let mut decode_tps = Vec::with_capacity(args.runs);
    let sampling = SamplingParams::greedy();
    for run_idx in 0..args.runs {
        loaded.model.reset()?;
        // `stop_on_eos: false` forces exactly `decode_tokens` tokens every
        // run regardless of what the (synthetic, meaningless) filler prompt
        // happens to continue with, so every run measures the same amount
        // of decode work.
        let stats = generate_sampled(
            &mut loaded.model,
            &loaded.tokenizer,
            &prompt_ids,
            args.decode_tokens,
            false,
            &sampling,
            |_| {},
        )?;
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
        println!(
            "{}",
            json!({
                "model": common::model_id(&resolved),
                "prompt_tokens": prompt_ids.len(),
                "decode_tokens": args.decode_tokens,
                "runs": args.runs,
                "prompt_tokens_per_sec_median": prompt_median,
                "decode_tokens_per_sec_median": decode_median,
            })
        );
    } else {
        println!(
            "prompt: {:.1} tok/s (median of {}); decode: {:.1} tok/s (median of {})",
            prompt_median, args.runs, decode_median, args.runs
        );
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

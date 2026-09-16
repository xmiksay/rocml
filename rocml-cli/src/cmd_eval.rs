//! `rocml-cli eval`: issue #15's agentic quality-eval harness. Loads a model
//! the same way `chat`/`generate` do, runs a fixed scenario set
//! (`eval::scenario`) through greedy decode + the real chat protocol
//! (`eval::runner`), scores each reply (`eval::scorer`), and adds a
//! teacher-forced perplexity secondary signal (`eval::ppl`) over a small
//! checked-in corpus. Writes `--out` after every scenario so a run
//! interrupted partway through (e.g. a Bash timeout wrapping a 30-60 minute
//! eval) can pick back up with `--resume` instead of restarting.

use std::path::PathBuf;

use clap::Args;
use rocml::snapshot::KvConfigStamp;
use rocml::{LoadOptions, RocmlError};

use crate::common::{self, ModelArgs};
use crate::eval::ppl::compute_ppl;
use crate::eval::results::{EngineConfig, EvalResults, SamplingSummary};
use crate::eval::runner::run_scenario;
use crate::eval::scenario::load_scenarios;

/// Long-context scenarios need ~8K tokens of filler plus thinking/answer
/// headroom; this is also this command's fallback when neither `--ctx` nor
/// a resolved registry preset applies (a path-based `--model`).
const DEFAULT_CTX: usize = 16384;

#[derive(Args, Debug)]
pub struct EvalArgs {
    #[command(flatten)]
    model_args: ModelArgs,
    /// Short name identifying this run in the results JSON (e.g.
    /// `ornith-q6k-fp16kv`).
    #[arg(long)]
    label: String,
    /// Where to write results — also read back on `--resume`.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "bench/eval/scenarios.json")]
    scenarios: PathBuf,
    #[arg(long, default_value = "bench/eval/corpus.txt")]
    corpus: PathBuf,
    /// Context length; unset falls back to the resolved model's registry
    /// preset, then [`DEFAULT_CTX`] — either way clamped to this
    /// checkpoint's estimated VRAM budget (`rocml::registry::clamp_ctx`).
    #[arg(long)]
    ctx: Option<usize>,
    /// Per-turn generation cap.
    #[arg(long = "max-gen-tokens", default_value_t = 2048)]
    max_gen_tokens: usize,
    /// Skip scenarios already present in `--out` (and skip PPL if it's
    /// already recorded there) instead of starting over.
    #[arg(long)]
    resume: bool,
}

pub fn run(args: &EvalArgs) -> Result<(), RocmlError> {
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

    let scenarios = load_scenarios(&args.scenarios)?;
    eprintln!(
        "loaded {} scenarios from {}",
        scenarios.len(),
        args.scenarios.display()
    );

    let resumed = if args.resume {
        EvalResults::load_if_exists(&args.out)?
    } else {
        None
    };
    let already_scored = resumed
        .as_ref()
        .map(EvalResults::already_scored_ids)
        .unwrap_or_default();
    if !already_scored.is_empty() {
        eprintln!(
            "--resume: {} scenario(s) already scored in {}, skipping",
            already_scored.len(),
            args.out.display()
        );
    }

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
            moe_cache_slots: args.model_args.moe_cache_slots,
            moe_decode_overlap: args.model_args.moe_decode_overlap,
        },
    )?;
    eprintln!(
        "ready: {} layers, hidden={}, vocab={}",
        loaded.model.block_count(),
        loaded.model.embedding_length(),
        loaded.model.vocab_size(),
    );

    let engine_config = EngineConfig {
        ctx,
        kv_cache: kv_cache.as_flag_str().to_string(),
        max_gen_tokens: args.max_gen_tokens,
        // Determinism (issue #15): greedy argmax regardless of this
        // checkpoint's registry sampling preset — recorded here rather than
        // read from `SamplingParams::greedy()` so the JSON is explicit about
        // what ran even if that default ever changes.
        sampling: SamplingSummary {
            temperature: 0.0,
            top_k: None,
            top_p: None,
            seed: 0,
        },
    };
    let mut results = resumed.unwrap_or_else(|| {
        EvalResults::new(
            args.label.clone(),
            resolved.path.to_string_lossy().into_owned(),
            engine_config.clone(),
        )
    });

    let model_stamp = common::model_stamp(&resolved.path)?;
    let kv_config = KvConfigStamp {
        mode: kv_cache,
        ctx,
    };

    let total = scenarios.len();
    for (index, scenario) in scenarios.iter().enumerate() {
        if already_scored.contains(scenario.id()) {
            continue;
        }
        eprint!(
            "[{}/{total}] {} ({})... ",
            index + 1,
            scenario.id(),
            scenario.kind()
        );
        let result = run_scenario(
            &mut loaded,
            &model_stamp,
            &kv_config,
            args.max_gen_tokens,
            scenario,
        )?;
        eprintln!(
            "{} ({:.1}s){}",
            if result.pass { "PASS" } else { "FAIL" },
            result.wall_seconds,
            if result.pass {
                String::new()
            } else {
                format!(" — {}", result.detail)
            },
        );
        results.push_scenario_result(result);
        results.save(&args.out)?;
    }

    if results.ppl.is_none() {
        eprintln!("computing PPL over {}...", args.corpus.display());
        let (ppl, wall_seconds) = compute_ppl(&mut loaded, &args.corpus)?;
        eprintln!("ppl = {ppl:.4} ({wall_seconds:.1}s)");
        results.ppl = Some(ppl);
        results.ppl_wall_seconds = Some(wall_seconds);
        results.save(&args.out)?;
    }

    print_summary(&results);
    Ok(())
}

fn print_summary(results: &EvalResults) {
    eprintln!("\n=== {} ===", results.label);
    eprintln!(
        "overall: {}/{} ({:.1}%)",
        results.aggregates.overall_passed,
        results.aggregates.overall_total,
        results.aggregates.overall_fraction * 100.0,
    );
    for (kind, agg) in &results.aggregates.by_kind {
        eprintln!(
            "  {kind:<12} {}/{} ({:.1}%)",
            agg.passed,
            agg.total,
            agg.fraction * 100.0,
        );
    }
    if let Some(ppl) = results.ppl {
        eprintln!("ppl: {ppl:.4}");
    }
}

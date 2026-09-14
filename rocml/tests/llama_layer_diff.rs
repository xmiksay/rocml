//! Issue #10's llama.cpp-reference per-layer diff: runs Ornith-1.0-9B's
//! real chunked-prefill path with `LayerCapture`, ingests a llama.cpp CPU
//! reference dump (produced out-of-band by the scratchpad `rocml-dump`
//! tool — see `docs/llama-diff.md` for the exact build/run recipe and the
//! commit it was built against), converts it via
//! `support::llama_ref::convert`, and reports per-layer/per-tensor
//! max/mean relative error against rocml's own capture — the localization
//! step this issue asked for, now pointed at an independent reference
//! implementation instead of rocml's own MMQ-vs-WMMA comparison
//! (`mmq_layer_diff.rs`).
//!
//! `#[ignore]`d diagnostic tool, not a correctness gate: llama.cpp-CPU and
//! rocml-GPU inherently differ by reduction order/precision (see
//! `docs/llama-diff.md`'s recorded noise floor), so there is no tight
//! tolerance to assert here, only a report to compare a future run
//! against. Run via `make llama-layer-diff`. Requires the real
//! Ornith-1.0-9B-Q6_K checkpoint and a pre-generated `LLAMA_DUMP` file;
//! skips itself if either is missing.

mod support;

use rocml::qwen35::forward::layer_capture::{diff_dumps, LayerCapture};
use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;
use support::llama_ref::convert::convert;
use support::llama_ref::parse::parse;

/// The pinned default (Ornith-1.0-9B, the flagship checkpoint) — override
/// with `LLAMA_DIFF_GGUF` (a `testpaths::checkpoint`-relative path) to run
/// the same harness against a different qwen35-arch checkpoint, e.g. when
/// VRAM headroom for the 9B model isn't available (see docs/llama-diff.md).
const DEFAULT_GGUF_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf";
const CORPUS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/eval/corpus.txt");
/// The pinned reference prompt: the first `PROMPT_LEN` tokens of
/// `bench/eval/corpus.txt` (the same corpus `mmq-layer-diff` uses), kept
/// well under `mmq-layer-diff`'s 128 (never mind `qwen35_chunked_prefill_
/// parity`'s multi-thousand-token gate) since a llama.cpp *CPU* forward
/// pass on a 9B model, and the resulting full-precision text dump, both
/// scale with it directly — see `docs/llama-diff.md` for the measured
/// wall-clock/dump-size cost this was picked against.
const PROMPT_LEN: usize = 64;

fn env_path(var: &str) -> Option<std::path::PathBuf> {
    std::env::var_os(var)
        .map(std::path::PathBuf::from)
        .filter(|p| p.exists())
}

#[test]
#[ignore]
fn ornith_9b_layer_diff_against_llama_cpp_cpu_reference() {
    let gguf_rel =
        std::env::var("LLAMA_DIFF_GGUF").unwrap_or_else(|_| DEFAULT_GGUF_REL.to_string());
    let Some(gguf_path) = checkpoint(&gguf_rel) else {
        eprintln!(
            "skipping ornith_9b_layer_diff_against_llama_cpp_cpu_reference: {gguf_rel} not found"
        );
        return;
    };
    let Some(dump_path) = env_path("LLAMA_DUMP") else {
        eprintln!(
            "skipping ornith_9b_layer_diff_against_llama_cpp_cpu_reference: \
             set LLAMA_DUMP to a rocml-dump reference file (see docs/llama-diff.md)"
        );
        return;
    };

    let corpus = std::fs::read_to_string(CORPUS_PATH).expect("failed to read eval corpus");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&gguf_path).expect("gguf open failed"))
        .expect("tokenizer load failed");
    let mut prompt_ids = tokenizer.encode(&corpus);
    prompt_ids.truncate(PROMPT_LEN);
    assert!(
        prompt_ids.len() == PROMPT_LEN,
        "eval corpus too short to tokenize to {PROMPT_LEN} tokens (got {})",
        prompt_ids.len()
    );

    let opts = LoadOptions::new(4096).with_kv_cache(KvCacheMode::F32);
    let mut model = Model::load(&gguf_path, opts).expect("Model::load failed");
    let hybrid = model
        .as_hybrid_mut()
        .expect("Ornith-1.0-9B must be the qwen35 hybrid architecture");
    let layer_kinds = hybrid.config().layer_kinds.clone();

    let mut capture = LayerCapture::new();
    hybrid
        .forward_prompt_chunked_captured(&prompt_ids, None, &mut capture)
        .expect("forward_prompt_chunked_captured failed");

    let llama_text = std::fs::read_to_string(&dump_path).expect("read LLAMA_DUMP");
    let raw = parse(&llama_text).expect("parse LLAMA_DUMP");
    let (llama_dump, report) = convert(&raw, &layer_kinds);

    if !report.unmapped_llama_nodes.is_empty() {
        eprintln!(
            "unmapped llama.cpp nodes (expected, not errors): {:?}",
            report.unmapped_llama_nodes
        );
    }
    if !report.missing_rocml_keys.is_empty() {
        eprintln!(
            "missing rocml keys (LLAMA_DUMP too narrow for {}): {:?}",
            gguf_rel, report.missing_rocml_keys
        );
    }

    let diffs = diff_dumps(&llama_dump, capture.dump());
    assert!(
        !diffs.is_empty(),
        "no comparable tensors between the llama.cpp dump and rocml's capture \
         (check ROCML_DUMP_FILTER covered every mapped node name)"
    );

    eprintln!(
        "\n{:>5}  {:<16}  {:>10}  {:>10}  {:>12}  {:>12}",
        "layer", "tensor", "max_rel", "mean_rel", "max_abs", "mean_abs"
    );
    for d in &diffs {
        eprintln!(
            "{:>5}  {:<16}  {:>10.6}  {:>10.6}  {:>12.6}  {:>12.6}",
            d.layer_idx, d.tensor, d.max_rel, d.mean_rel, d.max_abs, d.mean_abs
        );
    }
    let worst = &diffs[0];
    eprintln!(
        "\nworst offender: layer {} ({}), max_rel={:.4}",
        worst.layer_idx, worst.tensor, worst.max_rel
    );
}

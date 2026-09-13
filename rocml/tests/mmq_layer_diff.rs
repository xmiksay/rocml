//! Issue #10's per-layer diff harness, built while root-causing the int8
//! MMQ precision regression (`.claude/CLAUDE.md`'s MMQ section; the
//! `mmq_precision` investigation this test was written for): runs the same
//! real-text prompt through Qwen3.5-2B's chunked-prefill path twice — MMQ
//! off (f16 WMMA, the shipped default) vs MMQ on (`--mmq`, off by default
//! per that section's validation-ladder failure) — capturing every layer's
//! post-residual-add tensor *and* the intermediates on either side of each
//! MMQ-eligible matmul for the final chunk via
//! `qwen35::forward::layer_capture::LayerCapture`, then reports the
//! max/mean relative error between the two runs for every captured tensor,
//! sorted worst first. This is the localization step: does error grow
//! smoothly with depth (compounding reduction-order/precision noise), or
//! spike at a specific matmul (a real per-op defect or a genuinely
//! outlier-heavy activation)? This round's finding: GDN's `ssm_out`
//! projection (fed by the SiLU-gated `gdn_y_silu` activation) shows by far
//! the largest single-matmul jump in relative error at every layer — see
//! `weights/linear.rs`'s `mmq_eligible_by_name` for the real-data outlier
//! evidence and the resulting fix.
//!
//! `#[ignore]`d diagnostic tool, not a correctness gate (mirrors
//! `gdn_conv_microbench.rs`'s pattern) — run explicitly via `make
//! mmq-layer-diff`. Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint
//! required; skips itself if absent. `--release` recommended (debug-mode
//! chunked prefill over 128 tokens is slow but not unbearable, unlike the
//! multi-thousand-token parity suites).

use rocml::qwen35::forward::layer_capture::{diff_dumps, LayerCapture};
use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const CORPUS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/eval/corpus.txt");
/// Single chunk, `>= GEMM_WMMA_TILE_ROWS`(128) so every eligible projection
/// actually engages the WMMA/MMQ path rather than falling back to scalar —
/// matches the smallest failing length in the round's own report
/// (`{128,129,500,2048}`).
const PROMPT_LEN: usize = 128;

/// Runs the real prompt through the chunked-prefill path once, with MMQ
/// forced to `use_mmq`, capturing every layer's final-chunk residual
/// stream. `LoadOptions::with_kv_cache(F32)` pins the pre-issue-#3
/// reference KV numerics (same choice `qwen35_chunked_prefill_parity.rs`
/// makes) so this diff isolates the MMQ GEMM path's own error, not
/// unrelated fp16-KV rounding.
fn run_captured(gguf_path: &std::path::Path, prompt_ids: &[u32], use_mmq: bool) -> LayerCapture {
    let opts = LoadOptions::new(4096)
        .with_kv_cache(KvCacheMode::F32)
        .with_mmq(use_mmq);
    let mut model = Model::load(gguf_path, opts).expect("Model::load failed");
    let hybrid = model
        .as_hybrid_mut()
        .expect("Qwen3.5-2B must be the qwen35 hybrid architecture");
    let mut capture = LayerCapture::new();
    hybrid
        .forward_prompt_chunked_captured(prompt_ids, None, &mut capture)
        .expect("forward_prompt_chunked_captured failed");
    capture
}

#[test]
#[ignore]
fn mmq_vs_wmma_per_layer_diff() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping mmq_vs_wmma_per_layer_diff: {GGUF_REL} not found");
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

    eprintln!("running MMQ-off (f16 WMMA) reference pass...");
    let off = run_captured(&gguf_path, &prompt_ids, false);
    eprintln!("running MMQ-on pass...");
    let on = run_captured(&gguf_path, &prompt_ids, true);

    let out_dir = std::env::temp_dir().join("rocml_mmq_layer_diff");
    std::fs::create_dir_all(&out_dir).expect("create output dir");
    off.write_json(out_dir.join("mmq_off.json"))
        .expect("write mmq_off.json");
    on.write_json(out_dir.join("mmq_on.json"))
        .expect("write mmq_on.json");
    eprintln!("raw dumps written under {}", out_dir.display());

    let diffs = diff_dumps(off.dump(), on.dump());
    assert!(!diffs.is_empty(), "no comparable tensors captured");

    eprintln!(
        "\n{:>5}  {:<12}  {:>10}  {:>10}  {:>12}  {:>12}",
        "layer", "tensor", "max_rel", "mean_rel", "max_abs", "mean_abs"
    );
    for d in &diffs {
        eprintln!(
            "{:>5}  {:<12}  {:>10.6}  {:>10.6}  {:>12.6}  {:>12.6}",
            d.layer_idx, d.tensor, d.max_rel, d.mean_rel, d.max_abs, d.mean_abs
        );
    }
    let worst = &diffs[0];
    eprintln!(
        "\nworst offender: layer {} ({}), max_rel={:.4}",
        worst.layer_idx, worst.tensor, worst.max_rel
    );
}

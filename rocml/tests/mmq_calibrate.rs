//! Issue #17's calibration harness: captures per-input-channel amax for
//! every MMQ-eligible matmul input `LayerCapture` already knows how to
//! record (`gdn_xn`, `gdn_y_silu`, `ffn_xn`, `ffn_gate_silu`) over a
//! real-text calibration run, and persists the result as a JSON sidecar
//! (`CalibrationDump`) keyed by `"{layer_idx}:{tensor}"`.
//!
//! Runs on the actual flagship checkpoint (Ornith-1.0-9B-Q4_K_M), not the
//! smaller Qwen3.5-2B the earlier `mmq_precision` root-cause round used —
//! calibration must come from the same model/shapes the weight measurement
//! uses, since a per-channel scale only makes sense against the matching
//! input dimension. Confirmed by inspection (`rocml_core::gguf::GgufFile`,
//! not guessed): in this exact GGUF, `ssm_out.weight` is Q4_K on every GDN
//! layer, `ffn_down.weight` is a per-layer mix of Q4_K and Q6_K (llama.cpp's
//! own `Q4_K_M` importance-based upgrade heuristic) — the measurement test
//! below picks a Q4_K-quantized layer for both tensors so a single fold/
//! requant implementation covers both.
//!
//! Also writes a raw-values sidecar (real per-token activations, not just
//! their per-channel amax) for the two specific layer/tensor pairs the
//! standalone measurement test (`mmq_smoothquant_measure.rs`) examines —
//! a calibration amax summary alone can't reconstruct per-block amax/mean
//! *after* dividing by `s`, so the before/after flatness check needs actual
//! rows. Deliberately scoped to just those two keys rather than the full
//! `LayerDump` (every tensor, every layer): the full dump is tens of GB of
//! JSON at 500 real tokens x up to 12288 columns x dozens of layers, and
//! this round's own measurement only ever looks at one representative
//! layer per tensor anyway (see [`RAW_DUMP_KEYS`]'s doc).
//!
//! `#[ignore]`d diagnostic tool, not a correctness gate (mirrors
//! `mmq_layer_diff.rs`'s pattern) — run explicitly via `make mmq-calibrate`.
//! Real hardware + the real Ornith-1.0-9B-Q4_K_M checkpoint required; skips
//! itself if absent.

mod support;

use rocml::qwen35::forward::layer_capture::{CalibrationDump, LayerCapture, LayerDump};
use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;
use support::smoothquant::paths::{calibration_dir, calibration_path, raw_dump_path};
use support::smoothquant::stats::{percentile, pooled_block_outlier_ratios};

const GGUF_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf";
const CORPUS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/eval/corpus.txt");
/// A single `PREFILL_CHUNK_SIZE`(512)-token chunk of real text — "a few
/// hundred tokens", matching the issue's own calibration-size guidance, and
/// small enough that `LayerCapture`'s final-chunk-only recording captures
/// every one of these tokens (not just a tail).
const CALIB_TOKENS: usize = 500;

/// The two layer/tensor pairs `mmq_smoothquant_measure.rs` examines: layer
/// 0 (a GDN layer; `ssm_out.weight` there is Q4_K) for the worst offender's
/// input, layer 4 (`ffn_down.weight` there is Q4_K, per this file's module
/// doc) for the moderate offender's input — both Q4_K so a single fold/
/// requant implementation covers both weights.
const RAW_DUMP_KEYS: &[&str] = &["0:gdn_y_silu", "4:ffn_gate_silu"];

#[test]
#[ignore]
fn calibrate_channel_amax() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping calibrate_channel_amax: {GGUF_REL} not found");
        return;
    };

    let corpus = std::fs::read_to_string(CORPUS_PATH).expect("failed to read eval corpus");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&gguf_path).expect("gguf open failed"))
        .expect("tokenizer load failed");
    let mut prompt_ids = tokenizer.encode(&corpus);
    prompt_ids.truncate(CALIB_TOKENS);
    assert!(
        prompt_ids.len() == CALIB_TOKENS,
        "eval corpus too short to tokenize to {CALIB_TOKENS} tokens (got {})",
        prompt_ids.len()
    );

    let opts = LoadOptions::new(4096).with_kv_cache(KvCacheMode::F32);
    let mut model = Model::load(&gguf_path, opts).expect("Model::load failed");
    let hybrid = model
        .as_hybrid_mut()
        .expect("Ornith-1.0-9B must be the qwen35 hybrid architecture");

    let mut capture = LayerCapture::new();
    hybrid
        .forward_prompt_chunked_captured(&prompt_ids, None, &mut capture)
        .expect("forward_prompt_chunked_captured failed");

    let calib = CalibrationDump::from_layer_dump(capture.dump());
    std::fs::create_dir_all(calibration_dir()).expect("create calibration output dir");
    calib
        .write_json(calibration_path())
        .expect("write calibration json");
    let raw_subset = LayerDump {
        tensors: capture
            .dump()
            .tensors
            .iter()
            .filter(|(k, _)| RAW_DUMP_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    };
    assert_eq!(
        raw_subset.tensors.len(),
        RAW_DUMP_KEYS.len(),
        "expected every key in RAW_DUMP_KEYS to have been captured"
    );
    raw_subset
        .write_json(raw_dump_path())
        .expect("write raw dump json");
    eprintln!(
        "wrote calibration for {} tensors to {}, raw dump to {}",
        calib.channels.len(),
        calibration_path().display(),
        raw_dump_path().display()
    );

    // Sanity summary: p50/p90 of the per-block (32-wide) amax/mean(|x|)
    // ratio for the two tensors this round cares about, pooled across every
    // captured layer — reproduces the root-cause round's own reporting
    // shape (p50/p90 of the outlier ratio) as a load-bearing sanity check
    // that this calibration run reproduces the known failure mode before
    // anything downstream trusts it.
    for tensor in ["gdn_y_silu", "ffn_gate_silu"] {
        let mut ratios = Vec::new();
        for (key, t) in &capture.dump().tensors {
            if !key.ends_with(&format!(":{tensor}")) {
                continue;
            }
            ratios.extend(pooled_block_outlier_ratios(
                &t.values,
                t.rows as usize,
                t.cols as usize,
            ));
        }
        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !ratios.is_empty() {
            eprintln!(
                "{tensor}: per-32-block amax/mean(|x|) p50={:.2} p90={:.2} (n={})",
                percentile(&ratios, 0.5),
                percentile(&ratios, 0.9),
                ratios.len()
            );
        }
    }
}

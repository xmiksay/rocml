//! Issue #2's per-head boundary-skip measurement: "the ~2% lowest-entropy
//! heads" hint from the issue asks whether skipping quantization for a
//! small subset of KV heads (keeping them fp16, like the boundary layers)
//! would meaningfully reduce quantization error. Ornith-1.0-9B has only 8
//! full-attention layers x `head_count_kv` KV heads each (32 total, 24
//! outside the two always-fp16 boundary layers) — far too few for a literal
//! "2%" to mean anything, so this measures the actual per-head K/V
//! quantization error distribution directly and reports whether a small
//! number of heads dominate it (implement a skip) or error is roughly
//! uniform (honest negative, per the task's own "measure first" instruction
//! — mirrors this codebase's disposition for the int8-MMQ investigation:
//! measure, report the real numbers, and only add a feature if it
//! demonstrably wins).
//!
//! Method: run a real prompt through Ornith-1.0-9B's fp16-KV decode path
//! (every full-attention layer captured as a dense fp16 K/V plane, no
//! quantization involved yet), pull the exact K/V vectors back via the
//! snapshot layer's `capture_snapshot` (issue #1's existing D2H capture
//! path — reused here purely as a data-extraction tool, not for snapshot
//! testing), then run the *same* CPU quantize/dequantize reference
//! (`rocml::kv_quant::quant_math`) the production kernels are checked
//! against, per `WINDOW_LEN`-sized block, per (layer, head), and compare
//! against the captured fp16-rounded original. This measures the exact
//! quantization error the production mixed cache would introduce, without
//! needing a second (mixed-KV) model load or any new capture plumbing.
//!
//! `#[ignore]`d diagnostic tool, not a correctness gate (mirrors
//! `mmq_layer_diff.rs`'s pattern) — run explicitly via `make
//! kv-head-error-measure`. Real hardware + the real Ornith-1.0-9B checkpoint
//! required; skips itself if absent. `--release` recommended.

use half::f16;
use rocml::kv_quant::quant_math::{
    dequantize_k_per_channel, dequantize_v_per_token_q4, dequantize_v_per_token_q8,
    quantize_k_per_channel, quantize_v_per_token_q4, quantize_v_per_token_q8,
};
use rocml::kv_quant::WINDOW_LEN;
use rocml::qwen35::config::LayerKind;
use rocml::snapshot::AttnLayerBytes;
use rocml::{LoadOptions, Model};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;

const GGUF_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf";
const CORPUS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/eval/corpus.txt");
const CTX: usize = 4096;
/// Past `SINK_LEN(32) + WINDOW_LEN(128)*4`, so several full evicted blocks
/// exist per mixed-eligible layer, giving the per-head aggregation enough
/// samples to be more than noise.
const DECODE_TOKENS: usize = 700;

/// Per-(layer, head) accumulated squared error/magnitude — pooled across
/// every full block seen, so a layer/head's final ratio is a genuine
/// (not block-averaged) relative RMSE: `sqrt(sum_sq_err / sum_sq_orig)`.
#[derive(Debug, Clone, Copy, Default)]
struct ErrAcc {
    sum_sq_err: f64,
    sum_sq_orig: f64,
}

impl ErrAcc {
    fn add(&mut self, orig: &[f32], deq: &[f32]) {
        for (&o, &d) in orig.iter().zip(deq) {
            let e = (o - d) as f64;
            self.sum_sq_err += e * e;
            self.sum_sq_orig += (o as f64) * (o as f64);
        }
    }

    /// Relative RMSE: 0 for a perfect (or all-zero) reconstruction.
    fn relative_rmse(&self) -> f64 {
        if self.sum_sq_orig <= 0.0 {
            return 0.0;
        }
        (self.sum_sq_err / self.sum_sq_orig).sqrt()
    }
}

fn f16_to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

/// Extracts one `[n_kv_heads, WINDOW_LEN, head_dim]` block (positions
/// `[block_start, block_start+WINDOW_LEN)`) out of a captured layer's
/// `[n_kv_heads, filled, head_dim]` K or V buffer.
fn extract_block(
    full: &[f32],
    n_kv_heads: usize,
    filled: usize,
    head_dim: usize,
    block_start: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; n_kv_heads * WINDOW_LEN as usize * head_dim];
    for h in 0..n_kv_heads {
        let src_base = h * filled * head_dim + block_start * head_dim;
        let dst_base = h * WINDOW_LEN as usize * head_dim;
        out[dst_base..dst_base + WINDOW_LEN as usize * head_dim]
            .copy_from_slice(&full[src_base..src_base + WINDOW_LEN as usize * head_dim]);
    }
    out
}

#[test]
#[ignore]
fn ornith_per_head_kv_quant_error_measurement() {
    let Some(gguf_path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping ornith_per_head_kv_quant_error_measurement: {GGUF_REL} not found");
        return;
    };

    let corpus = std::fs::read_to_string(CORPUS_PATH).expect("failed to read eval corpus");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&gguf_path).expect("gguf open failed"))
        .expect("tokenizer load failed");
    let mut prompt_ids = tokenizer.encode(&corpus);
    // Repeat the corpus if it tokenizes shorter than DECODE_TOKENS — this
    // is a quantization-error measurement over real activation statistics,
    // not a language-modeling quality check, so repeated text is fine.
    while prompt_ids.len() < DECODE_TOKENS {
        let more = prompt_ids.clone();
        prompt_ids.extend(more);
    }
    prompt_ids.truncate(DECODE_TOKENS);

    // Default LoadOptions: fp16 KV, every full-attention layer captured as
    // a plain dense fp16 plane — no quantization in this load at all.
    let mut model = Model::load(&gguf_path, LoadOptions::new(CTX)).expect("Model::load failed");
    let hybrid = model
        .as_hybrid_mut()
        .expect("Ornith-1.0-9B must be the qwen35 hybrid architecture");

    for &id in &prompt_ids {
        hybrid.forward_token(id).expect("forward_token failed");
    }
    let filled = hybrid.position() as usize;
    assert_eq!(filled, DECODE_TOKENS);

    let cfg = hybrid.config().clone();
    let n_kv_heads = cfg.head_count_kv as usize;
    let head_dim = cfg.head_dim as usize;
    let attn_layer_indices: Vec<usize> = cfg
        .layer_kinds
        .iter()
        .enumerate()
        .filter(|(_, k)| **k == LayerKind::FullAttention)
        .map(|(i, _)| i)
        .collect();
    let (first_boundary, last_boundary) = (
        attn_layer_indices.first().copied(),
        attn_layer_indices.last().copied(),
    );
    eprintln!(
        "Ornith-1.0-9B: {} full-attention layers at {attn_layer_indices:?}, {n_kv_heads} kv \
         heads each (head_dim {head_dim}) -> {} mixed-eligible layers x {n_kv_heads} heads = {} \
         mixed KV heads (boundary layers {first_boundary:?}/{last_boundary:?} excluded, always \
         fp16 regardless of this measurement)",
        attn_layer_indices.len(),
        attn_layer_indices.len().saturating_sub(2),
        attn_layer_indices.len().saturating_sub(2) * n_kv_heads
    );

    let token_ids: Vec<u32> = prompt_ids.clone();
    let snap = hybrid
        .capture_snapshot(token_ids)
        .expect("capture_snapshot failed");

    let n_blocks = (filled - rocml::kv_quant::SINK_LEN as usize) / WINDOW_LEN as usize;
    assert!(
        n_blocks >= 2,
        "need at least 2 full evicted-block-equivalents past SINK_LEN+WINDOW_LEN for a \
         meaningful measurement; got {n_blocks} at filled={filled} — raise DECODE_TOKENS"
    );

    // Pooled per-(layer, head) accumulators, K (always Q8 in production)
    // and V at both bit widths (Q8 for `--kv-cache q8`, Q4 for the
    // production-recommended `q4-mixed`).
    let mut k_err: Vec<Vec<ErrAcc>> = Vec::new();
    let mut v_q8_err: Vec<Vec<ErrAcc>> = Vec::new();
    let mut v_q4_err: Vec<Vec<ErrAcc>> = Vec::new();
    let mut measured_layers: Vec<usize> = Vec::new();

    for &layer_idx in &attn_layer_indices {
        if Some(layer_idx) == first_boundary || Some(layer_idx) == last_boundary {
            continue; // Always fp16 in production — not what this measures.
        }
        let Some(AttnLayerBytes::DenseF16 { k, v }) = &snap.attn[layer_idx] else {
            panic!("layer {layer_idx}: expected a DenseF16 capture (fp16 KV, non-boundary)");
        };
        let k_full = f16_to_f32(k);
        let v_full = f16_to_f32(v);

        let mut layer_k_err = vec![ErrAcc::default(); n_kv_heads];
        let mut layer_v_q8_err = vec![ErrAcc::default(); n_kv_heads];
        let mut layer_v_q4_err = vec![ErrAcc::default(); n_kv_heads];

        for block in 0..n_blocks {
            let block_start = rocml::kv_quant::SINK_LEN as usize + block * WINDOW_LEN as usize;
            let k_block = extract_block(&k_full, n_kv_heads, filled, head_dim, block_start);
            let v_block = extract_block(&v_full, n_kv_heads, filled, head_dim, block_start);

            let (k_codes, k_scales) =
                quantize_k_per_channel(&k_block, n_kv_heads, WINDOW_LEN as usize, head_dim);
            let k_deq = dequantize_k_per_channel(
                &k_codes,
                &k_scales,
                n_kv_heads,
                WINDOW_LEN as usize,
                head_dim,
            );

            let (v8_codes, v8_scales) =
                quantize_v_per_token_q8(&v_block, n_kv_heads, WINDOW_LEN as usize, head_dim);
            let v8_deq = dequantize_v_per_token_q8(
                &v8_codes,
                &v8_scales,
                n_kv_heads,
                WINDOW_LEN as usize,
                head_dim,
            );

            let (v4_codes, v4_scales) =
                quantize_v_per_token_q4(&v_block, n_kv_heads, WINDOW_LEN as usize, head_dim);
            let v4_deq = dequantize_v_per_token_q4(
                &v4_codes,
                &v4_scales,
                n_kv_heads,
                WINDOW_LEN as usize,
                head_dim,
            );

            for h in 0..n_kv_heads {
                let range =
                    h * WINDOW_LEN as usize * head_dim..(h + 1) * WINDOW_LEN as usize * head_dim;
                layer_k_err[h].add(&k_block[range.clone()], &k_deq[range.clone()]);
                layer_v_q8_err[h].add(&v_block[range.clone()], &v8_deq[range.clone()]);
                layer_v_q4_err[h].add(&v_block[range.clone()], &v4_deq[range]);
            }
        }

        measured_layers.push(layer_idx);
        k_err.push(layer_k_err);
        v_q8_err.push(layer_v_q8_err);
        v_q4_err.push(layer_v_q4_err);
    }

    eprintln!(
        "\n=== per-(layer,head) relative RMSE over {n_blocks} evicted blocks ({} decode tokens) \
         ===",
        filled
    );
    eprintln!("layer  head  K(q8)     V(q8)     V(q4)");
    for (row, &layer_idx) in measured_layers.iter().enumerate() {
        for h in 0..n_kv_heads {
            eprintln!(
                "{layer_idx:5}  {h:4}  {:.6}  {:.6}  {:.6}",
                k_err[row][h].relative_rmse(),
                v_q8_err[row][h].relative_rmse(),
                v_q4_err[row][h].relative_rmse(),
            );
        }
    }

    // Aggregate per head *index* (pooled across every mixed-eligible layer)
    // — this is the axis a per-head skip bitmask would actually key on
    // (the mixed kernels index heads within a layer the same way at every
    // layer), so it's the one that matters for "does a small subset of
    // heads dominate error".
    let mut pooled_by_head = vec![ErrAcc::default(); n_kv_heads];
    for layer_v4 in &v_q4_err {
        for h in 0..n_kv_heads {
            pooled_by_head[h].sum_sq_err += layer_v4[h].sum_sq_err;
            pooled_by_head[h].sum_sq_orig += layer_v4[h].sum_sq_orig;
        }
    }
    let per_head_rmse: Vec<f64> = pooled_by_head.iter().map(ErrAcc::relative_rmse).collect();
    let median = {
        let mut sorted = per_head_rmse.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    };
    let max = per_head_rmse.iter().cloned().fold(0.0f64, f64::max);
    eprintln!(
        "\n=== V(q4) relative RMSE pooled by head index across all {} mixed-eligible layers ===",
        measured_layers.len()
    );
    for (h, &rmse) in per_head_rmse.iter().enumerate() {
        eprintln!(
            "head {h}: {rmse:.6} ({:.2}x median)",
            rmse / median.max(1e-12)
        );
    }
    eprintln!(
        "median={median:.6} max={max:.6} max/median={:.2}x",
        max / median.max(1e-12)
    );
}

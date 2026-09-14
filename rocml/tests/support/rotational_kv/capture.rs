//! Real K/V capture off Ornith-1.0-9B's fp16-KV decode path — the same
//! technique `kv_head_error_measure.rs` (issue #2) established (run a real
//! prompt through decode, pull exact K/V vectors back via the snapshot
//! layer's `capture_snapshot`), generalized here into a reusable helper for
//! issue #14's calibration (`rotational_kv_calibrate.rs`) and tensor-level
//! measurement (`rotational_kv_measure.rs`) tests.

use half::f16;
use rocml::qwen35::config::LayerKind;
use rocml::snapshot::AttnLayerBytes;
use rocml::{LoadOptions, Model};
use rocml_core::gguf::GgufFile;
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;

pub const GGUF_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf";
pub const CORPUS_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../bench/eval/corpus.txt");

/// One mixed-eligible full-attention layer's captured fp16 K/V, converted
/// to f32, each `[n_kv_heads, filled, head_dim]` row-major.
pub struct CapturedLayer {
    pub layer_idx: usize,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
}

pub struct Capture {
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub filled: usize,
    pub sink_len: usize,
    pub layers: Vec<CapturedLayer>,
}

/// Loads Ornith-1.0-9B-Q4_K_M, decodes `decode_tokens` real tokens (the
/// checked-in eval corpus, repeated if shorter), and captures every
/// mixed-eligible full-attention layer's K/V — `None` if the checkpoint
/// isn't present (the caller should skip its test in that case, matching
/// every other real-GGUF test in this suite).
pub fn capture_ornith_kv(decode_tokens: usize) -> Option<Capture> {
    let gguf_path = checkpoint(GGUF_REL)?;

    let corpus = std::fs::read_to_string(CORPUS_PATH).expect("failed to read eval corpus");
    let tokenizer = BpeTokenizer::from_gguf(&GgufFile::open(&gguf_path).expect("gguf open failed"))
        .expect("tokenizer load failed");
    let mut prompt_ids = tokenizer.encode(&corpus);
    while prompt_ids.len() < decode_tokens {
        let more = prompt_ids.clone();
        prompt_ids.extend(more);
    }
    prompt_ids.truncate(decode_tokens);

    let ctx = (decode_tokens + 256).max(4096);
    let mut model = Model::load(&gguf_path, LoadOptions::new(ctx)).expect("Model::load failed");
    let hybrid = model
        .as_hybrid_mut()
        .expect("Ornith-1.0-9B must be the qwen35 hybrid architecture");

    for &id in &prompt_ids {
        hybrid.forward_token(id).expect("forward_token failed");
    }
    let filled = hybrid.position() as usize;
    assert_eq!(filled, decode_tokens);

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

    let snap = hybrid
        .capture_snapshot(prompt_ids.clone())
        .expect("capture_snapshot failed");

    let mut layers = Vec::new();
    for &layer_idx in &attn_layer_indices {
        if Some(layer_idx) == first_boundary || Some(layer_idx) == last_boundary {
            continue; // Always fp16 in production — not calibration data.
        }
        let Some(AttnLayerBytes::DenseF16 { k, v }) = &snap.attn[layer_idx] else {
            panic!("layer {layer_idx}: expected a DenseF16 capture (fp16 KV, non-boundary)");
        };
        layers.push(CapturedLayer {
            layer_idx,
            k: f16_to_f32(k),
            v: f16_to_f32(v),
        });
    }

    Some(Capture {
        n_kv_heads,
        head_dim,
        filled,
        sink_len: rocml::kv_quant::SINK_LEN as usize,
        layers,
    })
}

fn f16_to_f32(v: &[f16]) -> Vec<f32> {
    v.iter().map(|x| x.to_f32()).collect()
}

impl Capture {
    /// One head's contiguous `[head_dim]`-per-position slice over
    /// `[start, end)` — a single head's positions are contiguous in this
    /// capture's `[n_kv_heads, filled, head_dim]` row-major layout, so this
    /// is a plain sub-slice, no copy.
    pub fn head_slice<'a>(
        &self,
        layer: &'a CapturedLayer,
        is_k: bool,
        head: usize,
        start: usize,
        end: usize,
    ) -> &'a [f32] {
        let src = if is_k { &layer.k } else { &layer.v };
        let base = head * self.filled * self.head_dim;
        &src[base + start * self.head_dim..base + end * self.head_dim]
    }

    /// Pools every `[head_dim]` vector across every captured layer/head at
    /// positions in `[start, end)` into one flat row-major buffer, for
    /// either `k` or `v` (`is_k`). Returns `(data, n_vectors)`.
    pub fn flatten(&self, is_k: bool, start: usize, end: usize) -> (Vec<f32>, usize) {
        let mut out = Vec::new();
        let mut n_vectors = 0usize;
        for layer in &self.layers {
            let src = if is_k { &layer.k } else { &layer.v };
            for h in 0..self.n_kv_heads {
                for pos in start..end {
                    let base = h * self.filled * self.head_dim + pos * self.head_dim;
                    out.extend_from_slice(&src[base..base + self.head_dim]);
                    n_vectors += 1;
                }
            }
        }
        (out, n_vectors)
    }
}

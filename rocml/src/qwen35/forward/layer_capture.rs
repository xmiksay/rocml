//! Per-layer activation capture for the qwen35 hybrid chunked-prefill path
//! — issue #10's per-layer diff harness, built while root-causing the
//! int8-MMQ precision regression (see `.claude/CLAUDE.md`'s MMQ section).
//! Threaded through `chunk_forward.rs` exactly like [`Profiler`]
//! (`crate::profile::Profiler`): an `Option<&mut LayerCapture>` that's a
//! plain no-op check when `None`, so capture-off costs nothing on the hot
//! decode/prefill path — decode never passes one at all (only
//! `forward_prompt_chunked_captured` does), and prefill's cost is one
//! `Option::is_none()` branch per layer.
//!
//! Records intra-layer intermediate tensors for the *final* chunk only —
//! the one whose output actually determines the prompt's logits, and the
//! one the reported precision failure is measured against — not just each
//! layer's post-residual-add output (`"resid_post"`, `ChunkScratch::x`
//! after that layer's FFN) but also the intermediates on either side of
//! each MMQ-eligible matmul (`gdn_chunk.rs`'s `"gdn_xn"`/`"gdn_qkv_raw"`/
//! `"gdn_y_silu"`/`"gdn_ssm_out_raw"`, `ffn_chunk.rs`'s `"resid_pre_ffn"`/
//! `"ffn_xn"`/`"ffn_gate_silu"`/`"ffn_down_out"`) — this is what let the
//! `mmq_precision` investigation localize the regression to a *specific*
//! matmul (GDN's `ssm_out` projection) rather than just a layer. Serializes
//! to a small JSON schema (`LayerDump`) keyed by `"{layer_idx}:{tensor_name}"`
//! deliberately generic enough that a later round can point the same
//! comparison logic (`diff_dumps`) at dumps produced by llama.cpp's
//! `eval-callback` example for issue #10 proper, once that tensor-name
//! convention is decided.
use std::collections::BTreeMap;
use std::path::Path;

use rocml_hip::DeviceBuffer;
use serde::{Deserialize, Serialize};

use crate::error::RocmlError;

/// One captured `[rows, cols]` row-major f32 tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapturedTensor {
    pub rows: u32,
    pub cols: u32,
    pub values: Vec<f32>,
}

impl CapturedTensor {
    pub fn row(&self, r: usize) -> &[f32] {
        let cols = self.cols as usize;
        &self.values[r * cols..(r + 1) * cols]
    }
}

/// `"{layer_idx}:{tensor}"` -> tensor, `BTreeMap` for deterministic JSON key
/// order (stable byte-for-byte dumps across runs, diffable by eye).
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct LayerDump {
    pub tensors: BTreeMap<String, CapturedTensor>,
}

impl LayerDump {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, RocmlError> {
        let bytes = std::fs::read(path.as_ref()).map_err(|e| {
            RocmlError::Config(format!("layer dump: read {}: {e}", path.as_ref().display()))
        })?;
        serde_json::from_slice(&bytes)
            .map_err(|e| RocmlError::Config(format!("layer dump: parse json: {e}")))
    }
}

/// Recording side: owns the in-progress dump, fed by `record` calls from
/// `chunk_forward.rs`'s per-layer loop.
pub struct LayerCapture {
    dump: LayerDump,
}

impl Default for LayerCapture {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerCapture {
    pub fn new() -> Self {
        Self {
            dump: LayerDump::default(),
        }
    }

    /// Copies `buf`'s first `rows*cols` elements to host and records them
    /// under `"{layer_idx}:{tensor}"`. Only ever called on the final
    /// chunk's own tensors (`chunk_len` rows, not `CHUNK_CAP`), so this
    /// stays cheap relative to the layer's own kernel launches.
    pub fn record(
        &mut self,
        layer_idx: u32,
        tensor: &str,
        buf: &DeviceBuffer<f32>,
        rows: u32,
        cols: u32,
    ) -> Result<(), RocmlError> {
        let mut host = vec![0f32; (rows * cols) as usize];
        buf.copy_range_to_host(0, &mut host)?;
        self.dump.tensors.insert(
            format!("{layer_idx}:{tensor}"),
            CapturedTensor {
                rows,
                cols,
                values: host,
            },
        );
        Ok(())
    }

    pub fn dump(&self) -> &LayerDump {
        &self.dump
    }

    pub fn write_json(&self, path: impl AsRef<Path>) -> Result<(), RocmlError> {
        let json = serde_json::to_string(&self.dump)
            .map_err(|e| RocmlError::Config(format!("layer capture: serialize json: {e}")))?;
        std::fs::write(path.as_ref(), json).map_err(|e| {
            RocmlError::Config(format!(
                "layer capture: write {}: {e}",
                path.as_ref().display()
            ))
        })
    }
}

/// Per-layer max/mean relative error between two dumps of the same tensor
/// name, over every row — the localization tool issue #10 asked for. `rel`
/// is `|a-b| / max(|a|, eps)`, `eps` guarding near-zero reference elements
/// (heavy-cancellation outputs, common near a residual stream's zero
/// crossings) from blowing up an otherwise-tiny absolute difference into a
/// meaningless relative one.
#[derive(Debug, Clone, Serialize)]
pub struct LayerDiff {
    pub layer_idx: u32,
    pub tensor: String,
    pub max_rel: f32,
    pub mean_rel: f32,
    pub max_abs: f32,
    pub mean_abs: f32,
}

const REL_EPS: f32 = 1e-3;

/// Diffs every tensor present in both dumps, sorted by descending
/// `max_rel` (the worst offender first) — `diff_dumps(mmq_off, mmq_on)` is
/// the intended call shape (`a` is the reference).
pub fn diff_dumps(a: &LayerDump, b: &LayerDump) -> Vec<LayerDiff> {
    let mut out = Vec::new();
    for (key, ta) in &a.tensors {
        let Some(tb) = b.tensors.get(key) else {
            continue;
        };
        if ta.rows != tb.rows || ta.cols != tb.cols {
            continue;
        }
        let (mut max_rel, mut sum_rel, mut max_abs, mut sum_abs) = (0f32, 0f32, 0f32, 0f32);
        let n = ta.values.len().max(1);
        for (va, vb) in ta.values.iter().zip(tb.values.iter()) {
            let abs = (va - vb).abs();
            let rel = abs / va.abs().max(REL_EPS);
            max_rel = max_rel.max(rel);
            sum_rel += rel;
            max_abs = max_abs.max(abs);
            sum_abs += abs;
        }
        let (layer_idx, tensor) = split_key(key);
        out.push(LayerDiff {
            layer_idx,
            tensor,
            max_rel,
            mean_rel: sum_rel / n as f32,
            max_abs,
            mean_abs: sum_abs / n as f32,
        });
    }
    out.sort_by(|x, y| {
        y.max_rel
            .partial_cmp(&x.max_rel)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn split_key(key: &str) -> (u32, String) {
    match key.split_once(':') {
        Some((idx, tensor)) => (idx.parse().unwrap_or(u32::MAX), tensor.to_string()),
        None => (u32::MAX, key.to_string()),
    }
}

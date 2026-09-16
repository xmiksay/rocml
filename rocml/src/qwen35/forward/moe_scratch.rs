//! Preallocated scratch for the qwen35moe MoE FFN step (`super::moe`):
//! small router/gate buffers plus the reusable per-expert staging buffers
//! the M1 expert-offload path copies one expert's raw quantized bytes into
//! before running the existing fused dequant-GEMV kernels. Sized once at
//! model load and reused across every token and every MoE layer — mirrors
//! `Scratch`/`ChunkScratch`, but always processes exactly one row (token) at
//! a time regardless of caller (decode or one iteration of a chunked-prefill
//! per-token loop — see `moe::moe_ffn_step`), so it needs no `CHUNK_CAP`
//! sizing.

use rocml_hip::DeviceBuffer;

use super::super::config::{MoeConfig, Qwen35Config};
use super::super::weights::{LayerWeights, ModelWeights, MoeFfnWeights};
use crate::error::RocmlError;

pub struct MoeScratch {
    pub xn: DeviceBuffer<f32>,
    pub router_logits: DeviceBuffer<f32>,
    pub topk_idx: DeviceBuffer<i32>,
    pub topk_weight: DeviceBuffer<f32>,
    pub shared_gate_logit: DeviceBuffer<f32>,
    /// Sized to `max(expert_ff_len, shared_ff_len)` — reused for both the
    /// shared expert's and every routed expert's gate/up projections.
    pub ffn_a: DeviceBuffer<f32>,
    pub ffn_b: DeviceBuffer<f32>,
    /// One expert's (shared or routed) down-projection output, `[hidden]`.
    pub expert_out: DeviceBuffer<f32>,
    /// The FFN step's output accumulator, `[hidden]` — the shared expert's
    /// gated contribution is always the first write, each selected routed
    /// expert's weighted contribution accumulates on top.
    pub accum: DeviceBuffer<f32>,
    /// Sum of only the *routed* experts' weighted contributions (excludes
    /// the shared expert) — exists purely for `layer_capture`'s
    /// `"moe_routed_sum"` key (see `super::moe`'s capture code), which maps
    /// onto llama.cpp's `ffn_moe_out` node (the routed-only sum, computed
    /// before that reference adds its own shared-expert term). `accum`
    /// itself already equals llama's `ffn_out` (shared + routed combined),
    /// since rocml writes the shared contribution into `accum` first and
    /// accumulates routed experts on top of it — this second buffer is the
    /// only way to observe the routed-only partial sum without changing
    /// `accum`'s own write order (which every other correctness gate
    /// depends on staying exactly as it is). Zeroed and (when capturing)
    /// accumulated into per routed expert; unused when not capturing.
    pub routed_sum: DeviceBuffer<f32>,
    /// Reused across every expert this token selects when the LRU cache
    /// (`super::moe_cache::ExpertCache`) is disabled (capacity 0 — see
    /// `Model::load`'s doc comment) or the cache reports a miss's staging
    /// went through here instead. Sized to the largest per-expert byte size
    /// across every MoE layer in the model (an oversized buffer is
    /// harmless — `gemv_quant` only ever reads the `m*n`-element prefix its
    /// own shape needs, per `LinearWeight`'s own design).
    pub stage_gate: DeviceBuffer<u8>,
    pub stage_up: DeviceBuffer<u8>,
    pub stage_down: DeviceBuffer<u8>,
}

impl MoeScratch {
    pub fn new(
        cfg: &Qwen35Config,
        moe_cfg: &MoeConfig,
        weights: &ModelWeights,
    ) -> Result<Self, RocmlError> {
        let hidden = cfg.embedding_length as usize;
        let ffn_dim = (moe_cfg.expert_ff_len.max(moe_cfg.shared_ff_len)) as usize;
        let expert_count = moe_cfg.expert_count as usize;
        let top_k = moe_cfg.expert_used_count as usize;
        let max_expert_bytes = max_expert_bytes(weights);

        Ok(Self {
            xn: DeviceBuffer::new(hidden)?,
            router_logits: DeviceBuffer::new(expert_count)?,
            topk_idx: DeviceBuffer::new(top_k)?,
            topk_weight: DeviceBuffer::new(top_k)?,
            shared_gate_logit: DeviceBuffer::new(1)?,
            ffn_a: DeviceBuffer::new(ffn_dim)?,
            ffn_b: DeviceBuffer::new(ffn_dim)?,
            expert_out: DeviceBuffer::new(hidden)?,
            accum: DeviceBuffer::new(hidden)?,
            routed_sum: DeviceBuffer::new(hidden)?,
            stage_gate: DeviceBuffer::new(max_expert_bytes)?,
            stage_up: DeviceBuffer::new(max_expert_bytes)?,
            stage_down: DeviceBuffer::new(max_expert_bytes)?,
        })
    }
}

/// Largest per-expert byte size across every MoE layer's gate/up/down
/// tensors (they can differ layer-to-layer since llama.cpp's `Q4_K_M`
/// importance heuristic picks Q4_K vs. Q6_K per layer for `ffn_down_exps`).
fn max_expert_bytes(weights: &ModelWeights) -> usize {
    weights
        .layers
        .iter()
        .filter_map(|l| match l {
            LayerWeights::Gdn(g) => match &g.ffn {
                super::super::weights::Ffn::Moe(m) => Some(m.max_expert_bytes()),
                super::super::weights::Ffn::Dense(_) => None,
            },
            LayerWeights::Attention(a) => match &a.ffn {
                super::super::weights::Ffn::Moe(m) => Some(m.max_expert_bytes()),
                super::super::weights::Ffn::Dense(_) => None,
            },
        })
        .max()
        .unwrap_or(0)
}

/// Every MoE layer's `MoeFfnWeights`, in layer order — the iterator
/// `super::moe_cache::ExpertCache`'s sizing (in `Model::load`) and
/// `max_expert_bytes` above both need, factored out so there's one place
/// that knows how to find a `Ffn::Moe` inside either layer-weight variant.
pub(crate) fn moe_layers(weights: &ModelWeights) -> impl Iterator<Item = &MoeFfnWeights> {
    weights.layers.iter().filter_map(|l| match l {
        LayerWeights::Gdn(g) => match &g.ffn {
            super::super::weights::Ffn::Moe(m) => Some(m.as_ref()),
            super::super::weights::Ffn::Dense(_) => None,
        },
        LayerWeights::Attention(a) => match &a.ffn {
            super::super::weights::Ffn::Moe(m) => Some(m.as_ref()),
            super::super::weights::Ffn::Dense(_) => None,
        },
    })
}

/// Model-wide max per-expert byte size for each of gate/up/down separately
/// (unlike `max_expert_bytes`, which takes the max across all three
/// combined for a single shared staging buffer) — the three VRAM pools'
/// per-slot strides in `super::moe_cache::ExpertCache`.
pub(crate) fn per_tensor_max_bytes(weights: &ModelWeights) -> (usize, usize, usize) {
    moe_layers(weights).fold((0, 0, 0), |(g, u, d), m| {
        (
            g.max(m.gate.per_expert_bytes),
            u.max(m.up.per_expert_bytes),
            d.max(m.down.per_expert_bytes),
        )
    })
}

/// Total distinct `(layer, expert)` keys across the whole model — the
/// natural ceiling on `ExpertCache` capacity (caching more slots than there
/// are experts can never help).
pub(crate) fn total_distinct_experts(weights: &ModelWeights, expert_count: u32) -> usize {
    moe_layers(weights).count() * expert_count as usize
}

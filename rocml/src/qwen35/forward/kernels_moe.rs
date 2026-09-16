//! Typed launch helpers for `kernels/moe.hip` (qwen35moe's mixture-of-experts
//! router + accumulate ops). Kept separate from `kernels.rs`/
//! `crate::forward::kernels` purely for the workspace's 400-line file cap,
//! mirroring `kernels_mixed.rs`.

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

/// Block size `moe_route_topk_f32` is compiled for — must match `MOE_BLOCK`
/// in `kernels/moe.hip`.
const MOE_ROUTE_BLOCK: u32 = 256;
/// Grid-stride block size for the plain elementwise ops below — no
/// power-of-two constraint, mirrors `crate::forward::kernels`'s own
/// (private-to-that-module) constant of the same name.
const LINEAR_BLOCK: u32 = 256;
/// Shared-memory cap `moe_route_topk_f32` is compiled for — must match
/// `MOE_MAX_EXPERTS` in `kernels/moe.hip`.
pub const MOE_MAX_EXPERTS: u32 = 1024;

pub struct MoeKernels {
    _mod_route: Module,
    route_fn: rocml_hip::Function,
    _mod_shared_gate: Module,
    shared_gate_fn: rocml_hip::Function,
    _mod_accum: Module,
    accum_fn: rocml_hip::Function,
}

impl MoeKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_route, route_fn) = load(
            rocml_kernels::MOE_ROUTE_TOPK_F32_HSACO,
            rocml_kernels::MOE_ROUTE_TOPK_F32_KERNEL,
        )?;
        let (_mod_shared_gate, shared_gate_fn) = load(
            rocml_kernels::MOE_SHARED_GATE_WRITE_F32_HSACO,
            rocml_kernels::MOE_SHARED_GATE_WRITE_F32_KERNEL,
        )?;
        let (_mod_accum, accum_fn) = load(
            rocml_kernels::MOE_WEIGHTED_ACCUM_F32_HSACO,
            rocml_kernels::MOE_WEIGHTED_ACCUM_F32_KERNEL,
        )?;
        Ok(Self {
            _mod_route,
            route_fn,
            _mod_shared_gate,
            shared_gate_fn,
            _mod_accum,
            accum_fn,
        })
    }

    /// `moe_route_topk_f32(logits, out_idx, out_weight, rows, expert_count,
    /// top_k, min_sum)`: softmax + top-k + renormalize, one block per row.
    /// `expert_count` must be `<= MOE_MAX_EXPERTS`.
    #[allow(clippy::too_many_arguments)]
    pub fn route_topk(
        &self,
        logits: DevPtr,
        out_idx: DevPtr,
        out_weight: DevPtr,
        rows: u32,
        expert_count: u32,
        top_k: u32,
        min_sum: f32,
    ) -> Result<(), RocmlError> {
        debug_assert!(
            expert_count <= MOE_MAX_EXPERTS,
            "expert_count {expert_count} exceeds moe_route_topk_f32's MOE_MAX_EXPERTS \
             ({MOE_MAX_EXPERTS})"
        );
        let cfg = LaunchConfig {
            grid: (rows, 1, 1),
            block: (MOE_ROUTE_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(
            logits,
            out_idx,
            out_weight,
            rows,
            expert_count,
            top_k,
            min_sum
        );
        // SAFETY: params matches moe_route_topk_f32's signature (const
        // float*, int*, float*, unsigned, unsigned, unsigned, float);
        // `expert_count` fits the kernel's fixed shared-memory arrays
        // (checked above).
        unsafe { self.route_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `moe_shared_gate_write_f32(y, gate_logit, out, n)`: `out[i] = y[i] *
    /// sigmoid(*gate_logit)` — the shared expert's contribution, always the
    /// accumulator's first write.
    pub fn shared_gate_write(
        &self,
        y: DevPtr,
        gate_logit: DevPtr,
        out: DevPtr,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(y, gate_logit, out, n);
        // SAFETY: params matches moe_shared_gate_write_f32's signature
        // (const float*, const float*, float*, unsigned); no block-size
        // constraint.
        unsafe { self.shared_gate_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `moe_weighted_accum_f32(y, weight, out, n)`: `out[i] += y[i] *
    /// (*weight)` — one routed expert's weighted accumulate.
    pub fn weighted_accum(
        &self,
        y: DevPtr,
        weight: DevPtr,
        out: DevPtr,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(y, weight, out, n);
        // SAFETY: params matches moe_weighted_accum_f32's signature (const
        // float*, const float*, float*, unsigned); no block-size constraint.
        unsafe { self.accum_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}

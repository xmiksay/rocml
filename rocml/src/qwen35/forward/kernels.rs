//! Typed launch helpers for the kernels the qwen35 hybrid forward pass needs
//! beyond what `crate::forward::kernels::Kernels` (rmsnorm/gemv/rope/softmax/
//! silu_mul/embedding/add_inplace, shared with the dense Qwen3 path) already
//! covers: the GDN decode-step kernels, partial rope, and the attention
//! output's sigmoid gate.

use std::mem::size_of;

use rocml_hip::{kernel_params, LaunchConfig, Module};

use crate::error::RocmlError;
use crate::forward::kernels::{load, DevPtr};

const LINEAR_BLOCK: u32 = 256;

pub struct HybridKernels {
    _mod_rope_partial: Module,
    rope_partial_fn: rocml_hip::Function,
    _mod_conv: Module,
    conv_fn: rocml_hip::Function,
    _mod_gate: Module,
    gate_fn: rocml_hip::Function,
    _mod_recurrence: Module,
    recurrence_fn: rocml_hip::Function,
    _mod_sigmoid_mul: Module,
    sigmoid_mul_fn: rocml_hip::Function,
}

impl HybridKernels {
    pub fn load_all() -> Result<Self, RocmlError> {
        let (_mod_rope_partial, rope_partial_fn) = load(
            rocml_kernels::ROPE_NEOX_PARTIAL_F32_HSACO,
            rocml_kernels::ROPE_NEOX_PARTIAL_F32_KERNEL,
        )?;
        let (_mod_conv, conv_fn) = load(
            rocml_kernels::GDN_CONV1D_DECODE_F32_HSACO,
            rocml_kernels::GDN_CONV1D_DECODE_F32_KERNEL,
        )?;
        let (_mod_gate, gate_fn) = load(
            rocml_kernels::GDN_GATE_F32_HSACO,
            rocml_kernels::GDN_GATE_F32_KERNEL,
        )?;
        let (_mod_recurrence, recurrence_fn) = load(
            rocml_kernels::GDN_RECURRENCE_DECODE_F32_HSACO,
            rocml_kernels::GDN_RECURRENCE_DECODE_F32_KERNEL,
        )?;
        let (_mod_sigmoid_mul, sigmoid_mul_fn) = load(
            rocml_kernels::ELEMENTWISE_HSACO,
            rocml_kernels::SIGMOID_MUL_F32_KERNEL,
        )?;

        Ok(Self {
            _mod_rope_partial,
            rope_partial_fn,
            _mod_conv,
            conv_fn,
            _mod_gate,
            gate_fn,
            _mod_recurrence,
            recurrence_fn,
            _mod_sigmoid_mul,
            sigmoid_mul_fn,
        })
    }

    /// In-place partial NEOX rope over `x` viewed as `[tokens, heads,
    /// head_dim]`: only the first `rot_dim` components of each head rotate.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_partial(
        &self,
        x: DevPtr,
        tokens: u32,
        heads: u32,
        head_dim: u32,
        rot_dim: u32,
        pos_base: u32,
        theta_base: f32,
    ) -> Result<(), RocmlError> {
        let half_rot = rot_dim / 2;
        let total = tokens * heads * half_rot;
        let cfg = LaunchConfig {
            grid: (total.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(x, tokens, heads, head_dim, rot_dim, pos_base, theta_base);
        // SAFETY: params matches rope_neox_partial_f32's signature (float*,
        // unsigned x4, unsigned, float); rot_dim is even (validated by
        // Qwen35Config).
        unsafe { self.rope_partial_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `causal_conv1d_decode_f32`: one GDN causal-conv decode step over
    /// `channels`, mutating `conv_state` in place.
    pub fn gdn_conv1d_decode(
        &self,
        x_new: DevPtr,
        conv_state: DevPtr,
        weight: DevPtr,
        out: DevPtr,
        channels: u32,
        kernel_size: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (channels.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(x_new, conv_state, weight, out, channels, kernel_size);
        // SAFETY: params matches causal_conv1d_decode_f32's signature (const
        // float*, float*, const float*, float*, unsigned, unsigned).
        unsafe { self.conv_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_gate_f32`: per-head write-strength beta and decay g, one step.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_gate(
        &self,
        a_raw: DevPtr,
        b_raw: DevPtr,
        a_log: DevPtr,
        dt_bias: DevPtr,
        beta_out: DevPtr,
        g_out: DevPtr,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(a_raw, b_raw, a_log, dt_bias, beta_out, g_out, n);
        // SAFETY: params matches gdn_gate_f32's signature (four const
        // float*, two float*, unsigned).
        unsafe { self.gate_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `gdn_recurrence_decode_f32`: fused state update + readout, one head
    /// per block, block size `max(head_k_dim, head_v_dim)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_recurrence_decode(
        &self,
        state: DevPtr,
        q: DevPtr,
        k: DevPtr,
        v: DevPtr,
        beta: DevPtr,
        g: DevPtr,
        y: DevPtr,
        num_heads: u32,
        head_k_dim: u32,
        head_v_dim: u32,
    ) -> Result<(), RocmlError> {
        let block = head_k_dim.max(head_v_dim);
        let cfg = LaunchConfig {
            grid: (num_heads, 1, 1),
            block: (block, 1, 1),
            shared_mem_bytes: 2 * (head_k_dim + head_v_dim) * size_of::<f32>() as u32,
        };
        let mut params =
            kernel_params!(state, q, k, v, beta, g, y, num_heads, head_k_dim, head_v_dim);
        // SAFETY: params matches gdn_recurrence_decode_f32's signature
        // (float*, four const float*, const float*, float*, three
        // unsigned); block size is max(head_k_dim, head_v_dim) as required.
        unsafe { self.recurrence_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// `sigmoid_mul_f32(x, gate, out, n)`: out = x * sigmoid(gate).
    pub fn sigmoid_mul(
        &self,
        x: DevPtr,
        gate: DevPtr,
        out: DevPtr,
        n: u32,
    ) -> Result<(), RocmlError> {
        let cfg = LaunchConfig {
            grid: (n.div_ceil(LINEAR_BLOCK), 1, 1),
            block: (LINEAR_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(x, gate, out, n);
        // SAFETY: params matches sigmoid_mul_f32's signature (const float*,
        // const float*, float*, unsigned).
        unsafe { self.sigmoid_mul_fn.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}

//! Per-layer KV cache, one contiguous buffer per layer laid out as
//! `[kv_head][max_seq][head_dim]` — a per-(layer, kv head) plane is exactly
//! the `[cur_len, head_dim]` row-major slice the attention kernels expect,
//! so decode attention needs no gather step, just a head-plane offset.
//!
//! Storage dtype is chosen at load time (`KvDtype`, issue #3): `F16`
//! (default) halves the cache vs `F32` with negligible quality impact at
//! these scales (see the module doc on `attn_decode.hip`'s templated
//! `load_kv` seam for how the fused kernels read either dtype without ever
//! materializing a dequantized copy); `F32` exists purely so the parity
//! test suites can pin the pre-issue-#3 reference numerics exactly (see
//! `crate::model::LoadOptions`).
//!
//! No hardcoded cap on `max_seq` any more: issue #3 removed the old
//! `MAX_SEQ_CAP` constant in favor of an up-front VRAM budget check (see
//! `crate::budget` and `crate::registry::clamp_ctx`) — the caller supplies
//! whatever context length the budgeter approved, and `KvCache::new` just
//! allocates it.

use half::f16;
use rocml_hip::DeviceBuffer;

use crate::config::ModelConfig;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtype {
    F16,
    F32,
}

enum LayerCache {
    F16 {
        k: DeviceBuffer<f16>,
        v: DeviceBuffer<f16>,
    },
    F32 {
        k: DeviceBuffer<f32>,
        v: DeviceBuffer<f32>,
    },
}

impl LayerCache {
    fn new(dtype: KvDtype, plane_len: usize) -> Result<Self, RocmlError> {
        Ok(match dtype {
            KvDtype::F16 => Self::F16 {
                k: DeviceBuffer::new(plane_len)?,
                v: DeviceBuffer::new(plane_len)?,
            },
            KvDtype::F32 => Self::F32 {
                k: DeviceBuffer::new(plane_len)?,
                v: DeviceBuffer::new(plane_len)?,
            },
        })
    }

    fn k_ptr(&self) -> DevPtr {
        match self {
            Self::F16 { k, .. } => offset(k, 0),
            Self::F32 { k, .. } => offset(k, 0),
        }
    }

    fn v_ptr(&self) -> DevPtr {
        match self {
            Self::F16 { v, .. } => offset(v, 0),
            Self::F32 { v, .. } => offset(v, 0),
        }
    }

    fn dtype(&self) -> KvDtype {
        match self {
            Self::F16 { .. } => KvDtype::F16,
            Self::F32 { .. } => KvDtype::F32,
        }
    }

    /// Appends one time step's `[n_kv_heads, head_dim]` k/v vectors (always
    /// f32 — the scratch buffers' native dtype) at `pos`. `F32` storage does
    /// the previous raw device-to-device copy per head; `F16` storage casts
    /// through `Kernels::cast_f32_f16` per head instead (n_kv_heads tiny
    /// launches per layer per step — 8 on Ornith — amortized against every
    /// future decode/prefill read of that position via the fused kernels'
    /// dequant-on-load seam, never materialized back to f32 in memory).
    #[allow(clippy::too_many_arguments)]
    fn append(
        &mut self,
        kernels: &Kernels,
        pos: u32,
        max_seq: u32,
        n_kv_heads: u32,
        head_dim: u32,
        k_src: &DeviceBuffer<f32>,
        v_src: &DeviceBuffer<f32>,
    ) -> Result<(), RocmlError> {
        let head_dim_u = head_dim as usize;
        let max_seq_u = max_seq as usize;
        match self {
            Self::F32 { k, v } => {
                for h in 0..n_kv_heads as usize {
                    let dst = h * max_seq_u * head_dim_u + pos as usize * head_dim_u;
                    let src = h * head_dim_u;
                    k.copy_from_device(dst, k_src, src, head_dim_u)?;
                    v.copy_from_device(dst, v_src, src, head_dim_u)?;
                }
                Ok(())
            }
            Self::F16 { k, v } => {
                for h in 0..n_kv_heads as usize {
                    let dst = h * max_seq_u * head_dim_u + pos as usize * head_dim_u;
                    let src = h * head_dim_u;
                    kernels.cast_f32_f16(offset(k_src, src), offset(k, dst), head_dim)?;
                    kernels.cast_f32_f16(offset(v_src, src), offset(v, dst), head_dim)?;
                }
                Ok(())
            }
        }
    }
}

pub struct KvCache {
    max_seq: u32,
    head_dim: u32,
    n_kv_heads: u32,
    layers: Vec<LayerCache>,
}

impl KvCache {
    /// `ctx` is the caller's already-budgeted context length (see
    /// `crate::registry::clamp_ctx`) — clamped once more here against the
    /// model's own declared `context_length` as a final sanity bound.
    pub fn new(config: &ModelConfig, ctx: usize, dtype: KvDtype) -> Result<Self, RocmlError> {
        let max_seq = ctx.min(config.context_length as usize).max(1) as u32;
        let n_kv_heads = config.head_count_kv;
        let head_dim = config.head_dim;
        let plane_len = (n_kv_heads as usize) * (max_seq as usize) * (head_dim as usize);

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for _ in 0..config.block_count {
            layers.push(LayerCache::new(dtype, plane_len)?);
        }

        Ok(Self {
            max_seq,
            head_dim,
            n_kv_heads,
            layers,
        })
    }

    pub fn max_seq(&self) -> u32 {
        self.max_seq
    }

    /// Storage dtype every layer shares (`KvCache::new` allocates all
    /// layers with the same `dtype`) — `None` if there are no layers
    /// (never true for a real model config, `block_count >= 1`).
    pub fn dtype(&self) -> KvDtype {
        self.layers
            .first()
            .map(LayerCache::dtype)
            .unwrap_or(KvDtype::F16)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &mut self,
        kernels: &Kernels,
        layer_idx: usize,
        pos: u32,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
    ) -> Result<(), RocmlError> {
        if pos >= self.max_seq {
            return Err(RocmlError::ContextOverflow {
                requested: pos + 1,
                max_seq: self.max_seq,
            });
        }
        let (max_seq, n_kv_heads, head_dim) = (self.max_seq, self.n_kv_heads, self.head_dim);
        let layer = self.layers.get_mut(layer_idx).ok_or_else(|| {
            RocmlError::Config(format!("cache: layer index {layer_idx} out of range"))
        })?;
        layer.append(kernels, pos, max_seq, n_kv_heads, head_dim, k, v)
    }

    /// Element offset where kv head `kvh`'s `[max_seq, head_dim]` plane
    /// starts within a layer's K/V buffer.
    pub fn head_plane_offset(&self, kvh: u32) -> usize {
        kvh as usize * self.max_seq as usize * self.head_dim as usize
    }

    pub fn k_ptr(&self, layer_idx: usize) -> Result<DevPtr, RocmlError> {
        self.layers
            .get(layer_idx)
            .map(LayerCache::k_ptr)
            .ok_or_else(|| {
                RocmlError::Config(format!("cache: layer index {layer_idx} out of range"))
            })
    }

    pub fn v_ptr(&self, layer_idx: usize) -> Result<DevPtr, RocmlError> {
        self.layers
            .get(layer_idx)
            .map(LayerCache::v_ptr)
            .ok_or_else(|| {
                RocmlError::Config(format!("cache: layer index {layer_idx} out of range"))
            })
    }
}

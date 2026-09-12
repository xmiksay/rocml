//! Per-layer decode state for the hybrid model: GDN layers carry a causal
//! conv1d ring buffer plus the gated-delta-rule recurrence state (both must
//! be zeroed between unrelated generations, and are O(1) in context length —
//! unaffected by issue #3/#2's long-context work); full-attention layers
//! carry a `[kv_head][max_seq][head_dim]` K/V plane, the same layout and
//! `KvDtype` choice `crate::cache::KvCache` uses for the dense model (never
//! explicitly reset — stale bytes past the current position are never read,
//! see `crate::forward::Model::reset`'s doc comment).
//!
//! No hardcoded cap on `max_seq` any more — see `crate::cache`'s module doc
//! for the issue #3 budgeting story this mirrors.

use half::f16;
use rocml_hip::DeviceBuffer;

use super::config::{GdnConfig, LayerKind, Qwen35Config};
use crate::cache::KvDtype;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels};

pub struct GdnLayerState {
    /// `[conv_dim, kernel-1]` row-major: per-channel history, oldest first.
    pub conv_state: DeviceBuffer<f32>,
    /// `[num_v_heads, head_k_dim, head_v_dim]` row-major recurrence state.
    pub state: DeviceBuffer<f32>,
    conv_len: usize,
    state_len: usize,
}

impl GdnLayerState {
    fn new(gdn: &GdnConfig) -> Result<Self, RocmlError> {
        let conv_len = gdn.conv_dim as usize * (gdn.conv_kernel as usize - 1);
        let state_len =
            gdn.num_v_heads as usize * gdn.head_k_dim as usize * gdn.head_v_dim as usize;
        let mut conv_state = DeviceBuffer::new(conv_len)?;
        let mut state = DeviceBuffer::new(state_len)?;
        conv_state.copy_from_host(&vec![0.0f32; conv_len])?;
        state.copy_from_host(&vec![0.0f32; state_len])?;
        Ok(Self {
            conv_state,
            state,
            conv_len,
            state_len,
        })
    }

    fn reset(&mut self) -> Result<(), RocmlError> {
        self.conv_state
            .copy_from_host(&vec![0.0f32; self.conv_len])?;
        self.state.copy_from_host(&vec![0.0f32; self.state_len])?;
        Ok(())
    }
}

enum PlaneStorage {
    F16 {
        k: DeviceBuffer<f16>,
        v: DeviceBuffer<f16>,
    },
    F32 {
        k: DeviceBuffer<f32>,
        v: DeviceBuffer<f32>,
    },
}

pub struct AttnPlane {
    storage: PlaneStorage,
}

impl AttnPlane {
    fn new(
        n_kv_heads: u32,
        max_seq: u32,
        head_dim: u32,
        dtype: KvDtype,
    ) -> Result<Self, RocmlError> {
        let plane_len = (n_kv_heads as usize) * (max_seq as usize) * (head_dim as usize);
        let storage = match dtype {
            KvDtype::F16 => PlaneStorage::F16 {
                k: DeviceBuffer::new(plane_len)?,
                v: DeviceBuffer::new(plane_len)?,
            },
            KvDtype::F32 => PlaneStorage::F32 {
                k: DeviceBuffer::new(plane_len)?,
                v: DeviceBuffer::new(plane_len)?,
            },
        };
        Ok(Self { storage })
    }

    pub fn dtype(&self) -> KvDtype {
        match &self.storage {
            PlaneStorage::F16 { .. } => KvDtype::F16,
            PlaneStorage::F32 { .. } => KvDtype::F32,
        }
    }

    /// Appends this decode step's `[n_kv_heads, head_dim]` k/v vectors at
    /// `pos` — see `crate::cache::KvCache::append`'s doc comment for the
    /// f16-vs-f32 storage split (identical logic, duplicated because the
    /// dense and hybrid caches otherwise share no code).
    #[allow(clippy::too_many_arguments)]
    pub fn append(
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
        match &mut self.storage {
            PlaneStorage::F32 { k, v } => {
                for h in 0..n_kv_heads as usize {
                    let dst = h * max_seq_u * head_dim_u + pos as usize * head_dim_u;
                    let src = h * head_dim_u;
                    k.copy_from_device(dst, k_src, src, head_dim_u)?;
                    v.copy_from_device(dst, v_src, src, head_dim_u)?;
                }
            }
            PlaneStorage::F16 { k, v } => {
                for h in 0..n_kv_heads as usize {
                    let dst = h * max_seq_u * head_dim_u + pos as usize * head_dim_u;
                    let src = h * head_dim_u;
                    kernels.cast_f32_f16(offset(k_src, src), offset(k, dst), head_dim)?;
                    kernels.cast_f32_f16(offset(v_src, src), offset(v, dst), head_dim)?;
                }
            }
        }
        Ok(())
    }

    pub fn head_plane_offset(&self, kvh: u32, max_seq: u32, head_dim: u32) -> usize {
        kvh as usize * max_seq as usize * head_dim as usize
    }

    pub fn k_ptr(&self) -> DevPtr {
        match &self.storage {
            PlaneStorage::F16 { k, .. } => offset(k, 0),
            PlaneStorage::F32 { k, .. } => offset(k, 0),
        }
    }

    pub fn v_ptr(&self) -> DevPtr {
        match &self.storage {
            PlaneStorage::F16 { v, .. } => offset(v, 0),
            PlaneStorage::F32 { v, .. } => offset(v, 0),
        }
    }
}

pub struct HybridCache {
    max_seq: u32,
    gdn: Vec<Option<GdnLayerState>>,
    attn: Vec<Option<AttnPlane>>,
}

impl HybridCache {
    /// `ctx` is the caller's already-budgeted context length (see
    /// `crate::registry::clamp_ctx`) — clamped once more here against the
    /// model's own declared `context_length` as a final sanity bound.
    pub fn new(cfg: &Qwen35Config, ctx: usize, dtype: KvDtype) -> Result<Self, RocmlError> {
        let max_seq = ctx.min(cfg.context_length as usize).max(1) as u32;
        let mut gdn = Vec::with_capacity(cfg.layer_kinds.len());
        let mut attn = Vec::with_capacity(cfg.layer_kinds.len());
        for &kind in &cfg.layer_kinds {
            match kind {
                LayerKind::LinearAttention => {
                    gdn.push(Some(GdnLayerState::new(&cfg.gdn)?));
                    attn.push(None);
                }
                LayerKind::FullAttention => {
                    gdn.push(None);
                    attn.push(Some(AttnPlane::new(
                        cfg.head_count_kv,
                        max_seq,
                        cfg.head_dim,
                        dtype,
                    )?));
                }
            }
        }
        Ok(Self { max_seq, gdn, attn })
    }

    pub fn max_seq(&self) -> u32 {
        self.max_seq
    }

    /// Zeroes every GDN layer's conv/recurrence state for a fresh sequence.
    pub fn reset(&mut self) -> Result<(), RocmlError> {
        for slot in self.gdn.iter_mut().flatten() {
            slot.reset()?;
        }
        Ok(())
    }

    pub fn gdn_mut(&mut self, layer_idx: usize) -> Result<&mut GdnLayerState, RocmlError> {
        self.gdn
            .get_mut(layer_idx)
            .and_then(Option::as_mut)
            .ok_or_else(|| RocmlError::Config(format!("cache: layer {layer_idx} has no GDN state")))
    }

    pub fn attn_mut(&mut self, layer_idx: usize) -> Result<&mut AttnPlane, RocmlError> {
        self.attn
            .get_mut(layer_idx)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                RocmlError::Config(format!("cache: layer {layer_idx} has no attention plane"))
            })
    }
}

//! Per-layer decode state for the hybrid model: GDN layers carry a causal
//! conv1d ring buffer plus the gated-delta-rule recurrence state (both must
//! be zeroed between unrelated generations); full-attention layers carry a
//! `[kv_head][max_seq][head_dim]` K/V plane, the same layout
//! `crate::cache::KvCache` uses for the dense model (never explicitly reset —
//! stale bytes past the current position are never read, see
//! `crate::forward::Model::reset`'s doc comment).

use rocml_hip::DeviceBuffer;

use super::config::{GdnConfig, LayerKind, Qwen35Config};
use crate::cache::MAX_SEQ_CAP;
use crate::error::RocmlError;

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

pub struct AttnPlane {
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
}

impl AttnPlane {
    fn new(n_kv_heads: u32, max_seq: u32, head_dim: u32) -> Result<Self, RocmlError> {
        let plane_len = (n_kv_heads as usize) * (max_seq as usize) * (head_dim as usize);
        Ok(Self {
            k: DeviceBuffer::new(plane_len)?,
            v: DeviceBuffer::new(plane_len)?,
        })
    }

    /// Appends this step's `[n_kv_heads, head_dim]` k/v vectors at `pos`.
    pub fn append(
        &mut self,
        pos: u32,
        max_seq: u32,
        n_kv_heads: u32,
        head_dim: u32,
        k: &DeviceBuffer<f32>,
        v: &DeviceBuffer<f32>,
    ) -> Result<(), RocmlError> {
        let head_dim = head_dim as usize;
        let max_seq = max_seq as usize;
        for h in 0..n_kv_heads as usize {
            let dst_offset = h * max_seq * head_dim + pos as usize * head_dim;
            let src_offset = h * head_dim;
            self.k
                .copy_from_device(dst_offset, k, src_offset, head_dim)?;
            self.v
                .copy_from_device(dst_offset, v, src_offset, head_dim)?;
        }
        Ok(())
    }

    pub fn head_plane_offset(&self, kvh: u32, max_seq: u32, head_dim: u32) -> usize {
        kvh as usize * max_seq as usize * head_dim as usize
    }

    pub fn k_buffer(&self) -> &DeviceBuffer<f32> {
        &self.k
    }

    pub fn v_buffer(&self) -> &DeviceBuffer<f32> {
        &self.v
    }
}

pub struct HybridCache {
    max_seq: u32,
    gdn: Vec<Option<GdnLayerState>>,
    attn: Vec<Option<AttnPlane>>,
}

impl HybridCache {
    pub fn new(cfg: &Qwen35Config) -> Result<Self, RocmlError> {
        let max_seq = cfg.context_length.min(MAX_SEQ_CAP);
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

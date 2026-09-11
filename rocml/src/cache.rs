//! Per-layer KV cache, f32, one contiguous buffer per layer laid out as
//! `[kv_head][max_seq][head_dim]` — a per-(layer, kv head) plane is exactly
//! the `[cur_len, head_dim]` row-major slice `gemv_f32`/`gemv_t_f32` expect,
//! so decode attention needs no gather step, just a head-plane offset.

use rocml_hip::DeviceBuffer;

use crate::config::ModelConfig;
use crate::error::RocmlError;

/// Cap on cache length regardless of the model's own `context_length`, per
/// milestone scope (a later milestone can grow this / make it configurable).
pub const MAX_SEQ_CAP: u32 = 4096;

struct LayerCache {
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
}

pub struct KvCache {
    max_seq: u32,
    head_dim: u32,
    n_kv_heads: u32,
    layers: Vec<LayerCache>,
}

impl KvCache {
    pub fn new(config: &ModelConfig) -> Result<Self, RocmlError> {
        let max_seq = config.context_length.min(MAX_SEQ_CAP);
        let n_kv_heads = config.head_count_kv;
        let head_dim = config.head_dim;
        let plane_len = (n_kv_heads as usize) * (max_seq as usize) * (head_dim as usize);

        let mut layers = Vec::with_capacity(config.block_count as usize);
        for _ in 0..config.block_count {
            layers.push(LayerCache {
                k: DeviceBuffer::new(plane_len)?,
                v: DeviceBuffer::new(plane_len)?,
            });
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

    /// Appends one time step's k/v vectors — each `[n_kv_heads, head_dim]`
    /// contiguous (heads outermost) — at position `pos` for layer
    /// `layer_idx`.
    pub fn append(
        &mut self,
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
        let layer = self.layers.get_mut(layer_idx).ok_or_else(|| {
            RocmlError::Config(format!("cache: layer index {layer_idx} out of range"))
        })?;

        let head_dim = self.head_dim as usize;
        let max_seq = self.max_seq as usize;
        for h in 0..self.n_kv_heads as usize {
            let dst_offset = h * max_seq * head_dim + pos as usize * head_dim;
            let src_offset = h * head_dim;
            layer
                .k
                .copy_from_device(dst_offset, k, src_offset, head_dim)?;
            layer
                .v
                .copy_from_device(dst_offset, v, src_offset, head_dim)?;
        }
        Ok(())
    }

    /// Element offset where kv head `kvh`'s `[max_seq, head_dim]` plane
    /// starts within a layer's K/V buffer.
    pub fn head_plane_offset(&self, kvh: u32) -> usize {
        kvh as usize * self.max_seq as usize * self.head_dim as usize
    }

    pub fn k_buffer(&self, layer_idx: usize) -> Result<&DeviceBuffer<f32>, RocmlError> {
        self.layers.get(layer_idx).map(|l| &l.k).ok_or_else(|| {
            RocmlError::Config(format!("cache: layer index {layer_idx} out of range"))
        })
    }

    pub fn v_buffer(&self, layer_idx: usize) -> Result<&DeviceBuffer<f32>, RocmlError> {
        self.layers.get(layer_idx).map(|l| &l.v).ok_or_else(|| {
            RocmlError::Config(format!("cache: layer index {layer_idx} out of range"))
        })
    }
}

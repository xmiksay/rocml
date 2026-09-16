//! Mixture-of-experts FFN weights (qwen35moe, M1). Unlike every other
//! weight this crate loads, the routed-expert tensors (`ffn_{gate,up}_exps`/
//! `ffn_down_exps`, one 3D `[hidden, expert_ff, expert_count]`-shaped tensor
//! each) are **never uploaded to the GPU as a whole** — at 18.16 GiB for
//! Ornith-1.5-35B-A3B they alone exceed a 16GB card. Instead [`ExpertTensorMeta`]
//! keeps a zero-copy view into the GGUF's mmap and hands out one expert's
//! raw quantized bytes on demand (`expert_bytes`), which the forward pass
//! (`crate::qwen35::forward::moe`) copies into a small reusable device
//! staging buffer and runs through the same fused dequant-GEMV kernels
//! (`Kernels::gemv_quant`) every other quantized weight in this codebase
//! uses. This is the whole of M1's "expert offload": correct, simple,
//! optimized in a later milestone (an LRU VRAM-resident expert cache).
//!
//! GGUF's "ne" order stores a 3D `[n, m, experts]` tensor with `experts` as
//! the *slowest*-varying axis (see `rocml_core::gguf::tensor`'s module
//! doc), so each expert's own `[n, m]` 2D matrix (exactly `LinearWeight`'s
//! usual per-tensor shape) sits at a fixed contiguous byte offset:
//! `e * per_expert_bytes`. `hidden * expert_ff` is always a whole multiple
//! of every supported quant's block width here (both are already
//! block-width multiples individually — the same precondition
//! `LinearWeight::load` requires of every 2D weight), so no expert's byte
//! range ever splits a quant block across the expert boundary.

use std::cell::RefCell;

use rocml_core::gguf::GgufFile;
use rocml_core::quant::GgmlDType;
use rocml_hip::DeviceBuffer;

use super::ffn::FfnWeights;
use crate::error::RocmlError;
use crate::qwen35::config::MoeConfig;
use crate::weights::{load_matrix_f32, load_vector_f32};

/// One routed-expert tensor's metadata plus a zero-copy accessor for a
/// single expert's raw bytes. Never holds a GPU buffer itself.
pub struct ExpertTensorMeta {
    name: String,
    pub dtype: GgmlDType,
    /// Output rows (`gemv_quant`'s `m`).
    pub m: u32,
    /// Reduction width (`gemv_quant`'s `n`).
    pub n: u32,
    pub per_expert_bytes: usize,
    /// Best-effort host registration (`rocml_hip::host_register_readonly`)
    /// of this tensor's whole byte range, attempted once on first access and
    /// never retried — see `crate::qwen35::forward::moe`'s doc comment for
    /// why a failure here is never propagated as an error.
    registered: RefCell<bool>,
}

impl ExpertTensorMeta {
    fn load(
        gguf: &GgufFile,
        name: &str,
        expected_m: u32,
        expected_n: u32,
        expected_experts: u32,
    ) -> Result<Self, RocmlError> {
        let view = gguf.tensor(name)?;
        let shape = view.shape();
        let &[n, m, experts] = shape else {
            return Err(RocmlError::Config(format!(
                "tensor {name:?}: must be 3D ([hidden, expert_ff, expert_count] in ne order), \
                 got {shape:?}"
            )));
        };
        let to_u32 = |value: u64, label: &str| {
            u32::try_from(value).map_err(|_| {
                RocmlError::Config(format!("tensor {name:?}: {label} {value} overflows u32"))
            })
        };
        let (n, m, experts) = (
            to_u32(n, "n")?,
            to_u32(m, "m")?,
            to_u32(experts, "experts")?,
        );
        if (m, n, experts) != (expected_m, expected_n, expected_experts) {
            return Err(RocmlError::Config(format!(
                "tensor {name:?}: shape ({n} x {m} x {experts}) doesn't match expected \
                 ({expected_n} x {expected_m} x {expected_experts})"
            )));
        }

        let dtype = view.dtype();
        let block_elems = dtype.block_elements();
        if block_elems == 0 {
            return Err(RocmlError::Config(format!(
                "tensor {name:?}: dtype {dtype:?} has no block layout (unsupported for MoE \
                 expert weights)"
            )));
        }
        let per_expert_elems = m as usize * n as usize;
        if !per_expert_elems.is_multiple_of(block_elems) {
            return Err(RocmlError::Config(format!(
                "tensor {name:?}: per-expert element count {per_expert_elems} is not a \
                 multiple of {dtype:?}'s block width {block_elems}"
            )));
        }
        let per_expert_bytes = (per_expert_elems / block_elems) * dtype.block_bytes();
        let expected_total = per_expert_bytes
            .checked_mul(experts as usize)
            .ok_or_else(|| RocmlError::Config(format!("tensor {name:?}: byte size overflow")))?;
        if view.data().len() != expected_total {
            return Err(RocmlError::Config(format!(
                "tensor {name:?}: {} bytes on disk, expected {expected_total} \
                 ({per_expert_bytes} bytes/expert x {experts} experts)",
                view.data().len()
            )));
        }

        Ok(Self {
            name: name.to_string(),
            dtype,
            m,
            n,
            per_expert_bytes,
            registered: RefCell::new(false),
        })
    }

    fn ensure_registered(&self, gguf: &GgufFile) {
        if *self.registered.borrow() {
            return;
        }
        // Set first, unconditionally: a registration attempt is at most
        // once per tensor regardless of outcome (see the module doc — a
        // failure here never blocks the plain hipMemcpy fallback).
        *self.registered.borrow_mut() = true;
        if let Ok(view) = gguf.tensor(&self.name) {
            let data = view.data();
            rocml_hip::host_register_readonly(data.as_ptr(), data.len());
        }
    }

    /// This expert's raw quantized bytes, straight out of the GGUF's mmap —
    /// exactly `per_expert_bytes` long, ready to copy H2D and hand to
    /// `Kernels::gemv_quant`/`gemv_quant`'s matching `dtype`.
    pub fn expert_bytes<'a>(
        &self,
        gguf: &'a GgufFile,
        expert: u32,
    ) -> Result<&'a [u8], RocmlError> {
        self.ensure_registered(gguf);
        let view = gguf.tensor(&self.name)?;
        let data = view.data();
        let start = expert as usize * self.per_expert_bytes;
        let end = start + self.per_expert_bytes;
        data.get(start..end).ok_or_else(|| {
            RocmlError::Config(format!(
                "expert {expert} out of range for tensor {:?} ({} bytes, {} bytes/expert)",
                self.name,
                data.len(),
                self.per_expert_bytes
            ))
        })
    }
}

/// One qwen35moe layer's FFN weights: the router, the always-on shared
/// expert (a small dense SwiGLU FFN, uploaded normally like every other
/// linear layer), and the three routed-expert tensor views.
pub struct MoeFfnWeights {
    /// (m=expert_count, n=hidden) — `ffn_gate_inp.weight`.
    pub router: DeviceBuffer<f32>,
    /// (n=hidden,) dot-product vector — `ffn_gate_inp_shexp.weight`.
    pub shared_gate: DeviceBuffer<f32>,
    pub shared: FfnWeights,
    pub gate: ExpertTensorMeta,
    pub up: ExpertTensorMeta,
    pub down: ExpertTensorMeta,
}

impl MoeFfnWeights {
    pub(crate) fn load(
        gguf: &GgufFile,
        prefix: &str,
        cfg: &MoeConfig,
        hidden: u32,
    ) -> Result<Self, RocmlError> {
        let router = load_matrix_f32(
            gguf,
            &format!("{prefix}.ffn_gate_inp.weight"),
            cfg.expert_count,
            hidden,
        )?;
        let shared_gate =
            load_vector_f32(gguf, &format!("{prefix}.ffn_gate_inp_shexp.weight"), hidden)?;
        let shared = FfnWeights::load_named(
            gguf,
            hidden,
            cfg.shared_ff_len,
            &format!("{prefix}.ffn_gate_shexp.weight"),
            &format!("{prefix}.ffn_up_shexp.weight"),
            &format!("{prefix}.ffn_down_shexp.weight"),
        )?;
        let gate = ExpertTensorMeta::load(
            gguf,
            &format!("{prefix}.ffn_gate_exps.weight"),
            cfg.expert_ff_len,
            hidden,
            cfg.expert_count,
        )?;
        let up = ExpertTensorMeta::load(
            gguf,
            &format!("{prefix}.ffn_up_exps.weight"),
            cfg.expert_ff_len,
            hidden,
            cfg.expert_count,
        )?;
        let down = ExpertTensorMeta::load(
            gguf,
            &format!("{prefix}.ffn_down_exps.weight"),
            hidden,
            cfg.expert_ff_len,
            cfg.expert_count,
        )?;

        Ok(Self {
            router,
            shared_gate,
            shared,
            gate,
            up,
            down,
        })
    }

    /// Largest of this layer's three expert tensors' per-expert byte size —
    /// used by `MoeScratch::new` to size the reusable staging buffers.
    pub(crate) fn max_expert_bytes(&self) -> usize {
        self.gate
            .per_expert_bytes
            .max(self.up.per_expert_bytes)
            .max(self.down.per_expert_bytes)
    }
}

/// Pins `ExpertTensorMeta`'s zero-copy per-expert byte-slicing arithmetic
/// against the real Ornith-1.5-35B-A3B checkpoint — no GPU needed (pure
/// GGUF/mmap reads), so this runs in the default `cargo test --workspace`
/// pass, skipping itself if the checkpoint isn't present (same convention
/// as `weights::linear::tests`).
#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::Qwen35Config;
    use rocml_core::testpaths::checkpoint;

    const GGUF_REL: &str = "Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf";

    fn load_gate_meta(gguf: &GgufFile, cfg: &Qwen35Config) -> ExpertTensorMeta {
        let moe = cfg.moe.expect("qwen35moe checkpoint");
        ExpertTensorMeta::load(
            gguf,
            "blk.0.ffn_gate_exps.weight",
            moe.expert_ff_len,
            cfg.embedding_length,
            moe.expert_count,
        )
        .expect("ExpertTensorMeta::load")
    }

    #[test]
    fn expert_slices_are_correctly_sized_and_non_overlapping() {
        let Some(path) = checkpoint(GGUF_REL) else {
            return;
        };
        let gguf = GgufFile::open(&path).expect("open gguf");
        let cfg = Qwen35Config::from_gguf(&gguf).expect("parse config");
        let meta = load_gate_meta(&gguf, &cfg);
        let moe = cfg.moe.unwrap();

        let expected_per_expert =
            (moe.expert_ff_len as usize * cfg.embedding_length as usize / 256) * 144; // Q4_K
        assert_eq!(meta.per_expert_bytes, expected_per_expert);

        let full = gguf.tensor("blk.0.ffn_gate_exps.weight").unwrap();
        assert_eq!(
            full.data().len(),
            meta.per_expert_bytes * moe.expert_count as usize
        );

        // A handful of experts across the range: each slice is exactly
        // `per_expert_bytes` long, matches a manually-computed offset into
        // the tensor's raw data, and distinct experts never overlap.
        let sample_experts = [0u32, 1, 42, 128, moe.expert_count - 1];
        let mut slices = Vec::new();
        for &e in &sample_experts {
            let got = meta.expert_bytes(&gguf, e).expect("expert_bytes");
            assert_eq!(got.len(), meta.per_expert_bytes);
            let start = e as usize * meta.per_expert_bytes;
            let want = &full.data()[start..start + meta.per_expert_bytes];
            assert_eq!(got, want, "expert {e}: byte range mismatch");
            slices.push((start, start + meta.per_expert_bytes));
        }
        for i in 0..slices.len() {
            for j in (i + 1)..slices.len() {
                let (a0, a1) = slices[i];
                let (b0, b1) = slices[j];
                assert!(a1 <= b0 || b1 <= a0, "expert byte ranges must not overlap");
            }
        }
    }

    #[test]
    fn expert_bytes_rejects_out_of_range_index() {
        let Some(path) = checkpoint(GGUF_REL) else {
            return;
        };
        let gguf = GgufFile::open(&path).expect("open gguf");
        let cfg = Qwen35Config::from_gguf(&gguf).expect("parse config");
        let meta = load_gate_meta(&gguf, &cfg);
        let moe = cfg.moe.unwrap();
        assert!(meta.expert_bytes(&gguf, moe.expert_count).is_err());
    }

    /// Host registration is best-effort by design (see the `registered`
    /// field's doc comment) — this pins that calling `expert_bytes` twice
    /// (the second hitting the "already tried" short-circuit) never panics
    /// or errors regardless of whether the first attempt actually succeeded.
    #[test]
    fn repeated_access_does_not_reregister_or_fail() {
        let Some(path) = checkpoint(GGUF_REL) else {
            return;
        };
        let gguf = GgufFile::open(&path).expect("open gguf");
        let cfg = Qwen35Config::from_gguf(&gguf).expect("parse config");
        let meta = load_gate_meta(&gguf, &cfg);
        for _ in 0..3 {
            assert!(meta.expert_bytes(&gguf, 0).is_ok());
        }
    }
}

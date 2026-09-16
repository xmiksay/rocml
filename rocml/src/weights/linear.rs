//! `LinearWeight`: a linear/matmul layer's weight matrix, either fully
//! dequantized to f16 (the pre-milestone-5b behavior) or kept in its raw
//! GGUF quantized block format in VRAM and matvec'd through a fused
//! dequant-GEMV kernel. Shared by both the dense Qwen3 and qwen3.5 hybrid
//! weight loaders/forward passes so the dispatch logic exists exactly once.
//!
//! Loader policy: a tensor whose dtype is Q8_0/Q4_K/Q5_K/Q6_K and whose row
//! length `n` is a multiple of that kernel's block width uploads as raw
//! bytes and runs through the matching fused kernel — this is what lets a
//! Q6_K model like Ornith-1.0-9B fit in 16GB VRAM at all (dequant-to-f16
//! would not). Every other case (F32/F16/BF16, an unsupported quant, or a
//! quant whose `n` doesn't divide the kernel's block width) falls back to
//! the original CPU-dequant-to-f32-then-f16-upload path.

use half::f16;
use rocml_core::gguf::GgufFile;
use rocml_core::quant::{dequantize, GgmlDType};
use rocml_hip::DeviceBuffer;

use super::matrix_dims;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, DevPtr, Kernels, MmqScratch, SplitKScratch};

/// One linear layer's weight matrix, ready for `y = W * x` via [`Self::matvec`].
pub enum LinearWeight {
    /// Fully dequantized, f16-cast, row-major `m x n`.
    F16(DeviceBuffer<f16>),
    /// Raw GGUF quantized blocks, row-major: each of the `m` rows is
    /// `n / dtype.block_elements()` contiguous on-disk blocks, uploaded
    /// byte-for-byte with no CPU-side transformation.
    Quant {
        dtype: GgmlDType,
        raw: DeviceBuffer<u8>,
        /// Whether this weight's `matmul` may route through the int8 MMQ
        /// path when `LoadOptions::use_mmq` is on (see [`Self::load`]'s doc
        /// for which tensor names are excluded and why) — `false` always
        /// falls back to WMMA/scalar regardless of the load-time flag.
        mmq_eligible: bool,
    },
}

/// Block-element width the fused kernel for `dtype` requires `n` to divide,
/// or `None` if `dtype` has no fused matvec kernel (falls back to dequant).
fn quant_block_elems(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q8_0 => Some(32),
        GgmlDType::Q4_K | GgmlDType::Q5_K | GgmlDType::Q6_K => Some(256),
        _ => None,
    }
}

/// **MMQ outlier-channel exclusion** (`mmq_precision` investigation,
/// follow-up to the int8-MMQ-integration round): two projections in every
/// layer read a SiLU/SwiGLU-gated activation whose per-32-element-block
/// outlier structure is measurably worse than the plain RMSNorm'd
/// activations (`attn_qkv`'s own input) feeding every other MMQ-eligible
/// projection (p50~3, p90~4-5.5) — the per-layer diff harness
/// (`rocml/tests/mmq_layer_diff.rs`) measured real Qwen3.5-2B block
/// `amax/mean(|x|)` ratios of p50=5.7-7.3/p90=13.8-16.7 (worst blocks over
/// 30x) for `qwen35`'s Gated Delta Net `ssm_out` projection's input
/// (`gdn_y = ssm_norm(v) * SiLU(z)`) and p50~4.6/p90~7.8 for `ffn_down`'s
/// SwiGLU-gated input. Per-block int8 quantization is inherently lossy
/// when one channel dominates a block's absmax — the other channels lose
/// most of their effective resolution — and `ssm_out` is exactly where the
/// MMQ-vs-WMMA per-layer diff shows the single biggest jump in relative
/// error anywhere in the model (mean relative error 1.5%->9.1% through
/// that one matmul alone, vs. `attn_qkv`'s ~0->2.4% on a bit-identical
/// input). Both `ssm_out` and `ffn_down` are therefore always excluded
/// from the int8 MMQ path regardless of `LoadOptions::use_mmq`, falling
/// back to WMMA/scalar like every non-eligible shape already does — see
/// `LinearWeight::matmul`'s `mmq_eligible` field. **Not sufficient on its
/// own**: `mmq-endtoend-measure` shows excluding only `ssm_out` barely
/// moves `qwen35_chunked_prefill_parity`'s own final-logits comparison
/// (12.65%-14.91% max relative error vs. the unmodified path's
/// 12.4%-15.1%); adding `ffn_down` helps more (8.46%-10.24%) but still
/// leaves the error 8-10x over that gate's `1e-2` tolerance — every other
/// MMQ-eligible matmul's own milder-but-nonzero outlier ratio compounds
/// over 24 layers regardless. `LoadOptions::use_mmq`/`--mmq` stays off by
/// default; this exclusion is a real, zero-risk-by-default precision
/// improvement for whichever future round revisits MMQ, not a fix for the
/// underlying regression.
fn mmq_eligible_by_name(name: &str) -> bool {
    !(name.ends_with(".ssm_out.weight") || name.ends_with(".ffn_down.weight"))
}

impl LinearWeight {
    /// Loads tensor `name`, validating its shape is exactly `(expected_m,
    /// expected_n)` in `gemv_f16`/`gemv_<quant>` terms (`m` output rows, `n`
    /// reduction width) — see the module doc for the quantized-vs-fallback
    /// policy.
    pub fn load(
        gguf: &GgufFile,
        name: &str,
        expected_m: u32,
        expected_n: u32,
    ) -> Result<Self, RocmlError> {
        let view = gguf.tensor(name)?;
        let (m, n) = matrix_dims(view.shape(), name)?;
        if (m, n) != (expected_m, expected_n) {
            return Err(RocmlError::Config(format!(
                "tensor {name:?}: shape ({m} x {n}) doesn't match expected ({expected_m} x {expected_n})"
            )));
        }

        let dtype = view.dtype();
        if let Some(block_elems) = quant_block_elems(dtype) {
            if (n as usize).is_multiple_of(block_elems) {
                let bytes = view.data();
                let mut raw = DeviceBuffer::<u8>::new(bytes.len())?;
                raw.copy_from_host(bytes)?;
                return Ok(Self::Quant {
                    dtype,
                    raw,
                    mmq_eligible: mmq_eligible_by_name(name),
                });
            }
        }

        let f32_data = dequantize(dtype, view.data())?;
        let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
        let mut buf = DeviceBuffer::<f16>::new(f16_data.len())?;
        buf.copy_from_host(&f16_data)?;
        Ok(Self::F16(buf))
    }

    /// Exact on-device byte size of this weight matrix — the raw quantized
    /// block bytes for `Quant`, or `2 * elements` for the f16 fallback.
    /// Used by `rocml::profile::cost::matvec_bytes` so the profiler's byte
    /// accounting reflects what's actually resident in VRAM (e.g. Q6_K's
    /// ~6.5 bits/weight, not a guessed constant) rather than re-deriving it
    /// from dtype metadata.
    pub fn byte_size(&self) -> u64 {
        match self {
            Self::F16(buf) => buf.len() as u64 * 2,
            Self::Quant { raw, .. } => raw.len() as u64,
        }
    }

    /// `y = W * x`: dispatches to the plain f16 gemv or the matching fused
    /// dequant-gemv kernel. `m`/`n` must match the shape this weight was
    /// loaded with (the same contract `Kernels::gemv_f16` already has).
    pub fn matvec(
        &self,
        kernels: &Kernels,
        x: DevPtr,
        y: DevPtr,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        match self {
            Self::F16(buf) => kernels.gemv_f16(offset(buf, 0), x, y, m, n),
            Self::Quant { dtype, raw, .. } => {
                kernels.gemv_quant(*dtype, offset(raw, 0), x, y, m, n)
            }
        }
    }

    /// `out[rows,m] = X[rows,n] * W^T`: the batched prefill-path sibling of
    /// [`Self::matvec`], dispatching to `gemm_xwt_f16`/`gemm_xwt_<quant>`.
    /// `m`/`n` must match the shape this weight was loaded with, same as
    /// `matvec`. `mmq_scratch` is only read for a `Quant` weight whose
    /// shape/dtype/load-time flag route it through the int8 MMQ path
    /// (`QuantKernels::gemm`'s dispatch) — every other case (including the
    /// `F16` arm) ignores it; callers always pass their chunk's scratch
    /// regardless, so every call site stays uniform (see
    /// `qwen35::forward::chunk_scratch::ChunkScratch::mmq_scratch`).
    /// `splitk_scratch` is the same always-pass-it-uniformly story for the
    /// split-K WMMA path (`ChunkScratch::splitk_scratch`) — only read when
    /// the shape's grid is narrow enough for `QuantKernels::gemm`'s dispatch
    /// to pick a split count `> 1`.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        kernels: &Kernels,
        x: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
        mmq_scratch: MmqScratch,
        splitk_scratch: SplitKScratch,
    ) -> Result<(), RocmlError> {
        match self {
            Self::F16(buf) => kernels.gemm_xwt_f16(x, offset(buf, 0), out, rows, m, n),
            Self::Quant {
                dtype,
                raw,
                mmq_eligible,
            } => kernels.gemm_quant(
                *dtype,
                x,
                offset(raw, 0),
                out,
                rows,
                m,
                n,
                mmq_scratch,
                *mmq_eligible,
                splitk_scratch,
                // Micro-tile WMMA (qwen35moe M4 lever 1) is opt-in per call
                // site, not per shape — every non-MoE caller reaches this
                // `matmul`, so it stays off here; only qwen35moe's grouped
                // GEMM (`moe_chunk.rs`) requests it directly.
                false,
            ),
        }
    }
}

/// Per-layer spot checks: `LinearWeight::load` + `matvec` against a handful
/// of real GGUF tensors (dense Qwen3, plus one qwen3.5 GDN layer and one
/// full-attention layer — the two hybrid layer kinds), each compared to
/// `rocml-core`'s proven-correct CPU dequant + a CPU dot product. This is
/// the loader/dispatch layer these tests exist to cover; the fused kernels
/// themselves already have dedicated synthetic + real-tensor coverage in
/// `rocml-kernels/tests/gemv_q*.rs`. Real hardware + real checkpoints
/// required; each test skips itself if its checkpoint isn't present on this
/// machine (same convention as `rocml/tests/*_greedy_parity.rs`).
#[cfg(test)]
mod tests {
    use std::path::Path;

    use rocml_core::testpaths::checkpoint;
    use rocml_hip::{Device, DeviceBuffer};

    use super::*;
    use crate::forward::kernels::{offset, Kernels};
    use crate::qwen35::config::{LayerKind, Qwen35Config};

    const QWEN3_GGUF_REL: &str = "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
    const QWEN35_GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
    const REL_TOL: f32 = 2e-3;

    fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
        for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
            let diff = (got - want).abs();
            let tol = REL_TOL * want.abs().max(1.0);
            assert!(
                diff <= tol,
                "{label}[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
            );
        }
    }

    /// CPU reference: dequantize each of `w_bytes`'s `m` rows independently
    /// (rocml-core's proven-correct dequant) and dot it against `x` with
    /// exact f64 accumulation.
    fn expected_gemv(dtype: GgmlDType, w_bytes: &[u8], x: &[f32], m: usize) -> Vec<f32> {
        let row_bytes = w_bytes.len() / m;
        (0..m)
            .map(|row| {
                let row_slice = &w_bytes[row * row_bytes..(row + 1) * row_bytes];
                let deq = dequantize(dtype, row_slice).expect("cpu dequantize failed");
                deq.iter()
                    .zip(x)
                    .map(|(a, b)| *a as f64 * *b as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    }

    /// Loads `tensor_name` from `gguf_path` through `LinearWeight::load` (its
    /// own shape, so the shape check is a no-op — this test is about the
    /// matvec dispatch, not that validation) and asserts its `matvec` output
    /// matches the CPU dequant+dot reference.
    fn spot_check(gguf_path: &Path, tensor_name: &str) {
        let _device = Device::new(0).expect("failed to select device 0");
        let gguf = GgufFile::open(gguf_path).expect("open GGUF");
        let view = gguf.tensor(tensor_name).expect("tensor not found");
        let &[n, m] = view.shape() else {
            panic!("expected a 2D tensor, got shape {:?}", view.shape());
        };
        let (m, n) = (m as u32, n as u32);

        let weight = LinearWeight::load(&gguf, tensor_name, m, n).expect("LinearWeight::load");
        let kernels = Kernels::load_all(false).expect("Kernels::load_all");

        // Realistic activation magnitudes (rmsnorm output ranges to ~+-4).
        let x_host: Vec<f32> = (0..n).map(|i| ((i % 29) as f32) * 0.29 - 4.0).collect();
        let mut x = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc x");
        x.copy_from_host(&x_host).expect("copy x");
        let y = DeviceBuffer::<f32>::new(m as usize).expect("hipMalloc y");

        weight
            .matvec(&kernels, offset(&x, 0), offset(&y, 0), m, n)
            .expect("matvec failed");

        let mut actual = vec![0.0f32; m as usize];
        y.copy_to_host(&mut actual).expect("copy y");

        let expected = expected_gemv(view.dtype(), view.data(), &x_host, m as usize);
        let mut max_abs = 0f32;
        let mut max_rel = 0f32;
        for (got, want) in actual.iter().zip(&expected) {
            let d = (got - want).abs();
            max_abs = max_abs.max(d);
            max_rel = max_rel.max(d / want.abs().max(1e-3));
        }
        println!("{tensor_name}: max_abs_err = {max_abs:.6e}, max_rel_err = {max_rel:.6e}");
        assert_close(&actual, &expected, tensor_name);
    }

    /// First layer index of `kind` in the qwen35 hybrid architecture's own
    /// layer-kind table — avoids hardcoding which block index is GDN vs.
    /// full-attention (a modeling detail, not something this test should
    /// assume).
    fn qwen35_layer_index(gguf: &GgufFile, kind: LayerKind) -> u32 {
        let cfg = Qwen35Config::from_gguf(gguf).expect("qwen35 config");
        cfg.layer_kinds
            .iter()
            .position(|&k| k == kind)
            .unwrap_or_else(|| panic!("no {kind:?} layer found")) as u32
    }

    #[test]
    fn dense_layer_matvec_matches_cpu_reference() {
        let Some(path) = checkpoint(QWEN3_GGUF_REL) else {
            return;
        };
        spot_check(&path, "blk.0.attn_q.weight");
    }

    #[test]
    fn ornith_q6k_lm_head_matvec_matches_cpu_reference() {
        let Some(path) = checkpoint("Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf") else {
            return;
        };
        spot_check(&path, "output.weight");
    }

    #[test]
    fn ornith_q6k_qkv_matvec_matches_cpu_reference() {
        let Some(path) = checkpoint("Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf") else {
            return;
        };
        spot_check(&path, "blk.0.attn_qkv.weight");
    }

    #[test]
    fn ornith_q6k_ffn_matvec_matches_cpu_reference() {
        let Some(path) = checkpoint("Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf") else {
            return;
        };
        spot_check(&path, "blk.0.ffn_down.weight");
    }

    #[test]
    fn ornith_q4km_qkv_matvec_matches_cpu_reference() {
        let Some(path) = checkpoint("Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf") else {
            return;
        };
        spot_check(&path, "blk.0.attn_qkv.weight");
    }

    #[test]
    fn qwen35_gdn_layer_matvec_matches_cpu_reference() {
        let Some(path) = checkpoint(QWEN35_GGUF_REL) else {
            return;
        };
        let gguf = GgufFile::open(&path).expect("open GGUF");
        let idx = qwen35_layer_index(&gguf, LayerKind::LinearAttention);
        spot_check(&path, &format!("blk.{idx}.attn_gate.weight"));
    }

    #[test]
    fn qwen35_attention_layer_matvec_matches_cpu_reference() {
        let Some(path) = checkpoint(QWEN35_GGUF_REL) else {
            return;
        };
        let gguf = GgufFile::open(&path).expect("open GGUF");
        let idx = qwen35_layer_index(&gguf, LayerKind::FullAttention);
        spot_check(&path, &format!("blk.{idx}.attn_output.weight"));
    }
}

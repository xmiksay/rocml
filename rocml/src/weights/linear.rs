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
use crate::forward::kernels::{offset, DevPtr, Kernels};

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
                return Ok(Self::Quant { dtype, raw });
            }
        }

        let f32_data = dequantize(dtype, view.data())?;
        let f16_data: Vec<f16> = f32_data.iter().map(|&v| f16::from_f32(v)).collect();
        let mut buf = DeviceBuffer::<f16>::new(f16_data.len())?;
        buf.copy_from_host(&f16_data)?;
        Ok(Self::F16(buf))
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
            Self::Quant { dtype, raw } => kernels.gemv_quant(*dtype, offset(raw, 0), x, y, m, n),
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

    use rocml_hip::{Device, DeviceBuffer};

    use super::*;
    use crate::forward::kernels::{offset, Kernels};
    use crate::qwen35::config::{LayerKind, Qwen35Config};

    const QWEN3_GGUF: &str = "/mnt/nvme/miksa/checkpoints/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
    const QWEN35_GGUF: &str = "/mnt/nvme/miksa/checkpoints/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
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
    /// (rocml-core's proven-correct dequant) and dot it against `x`.
    fn expected_gemv(dtype: GgmlDType, w_bytes: &[u8], x: &[f32], m: usize) -> Vec<f32> {
        let row_bytes = w_bytes.len() / m;
        (0..m)
            .map(|row| {
                let row_slice = &w_bytes[row * row_bytes..(row + 1) * row_bytes];
                let deq = dequantize(dtype, row_slice).expect("cpu dequantize failed");
                deq.iter().zip(x).map(|(a, b)| a * b).sum()
            })
            .collect()
    }

    /// Loads `tensor_name` from `gguf_path` through `LinearWeight::load` (its
    /// own shape, so the shape check is a no-op — this test is about the
    /// matvec dispatch, not that validation) and asserts its `matvec` output
    /// matches the CPU dequant+dot reference.
    fn spot_check(gguf_path: &str, tensor_name: &str) {
        if !Path::new(gguf_path).exists() {
            eprintln!("skipping: {gguf_path} not present on this machine");
            return;
        }
        let _device = Device::new(0).expect("failed to select device 0");
        let gguf = GgufFile::open(gguf_path).expect("open GGUF");
        let view = gguf.tensor(tensor_name).expect("tensor not found");
        let &[n, m] = view.shape() else {
            panic!("expected a 2D tensor, got shape {:?}", view.shape());
        };
        let (m, n) = (m as u32, n as u32);

        let weight = LinearWeight::load(&gguf, tensor_name, m, n).expect("LinearWeight::load");
        let kernels = Kernels::load_all().expect("Kernels::load_all");

        let x_host: Vec<f32> = (0..n).map(|i| ((i % 29) as f32) * 0.037 - 0.5).collect();
        let mut x = DeviceBuffer::<f32>::new(n as usize).expect("hipMalloc x");
        x.copy_from_host(&x_host).expect("copy x");
        let y = DeviceBuffer::<f32>::new(m as usize).expect("hipMalloc y");

        weight
            .matvec(&kernels, offset(&x, 0), offset(&y, 0), m, n)
            .expect("matvec failed");

        let mut actual = vec![0.0f32; m as usize];
        y.copy_to_host(&mut actual).expect("copy y");

        let expected = expected_gemv(view.dtype(), view.data(), &x_host, m as usize);
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
        spot_check(QWEN3_GGUF, "blk.0.attn_q.weight");
    }

    #[test]
    fn qwen35_gdn_layer_matvec_matches_cpu_reference() {
        if !Path::new(QWEN35_GGUF).exists() {
            eprintln!("skipping: {QWEN35_GGUF} not present on this machine");
            return;
        }
        let gguf = GgufFile::open(QWEN35_GGUF).expect("open GGUF");
        let idx = qwen35_layer_index(&gguf, LayerKind::LinearAttention);
        spot_check(QWEN35_GGUF, &format!("blk.{idx}.attn_gate.weight"));
    }

    #[test]
    fn qwen35_attention_layer_matvec_matches_cpu_reference() {
        if !Path::new(QWEN35_GGUF).exists() {
            eprintln!("skipping: {QWEN35_GGUF} not present on this machine");
            return;
        }
        let gguf = GgufFile::open(QWEN35_GGUF).expect("open GGUF");
        let idx = qwen35_layer_index(&gguf, LayerKind::FullAttention);
        spot_check(QWEN35_GGUF, &format!("blk.{idx}.attn_output.weight"));
    }
}

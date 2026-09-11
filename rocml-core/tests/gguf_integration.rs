//! Integration tests against the two real model files this milestone
//! targets. Each test is a no-op (with a message) if the file isn't present
//! on the machine running the suite, so CI/other checkouts aren't broken.

use rocml_core::gguf::GgufFile;
use rocml_core::quant::{dequantize, GgmlDType};
use rocml_core::testpaths::checkpoint;

const QWEN35_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const ORNITH_REL: &str = "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf";

/// The embedding tensor name used by both files under test (both are
/// `qwen35`-architecture models sharing llama.cpp's standard tensor naming).
const EMBEDDING_TENSOR: &str = "token_embd.weight";

fn check_model_file(rel: &str, expected_quant_dtype: GgmlDType) {
    let Some(path) = checkpoint(rel) else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("parse real GGUF file");

    let arch = gguf
        .get_str("general.architecture")
        .expect("general.architecture");
    assert!(!arch.is_empty());

    assert!(!gguf.tensors().is_empty(), "expected at least one tensor");

    let embed = gguf
        .tensor(EMBEDDING_TENSOR)
        .unwrap_or_else(|e| panic!("resolve {EMBEDDING_TENSOR}: {e}"));
    // ne-order: ne[0] is embedding_length (innermost), ne[1] is vocab size.
    assert_eq!(embed.shape().len(), 2, "expected a 2D embedding matrix");
    assert!(embed.shape().iter().all(|&d| d > 0));

    // Find some quantized tensor of the dtype this file is quantized to and
    // confirm it dequantizes to finite values.
    let quant_tensor = gguf
        .tensors()
        .iter()
        .find(|t| t.dtype == expected_quant_dtype)
        .unwrap_or_else(|| panic!("expected at least one {expected_quant_dtype:?} tensor"));
    let view = gguf.tensor(&quant_tensor.name).expect("resolve tensor");
    let values = dequantize(view.dtype(), view.data()).expect("dequantize");
    assert!(!values.is_empty());
    assert!(
        values.iter().all(|v| v.is_finite()),
        "dequantized {} contains non-finite values",
        quant_tensor.name
    );
}

#[test]
fn qwen35_2b_gguf_parses_and_dequantizes() {
    check_model_file(QWEN35_REL, GgmlDType::Q8_0);
}

#[test]
fn ornith_9b_gguf_parses_and_dequantizes() {
    check_model_file(ORNITH_REL, GgmlDType::Q6_K);
}

#[test]
fn qwen35_2b_architecture_is_qwen35_hybrid() {
    // Documents a real-file metadata quirk: both test files use ggml's
    // "qwen35" arch id for a hybrid attention/SSM (Mamba-style) model, not
    // a plain transformer — `qwen35.ssm.*` keys sit alongside the usual
    // `qwen35.attention.*` ones, and `full_attention_interval` says how
    // often a full-attention layer appears among the SSM layers.
    let Some(path) = checkpoint(QWEN35_REL) else {
        return;
    };
    let gguf = GgufFile::open(&path).unwrap();
    assert_eq!(gguf.get_str("general.architecture").unwrap(), "qwen35");
    assert!(gguf.get_u32("qwen35.ssm.state_size").is_ok());
    assert!(gguf.get_u32("qwen35.full_attention_interval").is_ok());
}

#[test]
fn all_k_quant_and_q8_0_tensors_dequantize_to_finite_values() {
    // Broader sweep than the single-tensor check above: every quantized
    // tensor this crate claims to support, across both real files, must
    // dequantize cleanly. This is the strongest real-data guarantee we have
    // that the block layouts/math match ggml's before any GPU kernel exists
    // to cross-check against. Capped at 25 tensors/file: an exhaustive sweep
    // over a 9B-parameter GGUF in an unoptimized debug build takes minutes
    // for no extra confidence once a couple dozen distinct layers/shapes
    // have passed.
    const MAX_TENSORS_PER_FILE: usize = 25;
    for rel in [QWEN35_REL, ORNITH_REL] {
        let Some(path) = checkpoint(rel) else {
            continue;
        };
        let gguf = GgufFile::open(&path).unwrap();
        let mut checked = 0usize;
        for info in gguf.tensors() {
            if checked >= MAX_TENSORS_PER_FILE {
                break;
            }
            if matches!(
                info.dtype,
                GgmlDType::Q8_0
                    | GgmlDType::Q2_K
                    | GgmlDType::Q3_K
                    | GgmlDType::Q4_K
                    | GgmlDType::Q5_K
                    | GgmlDType::Q6_K
            ) {
                let view = gguf.tensor(&info.name).unwrap();
                let values = dequantize(view.dtype(), view.data()).unwrap();
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "{}: tensor {} ({:?}) dequantized to a non-finite value",
                    path.display(),
                    info.name,
                    info.dtype
                );
                checked += 1;
            }
        }
        assert!(
            checked > 0,
            "{}: no quantized tensors found to check",
            path.display()
        );
    }
}

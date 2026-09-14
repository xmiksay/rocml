//! Issue #4: pins the quant-policy audit's actual findings against the real
//! registry checkpoints, so a future GGUF re-quant/re-upload that silently
//! changes the mix gets caught here rather than only in `docs/quant-policy.md`
//! prose. Pure GGUF-header/tensor-table reads (mmap, no dequant, no GPU) —
//! cheap enough to run in the default `cargo test --workspace` pass, unlike
//! the real-model decode tests `make test-model` gates separately.

use rocml::quant_policy::{audit, ArchFamily};
use rocml_core::gguf::GgufFile;
use rocml_core::quant::GgmlDType;
use rocml_core::testpaths::checkpoint;

/// Every 1D scan parameter and the conv1d weight must be full float in
/// every registry GGUF, regardless of quant scheme — this is the one
/// finding that must never regress silently (see docs/quant-policy.md).
fn assert_scan_params_are_float(gguf: &GgufFile) {
    for name in [
        "blk.0.ssm_a",
        "blk.0.ssm_dt.bias",
        "blk.0.ssm_conv1d.weight",
    ] {
        let dtype = gguf
            .tensor(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .dtype();
        assert_eq!(dtype, GgmlDType::F32, "{name} must stay F32");
    }
}

#[test]
fn ornith_q4_k_m_audit_matches_known_findings() {
    let Some(path) = checkpoint("Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    assert_scan_params_are_float(&gguf);

    let report = audit(&gguf, ArchFamily::Qwen35Hybrid);
    // Documented policy-vs-eval disagreement (docs/quant-policy.md): the
    // GDN 2D in-projections and ssm_out are generic Q4_K in this scheme, so
    // this checkpoint is NOT clean against the policy — that's the known,
    // reconciled finding, not a regression.
    assert!(
        !report.is_clean(),
        "expected the documented Q4_K_M GDN in-projection/ssm_out/attn_output/token_embd \
         violations; audit came back clean instead — did the file change?"
    );
    let labels: Vec<&str> = report.violations.iter().map(|v| v.label).collect();
    for expected in [
        "GDN alpha in-projection (2D, feeds decay logit)",
        "GDN beta in-projection (2D, feeds write-strength logit)",
        "ssm_out (GDN's down_proj analog, shapes residual stream)",
        "embedding table",
    ] {
        assert!(
            labels.contains(&expected),
            "expected a violation labeled {expected:?}, got {labels:?}"
        );
    }
    // lm_head is the one thing this scheme already bumps to Q6_K.
    assert_eq!(
        gguf.tensor("output.weight").unwrap().dtype(),
        GgmlDType::Q6_K
    );
}

#[test]
fn ornith_q6_k_audit_is_clean() {
    let Some(path) = checkpoint("Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    assert_scan_params_are_float(&gguf);

    let report = audit(&gguf, ArchFamily::Qwen35Hybrid);
    assert!(
        report.is_clean(),
        "expected a uniform-Q6_K checkpoint to pass every floor, got: {:?}",
        report.violations
    );
    assert!(
        report.checked > 0,
        "audit should have classified some tensors"
    );
}

#[test]
fn qwen35_2b_q8_0_audit_is_clean() {
    let Some(path) = checkpoint("Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    assert_scan_params_are_float(&gguf);

    let report = audit(&gguf, ArchFamily::Qwen35Hybrid);
    assert!(
        report.is_clean(),
        "expected a uniform-Q8_0 checkpoint to pass every floor, got: {:?}",
        report.violations
    );
}

#[test]
fn dense_qwen3_audit_has_no_gdn_tensors() {
    let Some(path) = checkpoint("Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf") else {
        return;
    };
    let gguf = GgufFile::open(&path).expect("open gguf");
    let report = audit(&gguf, ArchFamily::Qwen3Dense);
    assert!(
        report.is_clean(),
        "dense Qwen3-0.6B-Q8_0 should pass every common-rule floor, got: {:?}",
        report.violations
    );
    // No `ssm_*`/`attn_qkv` tensors exist on the dense architecture at all.
    assert!(gguf.tensor("blk.0.ssm_a").is_err());
}

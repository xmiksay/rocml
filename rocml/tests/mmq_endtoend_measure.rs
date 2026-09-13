//! Diagnostic-only measurement (not a gate — no assertions), built while
//! root-causing the int8-MMQ precision regression: reproduces
//! `qwen35_chunked_prefill_parity`'s chunked-vs-token-serial final-logits
//! comparison at each of that gate's own prompt lengths, but *measures and
//! prints* the max relative logit error and out-of-tolerance fraction
//! instead of asserting on it — this is how the int8-MMQ-integration
//! round's own 12.4%-15.1% figures were reproduced, and how the follow-up
//! `ssm_out`-exclusion fix (`LinearWeight`'s `mmq_eligible_by_name`) is
//! measured against the same yardstick without duplicating or weakening
//! the real gate. `#[ignore]`d, run via `make mmq-endtoend-measure`.

use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const PROMPT_LENGTHS: &[usize] = &[128, 129, 500, 2048];
const LOGITS_REL_TOL: f32 = 1e-2;

fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 5) % vocab_size).collect()
}

fn measure(model: &mut Model, prompt_ids: &[u32], label: &str) {
    model.reset().expect("reset failed");
    let mut serial_logits = Vec::new();
    for &id in prompt_ids {
        serial_logits = model.forward_token(id).expect("forward_token failed");
    }

    model.reset().expect("reset failed");
    let chunked_logits = model
        .forward_prompt(prompt_ids, None)
        .expect("forward_prompt failed");

    assert_eq!(chunked_logits.len(), serial_logits.len());
    let mut max_rel = 0f32;
    let mut n_over = 0usize;
    for (&got, &want) in chunked_logits.iter().zip(&serial_logits) {
        let diff = (got - want).abs();
        let rel = diff / want.abs().max(1.0);
        max_rel = max_rel.max(rel);
        if diff > LOGITS_REL_TOL * want.abs().max(1.0) {
            n_over += 1;
        }
    }
    eprintln!(
        "{label}: max_rel={:.4} n_over_tol={n_over}/{} ({:.1}%)",
        max_rel,
        chunked_logits.len(),
        100.0 * n_over as f32 / chunked_logits.len() as f32
    );
}

#[test]
#[ignore]
fn measure_mmq_on_vs_off_final_logits() {
    let Some(path) = checkpoint(GGUF_REL) else {
        eprintln!("skipping: {GGUF_REL} not found");
        return;
    };

    for &use_mmq in &[false, true] {
        let opts = LoadOptions::new(4096)
            .with_kv_cache(KvCacheMode::F32)
            .with_mmq(use_mmq);
        let mut model = Model::load(&path, opts).expect("Model::load failed");
        let vocab_size = model.vocab_size();
        eprintln!("--- use_mmq={use_mmq} ---");
        for &len in PROMPT_LENGTHS {
            let prompt_ids = synthetic_prompt(len, vocab_size);
            measure(&mut model, &prompt_ids, &format!("len={len}"));
        }
    }
}

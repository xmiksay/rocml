//! Secondary signal (issue #15): teacher-forced perplexity over a small
//! checked-in corpus (`bench/eval/corpus.txt`), computed sequentially
//! through the decode path — one forward pass per token, feeding the
//! corpus's own tokens back in regardless of what the model would have
//! predicted (teacher forcing), never sampling. Cheap enough at ~2-3K
//! tokens that a batched-prefill shortcut isn't worth the complexity here.

use std::path::Path;
use std::time::Instant;

use rocml::RocmlError;

use crate::common::Loaded;

/// Runs teacher-forced PPL over `corpus_path`'s tokens and returns
/// `(ppl, wall_seconds)`.
pub fn compute_ppl(loaded: &mut Loaded, corpus_path: &Path) -> Result<(f64, f64), RocmlError> {
    let start = Instant::now();
    let text = std::fs::read_to_string(corpus_path)?;
    let ids = loaded.tokenizer.encode(&text);
    if ids.len() < 2 {
        return Err(RocmlError::Eval(format!(
            "{}: corpus encodes to only {} token(s), need at least 2 for PPL",
            corpus_path.display(),
            ids.len()
        )));
    }

    loaded.model.reset()?;
    let mut nll_sum = 0.0f64;
    let mut count = 0usize;
    for window in ids.windows(2) {
        let (current, next) = (window[0], window[1] as usize);
        let logits = loaded.model.forward_token(current)?;
        nll_sum += negative_log_prob(&logits, next)?;
        count += 1;
    }

    let ppl = (nll_sum / count as f64).exp();
    Ok((ppl, start.elapsed().as_secs_f64()))
}

/// `-log_softmax(logits)[target]`, computed in the log-sum-exp form so it
/// never materializes the full softmax (irrelevant here at ~150K vocab, but
/// matches the numerically stable idiom used elsewhere in this codebase).
fn negative_log_prob(logits: &[f32], target: usize) -> Result<f64, RocmlError> {
    let target_logit = *logits.get(target).ok_or_else(|| {
        RocmlError::Eval(format!(
            "target token id {target} out of range for a {}-entry logits vector",
            logits.len()
        ))
    })? as f64;
    let max = logits
        .iter()
        .fold(f64::NEG_INFINITY, |m, &v| m.max(v as f64));
    let sum_exp: f64 = logits.iter().map(|&v| (v as f64 - max).exp()).sum();
    let log_z = max + sum_exp.ln();
    Ok(log_z - target_logit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negative_log_prob_of_the_argmax_is_small() {
        let logits = vec![0.0_f32, 5.0, 0.0];
        let nll = negative_log_prob(&logits, 1).unwrap();
        assert!(
            nll < 0.1,
            "nll {nll} should be small for the dominant logit"
        );
    }

    #[test]
    fn negative_log_prob_of_a_low_logit_is_large() {
        let logits = vec![0.0_f32, 5.0, 0.0];
        let nll = negative_log_prob(&logits, 0).unwrap();
        assert!(
            nll > 4.0,
            "nll {nll} should be large for a suppressed logit"
        );
    }

    #[test]
    fn negative_log_prob_out_of_range_target_is_a_typed_error() {
        let logits = vec![0.0_f32, 1.0];
        let err = negative_log_prob(&logits, 5).unwrap_err();
        assert!(matches!(err, RocmlError::Eval(_)));
    }

    #[test]
    fn uniform_logits_give_nll_equal_to_ln_vocab_size() {
        let logits = vec![0.0_f32; 4];
        let nll = negative_log_prob(&logits, 2).unwrap();
        assert!((nll - (4.0_f64).ln()).abs() < 1e-9);
    }
}

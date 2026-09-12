//! The compiled-in model catalog: the `ModelFamily`/`ModelSpec` shapes and
//! the [`REGISTRY`] table itself, kept separate from `resolve`'s lookup
//! logic (`super`) purely to stay under this workspace's 400-line file cap.

use crate::sample::SamplingParams;

/// Architecture family a registry entry loads through. Kept separate from
/// the GGUF's own `general.architecture` string (see `crate::model::Model`)
/// so the registry can describe a checkpoint without opening it — a future
/// family (e.g. a Gemma variant) just adds a variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Qwen3Dense,
    Qwen35Hybrid,
}

/// One known checkpoint: where it lives on Hugging Face, where it's expected
/// under the checkpoint dir, and the defaults callers should apply unless
/// the user explicitly overrides them.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub name: &'static str,
    pub family: ModelFamily,
    /// Path under `checkpoint_dir()`, e.g.
    /// `"Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf"`.
    pub gguf_rel: &'static str,
    pub hf_repo: &'static str,
    /// Filename to hand `hf download` (matches `gguf_rel`'s final segment).
    pub hf_file: &'static str,
    pub default_ctx: usize,
    pub sampling: SamplingParams,
    pub thinking_default: bool,
}

/// Thinking-mode sampling recommended by Qwen for both the Qwen3 dense
/// family (`generation_config.json`: temperature 0.6/top_p 0.95/top_k 20)
/// and Ornith, which inherits the same convention (its own model card
/// recommends the identical triple) — see each `ModelSpec` below for the
/// verification trail of every entry.
const QWEN_THINKING_SAMPLING: SamplingParams = SamplingParams {
    temperature: 0.6,
    top_k: Some(20),
    top_p: Some(0.95),
    seed: 0,
    repeat_penalty: None,
    repeat_penalty_window: 64,
};

/// Qwen3.5's own recommended non-thinking-mode, text-task sampling
/// (temperature 1.0/top_p 1.0/top_k 20 — the engine has no presence-penalty
/// knob, so that part of the upstream recommendation is dropped).
const QWEN35_NON_THINKING_SAMPLING: SamplingParams = SamplingParams {
    temperature: 1.0,
    top_k: Some(20),
    top_p: Some(1.0),
    seed: 0,
    repeat_penalty: None,
    repeat_penalty_window: 64,
};

/// The compiled-in catalog. Every entry's `hf_repo`/`hf_file` was verified
/// live against the Hugging Face API before being hardcoded here (see the
/// task report for the verification transcript); nothing below is a guess.
pub const REGISTRY: &[ModelSpec] = &[
    ModelSpec {
        name: "ornith-9b",
        family: ModelFamily::Qwen35Hybrid,
        gguf_rel: "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf",
        // `deepreinforce-ai/Ornith-1.0-9B-GGUF` (the design doc's guess)
        // redirects here — the org was renamed, not an unofficial re-upload.
        hf_repo: "ornith-ai/Ornith-1.0-9B-GGUF",
        hf_file: "ornith-1.0-9b-Q6_K.gguf",
        // The model card recommends ctx up to 256k; 8192 is a sane default
        // for single-GPU chat use. The engine's KV cache hard-caps at
        // `crate::cache::MAX_SEQ_CAP` (4096) today, so `resolve`'s callers
        // clamp this down with a warning via `clamp_ctx` — see issue #3.
        default_ctx: 8192,
        sampling: SamplingParams {
            temperature: 0.6,
            top_k: Some(20),
            top_p: Some(0.95),
            seed: 0,
            repeat_penalty: None,
            repeat_penalty_window: 64,
        },
        thinking_default: true,
    },
    ModelSpec {
        name: "ornith-9b-q4",
        family: ModelFamily::Qwen35Hybrid,
        gguf_rel: "Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q4_K_M.gguf",
        hf_repo: "ornith-ai/Ornith-1.0-9B-GGUF",
        hf_file: "ornith-1.0-9b-Q4_K_M.gguf",
        // Speed variant: ~2x decode vs Q6_K on this engine (the q6_k gemv is
        // ALU-bound, see issue #7) and 1.7 GB more VRAM headroom for KV.
        // Q6_K stays the default `ornith-9b` until the #15 agentic eval
        // rules on Q4_K_M's tool-calling quality.
        default_ctx: 8192,
        sampling: SamplingParams {
            temperature: 0.6,
            top_k: Some(20),
            top_p: Some(0.95),
            seed: 0,
            repeat_penalty: None,
            repeat_penalty_window: 64,
        },
        thinking_default: true,
    },
    ModelSpec {
        name: "qwen3.5-2b",
        family: ModelFamily::Qwen35Hybrid,
        gguf_rel: "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf",
        // No official Qwen-org GGUF repo exists for Qwen3.5-2B (verified:
        // `Qwen/Qwen3.5-2B` only ships safetensors). unsloth's conversion
        // has this exact filename and matches what's already on disk here.
        hf_repo: "unsloth/Qwen3.5-2B-GGUF",
        hf_file: "Qwen3.5-2B-Q8_0.gguf",
        default_ctx: 4096,
        sampling: QWEN35_NON_THINKING_SAMPLING,
        // Verified against the model card: "Qwen3.5-2B operates in
        // non-thinking mode by default" — deviates from the design doc's
        // "thinking on" guess. See `../README.md`'s `rocml-serve` section
        // for the pre-existing note about this model's template default.
        thinking_default: false,
    },
    ModelSpec {
        name: "qwen3.5-0.8b",
        family: ModelFamily::Qwen35Hybrid,
        gguf_rel: "Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf",
        hf_repo: "unsloth/Qwen3.5-0.8B-GGUF",
        hf_file: "Qwen3.5-0.8B-Q8_0.gguf",
        default_ctx: 4096,
        sampling: QWEN35_NON_THINKING_SAMPLING,
        // Same verified default as Qwen3.5-2B: "Qwen3.5-0.8B operates in
        // non-thinking mode by default".
        thinking_default: false,
    },
    ModelSpec {
        name: "qwen3-0.6b",
        family: ModelFamily::Qwen3Dense,
        gguf_rel: "Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
        hf_repo: "Qwen/Qwen3-0.6B-GGUF",
        hf_file: "Qwen3-0.6B-Q8_0.gguf",
        default_ctx: 4096,
        sampling: QWEN_THINKING_SAMPLING,
        thinking_default: true,
    },
    ModelSpec {
        name: "qwen3-1.7b",
        family: ModelFamily::Qwen3Dense,
        gguf_rel: "Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf",
        hf_repo: "Qwen/Qwen3-1.7B-GGUF",
        hf_file: "Qwen3-1.7B-Q8_0.gguf",
        default_ctx: 4096,
        sampling: QWEN_THINKING_SAMPLING,
        thinking_default: true,
    },
    ModelSpec {
        name: "qwen3-4b",
        family: ModelFamily::Qwen3Dense,
        gguf_rel: "Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf",
        hf_repo: "Qwen/Qwen3-4B-GGUF",
        // Q4_K_M rather than Q8_0 (both verified present): keeps a "step up
        // from 1.7B" entry comfortably inside a 16GB card alongside every
        // other checkpoint this registry might load in the same session.
        hf_file: "Qwen3-4B-Q4_K_M.gguf",
        default_ctx: 4096,
        sampling: QWEN_THINKING_SAMPLING,
        thinking_default: true,
    },
    ModelSpec {
        name: "qwen3-8b",
        family: ModelFamily::Qwen3Dense,
        gguf_rel: "Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
        hf_repo: "Qwen/Qwen3-8B-GGUF",
        hf_file: "Qwen3-8B-Q4_K_M.gguf",
        default_ctx: 4096,
        sampling: QWEN_THINKING_SAMPLING,
        thinking_default: true,
    },
];

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

    #[test]
    fn every_registry_name_is_unique() {
        let mut names: Vec<&str> = REGISTRY.iter().map(|s| s.name).collect();
        let len_before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), len_before, "duplicate name in REGISTRY");
    }

    #[test]
    fn every_hf_file_matches_gguf_rel_final_segment() {
        for spec in REGISTRY {
            let final_segment = Path::new(spec.gguf_rel)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            assert_eq!(
                final_segment, spec.hf_file,
                "{}: gguf_rel's filename must match hf_file so a download lands where resolve() expects it",
                spec.name
            );
        }
    }
}

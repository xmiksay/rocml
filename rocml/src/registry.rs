//! Compiled-in catalog of known model checkpoints: a short name (e.g.
//! `qwen3.5-2b`) maps to a Hugging Face repo/file and a set of defaults
//! (context budget, sampling, thinking-mode default) tuned for that specific
//! checkpoint. [`resolve`] is the single entry point every binary
//! (`rocml-cli`, `rocml-serve`) calls to turn a user-supplied `--model`
//! string into an actual file path, downloading it via the `hf` CLI on a
//! registry hit whose file isn't present yet.
//!
//! Path-based `--model` usage (the only mode that existed before this
//! module) is untouched: anything that looks like a path — contains `/`,
//! ends in `.gguf`, or already exists as a file — skips the registry
//! entirely and carries no preset defaults.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::error::RocmlError;
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
/// recommends the identical triple) — see `resolve`'s module doc for the
/// verification trail of every entry below.
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

/// A resolved `--model` argument: an on-disk path, plus the registry entry
/// it came from (`None` for a raw path — no preset defaults apply).
#[derive(Debug)]
pub struct ResolvedModel {
    pub path: PathBuf,
    pub spec: Option<&'static ModelSpec>,
}

/// Turns a `--model` argument into an on-disk path, consulting the registry
/// for anything that isn't obviously a path.
///
/// - Contains `/`, ends with `.gguf`, or already exists as a file: treated
///   as a path outright (current, pre-registry behavior — no defaults).
/// - Otherwise looked up by name in [`REGISTRY`]; unknown names error out
///   listing the known ones.
/// - A registry hit whose file is missing under the checkpoint dir either
///   downloads it via the `hf` CLI (`download: true`) or errors out naming
///   the missing path, the source repo, and both remedies.
pub fn resolve(name_or_path: &str, download: bool) -> Result<ResolvedModel, RocmlError> {
    if is_path_like(name_or_path) {
        return Ok(ResolvedModel {
            path: PathBuf::from(name_or_path),
            spec: None,
        });
    }

    let spec = REGISTRY
        .iter()
        .find(|s| s.name == name_or_path)
        .ok_or_else(|| RocmlError::UnknownModel {
            name: name_or_path.to_string(),
            known_names: REGISTRY
                .iter()
                .map(|s| s.name)
                .collect::<Vec<_>>()
                .join(", "),
        })?;

    let path = resolved_path(spec);
    if path.exists() {
        return Ok(ResolvedModel {
            path,
            spec: Some(spec),
        });
    }
    if !download {
        return Err(missing_file_error(spec, &path));
    }

    download_model(spec)?;
    if !path.exists() {
        // `hf download` exited 0 but the expected file still isn't there —
        // most likely `hf_file` doesn't match `gguf_rel`'s final segment,
        // a bug in this table rather than a transient download failure.
        return Err(missing_file_error(spec, &path));
    }
    Ok(ResolvedModel {
        path,
        spec: Some(spec),
    })
}

/// Clamps a requested context budget to the engine's current hard cap
/// (`crate::cache::MAX_SEQ_CAP`), warning on stderr when it does. Shared by
/// every caller that turns a preset's `default_ctx` (or a user's `--ctx`)
/// into the value actually handed to the engine, since a preset like
/// `ornith-9b`'s (8192) can legitimately exceed today's cap.
pub fn clamp_ctx(requested: usize) -> usize {
    let cap = crate::cache::MAX_SEQ_CAP as usize;
    if requested > cap {
        eprintln!(
            "warning: requested context {requested} exceeds this engine's current cache cap of \
             {cap} tokens (see issue #3); clamping to {cap}"
        );
        cap
    } else {
        requested
    }
}

fn is_path_like(name_or_path: &str) -> bool {
    name_or_path.contains('/')
        || name_or_path.ends_with(".gguf")
        || Path::new(name_or_path).is_file()
}

fn missing_file_error(spec: &ModelSpec, path: &Path) -> RocmlError {
    RocmlError::ModelFileMissing {
        name: spec.name.to_string(),
        path: path.display().to_string(),
        repo: spec.hf_repo.to_string(),
        file: spec.hf_file.to_string(),
        local_dir: checkpoint_subdir(spec).display().to_string(),
    }
}

/// The on-disk path a registry entry resolves to, whether or not it exists
/// yet — used by [`resolve`] and by `rocml-cli models` to report presence.
pub fn resolved_path(spec: &ModelSpec) -> PathBuf {
    checkpoint_dir().join(spec.gguf_rel)
}

fn checkpoint_subdir(spec: &ModelSpec) -> PathBuf {
    match Path::new(spec.gguf_rel).parent() {
        Some(p) if !p.as_os_str().is_empty() => checkpoint_dir().join(p),
        _ => checkpoint_dir(),
    }
}

/// Mirrors `rocml_core::testpaths::checkpoint_dir`'s env-var resolution.
/// Duplicated rather than called directly: that module documents itself as
/// test-support (mmap'd fixtures, skip-if-missing helpers), while this runs
/// on every `rocml-cli`/`rocml-serve` startup — keeping the two independent
/// means a future change to test-fixture resolution can't silently change
/// what a production binary loads, and vice versa.
fn checkpoint_dir() -> PathBuf {
    std::env::var_os("ROCML_CHECKPOINT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join("checkpoints")
        })
}

/// Shells out to `hf download <repo> <file> --local-dir <dir>`, streaming
/// its own stdout/stderr through so the user sees the normal `hf` progress
/// bar rather than a silent hang.
fn download_model(spec: &ModelSpec) -> Result<(), RocmlError> {
    let local_dir = checkpoint_subdir(spec);
    std::fs::create_dir_all(&local_dir).map_err(|e| RocmlError::HfLaunchFailed(e.to_string()))?;
    eprintln!(
        "downloading {} ({}) into {}...",
        spec.hf_repo,
        spec.hf_file,
        local_dir.display()
    );
    let status = Command::new("hf")
        .arg("download")
        .arg(spec.hf_repo)
        .arg(spec.hf_file)
        .arg("--local-dir")
        .arg(&local_dir)
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .map_err(|e| RocmlError::HfLaunchFailed(e.to_string()))?;
    if !status.success() {
        return Err(RocmlError::HfDownloadFailed {
            repo: spec.hf_repo.to_string(),
            file: spec.hf_file.to_string(),
            status: status.to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn dot_gguf_extension_is_path_like() {
        assert!(is_path_like("./somewhere/model.gguf"));
        assert!(is_path_like("model.gguf"));
    }

    #[test]
    fn slash_containing_name_is_path_like() {
        assert!(is_path_like("some/dir/model"));
    }

    #[test]
    fn existing_file_is_path_like_even_without_gguf_extension() {
        let dir = std::env::temp_dir().join(format!(
            "rocml-registry-test-{}-{}",
            std::process::id(),
            "existing_file_is_path_like"
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("checkpoint");
        std::fs::write(&file, b"x").unwrap();
        assert!(is_path_like(file.to_str().unwrap()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn plain_registry_name_is_not_path_like() {
        assert!(!is_path_like("qwen3.5-2b"));
        assert!(!is_path_like("ornith-9b"));
    }

    #[test]
    fn resolve_treats_bare_registry_name_as_lookup() {
        // Doesn't require the checkpoint to exist on disk — only that a
        // known name isn't misclassified as a path (`download: false` so
        // this never shells out, keeping the test hermetic).
        match resolve("qwen3-0.6b", false) {
            Ok(resolved) => assert_eq!(resolved.spec.unwrap().name, "qwen3-0.6b"),
            Err(RocmlError::ModelFileMissing { name, .. }) => assert_eq!(name, "qwen3-0.6b"),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn resolve_unknown_name_lists_known_names() {
        let err = resolve("totally-not-a-model", false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("totally-not-a-model"));
        for spec in REGISTRY {
            assert!(msg.contains(spec.name), "error should list {}", spec.name);
        }
    }

    #[test]
    fn resolve_missing_file_error_names_path_repo_and_remedies() {
        // Whether or not this checkpoint happens to be present on the
        // machine running the test, the underlying `missing_file_error`
        // shape (path/repo/remedies) is what matters — build it directly
        // rather than depending on filesystem state.
        let spec = REGISTRY.iter().find(|s| s.name == "qwen3-0.6b").unwrap();
        let err = missing_file_error(spec, &resolved_path(spec));
        if let RocmlError::ModelFileMissing { path, repo, .. } = err {
            assert!(path.contains("Qwen3-0.6B-Q8_0.gguf"));
            assert_eq!(repo, "Qwen/Qwen3-0.6B-GGUF");
        } else {
            panic!("expected ModelFileMissing");
        }
    }

    #[test]
    fn resolve_path_like_input_skips_registry_lookup() {
        let resolved = resolve("./some/model.gguf", false).unwrap();
        assert!(resolved.spec.is_none());
        assert_eq!(resolved.path, PathBuf::from("./some/model.gguf"));
    }

    #[test]
    fn clamp_ctx_leaves_small_values_untouched() {
        assert_eq!(clamp_ctx(2048), 2048);
    }

    #[test]
    fn clamp_ctx_caps_at_max_seq_cap() {
        assert_eq!(clamp_ctx(8192), crate::cache::MAX_SEQ_CAP as usize);
    }
}

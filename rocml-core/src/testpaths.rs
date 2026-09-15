//! Resolves machine-local model checkpoint and Hugging Face hub cache paths
//! for tests, so integration tests across the workspace don't hardcode
//! developer-machine absolute paths.
//!
//! ## Checkpoint directory
//!
//! [`checkpoint_dir`] is `$ROCML_CHECKPOINT_DIR` if set, else
//! `$HOME/checkpoints`. It's expected to contain:
//!
//! - `Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf`
//! - `Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf`
//! - `Ornith-1.0-9B-GGUF/ornith-1.0-9b-Q6_K.gguf`
//! - `Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf` (and `-Q6_K.gguf`)
//! - `Qwen3.5-2B-tokenizer/tokenizer.json`
//! - `Ornith-1.0-9B/chat_template.jinja`
//!
//! ## Hugging Face hub cache
//!
//! [`hf_hub_file`] resolves the standard HF hub cache layout
//! (`<hub_root>/<repo_dirname>/snapshots/<commit-hash>/<file>`), independent
//! of the checkpoint directory above.

use std::path::PathBuf;

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Root directory for model checkpoints: `$ROCML_CHECKPOINT_DIR` if set,
/// else `$HOME/checkpoints` (falling back to `.` if `$HOME` is also unset —
/// this never panics).
pub fn checkpoint_dir() -> PathBuf {
    std::env::var_os("ROCML_CHECKPOINT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home_dir().join("checkpoints"))
}

/// Resolves `rel` under [`checkpoint_dir`]. Returns `None` if the file isn't
/// present on this machine, after printing an actionable skip message
/// (resolved path + how to override it) so callers can keep their existing
/// skip-if-missing behavior.
pub fn checkpoint(rel: &str) -> Option<PathBuf> {
    let path = checkpoint_dir().join(rel);
    if path.exists() {
        Some(path)
    } else {
        eprintln!(
            "skipping: {} not present on this machine (set ROCML_CHECKPOINT_DIR to override; \
             default is $HOME/checkpoints)",
            path.display()
        );
        None
    }
}

/// Resolves a file cached under a Hugging Face hub repo:
/// `<hub_root>/<repo_dirname>/snapshots/<commit-hash>/<glob_rel>`, where
/// `hub_root` is `$HF_HOME/hub` if `HF_HOME` is set, else
/// `$HOME/.cache/huggingface/hub`. Returns the first snapshot directory that
/// has `glob_rel`, or `None` if there isn't one.
pub fn hf_hub_file(repo_dirname: &str, glob_rel: &str) -> Option<PathBuf> {
    let hub_root = std::env::var_os("HF_HOME")
        .map(|hf_home| PathBuf::from(hf_home).join("hub"))
        .unwrap_or_else(|| home_dir().join(".cache").join("huggingface").join("hub"));
    let snapshots_dir = hub_root.join(repo_dirname).join("snapshots");
    for entry in std::fs::read_dir(snapshots_dir).ok()?.flatten() {
        let candidate = entry.path().join(glob_rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

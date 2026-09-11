//! Error types for the rocml engine crate. Every fallible path from GGUF
//! loading through to a single generation step returns `RocmlError` rather
//! than panicking, per the workspace's I/O-reachable-panic ban.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum RocmlError {
    #[error(transparent)]
    Gguf(#[from] rocml_core::gguf::GgufError),

    #[error(transparent)]
    Quant(#[from] rocml_core::quant::QuantError),

    #[error(transparent)]
    Tokenizer(#[from] rocml_core::tokenizer::TokenizerError),

    #[error(transparent)]
    Hip(#[from] rocml_hip::HipError),

    #[error("unsupported architecture {found:?}, rocml only implements dense \"qwen3\"")]
    UnsupportedArchitecture { found: String },

    #[error("tensor {name:?}: expected {expected} dimensions, found {found}")]
    UnexpectedTensorRank {
        name: String,
        expected: usize,
        found: usize,
    },

    #[error(
        "tensor {name:?}: dimension {index} is {value}, expected a positive value that fits u32"
    )]
    InvalidTensorDim {
        name: String,
        index: usize,
        value: u64,
    },

    #[error("invalid model config: {0}")]
    Config(String),

    #[error("sequence length {requested} exceeds this model's cache capacity of {max_seq} tokens")]
    ContextOverflow { requested: u32, max_seq: u32 },

    #[error("tokenizer has no eos_token_id (tokenizer.ggml.eos_token_id missing from GGUF)")]
    MissingEosToken,

    #[error("unknown model {name:?}; known registry names: {known_names}")]
    UnknownModel { name: String, known_names: String },

    #[error(
        "model {name:?} not found at {path} — download it with `hf download {repo} {file} \
         --local-dir {local_dir}`, point ROCML_CHECKPOINT_DIR at a directory that already has \
         it, or re-run with downloading enabled (omit --no-download)"
    )]
    ModelFileMissing {
        name: String,
        path: String,
        repo: String,
        file: String,
        local_dir: String,
    },

    #[error("`hf download {repo} {file}` failed (exit: {status})")]
    HfDownloadFailed {
        repo: String,
        file: String,
        status: String,
    },

    #[error("failed to launch `hf`: {0} (is the huggingface_hub CLI installed and on PATH?)")]
    HfLaunchFailed(String),
}

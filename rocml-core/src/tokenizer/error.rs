//! Failure modes for building a tokenizer out of GGUF metadata. Once built,
//! `encode`/`decode` never fail (see `mod.rs` for how malformed input is
//! degraded instead of panicking).

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error(transparent)]
    Gguf(#[from] crate::gguf::GgufError),

    #[error("tokenizer.ggml.model is {found:?}, only \"gpt2\" (byte-level BPE) is supported")]
    UnsupportedModel { found: String },

    #[error(
        "tokenizer.ggml.merges entry {0:?} is not \"left right\" (expected exactly one space)"
    )]
    MalformedMerge(String),

    #[error("tokenizer.ggml.tokens is empty")]
    EmptyVocab,
}

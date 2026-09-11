//! Shared plumbing: GGUF/tokenizer/model loading and the sampling-flag set
//! every subcommand that generates text takes identically.

use std::path::Path;

use clap::Args;
use rocml::{Model, RocmlError, SamplingParams};
use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::BpeTokenizer;

/// Sampling flags shared by `chat` and `generate`. `bench` doesn't take
/// these — it always runs greedy, since it measures throughput, not output
/// quality, and greedy keeps every run's token count reproducible.
#[derive(Args, Debug, Clone)]
pub struct SamplingArgs {
    /// Sampling temperature; `0` (the default) is greedy decoding.
    #[arg(long, short = 't', default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub top_k: Option<usize>,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
}

impl SamplingArgs {
    pub fn to_sampling_params(&self) -> SamplingParams {
        SamplingParams {
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            seed: self.seed,
            ..SamplingParams::default()
        }
    }
}

pub struct Loaded {
    pub tokenizer: BpeTokenizer,
    pub model: Model,
}

/// Loads the tokenizer from GGUF metadata, then hands off to
/// `Model::load` (which reopens the file itself — see `Model::load`'s own
/// doc comment on why that's cheap and not worth threading a shared handle
/// through).
pub fn load(model_path: impl AsRef<Path>) -> Result<Loaded, RocmlError> {
    let path = model_path.as_ref();
    let gguf = GgufFile::open(path)?;
    let tokenizer = BpeTokenizer::from_gguf(&gguf)?;
    drop(gguf);
    let model = Model::load(path)?;
    Ok(Loaded { tokenizer, model })
}

/// The model file's stem, used as the served/reported model id (e.g.
/// `Qwen3.5-2B-Q8_0` from `.../Qwen3.5-2B-Q8_0.gguf`).
pub fn model_id(model_path: impl AsRef<Path>) -> String {
    model_path
        .as_ref()
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".to_string())
}

//! Shared plumbing: model-registry resolution, GGUF/tokenizer/model
//! loading, and the sampling-flag set every subcommand that generates text
//! takes identically.

use std::path::Path;

use clap::Args;
use rocml::{Model, ResolvedModel, RocmlError, SamplingParams};
use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::BpeTokenizer;

/// `--model`/`--no-download` flags shared by every subcommand that loads a
/// model.
#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    /// Registry name (e.g. `qwen3.5-2b`) or a path to a `.gguf` file.
    #[arg(long)]
    pub model: String,
    /// A registry hit whose file is missing errors out instead of
    /// downloading it via `hf` (the default).
    #[arg(long)]
    pub no_download: bool,
}

impl ModelArgs {
    pub fn resolve(&self) -> Result<ResolvedModel, RocmlError> {
        rocml::resolve(&self.model, !self.no_download)
    }
}

/// Sampling flags shared by `chat` and `generate`. `bench` doesn't take
/// these — it always runs greedy, since it measures throughput, not output
/// quality, and greedy keeps every run's token count reproducible.
///
/// Every field is `Option` (no `default_value_t`) so an unset flag is
/// distinguishable from an explicit one: [`SamplingArgs::to_sampling_params`]
/// needs that distinction to let a resolved model's registry preset supply
/// defaults without an unset flag clobbering them.
#[derive(Args, Debug, Clone)]
pub struct SamplingArgs {
    /// Sampling temperature; unset falls back to the resolved model's
    /// registry preset, then greedy decoding (`0.0`).
    #[arg(long, short = 't')]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub top_p: Option<f32>,
    #[arg(long)]
    pub top_k: Option<usize>,
    #[arg(long)]
    pub seed: Option<u64>,
}

impl SamplingArgs {
    /// Merges these explicit CLI overrides onto `preset` (a resolved
    /// model's registry sampling defaults) — an unset flag falls through to
    /// `preset`, an explicit flag always wins. `preset: None` (a path-based
    /// `--model` with no registry entry) falls through to
    /// `SamplingParams::default()` (greedy), matching this CLI's
    /// pre-registry behavior exactly.
    pub fn to_sampling_params(&self, preset: Option<&SamplingParams>) -> SamplingParams {
        let base = preset.copied().unwrap_or_default();
        SamplingParams {
            temperature: self.temperature.unwrap_or(base.temperature),
            top_k: self.top_k.or(base.top_k),
            top_p: self.top_p.or(base.top_p),
            seed: self.seed.unwrap_or(base.seed),
            ..base
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

/// The reported model id: the registry name for a resolved registry hit
/// (e.g. `qwen3.5-2b`), else the model file's stem (e.g. `Qwen3.5-2B-Q8_0`
/// from `.../Qwen3.5-2B-Q8_0.gguf`) for a path-based `--model`.
pub fn model_id(resolved: &ResolvedModel) -> String {
    resolved
        .spec
        .map(|s| s.name.to_string())
        .unwrap_or_else(|| {
            resolved
                .path
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| "model".to_string())
        })
}

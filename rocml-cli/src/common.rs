//! Shared plumbing: model-registry resolution, GGUF/tokenizer/model
//! loading, and the sampling-flag set every subcommand that generates text
//! takes identically.

use std::path::{Path, PathBuf};

use clap::{Args, ValueEnum};
use rocml::snapshot::{ModelStamp, SnapshotStore};
use rocml::{
    KvCacheMode, LoadOptions, Model, ModelSpec, ResolvedModel, RocmlError, SamplingParams,
};
use rocml_core::gguf::GgufFile;
use rocml_core::tokenizer::BpeTokenizer;

/// `--kv-cache` flag spelling, mapped to `rocml::KvCacheMode` — `F32` is
/// deliberately not exposed here (parity-test-only, see `KvCacheMode`'s doc
/// comment); quantized modes stay opt-in until issue #15's eval exists.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum KvCacheArg {
    Fp16,
    Q8,
    #[value(name = "q4-mixed")]
    Q4Mixed,
}

impl From<KvCacheArg> for KvCacheMode {
    fn from(arg: KvCacheArg) -> Self {
        match arg {
            KvCacheArg::Fp16 => KvCacheMode::Fp16,
            KvCacheArg::Q8 => KvCacheMode::Q8,
            KvCacheArg::Q4Mixed => KvCacheMode::Q4Mixed,
        }
    }
}

/// `--model`/`--no-download`/`--kv-cache` flags shared by every subcommand
/// that loads a model.
#[derive(Args, Debug, Clone)]
pub struct ModelArgs {
    /// Registry name (e.g. `qwen3.5-2b`) or a path to a `.gguf` file.
    #[arg(long)]
    pub model: String,
    /// A registry hit whose file is missing errors out instead of
    /// downloading it via `hf` (the default).
    #[arg(long)]
    pub no_download: bool,
    /// KV cache storage/quantization policy (issue #2/#3). Default `fp16`
    /// halves cache vs the old f32-only cache; `q8`/`q4-mixed` further
    /// shrink it (KIVI-style: fp16 attention sinks + recent window,
    /// quantized bulk, boundary attention layers left fp16) at the cost of
    /// a small, opt-in accuracy tradeoff.
    #[arg(long, value_enum, default_value = "fp16")]
    pub kv_cache: KvCacheArg,
    /// Routes qwen35 hybrid chunked-prefill matmuls through the int8 MMQ
    /// GEMM (issue #6's int8-MMQ-integration round) instead of f16 WMMA,
    /// where the shape is eligible. Off by default: int8 activation
    /// quantization is a real precision change beyond WMMA's f16 rounding,
    /// validated separately — see `rocml::LoadOptions::use_mmq`'s doc
    /// comment and `.claude/CLAUDE.md` for current status. No effect on
    /// the dense `qwen3` architecture or on decode.
    #[arg(long)]
    pub mmq: bool,
    /// Attention-sink length for a quantized (`q8`/`q4-mixed`) `--kv-cache`
    /// mode (issue #2 leftovers) — the leading positions that stay fp16
    /// forever, never evicted/quantized. No effect at the default
    /// `--kv-cache fp16`. Must be at least 1 and satisfy
    /// `--kv-sink + --kv-window < --ctx` (otherwise no history ever reaches
    /// the quantized bulk region).
    #[arg(long, default_value_t = rocml::kv_quant::SINK_LEN)]
    pub kv_sink: u32,
    /// Recent-window ring capacity (and quantize-on-evict batch size) for a
    /// quantized `--kv-cache` mode (issue #2 leftovers). Must be at least 1
    /// and at most 65535 (the mixed cache's quantize-evict kernel launches
    /// one HIP grid.y row per window position, and HIP's grid.y/z dimension
    /// is capped at 65535); no relationship to any prefill chunk size is
    /// required — the chunked-prefill eviction planner handles any window
    /// size, including one that spans multiple evictions within a single
    /// chunk.
    #[arg(long, default_value_t = rocml::kv_quant::WINDOW_LEN)]
    pub kv_window: u32,
    /// Issue #14 phase 2's debug model-level quality simulation: bits/coord
    /// (2/3/4) to round-trip V through rotational quantization (block-
    /// diagonal Givens + Lloyd-Max, `rocml::kv_quant::rotational`) at every
    /// mixed-cache window eviction, instead of the production scalar
    /// encoding — measures whether the technique's quality holds up on real
    /// tasks before investing in fast HIP kernels (phase 3). No effect
    /// unless `--kv-cache` is `q8`/`q4-mixed`. Performance is irrelevant on
    /// this path (a CPU round trip per evicted block) — research-only, not
    /// meant for `rocml-serve`.
    #[arg(long)]
    pub kv_rot_sim: Option<u8>,
    /// Also round-trips K (not just V) through the same simulation. No
    /// effect unless `--kv-rot-sim` is set.
    #[arg(long)]
    pub kv_rot_sim_k: bool,
}

impl ModelArgs {
    pub fn resolve(&self) -> Result<ResolvedModel, RocmlError> {
        rocml::resolve(&self.model, !self.no_download)
    }
}

/// Resolves the final context length for a load: an explicit `--ctx`, else
/// the resolved model's registry preset, else `default_ctx` — then clamped
/// to this checkpoint's estimated VRAM budget (`rocml::registry::clamp_ctx`,
/// issue #3) at the requested KV dtype.
pub fn resolve_ctx(
    explicit: Option<usize>,
    spec: Option<&ModelSpec>,
    default_ctx: usize,
    gguf_path: &Path,
    kv_cache: KvCacheMode,
    kv_sink: u32,
    kv_window: u32,
) -> Result<usize, RocmlError> {
    let requested = explicit.unwrap_or_else(|| spec.map_or(default_ctx, |s| s.default_ctx));
    rocml::registry::clamp_ctx(requested, gguf_path, kv_cache, kv_sink, kv_window)
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

/// `--snapshot-ram-mb`/`--snapshot-dir`/`--snapshot-disk-mb` — shared by
/// `chat` and `bench --turns` (issue #1). Mirrors `rocml-serve`'s own flags
/// of the same name.
#[derive(Args, Debug, Clone)]
pub struct SnapshotArgs {
    /// Conversation-state snapshot RAM budget in MiB (qwen35-hybrid-only).
    /// `0` disables the RAM tier.
    #[arg(long = "snapshot-ram-mb", default_value_t = 4096)]
    pub snapshot_ram_mb: usize,
    /// Optional NVMe persistence tier — content-addressed files under this
    /// directory, size-budgeted by `--snapshot-disk-mb`. Off by default.
    #[arg(long = "snapshot-dir")]
    pub snapshot_dir: Option<PathBuf>,
    #[arg(long = "snapshot-disk-mb", default_value_t = 20_000)]
    pub snapshot_disk_mb: u64,
}

impl SnapshotArgs {
    /// `None` when both tiers are off (`--snapshot-ram-mb 0` and no
    /// `--snapshot-dir`) — callers should pass that straight through to
    /// `run_turn` as `None` rather than a zero-budget store: `run_turn`
    /// still pays the D2H capture cost for a `Some` store even if nothing
    /// ends up stored, and a fully-disabled snapshot layer should skip that
    /// cost entirely, not just skip storing the result.
    pub fn build_store(&self) -> Result<Option<SnapshotStore>, RocmlError> {
        if self.snapshot_ram_mb == 0 && self.snapshot_dir.is_none() {
            return Ok(None);
        }
        SnapshotStore::new(
            self.snapshot_ram_mb * 1024 * 1024,
            self.snapshot_dir.clone(),
            self.snapshot_disk_mb * 1024 * 1024,
        )
        .map(Some)
    }
}

/// Cheap model identity stamp (issue #1) — wraps `ModelStamp::from_path`'s
/// `io::Error` into `RocmlError` for CLI call sites.
pub fn model_stamp(path: &Path) -> Result<ModelStamp, RocmlError> {
    ModelStamp::from_path(path)
        .map_err(|e| RocmlError::Config(format!("failed to stamp {}: {e}", path.display())))
}

pub struct Loaded {
    pub tokenizer: BpeTokenizer,
    pub model: Model,
}

/// Loads the tokenizer from GGUF metadata, then hands off to
/// `Model::load` (which reopens the file itself — see `Model::load`'s own
/// doc comment on why that's cheap and not worth threading a shared handle
/// through).
pub fn load(model_path: impl AsRef<Path>, opts: LoadOptions) -> Result<Loaded, RocmlError> {
    let path = model_path.as_ref();
    let gguf = GgufFile::open(path)?;
    let tokenizer = BpeTokenizer::from_gguf(&gguf)?;
    drop(gguf);
    let model = Model::load(path, opts)?;
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

/// A deterministic synthetic prompt of exactly `target_len.max(1)` tokens:
/// `filler` repeated enough times, tokenized once, then truncated. Encoding
/// once matters: re-encoding a growing string per repetition is quadratic
/// and took minutes at `bench --turns --depth 40000`.
pub fn synthetic_tokens(loaded: &Loaded, filler: &str, target_len: usize) -> Vec<u32> {
    let target = target_len.max(1);
    let per_filler = loaded.tokenizer.encode(filler).len().max(1);
    let mut reps = target / per_filler + 2;
    loop {
        let mut ids = loaded.tokenizer.encode(&filler.repeat(reps));
        if ids.len() >= target {
            ids.truncate(target);
            return ids;
        }
        reps *= 2;
    }
}

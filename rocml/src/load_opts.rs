//! Load-time configuration threaded from the CLI/server (or a test) down
//! into `Model::load`: the requested context length — already
//! budget-checked by `crate::registry::clamp_ctx` — and the KV cache's
//! storage policy (issue #3/#2).

use crate::cache::KvDtype;

/// KV cache storage policy selected at load time.
///
/// `Fp16` (default) and `F32` are issue #3's dense per-layer-plane cache —
/// `F32` exists solely so the parity test suites can pin the pre-issue-#3
/// reference numerics exactly (see each test's own `LoadOptions` usage);
/// nothing in the CLI/server ever selects it. `Q8`/`Q4Mixed` are issue #2's
/// KIVI-style quantized bulk + fp16 attention-sink/recent-window design —
/// see `crate::kv_quant`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub enum KvCacheMode {
    #[default]
    Fp16,
    F32,
    /// K per-channel Q8, V per-token Q8 (uniform bit width).
    Q8,
    /// K per-channel Q8, V per-token Q4 — issue #2's recommended default
    /// once the #15 agentic eval lands; opt-in until then.
    Q4Mixed,
}

impl KvCacheMode {
    /// The dense `KvDtype` every non-quantized layer/region uses regardless
    /// of mode: `F32` only for `KvCacheMode::F32` itself, `F16` for
    /// everything else — including `Q8`/`Q4Mixed`'s attention-sink,
    /// recent-window and boundary-layer regions, which are always f16 by
    /// design (issue #2).
    pub fn dense_dtype(self) -> KvDtype {
        match self {
            Self::F32 => KvDtype::F32,
            Self::Fp16 | Self::Q8 | Self::Q4Mixed => KvDtype::F16,
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Q8 | Self::Q4Mixed)
    }

    /// CLI/server flag spelling — see `rocml-cli`'s `--kv-cache` and
    /// `rocml-serve`'s same flag.
    pub fn as_flag_str(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::F32 => "f32",
            Self::Q8 => "q8",
            Self::Q4Mixed => "q4-mixed",
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Context length in tokens, already resolved/budget-clamped by the
    /// caller (`crate::registry::clamp_ctx`) — `Model::load` still performs
    /// its own authoritative post-weights-load budget check (see
    /// `crate::budget`), since the pre-load clamp only has an approximate,
    /// pre-weights-upload VRAM picture.
    pub ctx: usize,
    pub kv_cache: KvCacheMode,
}

impl LoadOptions {
    pub fn new(ctx: usize) -> Self {
        Self {
            ctx,
            kv_cache: KvCacheMode::default(),
        }
    }

    pub fn with_kv_cache(mut self, mode: KvCacheMode) -> Self {
        self.kv_cache = mode;
        self
    }
}

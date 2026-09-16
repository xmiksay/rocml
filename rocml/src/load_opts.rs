//! Load-time configuration threaded from the CLI/server (or a test) down
//! into `Model::load`: the requested context length — already
//! budget-checked by `crate::registry::clamp_ctx` — and the KV cache's
//! storage policy (issue #3/#2).

use crate::cache::KvDtype;
use crate::kv_quant::{SINK_LEN, WINDOW_LEN};

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
    /// Whether the qwen35 hybrid architecture's chunked-prefill GEMM
    /// dispatch (`forward::kernels_quant::QuantKernels::gemm`) may route a
    /// WMMA-eligible quantized matmul through the int8 MMQ path
    /// (`forward::kernels_mmq`) instead of f16 WMMA — issue #6's
    /// int8-MMQ-integration round. Off by default: int8 activation
    /// quantization is a real precision change beyond the WMMA f16
    /// rounding every quantized chunked-prefill matmul already accepts,
    /// and flips to on-by-default only once validated against the full
    /// parity suite and the issue-15 agentic eval — see `.claude/CLAUDE.md`
    /// for the current status. As of issue #16's dense chunked-prefill
    /// port, this now also gates the dense `qwen3` architecture's own
    /// chunked-prefill `matmul` dispatch (`forward::chunk_forward` reuses
    /// the identical `Kernels`/`QuantKernels` type) — it was never
    /// separately re-validated for the dense path's activation shapes
    /// (which have no GDN-style SiLU-gated recurrence output, the qwen35
    /// hybrid path's own documented MMQ failure mode), so it stays off by
    /// default here too. No effect on decode (token-serial decode uses
    /// `matvec`, never `matmul`).
    pub use_mmq: bool,
    /// Attention-sink length for a quantized (`Q8`/`Q4Mixed`) mixed KV
    /// layer — issue #2 leftovers' `--kv-sink`. Defaults to
    /// `crate::kv_quant::SINK_LEN` (32); no effect when `kv_cache` isn't
    /// quantized. Validated at load time by
    /// `crate::kv_quant::validate_sink_window` (positive, and
    /// `kv_sink + kv_window < ctx`) — `Model::load` surfaces a
    /// `RocmlError::Config` rather than panicking on an invalid value.
    pub kv_sink: u32,
    /// Recent-window capacity (and quantize-on-evict batch size) for a
    /// quantized mixed KV layer — issue #2 leftovers' `--kv-window`.
    /// Defaults to `crate::kv_quant::WINDOW_LEN` (128). See `kv_sink`'s doc
    /// comment for validation.
    pub kv_window: u32,
    /// Issue #14 phase 2's debug model-level quality simulation: when
    /// `Some(bpw)` and `kv_cache.is_quantized()`, every mixed-KV-cache
    /// window eviction (`MixedAttnPlane::append`/`append_chunk`) first
    /// round-trips V (rotate -> normalize -> Lloyd-Max quantize at `bpw` ->
    /// dequantize -> inverse-rotate, via `kv_quant::rotational`) on the
    /// host before the block is quantize-evicted through the normal
    /// production kernel — simulating rotational KV quantization's quality
    /// impact without any new fast kernels. Performance is irrelevant on
    /// this path (a full D2H/H2D round trip per evicted block); it exists
    /// purely to decide whether phase 3 (fast HIP kernels) is worth
    /// building. `None` (default): no effect, byte-identical to before this
    /// issue. Not exposed by `rocml-serve` — research-only, driven through
    /// `rocml-cli`.
    pub kv_rot_sim: Option<u8>,
    /// Also round-trips K (not just V) through the same simulation when
    /// `kv_rot_sim` is `Some`. No effect otherwise. See issue #14's own
    /// acceptance criteria: the primary decision gate is V-only at 3 bpw
    /// (K stays the real, already-shipped Q8 production encoding); this
    /// flag is the issue's own documented "if V passes, optionally 3 bpw
    /// both" follow-up run.
    pub kv_rot_sim_k: bool,
    /// M2's qwen35moe VRAM-resident expert cache slot count override — see
    /// `crate::qwen35::forward::Model::load`'s construction of the cache
    /// (its doc comment on the `expert_cache` field). `None` (default):
    /// derive the largest capacity that fits in whatever VRAM is left after
    /// every other allocation. `Some(0)`: disable the cache entirely,
    /// falling back to the M1 always-copy path — used by tests to compare
    /// cached vs. uncached runs bit-for-bit. `Some(n)` for `n > 0`: still
    /// capped at both the VRAM-derived capacity and the model's total
    /// distinct-expert count (a request larger than either is silently
    /// clamped, never an error — this is a performance knob, not a
    /// correctness one). No effect on a non-MoE checkpoint.
    pub moe_cache_slots: Option<usize>,
}

impl LoadOptions {
    pub fn new(ctx: usize) -> Self {
        Self {
            ctx,
            kv_cache: KvCacheMode::default(),
            use_mmq: false,
            kv_sink: SINK_LEN,
            kv_window: WINDOW_LEN,
            kv_rot_sim: None,
            kv_rot_sim_k: false,
            moe_cache_slots: None,
        }
    }

    pub fn with_moe_cache_slots(mut self, slots: Option<usize>) -> Self {
        self.moe_cache_slots = slots;
        self
    }

    pub fn with_kv_cache(mut self, mode: KvCacheMode) -> Self {
        self.kv_cache = mode;
        self
    }

    pub fn with_mmq(mut self, use_mmq: bool) -> Self {
        self.use_mmq = use_mmq;
        self
    }

    pub fn with_kv_sink(mut self, kv_sink: u32) -> Self {
        self.kv_sink = kv_sink;
        self
    }

    pub fn with_kv_window(mut self, kv_window: u32) -> Self {
        self.kv_window = kv_window;
        self
    }

    pub fn with_kv_rot_sim(mut self, bpw: Option<u8>, apply_to_k: bool) -> Self {
        self.kv_rot_sim = bpw;
        self.kv_rot_sim_k = apply_to_k;
        self
    }
}

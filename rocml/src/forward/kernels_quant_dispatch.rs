//! Batched prefill-path `gemm_xwt_<dtype>` dispatch for `QuantKernels`
//! (struct + `gemv`/loader in the sibling `kernels_quant` module — split
//! out purely for the 400-line file cap once the shape-aware dispatch round
//! below added a third kernel family to choose between).

use std::mem::size_of;

use rocml_core::quant::GgmlDType;
use rocml_hip::{kernel_params, Function, LaunchConfig};

use super::kernels::DevPtr;
use super::kernels_mmq::{MmqKernels, MmqScratch};
use super::kernels_quant::QuantKernels;
use super::kernels_splitk::SplitKScratch;
use crate::error::RocmlError;

/// Reduction elements staged into LDS per outer iteration by every
/// `gemm_xwt_q*` kernel — fixes the dynamic shared memory request
/// (`TILE_ELEMS * sizeof(f32)`) regardless of dtype (see
/// `kernels/gemm_xwt_quant.hip`'s module doc).
const GEMM_QUANT_TILE_ELEMS: u32 = 256;
/// Warps per `gemm_xwt_q*` workgroup — fixes the launch's block.y. Must
/// equal `GEMM_QUANT_TILE_ELEMS / 32` (the kernel's cooperative dequant
/// stages one element per thread per outer iteration in a single pass).
const GEMM_QUANT_WARPS_PER_BLOCK: u32 = 8;
/// Output rows one warp carries in registers per weight tile (mirrors the
/// kernel's `ROWS_PER_WARP`).
const GEMM_QUANT_ROWS_PER_WARP: u32 = 8;
/// Output rows one `gemm_xwt_q*` workgroup shares a weight row across —
/// fixes the launch's grid.y.
const GEMM_QUANT_TILE_ROWS: u32 = GEMM_QUANT_WARPS_PER_BLOCK * GEMM_QUANT_ROWS_PER_WARP;

/// Output rows (`X` rows) a `gemm_xwt_wmma_q*` workgroup tile covers —
/// mirrors the kernel's `TILE_ROWS` (fixes the launch's grid.y and doubles
/// as the dispatch threshold below: it exactly matches
/// `qwen35::forward::chunk_forward::PREFILL_CHUNK_SIZE`, so every full
/// chunked-prefill chunk fills one row-tile exactly and only a short last
/// chunk falls back to the scalar kernel). Shared by both the default and
/// narrow WMMA tile configs — only `TILE_M`/`WARPS_M`/`WARPS_N` vary
/// between them, see `gemm_xwt_wmma_impl.h`'s module doc.
const GEMM_WMMA_TILE_ROWS: u32 = 128;
/// Output columns (`W` rows) the default `gemm_xwt_wmma_q*` workgroup tile
/// covers — mirrors the kernel's `TILE_M`. 128 since the per-wave-
/// efficiency round — see `gemm_xwt_quant_wmma.hip`'s module doc.
const GEMM_WMMA_TILE_M: u32 = 128;
/// Reduction elements staged into LDS per outer iteration — mirrors the
/// kernel's `K_STAGE`; combined with `GEMM_WMMA_LDS_PAD` to size the LDS
/// request (the kernel's per-row stride is padded past `K_STAGE`).
const GEMM_WMMA_K_STAGE: u32 = 16;
/// LDS row-stride padding — must match the kernel's `LDS_PAD` (bank-conflict elimination).
const GEMM_WMMA_LDS_PAD: u32 = 8;
/// Warps per WMMA workgroup — fixes the launch's block.y. Identical for the
/// default (`WARPS_M=4,WARPS_N=4`) and narrow (`WARPS_M=8,WARPS_N=2`) tile
/// configs (both multiply out to 16), so one constant covers both.
const GEMM_WMMA_WARPS_PER_BLOCK: u32 = 16;

/// Output columns the narrow `gemm_xwt_wmma_q*_narrow` workgroup tile
/// covers (the original 1x2-fragment/`TILE_M`=64 config, from before the
/// per-wave-efficiency round's `TILE_M`=128 rewrite) — see
/// `gemm_xwt_quant_wmma_narrow.hip`'s module doc for the full measured
/// crossover table this file's `GEMM_WMMA_NARROW_THRESHOLD_M` is based on.
const GEMM_WMMA_NARROW_TILE_M: u32 = 64;
/// Shape-aware dispatch round (issue #6, fixing the per-wave-efficiency
/// round's accepted `m=1024` regression): `m < 2048` routes to the narrow
/// (`TILE_M`=64) WMMA config instead of the default (`TILE_M`=128) one.
/// Measured via a standalone interleaved hipcc harness (rows=512, q4_k,
/// both configs including the vec4-X-load lever): narrow wins by 6.6%-30.9%
/// for every `m` in {512,768,1024,1280,1536,1792}, default wins by 1.5%-8.4%
/// for every `m` in {2048,3072,4096} — see `gemm_xwt_quant_wmma_narrow.hip`
/// for the full table. Ornith-9b's real WMMA-eligible `m`s are {1024 (attn_k/
/// attn_v), 2048, 4096, 8192, 12288} — this threshold routes exactly the
/// regressed 1024 shape to the narrow config and leaves every other real
/// shape on the default one.
const GEMM_WMMA_NARROW_THRESHOLD_M: u32 = 2048;

/// Split-K WMMA GEMM (issue #6's split-K follow-up): adds a `blockIdx.z`
/// grid dimension to the default-tile (`TILE_M`=128) WMMA kernel that splits
/// the K reduction into independent partial sums, multiplying grid block
/// count by `num_splits` — a lever completely orthogonal to `m`/tile width,
/// so it applies to *any* WMMA-eligible shape whose plain grid
/// (`m.div_ceil(GEMM_WMMA_TILE_M) * rows.div_ceil(GEMM_WMMA_TILE_ROWS)`,
/// always computed against the *default* tile regardless of which plain
/// kernel this shape would otherwise use) is narrow. ffn-down's class
/// (`m=hidden, n=intermediate`, e.g. ornith-9b's m=4096/n=12288: grid.x=32,
/// only 128 blocks at rows=512 against gfx1101's 60 CUs) is the motivating
/// shape — see `rocml-kernels/kernels/gemm_xwt_wmma_splitk_impl.h`'s module
/// doc for the full diagnosis.
///
/// Measured via a same-process interleaved harness
/// (`rocml-kernels/tests/gemm_xwt_wmma_splitk_perf.rs`, rows=512, q4_k,
/// median of 5 interleaved rounds — this machine's ambient GPU clock state
/// swings enough between separate `cargo test` invocations to make
/// cross-process comparisons unusable, per `gemm_xwt_quant_wmma_perf.rs`'s
/// module doc, so every number below comes from one process):
///
/// | shape (m/n)           | plain-grid blocks | plain GFLOP/s | split(4) GFLOP/s | split(4) wins by |
/// |------------------------|--------------------|----------------|--------------------|--------------------|
/// | 4096/12288 (ffn-down)  | 128                | 14242.6        | 17104.2            | +20.1%             |
/// | 2048/4096              | 64                 | 12202.2        | 16635.3            | +36.3%             |
/// | 1024/4096              | 32                 | 7523.0         | 12962.3            | +72.3%             |
///
/// Every shape the issue asked to measure wins, and the win *grows* as the
/// plain grid narrows — the opposite of a lever that only helps at one
/// specific shape. This is why `splitk_num_splits` has no lower `m` bound
/// tied to `GEMM_WMMA_NARROW_THRESHOLD_M`: at `m=1024` (this codebase's only
/// real production shape below that threshold, ornith-9b's attn_k/attn_v),
/// split-K on the *default* tile (12962.3 GFLOP/s) also beats the narrow-
/// tile (`TILE_M`=64) kernel production currently dispatches there
/// (10100.2 GFLOP/s, same harness/session) by +28.3% — so split-K
/// eligibility is checked *before* the narrow-vs-default fork below, and
/// wins it outright wherever it applies. The narrow-tile kernel itself
/// stays as the fallback for any `m` too narrow for split-K to reach 2/4-way
/// alignment (see `SPLITK_CANDIDATE_SPLITS`) or whose grid already clears
/// `SPLITK_BLOCK_THRESHOLD`.
const SPLITK_BLOCK_THRESHOLD: u32 = 240;
/// Minimum reduction depth (`n`) for split-K to be worth its extra
/// reduce-pass launch and scratch round-trip. Every `n` this dispatch ever
/// sees in production is either `hidden` or `feed_forward_length` (each at
/// least 4096 on every registry model), and `n=4096` already shows the largest
/// measured win of the three crossover shapes at `m=1024` (+72.3%, see
/// `SPLITK_BLOCK_THRESHOLD`'s table) — no measured shape came anywhere close
/// to a break-even point, so this is set structurally (`2 *
/// GEMM_WMMA_K_STAGE * 4`, i.e. "large enough that even a 4-way split still
/// stages a real multi-iteration K reduction per split") rather than at a
/// measured crossover that was never observed to exist within real shapes.
const SPLITK_MIN_N: u32 = 256;
/// Fixed split counts split-K ever picks, tried widest-first — `n` must
/// divide evenly by `splits * GEMM_WMMA_K_STAGE` for every split's
/// `[k_start, k_end)` range to stay `K_STAGE`-aligned (see the split-K
/// kernel header's module doc for why that alignment matters: it's what
/// lets the kernel skip the short-stage zero-pad path entirely). A given
/// `(m, n)` shape always resolves to the same split count on every call —
/// the "fixed split count per shape" determinism the issue's numerics
/// section requires.
const SPLITK_CANDIDATE_SPLITS: [u32; 2] = [4, 2];

/// Output rows a `gemm_xwt_wmma_q*_micro` workgroup tile covers — mirrors
/// the kernel's `TR` (fixes the launch's grid.y). 16 is WMMA's minimum row
/// tile (one 16x16 fragment row) — see `gemm_xwt_quant_wmma_micro.hip`'s
/// module doc for why this exists (qwen35moe M4 lever 1: MoE's grouped-by-
/// expert batched GEMM averages ~2-16 rows per expert group, far below
/// `GEMM_WMMA_TILE_ROWS`).
const GEMM_WMMA_MICRO_TILE_ROWS: u32 = 16;
/// Output columns the micro `gemm_xwt_wmma_q*_micro` workgroup tile covers
/// — mirrors the kernel's `TM`. 64 (not 128) to maximize `grid.x` at this
/// lever's real shapes (`m` in {512, 2048} on Ornith-1.5-35B-A3B's expert
/// gate/up/down projections) — more, smaller blocks fill the GPU better
/// than fewer, larger ones when `grid.y` is already tiny.
const GEMM_WMMA_MICRO_TILE_M: u32 = 64;
/// Warps per micro WMMA workgroup — fixes the launch's block.y. `WM`=1
/// (forced: `TR/16`=1 has no other divisor) times `WN`=4
/// (`TM`/16/WN`=1`, one 16x16 subtile per warp).
const GEMM_WMMA_MICRO_WARPS_PER_BLOCK: u32 = 4;

/// Picks a fixed, deterministic split count for `(m, n, rows)` — `1` means
/// "don't split" (falls back to the narrow/default WMMA dispatch below,
/// unchanged). Pure function of the shape, so the same GEMM call always
/// resolves to the same split count and therefore the same summation order
/// run to run (this function has no runtime/scheduling input). `max_m` is
/// `SplitKScratch`'s allocated capacity (this model's `hidden`) — a shape
/// wider than that can never split regardless of its grid width, since
/// there's no scratch to hold its partials safely.
fn splitk_num_splits(m: u32, n: u32, rows: u32, max_m: u32) -> u32 {
    if m > max_m || n < SPLITK_MIN_N {
        return 1;
    }
    let plain_blocks = m.div_ceil(GEMM_WMMA_TILE_M) * rows.div_ceil(GEMM_WMMA_TILE_ROWS);
    if plain_blocks >= SPLITK_BLOCK_THRESHOLD {
        return 1;
    }
    for splits in SPLITK_CANDIDATE_SPLITS {
        if n.is_multiple_of(splits * GEMM_WMMA_K_STAGE) {
            return splits;
        }
    }
    1
}

impl QuantKernels {
    /// `gemm_xwt_<dtype>(x, w, out, rows, m, n)`: `out[rows,m] = X[rows,n] *
    /// dequant(W)^T`, the batched prefill-path sibling of
    /// [`QuantKernels::gemv`]. Same dtype restriction as `gemv`
    /// (Q8_0/Q4_K/Q5_K/Q6_K only).
    ///
    /// Dispatches to a WMMA matrix-core kernel (`gemm_xwt_quant_wmma.hip`/
    /// `_narrow.hip`, issue #6's follow-ups) whenever the shape can fill at
    /// least one full tile cleanly — `rows >= GEMM_WMMA_TILE_ROWS`, `m >=
    /// GEMM_WMMA_NARROW_TILE_M` (64, the smaller of the two WMMA tile
    /// configs' minimums), and `m`/`n` both multiples of 16 (WMMA's native
    /// fragment width) — and falls back to the scalar-FMA kernel otherwise
    /// (a short last chunked-prefill chunk, a shape WMMA can't tile at all,
    /// or an `m` narrower than even the narrow config's tile). `n` is
    /// already a multiple of 16 for every dtype this dispatches to (Q8_0/
    /// Q4_K/Q5_K/Q6_K block widths are 32/256/256/256), so the check below
    /// only ever turns on the fallback for `rows`/`m`.
    ///
    /// **Shape-aware WMMA-width dispatch** (this round): among
    /// WMMA-eligible shapes, `m < GEMM_WMMA_NARROW_THRESHOLD_M` (2048)
    /// further routes to the narrow-tile kernel instead of the default one
    /// — see that constant's doc for the measured crossover. Below
    /// `GEMM_WMMA_NARROW_TILE_M` (64) even the narrow config can't fill a
    /// tile column, so the scalar kernel (whose grid makes `m` itself a
    /// grid dimension, giving `m=32` still 32+ independent blocks) stays
    /// the fallback — this is the same reasoning the prefill-cleanup
    /// round's original `m >= TILE_M` clause established for Ornith's GDN
    /// per-head alpha/beta gate projections (`m = num_v_heads = 32`). The
    /// split-K round (below) checks its own eligibility *ahead* of this
    /// fork and, when it applies, wins outright at every `m` measured
    /// (including below `GEMM_WMMA_NARROW_THRESHOLD_M`) — this fork is only
    /// ever reached once split-K has already declined the shape.
    ///
    /// Known numeric consequence (measured, not a bug — see the kernel
    /// source's module doc and issue #6's synthetic correctness tests for
    /// why the WMMA kernel itself is verified correct against an
    /// f16-rounded reference): every quantized linear layer a chunked-
    /// prefill call routes through either WMMA path now rounds both
    /// operands to f16 before the matrix-core multiply. That's the expected
    /// ~5e-4 relative per-layer input-rounding error the issue anticipated,
    /// but it compounds across a ~30-layer model measurably more than the
    /// *pre-WMMA* pure-f32 pipeline this dispatch replaced for `rows >=
    /// 128` shapes — `rocml/tests/qwen35_chunked_prefill_parity.rs`'s and
    /// `rocml/tests/snapshot_equivalence.rs`'s final-logits tolerances were
    /// recalibrated to `1e-2` for it (measured max relative logit error
    /// ~0.16%-0.55%), while every *greedy-token* gate (with a near-tie
    /// escape hatch) still passes.
    ///
    /// **int8 MMQ (int8-MMQ-integration round)**: when `mmq_enabled` was
    /// set at load time (`LoadOptions::with_mmq`, off by default — see
    /// `.claude/CLAUDE.md` for the validation status that decides whether
    /// this ever flips to on by default) *and* the shape is WMMA-eligible
    /// (every weight kind this dispatch handles now has an MMQ kernel, so
    /// no further dtype gating is needed beyond `MmqKernels::supports`),
    /// this routes to the int8 integer-matrix-unit path instead of f16
    /// WMMA — quantizing `x` on the fly (`kernels_mmq.rs`'s
    /// `MmqKernels::gemm`) rather than rounding it to f16. This is a
    /// strictly larger precision change than the WMMA rounding above (int8
    /// activations, not f16), so it is gated separately and never turned on
    /// just because WMMA already is. `mmq_eligible=false`
    /// (`weights/linear.rs`'s `mmq_eligible_by_name`) forces WMMA/scalar.
    /// MMQ always uses the default (non-narrow) tile width — it was never
    /// swept against the narrow config, and MMQ stays off by default
    /// regardless (see its own validation-ladder writeup).
    ///
    /// **Split-K (split-K round)**: ahead of the plain default-tile WMMA
    /// path (below MMQ in priority — MMQ is a strictly different, opt-in
    /// numeric path and this dispatch never second-guesses it once enabled),
    /// `splitk_num_splits` picks a fixed split count from `(m, n, rows)`
    /// alone. `> 1` routes to `SplitKKernels::gemm` (the split-K WMMA pass
    /// plus its deterministic ascending-order reduce pass); `1` falls
    /// through to the narrow/default WMMA dispatch below unchanged. See
    /// `SPLITK_BLOCK_THRESHOLD`'s doc for the measured crossover.
    ///
    /// **Micro tile, `m >= n` gate (qwen35moe M4 lever 1)**: a standalone
    /// interleaved perf probe (`rocml-kernels/tests/
    /// gemm_xwt_quant_wmma_micro_perf.rs`, `rows` in {1,2,4,8,...,127}) at
    /// Ornith-1.5-35B-A3B's two real expert-projection shapes found the
    /// micro kernel's own wall time is dominated by `n` (its K-reduction
    /// depth, `n/K_STAGE`=128 sequential `__syncthreads`-gated iterations
    /// for `n`=2048) almost independent of `m`/grid width — widening the
    /// micro block's warp count (a scratch `TM`=128/`WN`=8 experiment) or
    /// `m` itself (a `m` sweep at fixed `n`=2048, rows=8: 512->2048 moved
    /// scalar's cost 78->330us but micro's stayed flat at ~300-302us) never
    /// moved this kernel's time — a genuine architectural property of this
    /// tiny-block (1-8 warps), heavily-`__syncthreads`-serialized design,
    /// not a tunable dispatch-layer parameter. Scalar's own cost instead
    /// scales with `m` (it always gets `m` grid blocks, queuing in waves
    /// once `m` exceeds the GPU's CU count). Net: down's shape (`m=hidden`
    /// 2048 `>= n=expert_ff_len` 512) measures a clean 1.3x-2.7x win across
    /// the whole `rows` range; gate/up's shape (`m=expert_ff_len` 512
    /// `< n=hidden` 2048) measures a 2.6x-3.8x **regression** at the same
    /// `rows`, closing to parity only once `m` grows to ~`n` (the `m`-sweep
    /// crosses 1.0x between `m`=1792 and `m`=2048 at `n`=2048). `m >= n`
    /// captures this measured crossover directly and routes each of this
    /// checkpoint's two real shapes to whichever kernel actually wins it,
    /// rather than a blanket "opt-in always dispatches" rule.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gemm(
        &self,
        dtype: GgmlDType,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
        mmq_scratch: MmqScratch,
        mmq_eligible: bool,
        splitk_scratch: SplitKScratch,
        allow_micro: bool,
    ) -> Result<(), RocmlError> {
        let wmma_eligible = rows >= GEMM_WMMA_TILE_ROWS
            && m >= GEMM_WMMA_NARROW_TILE_M
            && m.is_multiple_of(16)
            && n.is_multiple_of(16);
        let num_splits = if wmma_eligible {
            splitk_num_splits(m, n, rows, splitk_scratch.max_m)
        } else {
            1
        };
        // Micro tile (qwen35moe M4 lever 1): only reachable when the shape
        // isn't already WMMA-eligible above (`rows < GEMM_WMMA_TILE_ROWS`,
        // the scalar-fallback zone) and the caller opted in — see
        // `gemm_wmma_micro_cfg`'s doc for why this stays opt-in rather than
        // a blanket small-`rows` dispatch for every caller.
        let micro_eligible = !wmma_eligible
            && allow_micro
            && rows >= 1
            && m >= GEMM_WMMA_MICRO_TILE_M
            && m >= n
            && m.is_multiple_of(16)
            && n.is_multiple_of(16);
        if wmma_eligible && self.mmq_enabled && mmq_eligible && MmqKernels::supports(dtype) {
            self.mmq.gemm(dtype, x, w, out, rows, m, n, mmq_scratch)
        } else if wmma_eligible && num_splits > 1 {
            self.splitk
                .gemm(dtype, x, w, out, rows, m, n, num_splits, splitk_scratch)
        } else if wmma_eligible && m < GEMM_WMMA_NARROW_THRESHOLD_M {
            self.gemm_wmma_cfg(
                self.wmma_narrow_fn(dtype)?,
                GEMM_WMMA_NARROW_TILE_M,
                x,
                w,
                out,
                rows,
                m,
                n,
            )
        } else if wmma_eligible {
            self.gemm_wmma_cfg(
                self.wmma_fn(dtype)?,
                GEMM_WMMA_TILE_M,
                x,
                w,
                out,
                rows,
                m,
                n,
            )
        } else if micro_eligible {
            self.gemm_wmma_micro_cfg(self.wmma_micro_fn(dtype)?, x, w, out, rows, m, n)
        } else {
            self.gemm_scalar(dtype, x, w, out, rows, m, n)
        }
    }

    fn wmma_fn(&self, dtype: GgmlDType) -> Result<&Function, RocmlError> {
        match dtype {
            GgmlDType::Q8_0 => Ok(&self.gemm_wmma_q8_0_fn),
            GgmlDType::Q4_K => Ok(&self.gemm_wmma_q4_k_fn),
            GgmlDType::Q5_K => Ok(&self.gemm_wmma_q5_k_fn),
            GgmlDType::Q6_K => Ok(&self.gemm_wmma_q6_k_fn),
            other => Err(RocmlError::Config(format!(
                "gemm_quant: {other:?} has no fused WMMA kernel (internal loader bug)"
            ))),
        }
    }

    fn wmma_narrow_fn(&self, dtype: GgmlDType) -> Result<&Function, RocmlError> {
        match dtype {
            GgmlDType::Q8_0 => Ok(&self.gemm_wmma_q8_0_narrow_fn),
            GgmlDType::Q4_K => Ok(&self.gemm_wmma_q4_k_narrow_fn),
            GgmlDType::Q5_K => Ok(&self.gemm_wmma_q5_k_narrow_fn),
            GgmlDType::Q6_K => Ok(&self.gemm_wmma_q6_k_narrow_fn),
            other => Err(RocmlError::Config(format!(
                "gemm_quant: {other:?} has no fused narrow WMMA kernel (internal loader bug)"
            ))),
        }
    }

    fn wmma_micro_fn(&self, dtype: GgmlDType) -> Result<&Function, RocmlError> {
        match dtype {
            GgmlDType::Q8_0 => Ok(&self.gemm_wmma_q8_0_micro_fn),
            GgmlDType::Q4_K => Ok(&self.gemm_wmma_q4_k_micro_fn),
            GgmlDType::Q5_K => Ok(&self.gemm_wmma_q5_k_micro_fn),
            GgmlDType::Q6_K => Ok(&self.gemm_wmma_q6_k_micro_fn),
            other => Err(RocmlError::Config(format!(
                "gemm_quant: {other:?} has no fused micro WMMA kernel (internal loader bug)"
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gemm_scalar(
        &self,
        dtype: GgmlDType,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let function = match dtype {
            GgmlDType::Q8_0 => &self.gemm_q8_0_fn,
            GgmlDType::Q4_K => &self.gemm_q4_k_fn,
            GgmlDType::Q5_K => &self.gemm_q5_k_fn,
            GgmlDType::Q6_K => &self.gemm_q6_k_fn,
            other => {
                return Err(RocmlError::Config(format!(
                    "gemm_quant: {other:?} has no fused kernel (internal loader bug)"
                )))
            }
        };
        let cfg = LaunchConfig {
            grid: (m, rows.div_ceil(GEMM_QUANT_TILE_ROWS), 1),
            block: (32, GEMM_QUANT_WARPS_PER_BLOCK, 1),
            shared_mem_bytes: GEMM_QUANT_TILE_ELEMS * size_of::<f32>() as u32,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches every gemm_xwt_<quant>'s signature (const
        // float*, const void*, float*, unsigned x3); block = (32, 8, 1)
        // matches the kernel's fixed warp-per-`ROWS_PER_WARP`-rows tiling.
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// Shared WMMA launch: `tile_m` selects which tile config `function`
    /// was compiled for (128 for the default kernels, 64 for the narrow
    /// ones) — both use `GEMM_WMMA_WARPS_PER_BLOCK`=16, so only the grid.x
    /// and shared-memory computation vary.
    #[allow(clippy::too_many_arguments)]
    fn gemm_wmma_cfg(
        &self,
        function: &Function,
        tile_m: u32,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        // x2: double-buffered LDS K-stage tiles (padded rows, see LDS_PAD).
        let shared_mem_bytes = 2
            * (GEMM_WMMA_TILE_ROWS + tile_m)
            * (GEMM_WMMA_K_STAGE + GEMM_WMMA_LDS_PAD)
            * size_of::<u16>() as u32;
        let cfg = LaunchConfig {
            grid: (m.div_ceil(tile_m), rows.div_ceil(GEMM_WMMA_TILE_ROWS), 1),
            block: (32, GEMM_WMMA_WARPS_PER_BLOCK, 1),
            shared_mem_bytes,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches every gemm_xwt_wmma_<quant>[_narrow]'s
        // signature (const float*, const void*, float*, unsigned x3);
        // block/grid/shared_mem_bytes match `function`'s compiled-in
        // TILE_ROWS/tile_m/K_STAGE tiling (caller picks the `function`/
        // `tile_m` pair that match each other).
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }

    /// Micro-tile WMMA launch (qwen35moe M4 lever 1) — a separate helper
    /// from [`Self::gemm_wmma_cfg`] rather than a third `tile_rows`/
    /// `warps_per_block` parameter on it: the micro config's block shape
    /// (`GEMM_WMMA_MICRO_WARPS_PER_BLOCK`=4, vs 16 for the default/narrow
    /// configs) and row-tile (16, vs the fixed 128 `gemm_wmma_cfg` bakes
    /// into its grid.y/shared-mem formula) both differ, and there is only
    /// one micro config to dispatch to (unlike default-vs-narrow's genuine
    /// two-way `m`-based choice), so a dedicated function reads more
    /// directly than threading two more parameters through the shared one.
    #[allow(clippy::too_many_arguments)]
    fn gemm_wmma_micro_cfg(
        &self,
        function: &Function,
        x: DevPtr,
        w: DevPtr,
        out: DevPtr,
        rows: u32,
        m: u32,
        n: u32,
    ) -> Result<(), RocmlError> {
        let shared_mem_bytes = 2
            * (GEMM_WMMA_MICRO_TILE_ROWS + GEMM_WMMA_MICRO_TILE_M)
            * (GEMM_WMMA_K_STAGE + GEMM_WMMA_LDS_PAD)
            * size_of::<u16>() as u32;
        let cfg = LaunchConfig {
            grid: (
                m.div_ceil(GEMM_WMMA_MICRO_TILE_M),
                rows.div_ceil(GEMM_WMMA_MICRO_TILE_ROWS),
                1,
            ),
            block: (32, GEMM_WMMA_MICRO_WARPS_PER_BLOCK, 1),
            shared_mem_bytes,
        };
        let mut params = kernel_params!(x, w, out, rows, m, n);
        // SAFETY: params matches every gemm_xwt_wmma_<quant>_micro's
        // signature (const float*, const void*, float*, unsigned x3);
        // block/grid/shared_mem_bytes match the micro kernels' compiled-in
        // TR=16/TM=64/WM=1/WN=4 tiling.
        unsafe { function.launch(&cfg, &mut params, None) }.map_err(Into::into)
    }
}

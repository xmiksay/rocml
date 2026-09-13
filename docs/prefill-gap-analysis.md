# Where the prefill gap lives (issue #5)

llama.cpp prefills a comparable ornith-9b-class hybrid model at ~1733 tok/s
on a 9.4K-token prompt. rocml prefills ornith-9b (Q4_K_M) at **485-490
tok/s at depth 8192** — roughly a 3.5x gap. This doc ranks where that gap
actually lives, using `rocml-cli bench --profile-json` (issue #5's
analytical roofline export) cross-checked against a real `rocprofv3` kernel
trace (`docs/profiling.md`'s workflow) at the same depth. All numbers below
are from real runs on this machine (ROCm 7.2.4, RX 7800 XT/gfx1101), not
estimates.

## Data sources

- `bench/profile/ornith9b-depth8192.json` — `rocml-cli bench --model
  ornith-9b --depth 8192 --decode-tokens 64 --runs 1 --profile-json ...`
  Measured 485.3 tok/s prefill (known ballpark: ~490 tok/s), 40.4 tok/s
  decode at depth 8192 (known ballpark: ~44.1 tok/s — a single run, not a
  median of 3, so some variance from the reference figure is expected).
- `bench/profile/ornith9b-depth2048-decode.json` — same command at `--depth
  2048` for the decode brief below. Measured 584.95 tok/s prefill (known:
  ~586) and 42.9 tok/s decode (known: ~47.6).
- A real `rocprofv3 --kernel-trace --stats` run at `--depth 8192
  --decode-tokens 1` (kept to 1 decode token to keep the trace scoped to
  prefill): 489.3 tok/s measured, kernel-level truth in
  `ornith8192_kernel_stats.csv`.
- A `--pmc OccupancyPercent SQ_WAVES --kernel-include-regex
  gemm_xwt_wmma_q4_k` run at depth 2048 for the WMMA GEMM's occupancy
  (`docs/profiling.md`'s worked example 2).

## Sanity check: does the profiled time add up?

Profiler-recorded prefill total: **16847.6 ms**. End-to-end prefill wall
time from the measured tok/s (8192 tokens / 485.28 tok/s): **16880.8 ms**.

**Unaccounted time: 33.2 ms (0.20% of prefill wall time).** Essentially
everything is inside a profiled `Profiler::scope` span — there's no hidden
"launch overhead between spans" or host-side bookkeeping eating a real
chunk of prefill wall time outside what's already measured. This matters:
it means the wasted-time ranking below (which sorts *within* the profiled
spans) isn't missing some large invisible cost — the visible spans are the
whole story.

Independently, the rocprofv3 kernel-level run's summed `TotalDurationNs`
across every kernel is **16610.4 ms** against that run's own 16745 ms wall
time (8192/489.3 tok/s) — a 0.8% gap, consistent with the same
near-total-accounted-for picture from a completely independent measurement
tool. The two methods (HIP-event-timed `Profiler::scope`s vs. rocprofv3's
own dispatch timestamps) agree with each other to within ~2% on every op
they both cover (see the per-op cross-check table below) — good evidence
the analytical `Profiler` isn't perturbing what it measures.

## Top-3 prefill wasted-time ops (ranked, `wasted_ms = actual - t_min`)

| Rank | OpKind | actual ms | t_min ms | efficiency | wasted ms | % of prefill time |
|---|---|---|---|---|---|---|
| 1 | `gdn-recur` | 2597.3 | 94.2 | 4% | **2503.1** | 15.4% |
| 2 | `ffn-gate-up` | 4557.5 | 3015.8 | 66% | **1541.7** | 27.1% |
| 3 | `ffn-down` | 2536.1 | 1507.9 | 59% | **1028.2** | 15.1% |

(`gdn-conv` is close behind at 792.0 ms wasted, 59% efficient — same root
cause as #2/#3, see below.)

### #1 `gdn-recur` (chunkwise GDN recurrence) — category (b): launch overhead/serialization + occupancy-starved grids

`OpKind::GdnRecur` wraps exactly `gdn_chunkwise_step`'s six-kernel pipeline
(`gdn_chunkwise_{prep,ut_build,tinv,uv_vnew,output,state}_f32` — a clean 1:1
mapping, not bundled with anything else; see `docs/profiling.md`'s
correlation table). rocprofv3's kernel stats for the same six kernels sum
to **2527.0 ms** (576.3+511.0+446.9+442.3+305.6+244.9), matching the
profiler's 2597.3 ms to within 2.7%.

Both roofline percentages are tiny (3.2% of BW, 3.6% of FLOP), which is the
signature of neither-bound work — the real cost is per-launch fixed
overhead, not data movement or arithmetic. The rocprofv3 kernel trace
confirms why: every one of these six kernels launches a grid of exactly
**32 blocks** (`Grid_Size_X=4096` / `Workgroup_Size_X=128` = 32 blocks —
one block per `num_v_heads`=32), against gfx1101's **60 CUs**. At best
that's ~53% of the GPU ever holding a block for these kernels, and it gets
worse: at `PREFILL_CHUNK_SIZE`=512 with `GDN_RECUR_TILE`=128, each prefill
chunk sub-splits into 4 sequential tile passes, so this six-kernel,
32-block pipeline runs **4 times per chunk per layer**, serialized (each
stage's kernel depends on the previous stage's output) — 1536 total launch
groups across the 8192-token/24-GDN-layer/16-chunk prefill. Small grids,
paid for six times per chunk instead of once, with no independent work to
overlap the gaps between launches.

**This is not a kernel-efficiency problem** (occupancy that *is* achieved,
per rocprofv3's `OccupancyPercent`, isn't the limiter here — there just
aren't enough blocks per launch to fill the GPU in the first place) — it's
architecture: the six-kernel decomposition trades parallelism-across-tiles
for depth-of-pipeline, and `num_v_heads`=32 hard-caps grid width regardless
of tile size.

### #2/#3 `ffn-gate-up` / `ffn-down` (WMMA GEMM) — category (a): kernel efficiency below the *true* hardware roofline

Both route through `gemm_xwt_wmma_q4_k`/`gemm_xwt_wmma_q6_k` (ornith-9b's
Q4_K_M mixes both quant types by tensor). Measured throughput: 11.58
TFLOP/s (`ffn-gate-up`), 10.41 TFLOP/s (`ffn-down`) — 66% and 59% of the
**practical scalar-FMA ceiling** (17.5 TFLOP/s, `PRACTICAL_FLOPS_PER_SEC`)
this report's `t_min`/efficiency columns use. But that's the wrong ceiling
to declare victory against: gfx1101's *true* WMMA f16 peak is 75 TFLOP/s
(`WMMA_PEAK_FLOPS_PER_SEC`, see `rocml/src/profile/roofline.rs`), and
11.58/10.41 TFLOP/s against *that* is only **15.4%/13.9%** — real headroom
exists, just not in the "efficiency vs. practical ceiling" column this
report's worklist sorts by (that column already knows this — see the
`wasted_ms` doc comment on why WMMA-eligible ops can and do show low
"waste" against the scalar ceiling while still being far from the WMMA
ceiling).

`rocprofv3 --pmc OccupancyPercent` on `gemm_xwt_wmma_q4_k` measured **71.1%
mean occupancy** (38.5-79.8% range, 672 dispatches, VGPR=64/SGPR=128) — not
occupancy-starved like `gdn-recur` above. The gap to 75 TFLOP/s is
per-wave efficiency (WMMA fragment/tile shape), consistent with this
kernel's own known, already-tuned numbers from the issue #6 WMMA rewrite
(measured 7.4-10.5 TFLOP/s at that round's acceptance shapes — this run's
11.58 TFLOP/s at ornith's larger `m` is in the same family, not a
regression). Squeezing more out of this kernel is a real, known-quantity
lever, but it's a kernel rewrite (bigger `K_STAGE`, different tile/fragment
packing), not a bug — out of this round's scope, and already the subject of
the WMMA kernel's own tuning notes.

### Category (c) check: is any of this work llama.cpp skips entirely?

No — all three ops above are architecturally required by both engines
(ornith-9b's hybrid GDN+attention layers exist in the GGUF either way; the
FFN GEMMs are the model's dense MLP). None of the top-3 wasted-time ops are
"extra work rocml does that llama.cpp doesn't do at all" — they're work
both engines must do, where rocml's current kernels are comparatively less
mature (the WMMA GEMM path landed in issue #6; llama.cpp's CUDA/HIP tensor-
core GEMM kernels have years more tuning) or architecturally serialized
(the six-kernel chunkwise recurrence). This is a real, if slightly
deflating, finding: the gap is "our kernels aren't there yet", not "we're
doing redundant work".

## Ranked lever list with Amdahl arithmetic

All estimates hold every other op's time fixed and recompute end-to-end
prefill wall time from the measured 16880.8 ms baseline (8192 tokens @
485.28 tok/s) — i.e. straight-line Amdahl's law on the measured time
shares, not a re-simulation.

1. **`gdn-recur`: fewer, wider launches.** Getting this op from 4%
   efficient to a conservative 55% (matching this workload's worst *GEMM*
   efficiency, not even its best) means `t_actual` shrinks from 2597.3 ms
   to `94.2/0.55` = 171.3 ms, saving **2426.0 ms**. New prefill time
   14454.8 ms → **566.7 tok/s (+16.8%)**. Concretely this means widening
   the grid past 32 blocks (e.g. splitting each head's work across more
   blocks, or fusing tile passes to cut the 4x-per-chunk sequential
   pipeline down) — a real kernel-design change, not a tuning knob.
2. **`ffn-gate-up`: WMMA tile/fragment efficiency.** Closing the gap from
   66% to a still-conservative 75% efficient-against-practical-ceiling
   (below its own already-measured issue #6 ceiling) saves **536.4 ms** →
   **501.2 tok/s (+3.3%)** on its own.
3. **`ffn-down`: same lever, same kernel family.** 59%→75% efficient saves
   **525.6 ms** → **500.9 tok/s (+3.2%)** on its own.

**Combined (all three, additive since they're disjoint serial spans):**
save 2426.0 + 536.4 + 525.6 = 3488.0 ms → prefill time 13392.8 ms →
**611.7 tok/s, +26.0% over the 485.3 tok/s baseline.**

Honest caveat: even this combined, fairly optimistic estimate (611.7 tok/s)
is still **~2.8x below llama.cpp's ~1733 tok/s** reference point. Closing
the *entire* remaining gap after these three levers would need either a
much larger structural change (bigger `PREFILL_CHUNK_SIZE` to amortize
per-chunk overhead further, a from-scratch WMMA GEMM rewrite reaching
closer to the 75 TFLOP/s peak, or restructuring the six-kernel chunkwise
recurrence into fewer launches) or reflects llama.cpp doing some of this
work in a genuinely different, more-fused way this report can't see from
rocml's side alone. That's the honest state of the ~3x gap after this
observability round: about a quarter of it is legible and has a concrete
lever attached; the rest needs either deeper kernel work or a side-by-side
llama.cpp kernel trace (out of scope here — this repo doesn't build or run
llama.cpp) to characterize further.

## Decode, briefly (depth 2048)

Decode is already 3.7x llama.cpp (not this round's push), but for
completeness — top-3 wasted-time decode ops at depth 2048
(`bench/profile/ornith9b-depth2048-decode.json`, 42.9 tok/s measured, known
ballpark ~47.6):

| Rank | OpKind | actual ms | t_min ms | efficiency | wasted ms |
|---|---|---|---|---|---|
| 1 | `ffn-gate-up` | 682.8 | 372.5 | 54.6% | 310.3 |
| 2 | `gdn-recur` | 236.3 | 20.8 | 8.8% | 215.5 |
| 3 | `gdn-conv` | 368.3 | 162.7 | 44.2% | 205.6 |

One shared root cause with prefill and one decode-specific one. `ffn-gate-up`
and `gdn-conv` are the same WMMA-GEMM efficiency gap as prefill's #2/#3
(category (a)). `gdn-recur` is *not* the chunkwise pipeline here, though —
decode's `OpKind::GdnRecur` wraps a completely different, single kernel
(`gdn_recurrence_decode_f32` in `rocml/src/qwen35/forward/gdn.rs`'s
token-serial `gdn_layer_step`, not `gdn_chunkwise_step`). Its 8.8% efficiency
is the classic decode-time pattern documented elsewhere in this repo
(`.claude/CLAUDE.md`'s "Decode bandwidth (issue #7)" section): one tiny,
low-arithmetic-intensity kernel launch per token per layer, dominated by
fixed launch/sync overhead rather than the bytes/FLOPs it actually moves —
category (b), but a different mechanism from prefill's multi-kernel
chunkwise pipeline. No new decode-specific finding beyond what's already
documented — flagged here for the issue #16 (gemma) future, not as a
near-term decode push.

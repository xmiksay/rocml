# Kernel-level profiling with rocprofv3 (issue #5)

`rocml::profile::Profiler` (see `README.md`'s "Observability" section and
`rocml/src/profile/mod.rs`) gives an **analytical** roofline view: bytes and
FLOPs are computed from shapes, not measured, and wall time comes from HIP
events wrapping whole `Profiler::scope` spans — which sometimes bundle
several kernel launches together (see "Correlating OpKind spans to kernel
names" below). For the *actual* per-kernel truth — real measured kernel
duration, occupancy, VGPR/SGPR usage — use ROCm's own profiler, `rocprofv3`.

This machine has ROCm 7.2.4; both `rocprofv3` and the legacy `rocprof` are on
`PATH` at `/opt/rocm/bin/`. Everything below uses `rocprofv3` — the ROCm 7.x
default and the one with gfx1101-aware counter names (`OccupancyPercent`,
`MeanOccupancyPerActiveCU`, etc., visible via `rocprofv3 -L`).

## Quick reference

```
rocprofv3 [rocprofv3 options] -- <application> [application args]
```

Always run against the **release** binary (`target/release/rocml-cli`) —
profiling a debug build measures the debug build, not the kernel behavior
this repo ships.

## Worked example 1: kernel time table

```
$ rocprofv3 --kernel-trace --stats -d /tmp/rocprof_out -o run1 -f csv -- \
    ./target/release/rocml-cli bench --model qwen3.5-2b --depth 512 --decode-tokens 32 --runs 1
...
loaded: 24 layers; benchmarking 512 prompt tokens / 32 decode tokens x 1 run(s) (decode measured at depth 512)
  run 1/1: prompt 2306.5 tok/s, decode 103.5 tok/s
```

`--kernel-trace` records every kernel dispatch (`<output>_kernel_trace.csv`,
per-launch VGPR/SGPR/grid/timestamps); `--stats` additionally aggregates them
into `<output>_kernel_stats.csv` — the one to read first:

```
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"gemv_q8_0",5985,126417532,21122.394653,28.80,2560,903650,65663.521796
"gemm_xwt_wmma_q8_0",150,124643012,830953.413333,28.40,196122,1377375,362629.253807
"gdn_recurrence_decode_f32",576,34518290,59927.586806,7.86,58640,61161,417.831120
"attn_decode_partial_f16",192,29318728,152701.708333,6.68,147282,157482,2921.793647
"gdn_chunkwise_tinv_f32",72,27220739,378065.819444,6.20,368124,389724,7569.368118
"rmsnorm_f32",3759,20744149,5518.528598,4.73,2240,26240,3778.577936
```

`Calls`/`TotalDurationNs`/`Percentage` are exactly the {calls, wall time, %
of total} triple `rocml-cli bench --profile-json` reports per `OpKind` —
just at the real per-kernel granularity instead of the analytical
per-`Profiler::scope` one.

## Worked example 2: occupancy and register usage on the WMMA GEMM

`--kernel-trace`'s CSV already carries `VGPR_Count`/`SGPR_Count`/`LDS_Block_Size`
per dispatch (constant per kernel, since they're a compile-time property of
the code object):

```
$ python3 -c "
import csv
seen = {}
for row in csv.DictReader(open('/tmp/rocprof_out/run1_kernel_trace.csv')):
    seen.setdefault(row['Kernel_Name'], row)
r = seen['gemm_xwt_wmma_q8_0']
print(r['VGPR_Count'], r['SGPR_Count'], r['LDS_Block_Size'], r['Workgroup_Size_X'], r['Grid_Size_X'])
"
56 128 0 32 3072
```

`gemm_xwt_wmma_q8_0`: 56 VGPRs, 128 SGPRs, 32-wide workgroups (one wave32
wavefront each). `LDS_Block_Size` reads `0` for every kernel on this
ROCm 7.2.4 + gfx1101 combination even for kernels that clearly use LDS
(`gdn_chunkwise_*`'s in-LDS triangular-inverse solve) — a rocprofv3
reporting gap on this target, not a real zero; don't trust that column here.

For occupancy specifically, `-L`/`--list-avail` lists derived counters —
`OccupancyPercent` (`100*reduce(SQ_WAVE_CYCLES,sum)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM/32`)
is the one to use, collected with `--pmc` and (to keep the trace small)
`--kernel-include-regex` scoped to the kernel of interest:

```
$ rocprofv3 --pmc OccupancyPercent SQ_WAVES -d /tmp/rocprof_pmc -o wmma -f csv \
    --kernel-include-regex "gemm_xwt_wmma_q4_k" -- \
    ./target/release/rocml-cli bench --model ornith-9b --depth 2048 --decode-tokens 1 --runs 1
...
Unable to find all counters for agent 1 (gpu-0, gfx1101) in [OccupancyPercent, SQ_WAVES, VALUUtilization].
Found: [OccupancyPercent, SQ_WAVES]. Missing: [VALUUtilization]
```

(`VALUUtilization` isn't defined for gfx1101 — RDNA3 doesn't expose the
counters that derived metric needs; drop it and keep going with whatever
`-L` actually lists for this GPU.) Result, aggregated over 672 dispatches of
`gemm_xwt_wmma_q4_k` across a real 2048-token ornith-9b prefill:

```
n=672  mean_occupancy=71.1%  min=38.5%  max=79.8%   VGPR=64  SGPR=128
```

71% mean occupancy, 64 VGPRs/wave (vs. 56 for the f16-path `..._q8_0`
variant — Q4_K's extra unpack work costs 8 more registers/wave). Not
occupancy-starved, which matters for the gap analysis below: the WMMA GEMM's
gap to its FLOP roofline is not "not enough waves in flight", it's
per-wave efficiency (tile/fragment shape) — see
`docs/prefill-gap-analysis.md`.

## Correlating OpKind spans to kernel names

`Profiler::scope`'s `OpKind` labels are a coarser grouping than "one kernel"
— several bundle multiple launches under one span for event-count reasons
(see `rocml/src/profile/mod.rs`'s module doc). Reading a `--profile-json`
report next to a `rocprofv3 --stats` table requires knowing which is which:

| `OpKind` | Forward-pass call site | Kernel(s) actually launched |
|---|---|---|
| `Embed` | embedding lookup | `embedding_f16_f32` |
| `Norm` | `rmsnorm` | `rmsnorm_f32` |
| `Qkv` (qwen35 chunked prefill) | `attention_chunk.rs` | qkv/gate `gemm_xwt_{wmma_,}q*`/`gemm_xwt_f16`, `extract_heads_f32`, `rmsnorm_f32` (q/k-norm), `rope_neox_partial_f32` — **five+ kernels under one span** |
| `AttnDecode` (decode) | `Kernels::attn_decode` | `attn_decode_partial_f{16,32}` + `attn_decode_reduce_f32` |
| `AttnDecode` (qwen35 chunked **prefill**) | `attention_chunk.rs` | `scatter_kv_chunk[_f16]` + `attn_prefill_flash[_f16]` (`attn_prefill_flash_partial_f16` + `attn_prefill_flash_reduce_f32`) — **a different kernel family from the decode case above, reusing the same `OpKind`** (see the gotcha below) |
| `AttnOut` | attention output projection | `gemm_xwt_{wmma_,}q*`/`gemv_q*` + residual add |
| `GdnConv` | `gdn_chunk.rs` | **bundled**: `attn_qkv`/`attn_gate`/`ssm_alpha`/`ssm_beta` input-projection GEMMs + `causal_conv1d_chunk_f32` + `causal_conv1d_chunk_state_update_f32` + `gdn_gate_chunk_f32` |
| `GdnRecur` | `gdn_chunkwise_step` | `gdn_chunkwise_{prep,ut_build,tinv,uv_vnew,output,state}_f32` (six kernels, clean 1:1 with this span — not bundled with anything else) |
| `GdnOut` | GDN output projection | `rmsnorm_f32` + `gemm_xwt_{wmma_,}q*`/`gemv_q*` (ssm_out) |
| `FfnGateUp` | `ffn_chunk.rs`/`ffn.rs` | gate+up `gemm_xwt_{wmma_,}q*`/`gemv_q*` (two GEMMs) + `silu_mul_f32` |
| `FfnDown` | `ffn_chunk.rs`/`ffn.rs` | down-projection `gemm_xwt_{wmma_,}q*`/`gemv_q*` |
| `LmHead` | final projection | `gemv_f16`/`gemm_xwt_f16` (lm_head stays f16-dequantized, see `README.md`'s registry notes) |

**Gotcha**: `OpKind::AttnDecode` is reused for two structurally different
kernels depending on phase — the decode-time fused flash-decoding kernel
and the chunked-prefill row-tiled flash kernel (`kernels/attn_prefill.hip`).
This is harmless for the roofline math (each phase's rows are aggregated
separately, so a prefill `AttnDecode` row and a decode `AttnDecode` row never
mix), but it means the *label* alone doesn't tell you which kernel produced
a row — check the phase, or cross-reference against `rocprofv3`'s kernel
names filtered to that phase's time window (`Start_Timestamp`/`End_Timestamp`
in the kernel trace CSV can be intersected against the process's own
prefill/decode phase boundaries if you need to be precise; in practice
"prefill AttnDecode" vs "decode AttnDecode" is enough since `bench --depth`
runs prefill once then decode once, back to back, in that order).

`GdnConv`'s bundling is a known, previously-documented gap (see
`.claude/CLAUDE.md`'s "prefill-cleanup round" notes): a large chunk of what
that label's wall time reports is actually four GEMMs, not the conv1d
kernel its name suggests. Splitting it further wasn't in this round's
scope (shared with the decode path's `gdn.rs`) — this doc's job is making
the existing bundling legible via `rocprofv3`, not restructuring `OpKind`.

## Output formats and trace size

`-f csv` (used throughout this doc) is the easiest to script against;
`-f json`/`-f pftrace` are also available (`pftrace` opens directly in
[Perfetto UI](https://ui.perfetto.dev) for a visual timeline — useful for
seeing serialization/gaps between launches that a stats table can't show).
A full `--kernel-trace` at `--depth 8192` on ornith-9b produces a
multi-megabyte CSV (thousands of chunked-prefill kernel launches per layer);
prefer `--stats` alone (aggregated, small) unless you specifically need
per-dispatch VGPR/grid data or `--kernel-include-regex` to scope a trace
down to one kernel family.

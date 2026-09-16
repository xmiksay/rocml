# rocml vs llama.cpp: measured comparison (2026-09-16)

Every number here was measured on this machine on 2026-09-16 (RX 7800 XT /
gfx1101, ROCm 7.2.4, Ryzen 7 2700, 46 GiB DDR4, PCIe 4.0 x16), with both
engines loading the *same* GGUF files. Nothing is estimated or carried over
from an earlier round.

llama.cpp: Arch package `llama-cpp` 0.4.0-2 (build 10809, commit 5266f24da7),
installed 2026-09-11, unchanged since; ROCm backend. Note it is built with
asserts enabled, which handicaps it — its numbers are conservative.

## Single-stream throughput

Ornith-1.5-9B, ctx 16384, fp16 KV, `-fa on`/rocml defaults, median of 3:

| model / depth | engine | prefill tok/s | decode tok/s |
|---|---|---:|---:|
| Q4_K_M @ 2048 | llama.cpp | 1795.3 | 66.3 |
| Q4_K_M @ 2048 | rocml | 925.3 | 49.5 |
| Q4_K_M @ 8192 | llama.cpp | 1795.7 | 65.1 |
| Q4_K_M @ 8192 | rocml | 713.9 | 46.1 |
| Q6_K @ 8192 | llama.cpp | 1647.6 | 58.9 |
| Q6_K @ 8192 | rocml | 709.0 | 39.1 |

Ornith-1.5-35B-A3B Q4_K_M (MoE), ctx 128000, depth 8192:

| engine / config | prefill tok/s | decode tok/s |
|---|---:|---:|
| llama.cpp `--n-cpu-moe 14`, K/V q8_0 | 639.0 | 33.8 |
| rocml, q4-mixed KV, synthetic bench | 327.5 | 37.1 |
| rocml, q4-mixed KV, real prose (~10k tok) | 109.1 | 29.0 |

**llama.cpp's KV trap**: `-ctv q4_0` collapses it to 67 tok/s prefill / 15 tok/s
decode on this build. Use f16 or q8_0 KV. Every llama.cpp figure above does.

**rocml's synthetic-bench trap**: `bench --depth` uses a repetitive synthetic
prompt, which inflates the MoE expert-cache hit rate to ~99.9% vs ~91-95% on
real text — hence 37.1 vs 29.0 tok/s decode. Quote real-text numbers.

## Multi-turn agentic use (the case rocml is designed for)

6-turn growing conversation (~1.9K -> ~2.8K tokens), byte-identical content,
both over HTTP, cold server each time:

| metric | rocml-serve | llama-server | ratio |
|---|---:|---:|---|
| cumulative wall, 6 turns | 19.47 s | 11.12 s | llama.cpp 1.75x |
| TTFT turn 1 (cold) | 2616 ms | 1178 ms | 2.2x |
| TTFT turn 2 (first hit) | 1629 ms | 160 ms | 10.2x |
| TTFT turns 3-6 (steady) | 823-878 ms | 179-254 ms | 3.5-4.6x |
| decode tok/s | 48.5-49.1 | 64.1-64.4 | 1.3x |

llama.cpp uses a single persistent slot (`-np 1`) with `--cache-prompt` on by
default; no extra flags were needed for prefix reuse.

**The snapshot layer itself is not the problem — it works.** Isolated via
`bench --turns 6` with and without snapshots: GPU-tier restore costs
0.12-0.21 ms regardless of depth, turning turns 2-6 into a flat ~64-token
reprocess instead of a growing full re-prefill. Cumulative 24.35 s (on) vs
37.61 s (off), a 35.2% win over rocml's own baseline, and the gap widens with
depth. rocml loses on what happens *after* the hit: the small new suffix
(64-225 tokens) reprocesses at ~110-170 tok/s vs llama.cpp's ~700-1000, and
decode is ~30% slower per token.

## Where rocml's gap lives (leads, not conclusions)

1. **Small-batch prefill overhead** dominates the snapshot-hit path. A 64-token
   suffix costs ~570 ms. This is the largest single lever for agentic use.
2. **Decode ~30% behind** on the same weights — a dense/hybrid-path gap, not
   MoE-specific.
3. **MoE prefill**: expert groups average 2-16 rows, below the WMMA dispatch
   threshold, so they fall to scalar kernels. The M4 micro-tile (`TR=16`) kernel
   only wins when `m >= n`, leaving gate/up (2/3 of expert FLOPs) on scalar.
4. **MoE is not instrumented for profiling.** `moe.rs`/`moe_chunk.rs` have no
   `Profiler::scope` calls and `OpKind` has no MoE variants, so `bench --profile`
   silently omits all expert work; its percentages cover instrumented ops only.
   Fix this before optimizing MoE further.

## Historical note

An earlier informal comparison recorded llama.cpp decoding at ~13 tok/s, which
would have made rocml ~3x faster. That figure is not reproducible here and is
not recorded anywhere in this repo (unlike the ~1733 tok/s prefill reference in
`docs/prefill-gap-analysis.md`, which today's 1795 tok/s confirms). No package
was updated on this machine on 2026-09-16, and llama-cpp has been the same
build since 2026-09-11, so no update explains the difference. Most likely it
came from ollama (installed here, upgraded 2026-09-14) or a differently-flagged
run. Treat it as unverified.

## Practical recommendation

For day-to-day use of these checkpoints, run llama.cpp:

```
llama-server -m ~/checkpoints/Ornith-1.5-9B-GGUF/Ornith-1.5-9B-Q4_K_M.gguf \
  -ngl 99 -fa on -c 16384
llama-server -m ~/checkpoints/Ornith-1.5-35B-A3B-GGUF/Ornith-1.5-35B-Q4_K_M.gguf \
  -ngl 99 --n-cpu-moe 14 -c 128000 -fa on -ctk q8_0 -ctv q8_0
```

rocml remains the place where the kernels, the quantized-KV work and the
snapshot design live; it is not currently the faster way to run these models.

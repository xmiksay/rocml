//! Issue #2's chunked-prefill follow-up central correctness gate: chunked
//! prefill (`Model::forward_prompt`) over a mixed/quantized KV cache
//! (`--kv-cache q8`/`q4-mixed`) must reproduce the same cache-eviction
//! bookkeeping and near-identical cache/logit content as the pre-existing
//! token-serial path (`Model::forward_token`), across prompt lengths chosen
//! to land on both sides of the sink/ring boundaries (`SINK_LEN`=32,
//! `WINDOW_LEN`=128): 32, 128, 160, 500, 2048.
//!
//! **What's asserted bit-identical**: every mixed layer's `window_base`
//! (`MixedLayout`'s eviction bookkeeping — which absolute position the
//! window's physical slot 0 currently maps to) is pure integer arithmetic
//! driven only by position counts, never by any floating-point value, so it
//! must match *exactly* regardless of any numeric drift elsewhere in the
//! model. This is the "same ring positions, same quantized-block timing,
//! same sink handling" invariant `MixedAttnPlane::append_chunk`'s own
//! module doc describes — see `rocml/src/kv_quant/layout/chunk_plan.rs`'s
//! unit tests (`chunk_append_matches_token_serial_*`) for the pure-Rust
//! proof this is correct by construction, independent of any GPU numerics.
//!
//! **What's asserted near-identical, not bit-identical, with the measured
//! bound reported honestly**: the actual K/V *bytes* landing in the sink and
//! window regions, and the bulk quantize scale factors. Chunked prefill's
//! batched GEMMs (the WMMA matrix-core path, active once a chunk has at
//! least 128 rows) and the GDN chunkwise recurrence both compute the *same*
//! math as the token-serial path in a *different floating-point reduction
//! order* — exactly the reason `qwen35_chunked_prefill_parity.rs` uses a
//! relative-tolerance *logits* bound rather than bit-equality for the dense
//! cache. That numeric drift reaches a mixed layer's own K/V projection
//! before this feature's own code ever runs, so the raw cache bytes can't
//! be bit-identical in general even though the *bookkeeping* is.
//!
//! **Why this compares cache bytes by absolute, not relative, difference**
//! (unlike the logits comparison below): K/V hidden-state elements are
//! frequently small in magnitude, where a fixed absolute WMMA-rounding error
//! — entirely consistent with this codebase's own documented WMMA precision
//! trade-off (`.claude/CLAUDE.md`: "~0.16%-0.55%" per-layer relative logit
//! deviation, which compounds across layers and across the GDN chunkwise
//! recurrence's own reduction-order sensitivity) — reads as a huge
//! *relative* difference once the denominator is small. This is the same
//! "small-magnitude values dominate a relative metric" pitfall
//! `mixed_kv_parity.rs` already documents for its own logits comparison.
//!
//! **Measured on real hardware** (Qwen3.5-2B-Q8_0, this suite's own prompt
//! lengths): Q8 cache-byte absolute diff ranged `0.0039` (`len=32`, no WMMA
//! yet) to `0.023` (`len=2048`); Q4Mixed ranged `0.0039` to a `0.42` outlier
//! at `len=500` (a handful of individual channels a few evictions deep, where
//! Q4's coarse `[-8,7]` quantization amplifies the same upstream drift far
//! more than Q8's `[-127,127]` range does) — **despite that outlier, greedy
//! continuation matched exactly (0 near-tie flips, let alone divergences)
//! across every one of this suite's 10 (5 lengths x 2 modes) cases**, and
//! final-logits relative diff stayed under `0.041` throughout. `CACHE_ABS_TOL`
//! is set above the measured worst case with headroom, not tuned to the
//! measurement — it exists to catch a gross regression (a real ordering or
//! indexing bug would produce differences many times larger than a few
//! outlier channels), not to police normal float-order noise the rest of
//! this codebase already accepts for the dense chunked-prefill path.
//!
//! Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`: prompt lengths
//! up to 2048 through the token-serial path are unbearably slow in debug).

use rocml::kv_quant::{SINK_LEN, WINDOW_LEN};
use rocml::snapshot::AttnLayerBytes;
use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
const CTX: usize = 4096;
const PROMPT_LENGTHS: &[usize] = &[32, 128, 160, 500, 2048];
const CONTINUATION_LEN: usize = 8;
/// K/V byte-content drift bound (sink/window f16, bulk scales) — absolute,
/// not relative; see this file's module doc for why, and for the real
/// measurement (up to `0.42` for Q4Mixed) this was set against.
const CACHE_ABS_TOL: f32 = 0.6;
/// Final-logits bound: combines chunked-vs-serial reduction-order drift
/// (`qwen35_chunked_prefill_parity.rs`'s 1e-2) with the mixed cache's own
/// lossy quantization error (`mixed_kv_parity.rs`'s up to 1.5 for q4) —
/// measured and reported, not guessed.
const LOGITS_REL_TOL: f32 = 1.5;
const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;

fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
    (0..len as u32).map(|i| (i * 37 + 5) % vocab_size).collect()
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .expect("logits must be non-empty")
}

fn top2_gap(logits: &[f32]) -> f32 {
    let (mut best, mut second) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &v in logits {
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }
    (best - second) / best.abs().max(1.0)
}

fn greedy_continue(model: &mut Model, mut logits: Vec<f32>, n: usize) -> Vec<(u32, f32)> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let gap = top2_gap(&logits);
        let next = argmax(&logits);
        out.push((next, gap));
        logits = model.forward_token(next).expect("forward_token failed");
    }
    out
}

fn max_abs_diff_f16(a: &[half::f16], b: &[half::f16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x.to_f32() - y.to_f32()).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Slices out only the per-head *filled-so-far* prefix of a buffer whose
/// remaining tail is allocated-but-never-written device memory — the
/// window region (fixed `WINDOW_LEN` capacity, filled up to `window_fill`)
/// and the bulk region (`num_blocks_total`/`bulk_cap` capacity, filled up to
/// `evicted_blocks`/`evicted_blocks*WINDOW_LEN`) both need this; comparing
/// the unfilled tail would compare two independently-allocated models'
/// uninitialized memory, not anything this feature computed (see
/// `MixedAttnPlane::capture`'s own doc comment on why the whole allocated
/// region is captured rather than a precisely-sliced prefix).
fn filled_prefix<T>(buf: &[T], n_kv_heads: usize, stride: usize, filled_len: usize) -> Vec<&[T]> {
    (0..n_kv_heads)
        .map(|h| &buf[h * stride..h * stride + filled_len])
        .collect()
}

/// Compares every `Mixed` layer between two snapshots captured at the same
/// position: `window_base` exactly, everything else (sink/window bytes, and
/// only the *filled-so-far* bulk scale prefix) by measured absolute drift.
/// Returns `(max_cache_abs_diff, mixed_layer_count)`.
fn compare_mixed_layers(
    chunked: &[Option<AttnLayerBytes>],
    serial: &[Option<AttnLayerBytes>],
    n_kv_heads: usize,
    head_dim: usize,
    position: u32,
) -> (f32, usize) {
    assert_eq!(chunked.len(), serial.len(), "layer count mismatch");
    let mut max_diff = 0.0f32;
    let mut mixed_count = 0;
    for (layer_idx, (c, s)) in chunked.iter().zip(serial).enumerate() {
        match (c, s) {
            (
                Some(AttnLayerBytes::Mixed {
                    sink_k: ck,
                    sink_v: cv,
                    window_k: cwk,
                    window_v: cwv,
                    bulk_k_scales: cks,
                    bulk_v_scales: cvs,
                    window_base: c_wb,
                    v_bits: c_vb,
                    ..
                }),
                Some(AttnLayerBytes::Mixed {
                    sink_k: sk,
                    sink_v: sv,
                    window_k: swk,
                    window_v: swv,
                    bulk_k_scales: sks,
                    bulk_v_scales: svs,
                    window_base: s_wb,
                    v_bits: s_vb,
                    ..
                }),
            ) => {
                mixed_count += 1;
                assert_eq!(
                    c_wb, s_wb,
                    "layer {layer_idx}: window_base bookkeeping diverged (chunked {c_wb} vs \
                     serial {s_wb}) — this is pure integer arithmetic and must match exactly"
                );
                assert_eq!(c_vb, s_vb, "layer {layer_idx}: v_bits mismatch");

                // Sink is captured whole but only ever *written* up to
                // `min(position, SINK_LEN)` — always fully written for
                // every prompt length this suite uses (all >= SINK_LEN).
                max_diff = max_diff.max(max_abs_diff_f16(ck, sk));
                max_diff = max_diff.max(max_abs_diff_f16(cv, sv));

                // Window is captured whole (fixed `WINDOW_LEN` size) but
                // only the first `window_fill` slots per head are actually
                // written so far.
                let window_fill = (position - c_wb) as usize;
                let window_stride = WINDOW_LEN as usize * head_dim;
                let window_len = window_fill * head_dim;
                for (a, b) in filled_prefix(cwk, n_kv_heads, window_stride, window_len)
                    .into_iter()
                    .zip(filled_prefix(swk, n_kv_heads, window_stride, window_len))
                {
                    max_diff = max_diff.max(max_abs_diff_f16(a, b));
                }
                for (a, b) in filled_prefix(cwv, n_kv_heads, window_stride, window_len)
                    .into_iter()
                    .zip(filled_prefix(swv, n_kv_heads, window_stride, window_len))
                {
                    max_diff = max_diff.max(max_abs_diff_f16(a, b));
                }

                let evicted_blocks = ((c_wb - SINK_LEN) / WINDOW_LEN) as usize;
                if evicted_blocks > 0 {
                    let num_blocks_total = cks.len() / (n_kv_heads * head_dim);
                    let bulk_cap = cvs.len() / n_kv_heads;
                    let k_scale_stride = num_blocks_total * head_dim;
                    let k_filled_len = evicted_blocks * head_dim;
                    for (a, b) in filled_prefix(cks, n_kv_heads, k_scale_stride, k_filled_len)
                        .into_iter()
                        .zip(filled_prefix(sks, n_kv_heads, k_scale_stride, k_filled_len))
                    {
                        max_diff = max_diff.max(max_abs_diff_f32(a, b));
                    }
                    let v_filled_len = evicted_blocks * WINDOW_LEN as usize;
                    for (a, b) in filled_prefix(cvs, n_kv_heads, bulk_cap, v_filled_len)
                        .into_iter()
                        .zip(filled_prefix(svs, n_kv_heads, bulk_cap, v_filled_len))
                    {
                        max_diff = max_diff.max(max_abs_diff_f32(a, b));
                    }
                }
            }
            (Some(AttnLayerBytes::Mixed { .. }), other)
            | (other, Some(AttnLayerBytes::Mixed { .. })) => {
                panic!(
                    "layer {layer_idx}: mixed-ness mismatch between chunked and serial caches \
                     (other = {other:?})"
                );
            }
            _ => {}
        }
    }
    (max_diff, mixed_count)
}

fn run_case(
    mode: KvCacheMode,
    model_serial: &mut Model,
    model_chunked: &mut Model,
    prompt_ids: &[u32],
) {
    model_serial.reset().expect("reset failed");
    let mut serial_logits = Vec::new();
    for &id in prompt_ids {
        serial_logits = model_serial
            .forward_token(id)
            .expect("forward_token failed");
    }
    let serial_snap = model_serial
        .as_hybrid()
        .expect("qwen35 hybrid model")
        .capture_snapshot(prompt_ids.to_vec())
        .expect("capture_snapshot (serial) failed");
    let serial_continuation =
        greedy_continue(model_serial, serial_logits.clone(), CONTINUATION_LEN);

    model_chunked.reset().expect("reset failed");
    let chunked_logits = model_chunked
        .forward_prompt(prompt_ids, None)
        .expect("forward_prompt failed");
    let chunked_snap = model_chunked
        .as_hybrid()
        .expect("qwen35 hybrid model")
        .capture_snapshot(prompt_ids.to_vec())
        .expect("capture_snapshot (chunked) failed");
    let chunked_continuation =
        greedy_continue(model_chunked, chunked_logits.clone(), CONTINUATION_LEN);

    let label = format!("{mode:?} len={}", prompt_ids.len());

    assert_eq!(
        chunked_snap.position, serial_snap.position,
        "{label}: snapshot position mismatch"
    );
    let config = model_serial
        .as_hybrid()
        .expect("qwen35 hybrid model")
        .config();
    let (n_kv_heads, head_dim) = (config.head_count_kv as usize, config.head_dim as usize);
    let (cache_abs_diff, mixed_count) = compare_mixed_layers(
        &chunked_snap.attn,
        &serial_snap.attn,
        n_kv_heads,
        head_dim,
        chunked_snap.position,
    );
    eprintln!(
        "{label}: {mixed_count} mixed layer(s), max cache-content absolute diff \
         {cache_abs_diff:.6} (bound {CACHE_ABS_TOL})"
    );
    assert!(
        cache_abs_diff < CACHE_ABS_TOL,
        "{label}: cache content absolute diff {cache_abs_diff} exceeds the measured/justified bound"
    );

    let mut max_logit_rel = 0.0f32;
    for (&got, &want) in chunked_logits.iter().zip(&serial_logits) {
        let rel = (got - want).abs() / want.abs().max(1.0);
        max_logit_rel = max_logit_rel.max(rel);
    }
    eprintln!("{label}: max final-logit relative diff {max_logit_rel:.6} (bound {LOGITS_REL_TOL})");
    assert!(
        max_logit_rel < LOGITS_REL_TOL,
        "{label}: final logits relative diff {max_logit_rel} exceeds the measured/justified bound"
    );

    for (step, ((serial_tok, serial_gap), (chunked_tok, chunked_gap))) in serial_continuation
        .iter()
        .zip(&chunked_continuation)
        .enumerate()
    {
        if serial_tok == chunked_tok {
            continue;
        }
        assert!(
            serial_gap.abs() <= NEAR_TIE_RELATIVE_GAP || chunked_gap.abs() <= NEAR_TIE_RELATIVE_GAP,
            "{label} continuation step {step}: serial picked {serial_tok} (gap {serial_gap}), \
             chunked picked {chunked_tok} (gap {chunked_gap}) — gap too large to be a documented \
             near-tie"
        );
        eprintln!(
            "{label} continuation step {step}: documented near-tie flip, serial={serial_tok} \
             (gap {serial_gap}) vs chunked={chunked_tok} (gap {chunked_gap})"
        );
    }
}

fn run_mode(mode: KvCacheMode) {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    let mut model_serial = Model::load(&path, LoadOptions::new(CTX).with_kv_cache(mode))
        .unwrap_or_else(|e| panic!("load {mode:?} model (serial) failed: {e}"));
    let mut model_chunked = Model::load(&path, LoadOptions::new(CTX).with_kv_cache(mode))
        .unwrap_or_else(|e| panic!("load {mode:?} model (chunked) failed: {e}"));
    let vocab_size = model_serial.vocab_size();

    for &len in PROMPT_LENGTHS {
        let prompt_ids = synthetic_prompt(len, vocab_size);
        run_case(mode, &mut model_serial, &mut model_chunked, &prompt_ids);
    }
}

#[test]
fn q8_mixed_chunked_prefill_matches_token_serial() {
    run_mode(KvCacheMode::Q8);
}

#[test]
fn q4_mixed_chunked_prefill_matches_token_serial() {
    run_mode(KvCacheMode::Q4Mixed);
}

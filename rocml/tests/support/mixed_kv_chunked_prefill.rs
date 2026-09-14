//! Shared harness for `mixed_kv_chunked_prefill_parity.rs`'s chunked-vs-
//! token-serial parity gate — split out purely to keep that file (and any
//! sibling non-default-geometry test file) under the workspace's 400-line
//! cap. See that file's own module doc for the full design rationale
//! (what's asserted bit-identical vs measured-and-bounded, and why).

use rocml::snapshot::AttnLayerBytes;
use rocml::{KvCacheMode, LoadOptions, Model};
use rocml_core::testpaths::checkpoint;

pub const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";
pub const CTX: usize = 4096;
pub const CONTINUATION_LEN: usize = 8;
/// K/V byte-content drift bound (sink/window f16, bulk scales) — absolute,
/// not relative; see `mixed_kv_chunked_prefill_parity.rs`'s module doc for
/// why, and for the real measurement (up to `0.42` for Q4Mixed) this was
/// set against.
pub const CACHE_ABS_TOL: f32 = 0.6;
/// Final-logits bound: combines chunked-vs-serial reduction-order drift
/// (`qwen35_chunked_prefill_parity.rs`'s 1e-2) with the mixed cache's own
/// lossy quantization error (`mixed_kv_parity.rs`'s up to 1.5 for q4) —
/// measured and reported, not guessed.
pub const LOGITS_REL_TOL: f32 = 1.5;
pub const NEAR_TIE_RELATIVE_GAP: f32 = 1e-2;

pub fn synthetic_prompt(len: usize, vocab_size: u32) -> Vec<u32> {
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
/// window region (fixed `window_len` capacity, filled up to `window_fill`)
/// and the bulk region (`num_blocks_total`/`bulk_cap` capacity, filled up to
/// `evicted_blocks`/`evicted_blocks*window_len`) both need this; comparing
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
#[allow(clippy::too_many_arguments)]
fn compare_mixed_layers(
    chunked: &[Option<AttnLayerBytes>],
    serial: &[Option<AttnLayerBytes>],
    n_kv_heads: usize,
    head_dim: usize,
    position: u32,
    sink_len: u32,
    window_len: u32,
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
                // `min(position, sink_len)` — always fully written for
                // every prompt length this suite uses (all >= sink_len).
                max_diff = max_diff.max(max_abs_diff_f16(ck, sk));
                max_diff = max_diff.max(max_abs_diff_f16(cv, sv));

                // Window is captured whole (fixed `window_len` size) but
                // only the first `window_fill` slots per head are actually
                // written so far.
                let window_fill = (position - c_wb) as usize;
                let window_stride = window_len as usize * head_dim;
                let window_fill_len = window_fill * head_dim;
                for (a, b) in filled_prefix(cwk, n_kv_heads, window_stride, window_fill_len)
                    .into_iter()
                    .zip(filled_prefix(
                        swk,
                        n_kv_heads,
                        window_stride,
                        window_fill_len,
                    ))
                {
                    max_diff = max_diff.max(max_abs_diff_f16(a, b));
                }
                for (a, b) in filled_prefix(cwv, n_kv_heads, window_stride, window_fill_len)
                    .into_iter()
                    .zip(filled_prefix(
                        swv,
                        n_kv_heads,
                        window_stride,
                        window_fill_len,
                    ))
                {
                    max_diff = max_diff.max(max_abs_diff_f16(a, b));
                }

                let evicted_blocks = ((c_wb - sink_len) / window_len) as usize;
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
                    let v_filled_len = evicted_blocks * window_len as usize;
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

#[allow(clippy::too_many_arguments)]
fn run_case(
    mode: KvCacheMode,
    model_serial: &mut Model,
    model_chunked: &mut Model,
    prompt_ids: &[u32],
    sink_len: u32,
    window_len: u32,
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
        sink_len,
        window_len,
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

/// Runs the chunked-vs-serial parity gate for `mode` at `sink_len`/
/// `window_len` across every prompt length in `prompt_lengths`. No-ops
/// (returns immediately) if the real checkpoint isn't present.
pub fn run_mode_with_lens(
    mode: KvCacheMode,
    sink_len: u32,
    window_len: u32,
    prompt_lengths: &[usize],
) {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    let opts = LoadOptions::new(CTX)
        .with_kv_cache(mode)
        .with_kv_sink(sink_len)
        .with_kv_window(window_len);
    let mut model_serial = Model::load(&path, opts)
        .unwrap_or_else(|e| panic!("load {mode:?} model (serial) failed: {e}"));
    let mut model_chunked = Model::load(&path, opts)
        .unwrap_or_else(|e| panic!("load {mode:?} model (chunked) failed: {e}"));
    let vocab_size = model_serial.vocab_size();

    for &len in prompt_lengths {
        let prompt_ids = synthetic_prompt(len, vocab_size);
        run_case(
            mode,
            &mut model_serial,
            &mut model_chunked,
            &prompt_ids,
            sink_len,
            window_len,
        );
    }
}

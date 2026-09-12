//! GPU integration tests for the flash-attention-style tiled prefill kernel
//! (`attn_prefill_flash_partial_f32`/`_f16` + `attn_prefill_flash_reduce_f32`)
//! against an f64 CPU causal-softmax-attention reference — the round's
//! correctness gate (see `.claude/CLAUDE.md`'s flash-prefill section).
//! Mirrors `attn_prefill.rs`'s test style but additionally sweeps `n_splits`
//! (1, 2, and a "many splits" value) since this kernel's whole point is the
//! split-K/reduce path `attn_prefill.hip` doesn't have.
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1e-4;
/// Must match `BR` in `kernels/attn_prefill_flash.hip`.
const BR: u32 = 8;

fn assert_close(actual: &[f32], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got as f64 - want).abs();
        let tol = TOL as f64 * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "{label}[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

/// f64 causal softmax attention reference: query row `i` (global position
/// `pos_base + i`) attends to `[0, pos_base + i]` inclusive. `q`/output are
/// `[chunk_len, n_heads, head_dim]`; `k`/`v` are `[n_kv_heads, max_seq,
/// head_dim]`.
#[allow(clippy::too_many_arguments)]
fn cpu_reference_f64(
    q: &[f32],
    k: &[f64],
    v: &[f64],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    chunk_len: u32,
    pos_base: u32,
) -> Vec<f64> {
    let group = (n_heads / n_kv_heads) as usize;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let (hd, ms) = (head_dim as usize, max_seq as usize);
    let mut out = vec![0.0f64; chunk_len as usize * n_heads as usize * hd];
    for i in 0..chunk_len as usize {
        let end = pos_base as usize + i + 1;
        for h in 0..n_heads as usize {
            let kvh = h / group;
            let q_h = &q[(i * n_heads as usize + h) * hd..(i * n_heads as usize + h) * hd + hd];
            let mut scores = vec![0.0f64; end];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_row = &k[(kvh * ms + t) * hd..(kvh * ms + t) * hd + hd];
                let dot: f64 = q_h.iter().zip(k_row).map(|(&a, &b)| a as f64 * b).sum();
                *score = dot * scale;
            }
            let m = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|&s| (s - m).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let out_h =
                &mut out[(i * n_heads as usize + h) * hd..(i * n_heads as usize + h) * hd + hd];
            for (t, &e) in exps.iter().enumerate() {
                let prob = e / sum;
                let v_row = &v[(kvh * ms + t) * hd..(kvh * ms + t) * hd + hd];
                for (o, &vv) in out_h.iter_mut().zip(v_row) {
                    *o += prob * vv;
                }
            }
        }
    }
    out
}

/// Mirrors the Rust host launcher's split heuristic closely enough for
/// tests: callers pass `n_splits` directly (exercising 1/2/many explicitly)
/// rather than reproducing the occupancy-driven picker, so `split_len` is
/// derived the same way the kernel expects: from the *chunk's* deepest row.
fn split_len_for(pos_base: u32, chunk_len: u32, n_splits: u32) -> u32 {
    (pos_base + chunk_len).div_ceil(n_splits)
}

#[allow(clippy::too_many_arguments)]
fn run_attn_prefill_flash(
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    chunk_len: u32,
    pos_base: u32,
    n_splits: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F32_HSACO)
        .expect("partial module load failed");
    let partial_fn = module
        .get_function(rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F32_KERNEL)
        .expect("partial kernel lookup failed");
    let reduce_fn = module
        .get_function(rocml_kernels::ATTN_PREFILL_FLASH_REDUCE_F32_KERNEL)
        .expect("reduce kernel lookup failed");

    let group = n_heads / n_kv_heads;
    let (nh, nkv, hd, ms, cl) = (
        n_heads as usize,
        n_kv_heads as usize,
        head_dim as usize,
        max_seq as usize,
        chunk_len as usize,
    );

    let q: Vec<f32> = (0..cl * nh * hd)
        .map(|i| ((i % 23) as f32) * 0.05 - 0.55)
        .collect();
    // Fill the whole [n_kv_heads, max_seq, head_dim] plane (not just the
    // causally-visible prefix) so an out-of-bounds read past the causal
    // bound would corrupt the result if it ever happened.
    let k_f64: Vec<f64> = (0..nkv * ms * hd)
        .map(|i| ((i % 19) as f64) * 0.04 - 0.38)
        .collect();
    let v_f64: Vec<f64> = (0..nkv * ms * hd)
        .map(|i| ((i % 17) as f64) * 0.03 - 0.24)
        .collect();
    let k_f32: Vec<f32> = k_f64.iter().map(|&x| x as f32).collect();
    let v_f32: Vec<f32> = v_f64.iter().map(|&x| x as f32).collect();

    let expected = cpu_reference_f64(
        &q, &k_f64, &v_f64, n_heads, n_kv_heads, head_dim, max_seq, chunk_len, pos_base,
    );

    let mut buf_q = DeviceBuffer::<f32>::new(q.len()).expect("hipMalloc q failed");
    let mut buf_k = DeviceBuffer::<f32>::new(k_f32.len()).expect("hipMalloc k failed");
    let mut buf_v = DeviceBuffer::<f32>::new(v_f32.len()).expect("hipMalloc v failed");
    buf_q.copy_from_host(&q).expect("copy q failed");
    buf_k.copy_from_host(&k_f32).expect("copy k failed");
    buf_v.copy_from_host(&v_f32).expect("copy v failed");

    let num_row_tiles = chunk_len.div_ceil(BR) as usize;
    let partial_len = cl * nh * n_splits as usize;
    let mut buf_partial_out = DeviceBuffer::<f32>::new(partial_len * hd).expect("hipMalloc po");
    let buf_partial_m = DeviceBuffer::<f32>::new(partial_len).expect("hipMalloc pm");
    let buf_partial_l = DeviceBuffer::<f32>::new(partial_len).expect("hipMalloc pl");
    // Poison partial scratch so an under-write (e.g. a row-tile's tail rows
    // never touched) would show up as garbage in the final output instead
    // of silently reading zero-initialized memory.
    buf_partial_out
        .copy_from_host(&vec![f32::NAN; partial_len * hd])
        .unwrap();
    let buf_out = DeviceBuffer::<f32>::new(cl * nh * hd).expect("hipMalloc out failed");

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let split_len = split_len_for(pos_base, chunk_len, n_splits);

    let q_ptr: *mut c_void = buf_q.device_ptr();
    let k_ptr: *mut c_void = buf_k.device_ptr();
    let v_ptr: *mut c_void = buf_v.device_ptr();
    let po_ptr: *mut c_void = buf_partial_out.device_ptr();
    let pm_ptr: *mut c_void = buf_partial_m.device_ptr();
    let pl_ptr: *mut c_void = buf_partial_l.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();

    let mut partial_params = kernel_params!(
        q_ptr, k_ptr, v_ptr, po_ptr, pm_ptr, pl_ptr, n_kv_heads, group, head_dim, max_seq,
        chunk_len, pos_base, split_len, n_splits, scale
    );
    let smem = (2 * BC as usize * hd * std::mem::size_of::<f32>()) as u32;
    let partial_cfg = LaunchConfig {
        grid: (n_kv_heads, num_row_tiles as u32, n_splits),
        block: (32, group, BR),
        shared_mem_bytes: smem,
    };
    // SAFETY: params match attn_prefill_flash_partial_f32's signature
    // (three const float*, three float*, seven unsigned, float); block =
    // (32, group, BR); grid.z = n_splits; every buffer outlives this launch.
    unsafe { partial_fn.launch(&partial_cfg, &mut partial_params, None) }
        .expect("partial launch failed");

    let mut reduce_params =
        kernel_params!(po_ptr, pm_ptr, pl_ptr, out_ptr, n_heads, head_dim, n_splits);
    let reduce_cfg = LaunchConfig {
        grid: (chunk_len, n_heads, 1),
        block: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params match attn_prefill_flash_reduce_f32's signature (three
    // const float*, float*, three unsigned); grid = (chunk_len, n_heads).
    unsafe { reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }
        .expect("reduce launch failed");

    let mut actual = vec![0.0f32; cl * nh * hd];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    assert_close(&actual, &expected, "attn_prefill_flash out");
}

/// Must match `BC` in `kernels/attn_prefill_flash.hip`.
const BC: u32 = 16;

#[test]
fn degenerate_single_row_single_position() {
    run_attn_prefill_flash(16, 8, 128, 4, 1, 0, 1);
}

#[test]
fn degenerate_single_row_matches_decode_at_depth() {
    run_attn_prefill_flash(16, 8, 128, 128, 1, 99, 1);
}

#[test]
fn chunk_crosses_bc_boundary_single_split() {
    // BC=16: chunk_len=33 spans more than two K/V-staging tiles per row.
    run_attn_prefill_flash(16, 8, 128, 64, 33, 0, 1);
}

#[test]
fn chunk_from_fresh_start_no_split() {
    run_attn_prefill_flash(8, 2, 256, 300, 64, 0, 1);
}

#[test]
fn chunk_resumes_after_existing_prefix_no_split() {
    run_attn_prefill_flash(8, 2, 256, 512, 64, 200, 1);
}

#[test]
fn chunk_resumes_split_two() {
    run_attn_prefill_flash(8, 2, 256, 512, 64, 200, 2);
}

#[test]
fn chunk_resumes_split_many() {
    // n_splits=8: several splits fall entirely outside shallow rows'
    // causal range (exercising the `start >= end` empty-partial path) and
    // several straddle a row's own bound partway through.
    run_attn_prefill_flash(8, 2, 256, 512, 64, 200, 8);
}

#[test]
fn gqa_group_and_causal_bound_vary_per_row() {
    run_attn_prefill_flash(16, 4, 128, 256, 33, 50, 4);
}

#[test]
fn equal_head_counts_no_gqa_sharing() {
    run_attn_prefill_flash(8, 8, 128, 128, 40, 0, 2);
}

#[test]
fn large_chunk_deep_resume_many_splits() {
    run_attn_prefill_flash(16, 8, 128, 2048, 256, 1024, 8);
}

#[test]
fn chunk_len_not_multiple_of_br() {
    // BR=8: chunk_len=13 leaves a partial last row-tile.
    run_attn_prefill_flash(16, 4, 128, 32, 13, 0, 2);
}

#[test]
fn row_tile_spans_ornith_full_attention_shape() {
    // Ornith's real full-attention head config (16 Q / 4 KV heads, head_dim
    // 256) at a chunk length exercising several full BR=8 row-tiles plus a
    // partial one, deep enough that per-row causal bounds genuinely differ.
    run_attn_prefill_flash(16, 4, 256, 4096, 130, 2000, 8);
}

#[test]
fn head_dim_128_half_shape() {
    // Dense-model-style head_dim (128, half of Ornith's 256) exercising a
    // different LDS footprint.
    run_attn_prefill_flash(8, 4, 128, 1024, 96, 700, 4);
}

#[test]
fn deep_context_many_splits_approaches_16k() {
    run_attn_prefill_flash(16, 4, 256, 16512, 128, 16384 - 128, 16);
}

/// `attn_prefill_flash_partial_f16` (issue #3's default KV dtype): mirrors
/// `run_attn_prefill_flash` above but K/V are uploaded as f16 and the CPU
/// reference is computed against the f16-rounded values (widened to f64),
/// since the kernel is expected to reproduce exactly what the f16 cache
/// stores.
#[allow(clippy::too_many_arguments)]
fn run_attn_prefill_flash_f16(
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    chunk_len: u32,
    pos_base: u32,
    n_splits: u32,
) {
    use half::f16;

    const TOL_F16: f64 = 5e-3;
    let _device = Device::new(0).expect("failed to select device 0");
    let module = Module::load_from_bytes(rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F16_HSACO)
        .expect("partial module load failed");
    let partial_fn = module
        .get_function(rocml_kernels::ATTN_PREFILL_FLASH_PARTIAL_F16_KERNEL)
        .expect("partial kernel lookup failed");
    let reduce_fn = module
        .get_function(rocml_kernels::ATTN_PREFILL_FLASH_REDUCE_F32_KERNEL)
        .expect("reduce kernel lookup failed");

    let group = n_heads / n_kv_heads;
    let (nh, nkv, hd, ms, cl) = (
        n_heads as usize,
        n_kv_heads as usize,
        head_dim as usize,
        max_seq as usize,
        chunk_len as usize,
    );

    let q: Vec<f32> = (0..cl * nh * hd)
        .map(|i| ((i % 23) as f32) * 0.05 - 0.55)
        .collect();
    let k_f16: Vec<f16> = (0..nkv * ms * hd)
        .map(|i| f16::from_f32(((i % 19) as f32) * 0.04 - 0.38))
        .collect();
    let v_f16: Vec<f16> = (0..nkv * ms * hd)
        .map(|i| f16::from_f32(((i % 17) as f32) * 0.03 - 0.24))
        .collect();
    let k_roundtrip: Vec<f64> = k_f16.iter().map(|&x| x.to_f32() as f64).collect();
    let v_roundtrip: Vec<f64> = v_f16.iter().map(|&x| x.to_f32() as f64).collect();

    let expected = cpu_reference_f64(
        &q,
        &k_roundtrip,
        &v_roundtrip,
        n_heads,
        n_kv_heads,
        head_dim,
        max_seq,
        chunk_len,
        pos_base,
    );

    let mut buf_q = DeviceBuffer::<f32>::new(q.len()).expect("hipMalloc q failed");
    let mut buf_k = DeviceBuffer::<f16>::new(k_f16.len()).expect("hipMalloc k failed");
    let mut buf_v = DeviceBuffer::<f16>::new(v_f16.len()).expect("hipMalloc v failed");
    buf_q.copy_from_host(&q).expect("copy q failed");
    buf_k.copy_from_host(&k_f16).expect("copy k failed");
    buf_v.copy_from_host(&v_f16).expect("copy v failed");

    let num_row_tiles = chunk_len.div_ceil(BR) as usize;
    let partial_len = cl * nh * n_splits as usize;
    let buf_partial_out = DeviceBuffer::<f32>::new(partial_len * hd).expect("hipMalloc po");
    let buf_partial_m = DeviceBuffer::<f32>::new(partial_len).expect("hipMalloc pm");
    let buf_partial_l = DeviceBuffer::<f32>::new(partial_len).expect("hipMalloc pl");
    let buf_out = DeviceBuffer::<f32>::new(cl * nh * hd).expect("hipMalloc out failed");

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let split_len = split_len_for(pos_base, chunk_len, n_splits);

    let q_ptr: *mut c_void = buf_q.device_ptr();
    let k_ptr: *mut c_void = buf_k.device_ptr();
    let v_ptr: *mut c_void = buf_v.device_ptr();
    let po_ptr: *mut c_void = buf_partial_out.device_ptr();
    let pm_ptr: *mut c_void = buf_partial_m.device_ptr();
    let pl_ptr: *mut c_void = buf_partial_l.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();

    let mut partial_params = kernel_params!(
        q_ptr, k_ptr, v_ptr, po_ptr, pm_ptr, pl_ptr, n_kv_heads, group, head_dim, max_seq,
        chunk_len, pos_base, split_len, n_splits, scale
    );
    let smem = (2 * BC as usize * hd * std::mem::size_of::<f32>()) as u32;
    let partial_cfg = LaunchConfig {
        grid: (n_kv_heads, num_row_tiles as u32, n_splits),
        block: (32, group, BR),
        shared_mem_bytes: smem,
    };
    // SAFETY: params match attn_prefill_flash_partial_f16's signature
    // (const float*, two const half*, three float*, seven unsigned, float).
    unsafe { partial_fn.launch(&partial_cfg, &mut partial_params, None) }
        .expect("partial launch failed");

    let mut reduce_params =
        kernel_params!(po_ptr, pm_ptr, pl_ptr, out_ptr, n_heads, head_dim, n_splits);
    let reduce_cfg = LaunchConfig {
        grid: (chunk_len, n_heads, 1),
        block: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    unsafe { reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }
        .expect("reduce launch failed");

    let mut actual = vec![0.0f32; cl * nh * hd];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let diff = (got as f64 - want).abs();
        let tol = TOL_F16 * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "attn_prefill_flash f16 out[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

#[test]
fn f16_kv_chunk_from_fresh_start_matches_f16_rounded_reference() {
    run_attn_prefill_flash_f16(8, 2, 256, 300, 64, 0, 1);
}

#[test]
fn f16_kv_chunk_resumes_split_many_matches_f16_rounded_reference() {
    run_attn_prefill_flash_f16(8, 2, 256, 512, 64, 200, 8);
}

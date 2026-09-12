//! GPU integration tests for the batched prefill-attention kernel
//! (`attn_prefill_f32`) against a CPU causal-softmax-attention reference.
//! Mirrors `attn_decode.rs`'s reference style but with `chunk_len` query
//! rows, each with its own causal bound `pos_base + i + 1` — including the
//! `pos_base > 0` "resume" case (a chunk starting partway through an
//! already-populated cache, e.g. the second chunk of a multi-chunk prompt).
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1e-4;
/// Must match `TILE_T` in `kernels/attn_prefill.hip`.
const TILE_T: u32 = 8;

fn assert_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let diff = (got - want).abs();
        let tol = TOL * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "{label}[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

/// Causal batched softmax attention: query row `i` (global position
/// `pos_base + i`) attends to `[0, pos_base + i]` inclusive. `q`/`out` are
/// `[chunk_len, n_heads, head_dim]`; `k`/`v` are `[n_kv_heads, max_seq,
/// head_dim]`.
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    chunk_len: u32,
    pos_base: u32,
) -> Vec<f32> {
    let group = (n_heads / n_kv_heads) as usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (hd, ms) = (head_dim as usize, max_seq as usize);
    let mut out = vec![0.0f32; chunk_len as usize * n_heads as usize * hd];
    for i in 0..chunk_len as usize {
        let end = pos_base as usize + i + 1;
        for h in 0..n_heads as usize {
            let kvh = h / group;
            let q_h = &q[(i * n_heads as usize + h) * hd..(i * n_heads as usize + h) * hd + hd];
            let mut scores = vec![0.0f32; end];
            for (t, score) in scores.iter_mut().enumerate() {
                let k_row = &k[(kvh * ms + t) * hd..(kvh * ms + t) * hd + hd];
                let dot: f32 = q_h.iter().zip(k_row).map(|(a, b)| a * b).sum();
                *score = dot * scale;
            }
            let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
            let sum: f32 = exps.iter().sum();
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

#[allow(clippy::too_many_arguments)]
fn run_attn_prefill(
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    chunk_len: u32,
    pos_base: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::ATTN_PREFILL_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::ATTN_PREFILL_F32_KERNEL)
        .expect("kernel lookup failed");

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
    // Fill the whole [n_kv_heads, max_seq, head_dim] plane, matching the
    // real cache layout: positions past pos_base+chunk_len-1 must never be
    // read (the causal bound), and this catches it if they are.
    let k: Vec<f32> = (0..nkv * ms * hd)
        .map(|i| ((i % 19) as f32) * 0.04 - 0.38)
        .collect();
    let v: Vec<f32> = (0..nkv * ms * hd)
        .map(|i| ((i % 17) as f32) * 0.03 - 0.24)
        .collect();

    let expected = cpu_reference(
        &q, &k, &v, n_heads, n_kv_heads, head_dim, max_seq, chunk_len, pos_base,
    );

    let mut buf_q = DeviceBuffer::<f32>::new(q.len()).expect("hipMalloc q failed");
    let mut buf_k = DeviceBuffer::<f32>::new(k.len()).expect("hipMalloc k failed");
    let mut buf_v = DeviceBuffer::<f32>::new(v.len()).expect("hipMalloc v failed");
    buf_q.copy_from_host(&q).expect("copy q failed");
    buf_k.copy_from_host(&k).expect("copy k failed");
    buf_v.copy_from_host(&v).expect("copy v failed");
    let buf_out = DeviceBuffer::<f32>::new(cl * nh * hd).expect("hipMalloc out failed");

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_ptr: *mut c_void = buf_q.device_ptr();
    let k_ptr: *mut c_void = buf_k.device_ptr();
    let v_ptr: *mut c_void = buf_v.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(
        q_ptr, k_ptr, v_ptr, out_ptr, n_kv_heads, group, head_dim, max_seq, chunk_len, pos_base,
        scale
    );
    let cfg = LaunchConfig {
        grid: (n_kv_heads, chunk_len, 1),
        block: (32, group, 1),
        shared_mem_bytes: 2 * TILE_T * head_dim * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches attn_prefill_f32's parameter list (three const
    // float*, float*, six unsigned, float) in order; block = (32, group, 1)
    // per the kernel's warp-per-q-head design; every buffer outlives this
    // launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; cl * nh * hd];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    assert_close(&actual, &expected, "attn_prefill out");
}

#[test]
fn degenerate_single_row_single_position() {
    run_attn_prefill(16, 8, 128, 4, 1, 0);
}

#[test]
fn degenerate_single_row_matches_decode_at_depth() {
    // chunk_len=1 at pos_base=99: identical causal bound (end=100) to a lone
    // attn_decode step with cur_len=100.
    run_attn_prefill(16, 8, 128, 128, 1, 99);
}

#[test]
fn chunk_crosses_tile_t_boundary() {
    // TILE_T=8: chunk_len=17 spans more than two K/V-staging tiles per row.
    run_attn_prefill(16, 8, 128, 32, 17, 0);
}

#[test]
fn chunk_from_fresh_start() {
    run_attn_prefill(8, 2, 256, 300, 64, 0);
}

#[test]
fn chunk_resumes_after_existing_prefix() {
    // pos_base > 0: the second chunk of a multi-chunk prompt, resuming after
    // an already-populated cache prefix.
    run_attn_prefill(8, 2, 256, 512, 64, 200);
}

#[test]
fn gqa_group_and_causal_bound_vary_per_row() {
    run_attn_prefill(16, 4, 128, 256, 33, 50);
}

#[test]
fn equal_head_counts_no_gqa_sharing() {
    run_attn_prefill(8, 8, 128, 128, 40, 0);
}

#[test]
fn large_chunk_deep_resume() {
    run_attn_prefill(16, 8, 128, 2048, 256, 1024);
}

/// `attn_prefill_f16` (issue #3's default KV dtype): mirrors
/// `run_attn_prefill` above but K/V are uploaded as f16 and the CPU
/// reference is computed against the f16-rounded values, since the kernel
/// is expected to reproduce exactly what the f16 cache stores.
fn run_attn_prefill_f16(
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    chunk_len: u32,
    pos_base: u32,
) {
    use half::f16;

    const TOL_F16: f32 = 5e-3;
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::ATTN_PREFILL_F16_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::ATTN_PREFILL_F16_KERNEL)
        .expect("kernel lookup failed");

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
    let k_roundtrip: Vec<f32> = k_f16.iter().map(|&x| x.to_f32()).collect();
    let v_roundtrip: Vec<f32> = v_f16.iter().map(|&x| x.to_f32()).collect();

    let expected = cpu_reference(
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
    let buf_out = DeviceBuffer::<f32>::new(cl * nh * hd).expect("hipMalloc out failed");

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let q_ptr: *mut c_void = buf_q.device_ptr();
    let k_ptr: *mut c_void = buf_k.device_ptr();
    let v_ptr: *mut c_void = buf_v.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(
        q_ptr, k_ptr, v_ptr, out_ptr, n_kv_heads, group, head_dim, max_seq, chunk_len, pos_base,
        scale
    );
    let cfg = LaunchConfig {
        grid: (n_kv_heads, chunk_len, 1),
        block: (32, group, 1),
        shared_mem_bytes: 2 * TILE_T * head_dim * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches attn_prefill_f16's parameter list (const
    // float*, two const half*, float*, six unsigned, float) in order;
    // block = (32, group, 1); every buffer outlives this launch.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; cl * nh * hd];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        let diff = (got - want).abs();
        let tol = TOL_F16 * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "attn_prefill f16 out[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

#[test]
fn f16_kv_chunk_from_fresh_start_matches_f16_rounded_reference() {
    run_attn_prefill_f16(8, 2, 256, 300, 64, 0);
}

#[test]
fn f16_kv_chunk_resumes_after_existing_prefix_matches_f16_rounded_reference() {
    run_attn_prefill_f16(8, 2, 256, 512, 64, 200);
}

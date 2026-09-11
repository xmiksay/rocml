//! GPU integration tests for the fused decode-attention kernels
//! (`attn_decode_partial_f32` + `attn_decode_reduce_f32`) against a CPU
//! softmax-attention reference. Each case is run once with `n_splits == 1`
//! (the common shallow-decode path) and, for the deeper cases, again with
//! an explicit multi-split count chosen directly by the test — independent
//! of whatever split heuristic the `rocml` crate's launcher picks — so the
//! log-sum-exp merge in `attn_decode_reduce_f32` gets its own direct
//! coverage rather than only being exercised incidentally.
use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1e-4;
/// Must match `TILE_T` in `kernels/attn_decode.hip` — sizes the dynamic
/// shared memory the partial kernel's launch requests.
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

fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

/// Standard single-query causal softmax attention over the first `cur_len`
/// cached positions, one head at a time — the composition `attn_decode`
/// replaces (gemv scores -> softmax -> weighted V), computed the naive way
/// (not online) so it checks the fused kernel's *result*, not its algorithm.
#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    cur_len: u32,
) -> Vec<f32> {
    let group = (n_heads / n_kv_heads) as usize;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let (hd, ms, cl) = (head_dim as usize, max_seq as usize, cur_len as usize);
    let mut out = vec![0.0f32; n_heads as usize * hd];
    for h in 0..n_heads as usize {
        let kvh = h / group;
        let q_h = &q[h * hd..(h + 1) * hd];
        let mut scores = vec![0.0f32; cl];
        for (t, score) in scores.iter_mut().enumerate() {
            let k_row = &k[(kvh * ms + t) * hd..(kvh * ms + t) * hd + hd];
            let dot: f32 = q_h.iter().zip(k_row).map(|(a, b)| a * b).sum();
            *score = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let out_h = &mut out[h * hd..(h + 1) * hd];
        for (t, &e) in exps.iter().enumerate() {
            let prob = e / sum;
            let v_row = &v[(kvh * ms + t) * hd..(kvh * ms + t) * hd + hd];
            for (o, &vv) in out_h.iter_mut().zip(v_row) {
                *o += prob * vv;
            }
        }
    }
    out
}

fn run_attn_decode(
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    cur_len: u32,
    n_splits: u32,
) {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_mp, partial_fn) = load(
        rocml_kernels::ATTN_DECODE_PARTIAL_F32_HSACO,
        rocml_kernels::ATTN_DECODE_PARTIAL_F32_KERNEL,
    );
    let (_mr, reduce_fn) = load(
        rocml_kernels::ATTN_DECODE_REDUCE_F32_HSACO,
        rocml_kernels::ATTN_DECODE_REDUCE_F32_KERNEL,
    );

    let group = n_heads / n_kv_heads;
    let (nh, nkv, hd, ms) = (
        n_heads as usize,
        n_kv_heads as usize,
        head_dim as usize,
        max_seq as usize,
    );

    let q: Vec<f32> = (0..nh * hd)
        .map(|i| ((i % 23) as f32) * 0.05 - 0.55)
        .collect();
    // Fill the whole [n_kv_heads, max_seq, head_dim] plane, matching the
    // real cache layout, even though only the first cur_len rows per kv
    // head are read — catches any accidental out-of-range read.
    let k: Vec<f32> = (0..nkv * ms * hd)
        .map(|i| ((i % 19) as f32) * 0.04 - 0.38)
        .collect();
    let v: Vec<f32> = (0..nkv * ms * hd)
        .map(|i| ((i % 17) as f32) * 0.03 - 0.24)
        .collect();

    let expected = cpu_reference(&q, &k, &v, n_heads, n_kv_heads, head_dim, max_seq, cur_len);

    let mut buf_q = DeviceBuffer::<f32>::new(q.len()).expect("hipMalloc q failed");
    let mut buf_k = DeviceBuffer::<f32>::new(k.len()).expect("hipMalloc k failed");
    let mut buf_v = DeviceBuffer::<f32>::new(v.len()).expect("hipMalloc v failed");
    buf_q.copy_from_host(&q).expect("copy q failed");
    buf_k.copy_from_host(&k).expect("copy k failed");
    buf_v.copy_from_host(&v).expect("copy v failed");

    let buf_out = DeviceBuffer::<f32>::new(nh * hd).expect("hipMalloc out failed");
    let ns = n_splits as usize;
    let buf_partial_out =
        DeviceBuffer::<f32>::new(nh * ns * hd).expect("hipMalloc partial_out failed");
    let buf_partial_m = DeviceBuffer::<f32>::new(nh * ns).expect("hipMalloc partial_m failed");
    let buf_partial_l = DeviceBuffer::<f32>::new(nh * ns).expect("hipMalloc partial_l failed");

    let split_len = cur_len.div_ceil(n_splits);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q_ptr: *mut c_void = buf_q.device_ptr();
    let k_ptr: *mut c_void = buf_k.device_ptr();
    let v_ptr: *mut c_void = buf_v.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let partial_out_ptr: *mut c_void = buf_partial_out.device_ptr();
    let partial_m_ptr: *mut c_void = buf_partial_m.device_ptr();
    let partial_l_ptr: *mut c_void = buf_partial_l.device_ptr();

    let mut partial_params = kernel_params!(
        q_ptr,
        k_ptr,
        v_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        n_kv_heads,
        group,
        head_dim,
        max_seq,
        cur_len,
        split_len,
        n_splits,
        scale
    );
    let partial_cfg = LaunchConfig {
        grid: (n_kv_heads, n_splits, 1),
        block: (32, group, 1),
        shared_mem_bytes: 2 * TILE_T * head_dim * std::mem::size_of::<f32>() as u32,
    };
    // SAFETY: params matches attn_decode_partial_f32's parameter list (three
    // const float*, three float*, seven unsigned, float) in order; block =
    // (32, group, 1) per the kernel's warp-per-q-head design; every buffer
    // outlives this launch.
    unsafe { partial_fn.launch(&partial_cfg, &mut partial_params, None) }
        .expect("partial kernel launch failed");

    let mut reduce_params = kernel_params!(
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        out_ptr,
        head_dim,
        n_splits
    );
    let reduce_cfg = LaunchConfig {
        grid: (n_heads, 1, 1),
        block: (head_dim, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches attn_decode_reduce_f32's parameter list (three
    // const float*, float*, two unsigned) in order; block = head_dim, one
    // thread per output element; every buffer outlives this launch.
    unsafe { reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }
        .expect("reduce kernel launch failed");

    let mut actual = vec![0.0f32; nh * hd];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");
    assert_close(&actual, &expected, "attn_decode out");
}

#[test]
fn degenerate_single_position() {
    run_attn_decode(16, 8, 128, 4, 1, 1);
}

#[test]
fn cur_len_three_single_split() {
    run_attn_decode(16, 8, 128, 8, 3, 1);
}

#[test]
fn cur_len_33_non_pow2_single_split() {
    run_attn_decode(16, 8, 128, 64, 33, 1);
}

#[test]
fn cur_len_257_non_pow2_single_split() {
    run_attn_decode(16, 8, 128, 512, 257, 1);
}

#[test]
fn cur_len_257_multi_split_non_pow2_remainder() {
    // 257 positions / 4 splits = split_len 65 -> the last split is a
    // shorter, non-power-of-two remainder (257 - 3*65 = 62).
    run_attn_decode(16, 8, 128, 512, 257, 4);
}

#[test]
fn cur_len_2048_single_split() {
    run_attn_decode(16, 8, 128, 2048, 2048, 1);
}

#[test]
fn cur_len_2048_multi_split() {
    run_attn_decode(16, 8, 128, 2048, 2048, 8);
}

#[test]
fn gqa_8q_2kv_head_dim_256() {
    run_attn_decode(8, 2, 256, 400, 400, 1);
}

#[test]
fn gqa_8q_2kv_head_dim_256_multi_split() {
    run_attn_decode(8, 2, 256, 1024, 1024, 6);
}

#[test]
fn equal_head_counts_no_gqa_sharing() {
    // n_heads == n_kv_heads (group == 1): every q head is its own kv head.
    run_attn_decode(8, 8, 128, 300, 300, 1);
}

#[test]
fn cache_headroom_beyond_cur_len() {
    // max_seq > cur_len, the real cache's shape (preallocated to
    // MAX_SEQ_CAP, cur_len grows into it): confirms the kernel never reads
    // past position cur_len-1 regardless of the plane's total pitch.
    run_attn_decode(16, 8, 128, 4096, 100, 1);
}

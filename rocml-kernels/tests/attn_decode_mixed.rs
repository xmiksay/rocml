//! GPU integration test for the fused mixed-KV decode-attention kernels
//! (`attn_decode_partial_mixed_q8`/`_q4`, `kernels/attn_decode_mixed.hip`)
//! against a CPU reference. `sink_len`/`window_len` here are small test
//! values passed as ordinary kernel parameters — independent of
//! `rocml::kv_quant::layout`'s production `SINK_LEN`/`WINDOW_LEN`
//! constants, which only the Rust-side cache orchestration uses.
use half::f16;
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

const TOL: f32 = 1.5e-2; // quantization + f16 rounding, looser than the plain f16 kernel tests
const TILE_T: u32 = 8;

fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

fn synthetic(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x >> 8) as f32 / u32::MAX as f32 - 0.5) * 3.0
        })
        .collect()
}

fn quantize_k(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut codes = vec![0i8; block.len()];
    let mut scales = vec![0f32; n_kv_heads * head_dim];
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let mut max_abs = 0f32;
            for t in 0..window_len {
                max_abs = max_abs.max(block[(h * window_len + t) * head_dim + d].abs());
            }
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[h * head_dim + d] = scale;
            for t in 0..window_len {
                let idx = (h * window_len + t) * head_dim + d;
                codes[idx] = (block[idx] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (codes, scales)
}

fn quantize_v_q8(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<i8>, Vec<f32>) {
    let mut codes = vec![0i8; block.len()];
    let mut scales = vec![0f32; n_kv_heads * window_len];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let row = &block[(h * window_len + t) * head_dim..(h * window_len + t + 1) * head_dim];
            let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            scales[h * window_len + t] = scale;
            for d in 0..head_dim {
                let idx = (h * window_len + t) * head_dim + d;
                codes[idx] = (block[idx] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    (codes, scales)
}

fn quantize_v_q4(
    block: &[f32],
    n_kv_heads: usize,
    window_len: usize,
    head_dim: usize,
) -> (Vec<u8>, Vec<f32>) {
    let mut packed = vec![0u8; n_kv_heads * window_len * (head_dim / 2)];
    let mut scales = vec![0f32; n_kv_heads * window_len];
    for h in 0..n_kv_heads {
        for t in 0..window_len {
            let row = &block[(h * window_len + t) * head_dim..(h * window_len + t + 1) * head_dim];
            let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
            let scale = if max_abs > 0.0 { max_abs / 7.0 } else { 1.0 };
            scales[h * window_len + t] = scale;
            for pair in 0..head_dim / 2 {
                let lo = (row[2 * pair] / scale).round().clamp(-8.0, 7.0) as i8;
                let hi = (row[2 * pair + 1] / scale).round().clamp(-8.0, 7.0) as i8;
                packed[(h * window_len + t) * (head_dim / 2) + pair] =
                    ((lo as u8) & 0x0F) | (((hi as u8) & 0x0F) << 4);
            }
        }
    }
    (packed, scales)
}

fn f16_roundtrip(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| f16::from_f32(x).to_f32()).collect()
}

/// One-head-at-a-time causal softmax attention over `cur_len` positions,
/// given the *already resolved* per-position K/V rows (the caller supplies
/// exactly what each kernel is expected to reconstruct: f16-rounded for
/// sink/window positions, quantize-roundtripped for bulk positions).
fn cpu_reference(
    q: &[f32],
    k_by_pos: &[Vec<f32>],
    v_by_pos: &[Vec<f32>],
    n_heads: usize,
    group: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let cur_len = k_by_pos.len();
    let mut out = vec![0.0f32; n_heads * head_dim];
    for h in 0..n_heads {
        let kvh = h / group;
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; cur_len];
        for (t, score) in scores.iter_mut().enumerate() {
            let dot: f32 = q_h.iter().zip(&k_by_pos[t]).map(|(a, b)| a * b).sum();
            *score = dot * scale;
        }
        let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|&s| (s - m).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        for (t, &e) in exps.iter().enumerate() {
            let prob = e / sum;
            for (o, &vv) in out_h.iter_mut().zip(&v_by_pos[t]) {
                let _ = kvh; // kvh already selected via k_by_pos/v_by_pos construction
                *o += prob * vv;
            }
        }
    }
    out
}

struct Scenario {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    sink_len: u32,
    window_len: u32,
    /// How many positions past `window_base` are currently valid (< window_len).
    window_fill: u32,
    n_splits: u32,
}

fn run_mixed(scenario: Scenario, v_bits: u32) {
    let Scenario {
        n_heads,
        n_kv_heads,
        head_dim,
        sink_len,
        window_len,
        window_fill,
        n_splits,
    } = scenario;
    assert!(window_fill <= window_len);
    let group = (n_heads / n_kv_heads) as usize;
    let window_base = sink_len + window_len; // one full block already evicted
    let cur_len = window_base + window_fill;
    let (nkv, hd) = (n_kv_heads as usize, head_dim as usize);

    let _device = Device::new(0).expect("failed to select device 0");
    let (kernel_hsaco, kernel_name) = if v_bits == 8 {
        (
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q8_HSACO,
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q8_KERNEL,
        )
    } else {
        (
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q4_HSACO,
            rocml_kernels::ATTN_DECODE_PARTIAL_MIXED_Q4_KERNEL,
        )
    };
    let (_mp, partial_fn) = load(kernel_hsaco, kernel_name);
    let (_mr, reduce_fn) = load(
        rocml_kernels::ATTN_DECODE_REDUCE_F32_HSACO,
        rocml_kernels::ATTN_DECODE_REDUCE_F32_KERNEL,
    );

    let q: Vec<f32> = synthetic(n_heads as usize * hd, 1);
    let sink_k_f32 = synthetic(nkv * sink_len as usize * hd, 2);
    let sink_v_f32 = synthetic(nkv * sink_len as usize * hd, 3);
    let bulk_block_k = synthetic(nkv * window_len as usize * hd, 4);
    let bulk_block_v = synthetic(nkv * window_len as usize * hd, 5);
    let window_k_f32 = synthetic(nkv * window_len as usize * hd, 6);
    let window_v_f32 = synthetic(nkv * window_len as usize * hd, 7);

    let sink_k_f16 = f16_roundtrip(&sink_k_f32);
    let sink_v_f16 = f16_roundtrip(&sink_v_f32);
    let window_k_f16 = f16_roundtrip(&window_k_f32);
    let window_v_f16 = f16_roundtrip(&window_v_f32);

    let (k_codes, k_scales) = quantize_k(&bulk_block_k, nkv, window_len as usize, hd);
    let k_deq = {
        let mut out = vec![0f32; k_codes.len()];
        for h in 0..nkv {
            for t in 0..window_len as usize {
                for d in 0..hd {
                    let idx = (h * window_len as usize + t) * hd + d;
                    out[idx] = k_codes[idx] as f32 * k_scales[h * hd + d];
                }
            }
        }
        out
    };
    let (v_codes_q8, v_scales_q8) = quantize_v_q8(&bulk_block_v, nkv, window_len as usize, hd);
    let (v_codes_q4, v_scales_q4) = quantize_v_q4(&bulk_block_v, nkv, window_len as usize, hd);
    let (v_deq, v_scales_bulk, bulk_v_codes_bytes): (Vec<f32>, Vec<f32>, Vec<u8>) = if v_bits == 8 {
        let mut out = vec![0f32; v_codes_q8.len()];
        for h in 0..nkv {
            for t in 0..window_len as usize {
                let scale = v_scales_q8[h * window_len as usize + t];
                for d in 0..hd {
                    let idx = (h * window_len as usize + t) * hd + d;
                    out[idx] = v_codes_q8[idx] as f32 * scale;
                }
            }
        }
        (
            out,
            v_scales_q8,
            v_codes_q8.iter().map(|&c| c as u8).collect(),
        )
    } else {
        let mut out = vec![0f32; nkv * window_len as usize * hd];
        for h in 0..nkv {
            for t in 0..window_len as usize {
                let scale = v_scales_q4[h * window_len as usize + t];
                for pair in 0..hd / 2 {
                    let byte = v_codes_q4[(h * window_len as usize + t) * (hd / 2) + pair];
                    let lo =
                        ((byte & 0x0F) as i8).wrapping_sub(if byte & 0x08 != 0 { 16 } else { 0 });
                    let hi_raw = (byte >> 4) & 0x0F;
                    let hi = (hi_raw as i8).wrapping_sub(if hi_raw & 0x08 != 0 { 16 } else { 0 });
                    let base = (h * window_len as usize + t) * hd + 2 * pair;
                    out[base] = lo as f32 * scale;
                    out[base + 1] = hi as f32 * scale;
                }
            }
        }
        (out, v_scales_q4, v_codes_q4)
    };

    // Build the per-position K/V rows a correct kernel run must reproduce.
    let mut k_by_pos: Vec<Vec<f32>> = Vec::with_capacity(cur_len as usize);
    let mut v_by_pos: Vec<Vec<f32>> = Vec::with_capacity(cur_len as usize);
    for kvh in 0..1usize {
        let _ = kvh; // per-kv-head rows are selected below via group mapping
    }
    // Since this test uses a single kv head group at a time via `group`,
    // build rows per kv head and let cpu_reference select via h/group.
    // (This helper only supports n_kv_heads == 1 for reference simplicity;
    // scenarios below all use n_kv_heads == 1.)
    assert_eq!(n_kv_heads, 1, "reference builder assumes a single kv head");
    for pos in 0..cur_len as usize {
        if pos < sink_len as usize {
            k_by_pos.push(sink_k_f16[pos * hd..(pos + 1) * hd].to_vec());
            v_by_pos.push(sink_v_f16[pos * hd..(pos + 1) * hd].to_vec());
        } else if pos < window_base as usize {
            let rel = pos - sink_len as usize;
            k_by_pos.push(k_deq[rel * hd..(rel + 1) * hd].to_vec());
            v_by_pos.push(v_deq[rel * hd..(rel + 1) * hd].to_vec());
        } else {
            let widx = pos - window_base as usize;
            k_by_pos.push(window_k_f16[widx * hd..(widx + 1) * hd].to_vec());
            v_by_pos.push(window_v_f16[widx * hd..(widx + 1) * hd].to_vec());
        }
    }
    let expected = cpu_reference(&q, &k_by_pos, &v_by_pos, n_heads as usize, group, hd);

    // Upload everything.
    let mut buf_q = DeviceBuffer::<f32>::new(q.len()).unwrap();
    buf_q.copy_from_host(&q).unwrap();
    let sink_k_f16h: Vec<f16> = sink_k_f16.iter().map(|&x| f16::from_f32(x)).collect();
    let sink_v_f16h: Vec<f16> = sink_v_f16.iter().map(|&x| f16::from_f32(x)).collect();
    let window_k_f16h: Vec<f16> = window_k_f16.iter().map(|&x| f16::from_f32(x)).collect();
    let window_v_f16h: Vec<f16> = window_v_f16.iter().map(|&x| f16::from_f32(x)).collect();
    let mut buf_sink_k = DeviceBuffer::<f16>::new(sink_k_f16h.len()).unwrap();
    buf_sink_k.copy_from_host(&sink_k_f16h).unwrap();
    let mut buf_sink_v = DeviceBuffer::<f16>::new(sink_v_f16h.len()).unwrap();
    buf_sink_v.copy_from_host(&sink_v_f16h).unwrap();
    let mut buf_window_k = DeviceBuffer::<f16>::new(window_k_f16h.len()).unwrap();
    buf_window_k.copy_from_host(&window_k_f16h).unwrap();
    let mut buf_window_v = DeviceBuffer::<f16>::new(window_v_f16h.len()).unwrap();
    buf_window_v.copy_from_host(&window_v_f16h).unwrap();
    let mut buf_k_codes = DeviceBuffer::<i8>::new(k_codes.len()).unwrap();
    buf_k_codes.copy_from_host(&k_codes).unwrap();
    let mut buf_k_scales = DeviceBuffer::<f32>::new(k_scales.len()).unwrap();
    buf_k_scales.copy_from_host(&k_scales).unwrap();
    let mut buf_v_codes = DeviceBuffer::<u8>::new(bulk_v_codes_bytes.len()).unwrap();
    buf_v_codes.copy_from_host(&bulk_v_codes_bytes).unwrap();
    let mut buf_v_scales = DeviceBuffer::<f32>::new(v_scales_bulk.len()).unwrap();
    buf_v_scales.copy_from_host(&v_scales_bulk).unwrap();

    let buf_out = DeviceBuffer::<f32>::new(n_heads as usize * hd).unwrap();
    let buf_partial_out =
        DeviceBuffer::<f32>::new(n_heads as usize * n_splits as usize * hd).unwrap();
    let buf_partial_m = DeviceBuffer::<f32>::new(n_heads as usize * n_splits as usize).unwrap();
    let buf_partial_l = DeviceBuffer::<f32>::new(n_heads as usize * n_splits as usize).unwrap();

    let q_ptr = buf_q.device_ptr();
    let sink_k_ptr = buf_sink_k.device_ptr();
    let sink_v_ptr = buf_sink_v.device_ptr();
    let window_k_ptr = buf_window_k.device_ptr();
    let window_v_ptr = buf_window_v.device_ptr();
    let k_codes_ptr = buf_k_codes.device_ptr();
    let k_scales_ptr = buf_k_scales.device_ptr();
    let v_codes_ptr = buf_v_codes.device_ptr();
    let v_scales_ptr = buf_v_scales.device_ptr();
    let partial_out_ptr = buf_partial_out.device_ptr();
    let partial_m_ptr = buf_partial_m.device_ptr();
    let partial_l_ptr = buf_partial_l.device_ptr();
    let out_ptr = buf_out.device_ptr();

    let group_u32 = group as u32;
    let bulk_cap = window_len; // exactly one block's worth in this test
    let num_blocks_total = 1u32;
    let split_len = cur_len.div_ceil(n_splits);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let mut params = kernel_params!(
        q_ptr,
        sink_k_ptr,
        sink_v_ptr,
        window_k_ptr,
        window_v_ptr,
        k_codes_ptr,
        k_scales_ptr,
        v_codes_ptr,
        v_scales_ptr,
        partial_out_ptr,
        partial_m_ptr,
        partial_l_ptr,
        n_kv_heads,
        group_u32,
        head_dim,
        sink_len,
        window_len,
        window_base,
        bulk_cap,
        num_blocks_total,
        cur_len,
        split_len,
        n_splits,
        scale
    );
    let cfg = LaunchConfig {
        grid: (n_kv_heads, n_splits, 1),
        block: (32, group as u32, 1),
        shared_mem_bytes: 2 * TILE_T * head_dim * std::mem::size_of::<f32>() as u32,
    };
    unsafe { partial_fn.launch(&cfg, &mut params, None) }.expect("partial launch failed");

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
    unsafe { reduce_fn.launch(&reduce_cfg, &mut reduce_params, None) }
        .expect("reduce launch failed");

    let mut actual = vec![0.0f32; n_heads as usize * hd];
    buf_out.copy_to_host(&mut actual).unwrap();
    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
        let diff = (got - want).abs();
        let tol = TOL * want.abs().max(1.0);
        assert!(
            diff <= tol,
            "mixed decode (v_bits={v_bits}) out[{i}]: got {got}, want {want} (diff {diff}, tol {tol})"
        );
    }
}

#[test]
fn mixed_decode_q8_sink_bulk_and_partial_window() {
    run_mixed(
        Scenario {
            n_heads: 4,
            n_kv_heads: 1,
            head_dim: 64,
            sink_len: 4,
            window_len: 8,
            window_fill: 5,
            n_splits: 1,
        },
        8,
    );
}

#[test]
fn mixed_decode_q4_sink_bulk_and_partial_window() {
    run_mixed(
        Scenario {
            n_heads: 4,
            n_kv_heads: 1,
            head_dim: 64,
            sink_len: 4,
            window_len: 8,
            window_fill: 5,
            n_splits: 1,
        },
        4,
    );
}

#[test]
fn mixed_decode_q8_deep_multi_block_head_dim_256() {
    // head_dim 256 (Ornith's real shape) and a longer window to exercise
    // the tile loop across more than one TILE_T(8) iteration.
    run_mixed(
        Scenario {
            n_heads: 8,
            n_kv_heads: 1,
            head_dim: 256,
            sink_len: 32,
            window_len: 128,
            window_fill: 100,
            n_splits: 1,
        },
        8,
    );
}

#[test]
fn mixed_decode_q8_split_k_matches_single_split() {
    // n_splits > 1: exercises the split-K path attn_decode_splits picks at
    // real model depth (the model-level bug this test was added to isolate
    // showed up only here, not at n_splits == 1).
    run_mixed(
        Scenario {
            n_heads: 4,
            n_kv_heads: 1,
            head_dim: 256,
            sink_len: 32,
            window_len: 128,
            window_fill: 69,
            n_splits: 2,
        },
        8,
    );
}

#[test]
fn mixed_decode_q4_split_k_matches_single_split() {
    run_mixed(
        Scenario {
            n_heads: 4,
            n_kv_heads: 1,
            head_dim: 256,
            sink_len: 32,
            window_len: 128,
            window_fill: 69,
            n_splits: 2,
        },
        4,
    );
}

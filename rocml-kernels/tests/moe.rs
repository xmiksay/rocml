//! GPU integration tests for `kernels/moe.hip` (qwen35moe's router + expert
//! accumulate ops) against a plain CPU reference — see `kv_quant.rs`'s
//! module doc for why this crate's kernel tests always write their own
//! standalone reference rather than importing one from `rocml`.
use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

fn synthetic_logits(rows: usize, expert_count: usize, seed: u32) -> Vec<f32> {
    (0..rows * expert_count)
        .map(|i| {
            let x = (i as u32).wrapping_mul(2654435761).wrapping_add(seed);
            ((x >> 8) as f32 / u32::MAX as f32 - 0.5) * 8.0
        })
        .collect()
}

/// Plain f64 CPU reference: softmax, then `top_k` sequential argmax
/// extractions (ties broken toward the lower index, matching the kernel's
/// own reduction order), then renormalize with the sum clamped to
/// `min_sum`.
fn cpu_route_topk(
    logits: &[f32],
    rows: usize,
    expert_count: usize,
    top_k: usize,
    min_sum: f32,
) -> (Vec<i32>, Vec<f32>) {
    let mut out_idx = vec![0i32; rows * top_k];
    let mut out_weight = vec![0f32; rows * top_k];
    for r in 0..rows {
        let row = &logits[r * expert_count..(r + 1) * expert_count];
        let max = row
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, |a, b| a.max(b as f64));
        let mut probs: Vec<f64> = row.iter().map(|&v| (v as f64 - max).exp()).collect();
        let sum: f64 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }
        for k in 0..top_k {
            let mut best_idx = 0usize;
            let mut best_val = -1.0f64;
            for (i, &p) in probs.iter().enumerate() {
                if p > best_val {
                    best_val = p;
                    best_idx = i;
                }
            }
            out_idx[r * top_k + k] = best_idx as i32;
            out_weight[r * top_k + k] = best_val as f32;
            probs[best_idx] = -1.0;
        }
        let mut wsum: f64 = (0..top_k).map(|k| out_weight[r * top_k + k] as f64).sum();
        if wsum < min_sum as f64 {
            wsum = min_sum as f64;
        }
        for k in 0..top_k {
            out_weight[r * top_k + k] = (out_weight[r * top_k + k] as f64 / wsum) as f32;
        }
    }
    (out_idx, out_weight)
}

#[test]
fn route_topk_matches_cpu_reference_for_several_rows() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_module, function) = load(
        rocml_kernels::MOE_ROUTE_TOPK_F32_HSACO,
        rocml_kernels::MOE_ROUTE_TOPK_F32_KERNEL,
    );

    let rows = 5usize;
    let expert_count = 256usize;
    let top_k = 8usize;
    let min_sum = 6.1e-5f32;
    let logits_host = synthetic_logits(rows, expert_count, 0x1234_5678);

    let mut logits = DeviceBuffer::<f32>::new(rows * expert_count).expect("alloc logits");
    logits.copy_from_host(&logits_host).expect("copy logits");
    let out_idx = DeviceBuffer::<i32>::new(rows * top_k).expect("alloc idx");
    let out_weight = DeviceBuffer::<f32>::new(rows * top_k).expect("alloc weight");

    let cfg = LaunchConfig {
        grid: (rows as u32, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rows_u, expert_count_u, top_k_u) = (rows as u32, expert_count as u32, top_k as u32);
    let logits_ptr = logits.device_ptr();
    let out_idx_ptr = out_idx.device_ptr();
    let out_weight_ptr = out_weight.device_ptr();
    let mut params = kernel_params!(
        logits_ptr,
        out_idx_ptr,
        out_weight_ptr,
        rows_u,
        expert_count_u,
        top_k_u,
        min_sum
    );
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got_idx = vec![0i32; rows * top_k];
    let mut got_weight = vec![0f32; rows * top_k];
    out_idx.copy_to_host(&mut got_idx).expect("copy idx back");
    out_weight
        .copy_to_host(&mut got_weight)
        .expect("copy weight back");

    let (want_idx, want_weight) = cpu_route_topk(&logits_host, rows, expert_count, top_k, min_sum);
    assert_eq!(
        got_idx, want_idx,
        "selected expert indices must match exactly"
    );
    for (g, w) in got_weight.iter().zip(&want_weight) {
        assert!(
            (g - w).abs() < 1e-4,
            "renormalized weight mismatch: got {g}, want {w}"
        );
    }

    // Every row's renormalized weights must sum to ~1 (the whole point of
    // renormalization) and every selected index must be distinct.
    for r in 0..rows {
        let row_sum: f32 = (0..top_k).map(|k| got_weight[r * top_k + k]).sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-4,
            "row {r}: weights sum to {row_sum}"
        );
        let mut idxs = got_idx[r * top_k..(r + 1) * top_k].to_vec();
        idxs.sort_unstable();
        idxs.dedup();
        assert_eq!(
            idxs.len(),
            top_k,
            "row {r}: expected {top_k} distinct experts"
        );
    }
}

#[test]
fn route_topk_handles_a_single_row_and_small_expert_count() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_module, function) = load(
        rocml_kernels::MOE_ROUTE_TOPK_F32_HSACO,
        rocml_kernels::MOE_ROUTE_TOPK_F32_KERNEL,
    );

    let expert_count = 4usize;
    let top_k = 2usize;
    let min_sum = 6.1e-5f32;
    let logits_host = vec![1.0f32, 5.0, -3.0, 2.0];

    let mut logits = DeviceBuffer::<f32>::new(expert_count).expect("alloc logits");
    logits.copy_from_host(&logits_host).expect("copy logits");
    let out_idx = DeviceBuffer::<i32>::new(top_k).expect("alloc idx");
    let out_weight = DeviceBuffer::<f32>::new(top_k).expect("alloc weight");

    let cfg = LaunchConfig {
        grid: (1, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let (rows_u, expert_count_u, top_k_u) = (1u32, expert_count as u32, top_k as u32);
    let logits_ptr = logits.device_ptr();
    let out_idx_ptr = out_idx.device_ptr();
    let out_weight_ptr = out_weight.device_ptr();
    let mut params = kernel_params!(
        logits_ptr,
        out_idx_ptr,
        out_weight_ptr,
        rows_u,
        expert_count_u,
        top_k_u,
        min_sum
    );
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got_idx = vec![0i32; top_k];
    out_idx.copy_to_host(&mut got_idx).expect("copy idx back");
    // Logits [1, 5, -3, 2]: top-2 by value are index 1 (5.0) then index 3 (2.0).
    assert_eq!(got_idx, vec![1, 3]);
}

#[test]
fn shared_gate_write_matches_sigmoid_gated_reference() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_module, function) = load(
        rocml_kernels::MOE_SHARED_GATE_WRITE_F32_HSACO,
        rocml_kernels::MOE_SHARED_GATE_WRITE_F32_KERNEL,
    );

    let n = 37usize;
    let y_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 1.5).collect();
    let gate_logit_host = [0.42f32];

    let mut y = DeviceBuffer::<f32>::new(n).expect("alloc y");
    y.copy_from_host(&y_host).expect("copy y");
    let mut gate_logit = DeviceBuffer::<f32>::new(1).expect("alloc gate");
    gate_logit
        .copy_from_host(&gate_logit_host)
        .expect("copy gate");
    let out = DeviceBuffer::<f32>::new(n).expect("alloc out");

    let cfg = LaunchConfig {
        grid: (n.div_ceil(256) as u32, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u = n as u32;
    let y_ptr = y.device_ptr();
    let gate_logit_ptr = gate_logit.device_ptr();
    let out_ptr = out.device_ptr();
    let mut params = kernel_params!(y_ptr, gate_logit_ptr, out_ptr, n_u);
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got = vec![0f32; n];
    out.copy_to_host(&mut got).expect("copy out back");

    let w = 1.0 / (1.0 + (-gate_logit_host[0] as f64).exp());
    for (i, (&g, &y)) in got.iter().zip(&y_host).enumerate() {
        let want = (y as f64 * w) as f32;
        assert!((g - want).abs() < 1e-5, "index {i}: got {g}, want {want}");
    }
}

#[test]
fn weighted_accum_adds_scaled_contribution_on_top_of_existing_accumulator() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_module, function) = load(
        rocml_kernels::MOE_WEIGHTED_ACCUM_F32_HSACO,
        rocml_kernels::MOE_WEIGHTED_ACCUM_F32_KERNEL,
    );

    let n = 20usize;
    let y_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let weight_host = [0.25f32];
    let existing: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();

    let mut y = DeviceBuffer::<f32>::new(n).expect("alloc y");
    y.copy_from_host(&y_host).expect("copy y");
    let mut weight = DeviceBuffer::<f32>::new(1).expect("alloc weight");
    weight.copy_from_host(&weight_host).expect("copy weight");
    let mut out = DeviceBuffer::<f32>::new(n).expect("alloc out");
    out.copy_from_host(&existing).expect("seed out");

    let cfg = LaunchConfig {
        grid: (n.div_ceil(256) as u32, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u = n as u32;
    let y_ptr = y.device_ptr();
    let weight_ptr = weight.device_ptr();
    let out_ptr = out.device_ptr();
    let mut params = kernel_params!(y_ptr, weight_ptr, out_ptr, n_u);
    unsafe { function.launch(&cfg, &mut params, None) }.expect("launch failed");

    let mut got = vec![0f32; n];
    out.copy_to_host(&mut got).expect("copy out back");

    for i in 0..n {
        let want = existing[i] + y_host[i] * weight_host[0];
        assert!(
            (got[i] - want).abs() < 1e-5,
            "index {i}: got {}, want {want}",
            got[i]
        );
    }
}

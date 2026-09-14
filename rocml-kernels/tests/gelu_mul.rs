//! GPU integration test for `gelu_mul_f32` (issue #16's config-driven-
//! activation seam — see `kernels/silu_mul.hip`'s module doc). Checked
//! against a CPU f64 reference of the tanh-approximation GELU (HF's
//! "gelu_pytorch_tanh") to keep the tolerance tight despite `tanhf`'s
//! single-precision rounding.

use std::ffi::c_void;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module};

fn gelu_tanh_ref(x: f64) -> f64 {
    const K0: f64 = 0.7978845608028654; // sqrt(2/pi)
    const K1: f64 = 0.044715;
    0.5 * x * (1.0 + (K0 * (x + K1 * x * x * x)).tanh())
}

fn run_gelu_mul(n: u32) {
    let _device = Device::new(0).expect("failed to select device 0");
    let module =
        Module::load_from_bytes(rocml_kernels::SILU_MUL_F32_HSACO).expect("module load failed");
    let function = module
        .get_function(rocml_kernels::GELU_MUL_F32_KERNEL)
        .expect("kernel lookup failed");

    let gate: Vec<f32> = (0..n)
        .map(|i| ((i as f32) - (n as f32) / 2.0) * 0.1)
        .collect();
    let up: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.3 - 0.6).collect();
    let expected: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| (gelu_tanh_ref(g as f64) * u as f64) as f32)
        .collect();

    let mut buf_gate = DeviceBuffer::<f32>::new(gate.len()).expect("hipMalloc gate failed");
    let mut buf_up = DeviceBuffer::<f32>::new(up.len()).expect("hipMalloc up failed");
    let buf_out = DeviceBuffer::<f32>::new(gate.len()).expect("hipMalloc out failed");
    buf_gate.copy_from_host(&gate).expect("copy gate failed");
    buf_up.copy_from_host(&up).expect("copy up failed");

    let gate_ptr: *mut c_void = buf_gate.device_ptr();
    let up_ptr: *mut c_void = buf_up.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();
    let mut params = kernel_params!(gate_ptr, up_ptr, out_ptr, n);

    let cfg = LaunchConfig {
        grid: (n.div_ceil(256), 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: params matches gelu_mul_f32's signature (const float*, const
    // float*, float*, unsigned); no block-size constraint.
    unsafe { function.launch(&cfg, &mut params, None) }.expect("kernel launch failed");

    let mut actual = vec![0.0f32; gate.len()];
    buf_out.copy_to_host(&mut actual).expect("copy out failed");

    for (i, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        let diff = (got - want).abs();
        let tol = 1e-5 * want.abs().max(1.0);
        assert!(diff <= tol, "[{i}]: got {got}, want {want} (diff {diff})");
    }
}

#[test]
fn gelu_mul_matches_cpu_reference() {
    run_gelu_mul(1024);
}

#[test]
fn gelu_mul_non_multiple_of_blocksize() {
    run_gelu_mul(777);
}

#[test]
fn gelu_mul_degenerate_single_element() {
    run_gelu_mul(1);
}

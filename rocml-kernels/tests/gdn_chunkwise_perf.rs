//! Diagnostic perf probe (not part of any CI gate — `#[ignore]`d, run
//! explicitly) for the chunkwise GDN recurrence pipeline at a realistic
//! Ornith-1.0-9B shape (32 v-heads / 16 k-heads, head dims 128, tile 128):
//! times each of the six kernel stages individually to find which one
//! dominates wall time.
#[path = "gdn_chunkwise_support/mod.rs"]
mod support;

use std::ffi::c_void;

use rocml_hip::{elapsed_ms, kernel_params, Device, DeviceBuffer, Event, LaunchConfig};
use support::ChunkwiseKernels;

#[test]
#[ignore]
fn perf_probe_gdn_chunkwise_stages_ornith_shape() {
    let _device = Device::new(0).expect("failed to select device 0");
    let k = ChunkwiseKernels::load();

    let (h, hk, sk, sv, t): (u32, u32, u32, u32, u32) = (32, 16, 128, 128, 128);
    let key_dim = hk * sk;
    let value_dim = h * sv;
    let conv_dim = 2 * key_dim + value_dim;

    let conv_out = DeviceBuffer::<f32>::new((t * conv_dim) as usize).unwrap();
    let beta = DeviceBuffer::<f32>::new((t * h) as usize).unwrap();
    let g = DeviceBuffer::<f32>::new((t * h) as usize).unwrap();
    let state = DeviceBuffer::<f32>::new((h * sk * sv) as usize).unwrap();
    let q_norm = DeviceBuffer::<f32>::new((t * sk * h) as usize).unwrap();
    let k_norm = DeviceBuffer::<f32>::new((t * sk * h) as usize).unwrap();
    let k_beta = DeviceBuffer::<f32>::new((t * sk * h) as usize).unwrap();
    let g_cum = DeviceBuffer::<f32>::new((t * h) as usize).unwrap();
    let cum_decay_exp = DeviceBuffer::<f32>::new((t * h) as usize).unwrap();
    let kb = DeviceBuffer::<f32>::new((t * t * h) as usize).unwrap();
    let kq = DeviceBuffer::<f32>::new((t * t * h) as usize).unwrap();
    let v_new = DeviceBuffer::<f32>::new((t * sv * h) as usize).unwrap();
    let y = DeviceBuffer::<f32>::new((t * sv * h) as usize).unwrap();

    let conv_out_p: *mut c_void = conv_out.device_ptr();
    let beta_p: *mut c_void = beta.device_ptr();
    let g_p: *mut c_void = g.device_ptr();
    let state_p: *mut c_void = state.device_ptr();
    let q_norm_p: *mut c_void = q_norm.device_ptr();
    let k_norm_p: *mut c_void = k_norm.device_ptr();
    let k_beta_p: *mut c_void = k_beta.device_ptr();
    let g_cum_p: *mut c_void = g_cum.device_ptr();
    let cde_p: *mut c_void = cum_decay_exp.device_ptr();
    let kb_p: *mut c_void = kb.device_ptr();
    let kq_p: *mut c_void = kq.device_ptr();
    let vnew_p: *mut c_void = v_new.device_ptr();
    let y_p: *mut c_void = y.device_ptr();

    const ITERS: u32 = 200;

    macro_rules! time_stage {
        ($name:expr, $body:expr) => {{
            let start = Event::new().unwrap();
            let stop = Event::new().unwrap();
            start.record(None).unwrap();
            for _ in 0..ITERS {
                $body
            }
            stop.record(None).unwrap();
            let ms = elapsed_ms(&start, &stop).unwrap();
            println!(
                "{}: {:.4} ms/iter ({:.2} ms total)",
                $name,
                ms / ITERS as f64,
                ms
            );
        }};
    }

    time_stage!("A prep", {
        let cfg = LaunchConfig {
            grid: (h, 1, 1),
            block: (t, 1, 1),
            shared_mem_bytes: t * 4,
        };
        let l2_eps = 1e-6f32;
        let mut params = kernel_params!(
            conv_out_p, beta_p, g_p, q_norm_p, k_norm_p, k_beta_p, g_cum_p, cde_p, h, hk, sk,
            conv_dim, key_dim, t, l2_eps
        );
        unsafe { k.prep_fn().launch(&cfg, &mut params, None) }.unwrap();
    });

    time_stage!("B ut_build", {
        let cfg = LaunchConfig {
            grid: (h, t, 1),
            block: (32, 8, 1),
            shared_mem_bytes: 0,
        };
        let mut params =
            kernel_params!(q_norm_p, k_norm_p, k_beta_p, g_cum_p, kb_p, kq_p, h, sk, t);
        unsafe { k.ut_build_fn().launch(&cfg, &mut params, None) }.unwrap();
    });

    time_stage!("C tinv", {
        let cfg = LaunchConfig {
            grid: (h, 1, 1),
            block: (t, 1, 1),
            shared_mem_bytes: t * t * 4,
        };
        let mut params = kernel_params!(kb_p, h, t);
        unsafe { k.tinv_fn().launch(&cfg, &mut params, None) }.unwrap();
    });

    time_stage!("D+E uv_vnew", {
        let block = sk.max(sv);
        let cfg = LaunchConfig {
            grid: (h, t, 1),
            block: (block, 1, 1),
            shared_mem_bytes: sk * 4,
        };
        let mut params = kernel_params!(
            kb_p, conv_out_p, beta_p, k_beta_p, cde_p, state_p, vnew_p, h, hk, sk, sv, conv_dim,
            key_dim, t
        );
        unsafe { k.uv_vnew_fn().launch(&cfg, &mut params, None) }.unwrap();
    });

    time_stage!("F output", {
        let cfg = LaunchConfig {
            grid: (h, t, 1),
            block: (sv, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(q_norm_p, cde_p, state_p, kq_p, vnew_p, y_p, h, sk, sv, t);
        unsafe { k.output_fn().launch(&cfg, &mut params, None) }.unwrap();
    });

    time_stage!("G state_update", {
        let cfg = LaunchConfig {
            grid: (h, sk, 1),
            block: (sv, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(k_norm_p, g_cum_p, vnew_p, state_p, h, sk, sv, t);
        unsafe { k.state_update_fn().launch(&cfg, &mut params, None) }.unwrap();
    });
}

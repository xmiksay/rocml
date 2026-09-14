//! Informational perf probe (gdn-uvvnew round, issue #6) comparing scalar
//! vs. LDS-staged WMMA `uv_vnew` (fused D+E) at Ornith-1.0-9B's real shape
//! (32 v-heads/16 k-heads, head dims 128, tile 128) — not a correctness gate
//! (see `gdn_chunkwise_wmma_lds.rs` for that). Two-way, not three-way: unlike
//! `ut_build`/`output`, no naive (non-LDS-staged) WMMA variant of `uv_vnew`
//! was ever built — the prior two rounds evaluated and rejected it before
//! writing any code (see `gdn_chunkwise_kernels_wmma.rs`'s module doc).
//!
//! Run with `cargo test --release -p rocml-kernels --test
//! gdn_chunkwise_uv_vnew_wmma_lds_perf -- --ignored --nocapture` (or `make
//! gdn-uvvnew-perf`). One process, interleaved rounds (this codebase's
//! established anti-clock-drift methodology — cross-process `cargo test`
//! runs on this machine show up to 2x swings in raw wall time from ambient
//! GPU clock state alone, see `gdn_chunkwise_wmma_lds_perf.rs`'s module doc
//! for the same reasoning).
use std::ffi::c_void;
use std::time::Instant;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const WARMUP_ITERS: u32 = 10;
const TIMED_ITERS: u32 = 100;
const ROUNDS: usize = 5;
const LDS_TILE_BYTES: u32 = 128 * 64 * 2;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn time_launch(stream: &Stream, mut launch: impl FnMut()) -> f64 {
    for _ in 0..WARMUP_ITERS {
        launch();
    }
    stream.synchronize().expect("sync failed");
    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        launch();
    }
    stream.synchronize().expect("sync failed");
    start.elapsed().as_secs_f64() / TIMED_ITERS as f64 * 1e6 // us/call
}

fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

#[test]
#[ignore]
fn perf_compare_uv_vnew_ornith_shape() {
    let _device = Device::new(0).expect("failed to select device 0");
    let stream = Stream::new().expect("failed to create stream");

    let (h, hk, sk, sv, t): (u32, u32, u32, u32, u32) = (32, 16, 128, 128, 128);
    let key_dim = hk * sk;
    let conv_dim = 2 * key_dim + h * sv;

    // Synthetic but finite, deterministic operand data — this probe only
    // measures wall time, not correctness (see gdn_chunkwise_wmma_lds.rs for
    // that), so the exact values don't matter beyond avoiding NaN/Inf.
    let tinv_h: Vec<f32> = (0..(t * t * h))
        .map(|i| (i % 11) as f32 * 0.02 - 0.1)
        .collect();
    let conv_out_h: Vec<f32> = (0..(t * conv_dim))
        .map(|i| (i % 23) as f32 * 0.07 - 0.75)
        .collect();
    let beta_h: Vec<f32> = (0..(t * h))
        .map(|i| 0.1 + (i as f32 % 7.0) * 0.09)
        .collect();
    let k_beta_h: Vec<f32> = (0..(t * sk * h))
        .map(|i| (i % 13) as f32 * 0.05 - 0.3)
        .collect();
    let g_cum_h: Vec<f32> = (0..(t * h)).map(|i| -(i as f32 % 23.0) * 0.01).collect();
    let cde_h: Vec<f32> = g_cum_h.iter().map(|&g| g.exp()).collect();
    let state_h: Vec<f32> = (0..(h * sk * sv))
        .map(|i| (i % 13) as f32 * 0.04 - 0.24)
        .collect();

    macro_rules! upload {
        ($v:expr) => {{
            let mut b = DeviceBuffer::<f32>::new($v.len()).unwrap();
            b.copy_from_host(&$v).unwrap();
            b
        }};
    }
    let tinv = upload!(tinv_h);
    let conv_out = upload!(conv_out_h);
    let beta = upload!(beta_h);
    let k_beta = upload!(k_beta_h);
    let cde = upload!(cde_h);
    let state = upload!(state_h);
    let v_new = DeviceBuffer::<f32>::new((t * sv * h) as usize).unwrap();

    let tinv_p: *mut c_void = tinv.device_ptr();
    let conv_out_p: *mut c_void = conv_out.device_ptr();
    let beta_p: *mut c_void = beta.device_ptr();
    let k_beta_p: *mut c_void = k_beta.device_ptr();
    let cde_p: *mut c_void = cde.device_ptr();
    let state_p: *mut c_void = state.device_ptr();
    let vnew_p: *mut c_void = v_new.device_ptr();

    let (_m_s, scalar_fn) = load(
        rocml_kernels::GDN_CW_UV_VNEW_F32_HSACO,
        rocml_kernels::GDN_CW_UV_VNEW_F32_KERNEL,
    );
    let (_m_l, lds_fn) = load(
        rocml_kernels::GDN_CW_UV_VNEW_WMMA_LDS_F32_HSACO,
        rocml_kernels::GDN_CW_UV_VNEW_WMMA_LDS_F32_KERNEL,
    );

    let scalar_block = sk.max(sv);
    let scalar_cfg = LaunchConfig {
        grid: (h, t, 1),
        block: (scalar_block, 1, 1),
        shared_mem_bytes: sk * 4,
    };
    let mut scalar_params = kernel_params!(
        tinv_p, conv_out_p, beta_p, k_beta_p, cde_p, state_p, vnew_p, h, hk, sk, sv, conv_dim,
        key_dim, t
    );
    let lds_cfg = LaunchConfig {
        grid: (h, 1, 1),
        block: (32, 16, 1),
        shared_mem_bytes: 3 * LDS_TILE_BYTES,
    };
    let mut lds_params = kernel_params!(
        tinv_p, conv_out_p, beta_p, k_beta_p, cde_p, state_p, vnew_p, h, hk, sk, sv, conv_dim,
        key_dim, t
    );

    let (mut s, mut l) = (vec![], vec![]);
    for _ in 0..ROUNDS {
        s.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_uv_vnew_f32's signature.
            unsafe { scalar_fn.launch(&scalar_cfg, &mut scalar_params, Some(&stream)) }
                .expect("uv_vnew scalar launch failed");
        }));
        l.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_uv_vnew_wmma_lds_f32's
            // signature; shared_mem_bytes matches its three staged tiles.
            unsafe { lds_fn.launch(&lds_cfg, &mut lds_params, Some(&stream)) }
                .expect("uv_vnew lds launch failed");
        }));
    }

    let (s, l) = (median(s), median(l));
    println!(
        "uv_vnew: scalar={s:.1}us lds_wmma={l:.1}us ({:+.1}% vs scalar)",
        (l / s - 1.0) * 100.0,
    );
}

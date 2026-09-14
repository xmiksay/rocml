//! Informational perf probe (gdn-wmma-lds round, issue #6) comparing three
//! implementations of GDN chunkwise stages B (`ut_build`) and F (`output`)
//! at Ornith-1.0-9B's real shape (32 v-heads/16 k-heads, head dims 128, tile
//! 128) — not a correctness gate (see `gdn_chunkwise_wmma_lds.rs` for that).
//! Run with `cargo test --release -p rocml-kernels --test
//! gdn_chunkwise_wmma_lds_perf -- --ignored --nocapture` (or `make
//! gdn-wmma-lds-perf`).
//!
//! One process, interleaved rounds (scalar, then naive WMMA
//! (`gdn_chunkwise_wmma.hip`), then LDS-staged WMMA
//! (`gdn_chunkwise_{ut_build,output}_wmma_lds.hip`), repeated `ROUNDS` times,
//! median per variant) — this codebase's established anti-clock-drift
//! methodology (see `gemm_xwt_wmma_splitk_perf.rs`'s module doc for the same
//! reasoning): cross-process `cargo test` runs on this machine show up to 2x
//! swings in raw wall time from ambient GPU clock state alone.
use std::ffi::c_void;
use std::time::Instant;

use rocml_hip::{kernel_params, Device, DeviceBuffer, LaunchConfig, Module, Stream};

const WARMUP_ITERS: u32 = 10;
const TIMED_ITERS: u32 = 100;
const ROUNDS: usize = 5;
const LDS_TILE_BYTES: u32 = 128 * 64 * 2;

/// Must match `kernels/gdn_chunkwise.hip`'s `#define UT_BUILD_J_PER_BLOCK`.
const UT_BUILD_J_PER_BLOCK: u32 = 8;

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
fn perf_compare_ut_build_and_output_ornith_shape() {
    let _device = Device::new(0).expect("failed to select device 0");
    let stream = Stream::new().expect("failed to create stream");

    // `hk`(16)/`num_k_heads` doesn't appear in either stage's own signature —
    // kept only in this comment for Ornith-shape documentation parity with
    // the other gdn_chunkwise perf probes.
    let (h, sk, sv, t): (u32, u32, u32, u32) = (32, 128, 128, 128);

    // Synthetic but finite, deterministic operand data — this probe only
    // measures wall time, not correctness (see gdn_chunkwise_wmma_lds.rs for
    // that), so the exact values don't matter beyond avoiding NaN/Inf.
    let q_norm_h: Vec<f32> = (0..(t * sk * h))
        .map(|i| (i % 17) as f32 * 0.05 - 0.4)
        .collect();
    let k_norm_h: Vec<f32> = (0..(t * sk * h))
        .map(|i| (i % 19) as f32 * 0.05 - 0.4)
        .collect();
    let k_beta_h: Vec<f32> = (0..(t * sk * h))
        .map(|i| (i % 13) as f32 * 0.05 - 0.3)
        .collect();
    let g_cum_h: Vec<f32> = (0..(t * h)).map(|i| -(i as f32 % 23.0) * 0.01).collect();
    let cde_h: Vec<f32> = g_cum_h.iter().map(|&g| g.exp()).collect();
    let kq_h: Vec<f32> = (0..(t * t * h))
        .map(|i| (i % 11) as f32 * 0.03 - 0.15)
        .collect();
    let v_new_h: Vec<f32> = (0..(t * sv * h))
        .map(|i| (i % 15) as f32 * 0.04 - 0.28)
        .collect();
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
    let q_norm = upload!(q_norm_h);
    let k_norm = upload!(k_norm_h);
    let k_beta = upload!(k_beta_h);
    let g_cum = upload!(g_cum_h);
    let cde = upload!(cde_h);
    let kq = upload!(kq_h);
    let v_new = upload!(v_new_h);
    let state = upload!(state_h);
    let kb = DeviceBuffer::<f32>::new((t * t * h) as usize).unwrap();
    let y = DeviceBuffer::<f32>::new((t * sv * h) as usize).unwrap();

    let q_norm_p: *mut c_void = q_norm.device_ptr();
    let k_norm_p: *mut c_void = k_norm.device_ptr();
    let k_beta_p: *mut c_void = k_beta.device_ptr();
    let g_cum_p: *mut c_void = g_cum.device_ptr();
    let cde_p: *mut c_void = cde.device_ptr();
    let kq_p: *mut c_void = kq.device_ptr();
    let vnew_p: *mut c_void = v_new.device_ptr();
    let state_p: *mut c_void = state.device_ptr();
    let kb_p: *mut c_void = kb.device_ptr();
    let y_p: *mut c_void = y.device_ptr();

    let (_m_ub_s, ub_scalar) = load(
        rocml_kernels::GDN_CW_UT_BUILD_F32_HSACO,
        rocml_kernels::GDN_CW_UT_BUILD_F32_KERNEL,
    );
    let (_m_ub_n, ub_naive) = load(
        rocml_kernels::GDN_CW_UT_BUILD_WMMA_F32_HSACO,
        rocml_kernels::GDN_CW_UT_BUILD_WMMA_F32_KERNEL,
    );
    let (_m_ub_l, ub_lds) = load(
        rocml_kernels::GDN_CW_UT_BUILD_WMMA_LDS_F32_HSACO,
        rocml_kernels::GDN_CW_UT_BUILD_WMMA_LDS_F32_KERNEL,
    );
    let (_m_o_s, out_scalar) = load(
        rocml_kernels::GDN_CW_OUTPUT_F32_HSACO,
        rocml_kernels::GDN_CW_OUTPUT_F32_KERNEL,
    );
    let (_m_o_n, out_naive) = load(
        rocml_kernels::GDN_CW_OUTPUT_WMMA_F32_HSACO,
        rocml_kernels::GDN_CW_OUTPUT_WMMA_F32_KERNEL,
    );
    let (_m_o_l, out_lds) = load(
        rocml_kernels::GDN_CW_OUTPUT_WMMA_LDS_F32_HSACO,
        rocml_kernels::GDN_CW_OUTPUT_WMMA_LDS_F32_KERNEL,
    );

    let ub_scalar_cfg = LaunchConfig {
        grid: (h, t, 1),
        block: (32, UT_BUILD_J_PER_BLOCK, 1),
        shared_mem_bytes: 0,
    };
    let mut ub_scalar_params =
        kernel_params!(q_norm_p, k_norm_p, k_beta_p, g_cum_p, kb_p, kq_p, h, sk, t);
    let ub_naive_cfg = LaunchConfig {
        grid: (h, t.div_ceil(16), t.div_ceil(16)),
        block: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut ub_naive_params =
        kernel_params!(q_norm_p, k_norm_p, k_beta_p, g_cum_p, kb_p, kq_p, h, sk, t);
    let ub_lds_cfg = LaunchConfig {
        grid: (h, 1, 1),
        block: (32, 16, 1),
        shared_mem_bytes: 3 * LDS_TILE_BYTES,
    };
    let mut ub_lds_params =
        kernel_params!(q_norm_p, k_norm_p, k_beta_p, g_cum_p, kb_p, kq_p, h, sk, t);

    let out_scalar_cfg = LaunchConfig {
        grid: (h, t, 1),
        block: (sv, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out_scalar_params =
        kernel_params!(q_norm_p, cde_p, state_p, kq_p, vnew_p, y_p, h, sk, sv, t);
    let out_naive_cfg = LaunchConfig {
        grid: (h, t.div_ceil(16), sv.div_ceil(16)),
        block: (32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut out_naive_params =
        kernel_params!(q_norm_p, cde_p, state_p, kq_p, vnew_p, y_p, h, sk, sv, t);
    let out_lds_cfg = LaunchConfig {
        grid: (h, 1, 1),
        block: (32, 16, 1),
        shared_mem_bytes: 2 * LDS_TILE_BYTES,
    };
    let mut out_lds_params =
        kernel_params!(q_norm_p, cde_p, state_p, kq_p, vnew_p, y_p, h, sk, sv, t);

    let (mut ub_s, mut ub_n, mut ub_l) = (vec![], vec![], vec![]);
    let (mut o_s, mut o_n, mut o_l) = (vec![], vec![], vec![]);
    for _ in 0..ROUNDS {
        ub_s.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_ut_build_f32's signature.
            unsafe { ub_scalar.launch(&ub_scalar_cfg, &mut ub_scalar_params, Some(&stream)) }
                .expect("ut_build scalar launch failed");
        }));
        ub_n.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_ut_build_wmma_f32's signature.
            unsafe { ub_naive.launch(&ub_naive_cfg, &mut ub_naive_params, Some(&stream)) }
                .expect("ut_build naive launch failed");
        }));
        ub_l.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_ut_build_wmma_lds_f32's
            // signature; shared_mem_bytes matches its three staged tiles.
            unsafe { ub_lds.launch(&ub_lds_cfg, &mut ub_lds_params, Some(&stream)) }
                .expect("ut_build lds launch failed");
        }));
        o_s.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_output_f32's signature.
            unsafe { out_scalar.launch(&out_scalar_cfg, &mut out_scalar_params, Some(&stream)) }
                .expect("output scalar launch failed");
        }));
        o_n.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_output_wmma_f32's signature.
            unsafe { out_naive.launch(&out_naive_cfg, &mut out_naive_params, Some(&stream)) }
                .expect("output naive launch failed");
        }));
        o_l.push(time_launch(&stream, || {
            // SAFETY: params match gdn_chunkwise_output_wmma_lds_f32's
            // signature; shared_mem_bytes matches its two staged tiles.
            unsafe { out_lds.launch(&out_lds_cfg, &mut out_lds_params, Some(&stream)) }
                .expect("output lds launch failed");
        }));
    }

    let (ub_s, ub_n, ub_l) = (median(ub_s), median(ub_n), median(ub_l));
    let (o_s, o_n, o_l) = (median(o_s), median(o_n), median(o_l));
    println!(
        "ut_build: scalar={ub_s:.1}us naive_wmma={ub_n:.1}us ({:+.1}%) lds_wmma={ub_l:.1}us ({:+.1}% vs scalar, {:+.1}% vs naive)",
        (ub_n / ub_s - 1.0) * 100.0,
        (ub_l / ub_s - 1.0) * 100.0,
        (ub_l / ub_n - 1.0) * 100.0,
    );
    println!(
        "output:   scalar={o_s:.1}us naive_wmma={o_n:.1}us ({:+.1}%) lds_wmma={o_l:.1}us ({:+.1}% vs scalar, {:+.1}% vs naive)",
        (o_n / o_s - 1.0) * 100.0,
        (o_l / o_s - 1.0) * 100.0,
        (o_l / o_n - 1.0) * 100.0,
    );
}

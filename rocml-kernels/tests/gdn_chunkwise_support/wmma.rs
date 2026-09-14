//! WMMA variant of the chunkwise GPU pipeline runner (gdn-wmma round, issue
//! #6): stages A/C/D+E are identical to [`super::run_chunkwise_gpu`] (the
//! scalar `ChunkwiseKernels`) — B (`ut_build`), F (`output`) and G
//! (`state_update`) are swapped for `kernels/gdn_chunkwise_wmma.hip`'s
//! matrix-core kernels. A separate file (not folded into `mod.rs`, which is
//! already near the 400-line cap) so this test-only pipeline variant
//! doesn't grow that file.
use std::ffi::c_void;

use rocml_hip::{kernel_params, DeviceBuffer, LaunchConfig, Module};

use super::{load, L2_EPS};

pub struct ChunkwiseWmmaKernels {
    _m1: Module,
    prep_point: rocml_hip::Function,
    _m1b: Module,
    prep_cumsum: rocml_hip::Function,
    _m2: Module,
    ut_build_wmma: rocml_hip::Function,
    _m3: Module,
    tinv: rocml_hip::Function,
    _m4: Module,
    uv_vnew: rocml_hip::Function,
    _m6: Module,
    output_wmma: rocml_hip::Function,
    _m7: Module,
    state_wmma: rocml_hip::Function,
}

impl ChunkwiseWmmaKernels {
    pub fn load() -> Self {
        let (_m1, prep_point) = load(
            rocml_kernels::GDN_CW_PREP_POINT_F32_HSACO,
            rocml_kernels::GDN_CW_PREP_POINT_F32_KERNEL,
        );
        let (_m1b, prep_cumsum) = load(
            rocml_kernels::GDN_CW_PREP_CUMSUM_F32_HSACO,
            rocml_kernels::GDN_CW_PREP_CUMSUM_F32_KERNEL,
        );
        let (_m2, ut_build_wmma) = load(
            rocml_kernels::GDN_CW_UT_BUILD_WMMA_F32_HSACO,
            rocml_kernels::GDN_CW_UT_BUILD_WMMA_F32_KERNEL,
        );
        let (_m3, tinv) = load(
            rocml_kernels::GDN_CW_TINV_F32_HSACO,
            rocml_kernels::GDN_CW_TINV_F32_KERNEL,
        );
        let (_m4, uv_vnew) = load(
            rocml_kernels::GDN_CW_UV_VNEW_F32_HSACO,
            rocml_kernels::GDN_CW_UV_VNEW_F32_KERNEL,
        );
        let (_m6, output_wmma) = load(
            rocml_kernels::GDN_CW_OUTPUT_WMMA_F32_HSACO,
            rocml_kernels::GDN_CW_OUTPUT_WMMA_F32_KERNEL,
        );
        let (_m7, state_wmma) = load(
            rocml_kernels::GDN_CW_STATE_WMMA_F32_HSACO,
            rocml_kernels::GDN_CW_STATE_WMMA_F32_KERNEL,
        );
        Self {
            _m1,
            prep_point,
            _m1b,
            prep_cumsum,
            _m2,
            ut_build_wmma,
            _m3,
            tinv,
            _m4,
            uv_vnew,
            _m6,
            output_wmma,
            _m7,
            state_wmma,
        }
    }
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[allow(clippy::too_many_arguments)]
pub fn run_chunkwise_gpu_wmma(
    k: &ChunkwiseWmmaKernels,
    num_heads: u32,
    num_k_heads: u32,
    head_k_dim: u32,
    head_v_dim: u32,
    chunk_len: u32,
    conv_out: &[f32],
    beta: &[f32],
    g: &[f32],
    init_state: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let (h, hk, sk, sv, t) = (num_heads, num_k_heads, head_k_dim, head_v_dim, chunk_len);
    let key_dim = hk * sk;
    let value_dim = h * sv;
    let conv_dim = 2 * key_dim + value_dim;

    let mut buf_conv_out = DeviceBuffer::<f32>::new(conv_out.len()).unwrap();
    buf_conv_out.copy_from_host(conv_out).unwrap();
    let mut buf_beta = DeviceBuffer::<f32>::new(beta.len()).unwrap();
    buf_beta.copy_from_host(beta).unwrap();
    let mut buf_g = DeviceBuffer::<f32>::new(g.len()).unwrap();
    buf_g.copy_from_host(g).unwrap();
    let mut buf_state = DeviceBuffer::<f32>::new(init_state.len()).unwrap();
    buf_state.copy_from_host(init_state).unwrap();

    let q_norm = DeviceBuffer::<f32>::new((t * sk * h) as usize).unwrap();
    let k_norm = DeviceBuffer::<f32>::new((t * sk * h) as usize).unwrap();
    let k_beta = DeviceBuffer::<f32>::new((t * sk * h) as usize).unwrap();
    let g_cum = DeviceBuffer::<f32>::new((t * h) as usize).unwrap();
    let cum_decay_exp = DeviceBuffer::<f32>::new((t * h) as usize).unwrap();
    let state_decay = DeviceBuffer::<f32>::new((t * h) as usize).unwrap();
    let kb = DeviceBuffer::<f32>::new((t * t * h) as usize).unwrap();
    let kq = DeviceBuffer::<f32>::new((t * t * h) as usize).unwrap();
    let v_new = DeviceBuffer::<f32>::new((t * sv * h) as usize).unwrap();
    let y = DeviceBuffer::<f32>::new((t * sv * h) as usize).unwrap();

    let conv_out_p: *mut c_void = buf_conv_out.device_ptr();
    let beta_p: *mut c_void = buf_beta.device_ptr();
    let g_p: *mut c_void = buf_g.device_ptr();
    let state_p: *mut c_void = buf_state.device_ptr();
    let q_norm_p: *mut c_void = q_norm.device_ptr();
    let k_norm_p: *mut c_void = k_norm.device_ptr();
    let k_beta_p: *mut c_void = k_beta.device_ptr();
    let g_cum_p: *mut c_void = g_cum.device_ptr();
    let cde_p: *mut c_void = cum_decay_exp.device_ptr();
    let sd_p: *mut c_void = state_decay.device_ptr();
    let kb_p: *mut c_void = kb.device_ptr();
    let kq_p: *mut c_void = kq.device_ptr();
    let vnew_p: *mut c_void = v_new.device_ptr();
    let y_p: *mut c_void = y.device_ptr();

    // A1/A2/B/C/D+E: identical to the scalar pipeline (`run_chunkwise_gpu`).
    {
        let cfg = LaunchConfig {
            grid: (h, t, 1),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let l2_eps = L2_EPS as f32;
        let mut params = kernel_params!(
            conv_out_p, beta_p, q_norm_p, k_norm_p, k_beta_p, h, hk, sk, conv_dim, key_dim, t,
            l2_eps
        );
        unsafe { k.prep_point.launch(&cfg, &mut params, None) }.expect("prep_point launch failed");
    }
    {
        let cfg = LaunchConfig {
            grid: (h, 1, 1),
            block: (t, 1, 1),
            shared_mem_bytes: t * 4,
        };
        let mut params = kernel_params!(g_p, g_cum_p, cde_p, sd_p, h, t);
        unsafe { k.prep_cumsum.launch(&cfg, &mut params, None) }
            .expect("prep_cumsum launch failed");
    }
    // B: ut_build, WMMA — grid (num_v_heads, ceil(tile_len/16), ceil(tile_len/16)).
    {
        let cfg = LaunchConfig {
            grid: (h, ceil_div(t, 16), ceil_div(t, 16)),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params =
            kernel_params!(q_norm_p, k_norm_p, k_beta_p, g_cum_p, kb_p, kq_p, h, sk, t);
        unsafe { k.ut_build_wmma.launch(&cfg, &mut params, None) }
            .expect("ut_build_wmma launch failed");
    }
    {
        let cfg = LaunchConfig {
            grid: (h, 1, 1),
            block: (t, 1, 1),
            shared_mem_bytes: t * t * 4,
        };
        let mut params = kernel_params!(kb_p, h, t);
        unsafe { k.tinv.launch(&cfg, &mut params, None) }.expect("tinv launch failed");
    }
    {
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
        unsafe { k.uv_vnew.launch(&cfg, &mut params, None) }.expect("uv_vnew launch failed");
    }
    // F: output, WMMA — grid (num_v_heads, ceil(tile_len/16), ceil(head_v_dim/16)).
    {
        let cfg = LaunchConfig {
            grid: (h, ceil_div(t, 16), ceil_div(sv, 16)),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(q_norm_p, cde_p, state_p, kq_p, vnew_p, y_p, h, sk, sv, t);
        unsafe { k.output_wmma.launch(&cfg, &mut params, None) }
            .expect("output_wmma launch failed");
    }
    // G: state update, WMMA — grid (num_v_heads, ceil(head_k_dim/16), ceil(head_v_dim/16)).
    {
        let cfg = LaunchConfig {
            grid: (h, ceil_div(sk, 16), ceil_div(sv, 16)),
            block: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(k_norm_p, cde_p, sd_p, vnew_p, state_p, h, sk, sv, t);
        unsafe { k.state_wmma.launch(&cfg, &mut params, None) }.expect("state_wmma launch failed");
    }

    let mut y_out = vec![0.0f32; (t * sv * h) as usize];
    y.copy_to_host(&mut y_out).unwrap();
    let mut state_out = vec![0.0f32; init_state.len()];
    buf_state.copy_to_host(&mut state_out).unwrap();
    (y_out, state_out)
}

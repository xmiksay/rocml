//! Shared support for the chunkwise (blocked delta-rule) Gated Delta Net
//! recurrence integration tests (`kernels/gdn_chunkwise.hip`): the f64 CPU
//! reference (a plain sequential recurrence — the chunkwise pipeline is an
//! *exact* algebraic reformulation of it, not an approximation) and the raw
//! seven-kernel GPU pipeline runner, launched in the same order
//! `gdn_chunkwise.rs::run_tile` (the `rocml` crate's host wrapper) uses.
#![allow(dead_code)] // each test binary only exercises a subset of this module

use std::ffi::c_void;

use rocml_hip::{kernel_params, DeviceBuffer, LaunchConfig, Module};

pub const TOL: f32 = 3e-3;
pub const L2_EPS: f64 = 1e-6;
/// Must match `kernels/gdn_chunkwise.hip`'s `#define UT_BUILD_J_PER_BLOCK`.
const UT_BUILD_J_PER_BLOCK: u32 = 8;

pub fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

pub fn assert_close(actual: &[f32], expected: &[f64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (got, want)) in actual.iter().zip(expected).enumerate() {
        let want = *want as f32;
        let diff = (got - want).abs();
        assert!(
            diff <= TOL * want.abs().max(1.0),
            "{label}[{i}]: got {got}, want {want} (diff {diff})"
        );
    }
}

// ── f64 CPU reference: plain sequential gated-delta-rule recurrence ────────

fn l2_norm_f64(raw: &[f64], extra_scale: f64) -> Vec<f64> {
    let sum_sq: f64 = raw.iter().map(|v| v * v).sum();
    let inv = (sum_sq + L2_EPS).sqrt().recip() * extra_scale;
    raw.iter().map(|v| v * inv).collect()
}

#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn reference_sequential(
    num_heads: usize,
    num_k_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    chunk_len: usize,
    conv_out: &[f64],
    beta: &[f64],
    g: &[f64],
    init_state: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let key_dim = num_k_heads * head_k_dim;
    let conv_dim = 2 * key_dim + num_heads * head_v_dim;
    let q_scale = 1.0 / (head_k_dim as f64).sqrt();
    let mut state = init_state.to_vec();
    let mut y = vec![0.0f64; chunk_len * num_heads * head_v_dim];

    for t in 0..chunk_len {
        let row = &conv_out[t * conv_dim..(t + 1) * conv_dim];
        for h in 0..num_heads {
            let kh = h % num_k_heads;
            let q_raw = &row[kh * head_k_dim..(kh + 1) * head_k_dim];
            let k_raw = &row[key_dim + kh * head_k_dim..key_dim + (kh + 1) * head_k_dim];
            let v_raw = &row[2 * key_dim + h * head_v_dim..2 * key_dim + (h + 1) * head_v_dim];
            let qn = l2_norm_f64(q_raw, q_scale);
            let kn = l2_norm_f64(k_raw, 1.0);
            let beta_h = beta[t * num_heads + h];
            let decay = g[t * num_heads + h].exp();

            let s = &mut state[h * head_k_dim * head_v_dim..(h + 1) * head_k_dim * head_v_dim];
            let mut kv_mem = vec![0.0f64; head_v_dim];
            for kk in 0..head_k_dim {
                for d in 0..head_v_dim {
                    let idx = kk * head_v_dim + d;
                    s[idx] *= decay;
                    kv_mem[d] += s[idx] * kn[kk];
                }
            }
            let delta: Vec<f64> = (0..head_v_dim)
                .map(|d| beta_h * (v_raw[d] - kv_mem[d]))
                .collect();
            let mut y_acc = vec![0.0f64; head_v_dim];
            for kk in 0..head_k_dim {
                for d in 0..head_v_dim {
                    let idx = kk * head_v_dim + d;
                    s[idx] += kn[kk] * delta[d];
                    y_acc[d] += s[idx] * qn[kk];
                }
            }
            let y_out =
                &mut y[(t * num_heads + h) * head_v_dim..(t * num_heads + h + 1) * head_v_dim];
            y_out.copy_from_slice(&y_acc);
        }
    }
    (y, state)
}

// ── GPU chunkwise pipeline runner ───────────────────────────────────────────

pub struct ChunkwiseKernels {
    _m1: Module,
    prep_point: rocml_hip::Function,
    _m1b: Module,
    prep_cumsum: rocml_hip::Function,
    _m2: Module,
    ut_build: rocml_hip::Function,
    _m3: Module,
    tinv: rocml_hip::Function,
    _m4: Module,
    uv_vnew: rocml_hip::Function,
    _m6: Module,
    output: rocml_hip::Function,
    _m7: Module,
    state_update: rocml_hip::Function,
}

impl ChunkwiseKernels {
    pub fn load() -> Self {
        let (_m1, prep_point) = load(
            rocml_kernels::GDN_CW_PREP_POINT_F32_HSACO,
            rocml_kernels::GDN_CW_PREP_POINT_F32_KERNEL,
        );
        let (_m1b, prep_cumsum) = load(
            rocml_kernels::GDN_CW_PREP_CUMSUM_F32_HSACO,
            rocml_kernels::GDN_CW_PREP_CUMSUM_F32_KERNEL,
        );
        let (_m2, ut_build) = load(
            rocml_kernels::GDN_CW_UT_BUILD_F32_HSACO,
            rocml_kernels::GDN_CW_UT_BUILD_F32_KERNEL,
        );
        let (_m3, tinv) = load(
            rocml_kernels::GDN_CW_TINV_F32_HSACO,
            rocml_kernels::GDN_CW_TINV_F32_KERNEL,
        );
        let (_m4, uv_vnew) = load(
            rocml_kernels::GDN_CW_UV_VNEW_F32_HSACO,
            rocml_kernels::GDN_CW_UV_VNEW_F32_KERNEL,
        );
        let (_m6, output) = load(
            rocml_kernels::GDN_CW_OUTPUT_F32_HSACO,
            rocml_kernels::GDN_CW_OUTPUT_F32_KERNEL,
        );
        let (_m7, state_update) = load(
            rocml_kernels::GDN_CW_STATE_F32_HSACO,
            rocml_kernels::GDN_CW_STATE_F32_KERNEL,
        );
        Self {
            _m1,
            prep_point,
            _m1b,
            prep_cumsum,
            _m2,
            ut_build,
            _m3,
            tinv,
            _m4,
            uv_vnew,
            _m6,
            output,
            _m7,
            state_update,
        }
    }

    pub fn prep_point_fn(&self) -> &rocml_hip::Function {
        &self.prep_point
    }
    pub fn prep_cumsum_fn(&self) -> &rocml_hip::Function {
        &self.prep_cumsum
    }
    pub fn ut_build_fn(&self) -> &rocml_hip::Function {
        &self.ut_build
    }
    pub fn tinv_fn(&self) -> &rocml_hip::Function {
        &self.tinv
    }
    pub fn uv_vnew_fn(&self) -> &rocml_hip::Function {
        &self.uv_vnew
    }
    pub fn output_fn(&self) -> &rocml_hip::Function {
        &self.output
    }
    pub fn state_update_fn(&self) -> &rocml_hip::Function {
        &self.state_update
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run_chunkwise_gpu(
    k: &ChunkwiseKernels,
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

    // A1: prep_point
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
    // A2: prep_cumsum
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
    // B: ut_build (block = (32, UT_BUILD_J_PER_BLOCK, 1) — must match the
    // kernel source's #define exactly)
    {
        let cfg = LaunchConfig {
            grid: (h, t, 1),
            block: (32, UT_BUILD_J_PER_BLOCK, 1),
            shared_mem_bytes: 0,
        };
        let mut params =
            kernel_params!(q_norm_p, k_norm_p, k_beta_p, g_cum_p, kb_p, kq_p, h, sk, t);
        unsafe { k.ut_build.launch(&cfg, &mut params, None) }.expect("ut_build launch failed");
    }
    // C: tinv
    {
        let cfg = LaunchConfig {
            grid: (h, 1, 1),
            block: (t, 1, 1),
            shared_mem_bytes: t * t * 4,
        };
        let mut params = kernel_params!(kb_p, h, t);
        unsafe { k.tinv.launch(&cfg, &mut params, None) }.expect("tinv launch failed");
    }
    // D+E (fused): uv_vnew
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
    // F: output
    {
        let cfg = LaunchConfig {
            grid: (h, t, 1),
            block: (sv, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(q_norm_p, cde_p, state_p, kq_p, vnew_p, y_p, h, sk, sv, t);
        unsafe { k.output.launch(&cfg, &mut params, None) }.expect("output launch failed");
    }
    // G: state update
    {
        let cfg = LaunchConfig {
            grid: (h, sk, 1),
            block: (sv, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut params = kernel_params!(k_norm_p, cde_p, sd_p, vnew_p, state_p, h, sk, sv, t);
        unsafe { k.state_update.launch(&cfg, &mut params, None) }
            .expect("state_update launch failed");
    }

    let mut y_out = vec![0.0f32; (t * sv * h) as usize];
    y.copy_to_host(&mut y_out).unwrap();
    let mut state_out = vec![0.0f32; init_state.len()];
    buf_state.copy_to_host(&mut state_out).unwrap();
    (y_out, state_out)
}

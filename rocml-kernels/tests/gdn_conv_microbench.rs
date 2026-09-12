//! Temporary standalone microbenchmark (not part of the normal suite --
//! `#[ignore]`d) directly timing `causal_conv1d_chunk_f32` +
//! `causal_conv1d_chunk_state_update_f32` at Ornith's real shape
//! (channels=8192, chunk_len=128, kernel_size=4) to isolate the conv
//! kernel's own wall time from the surrounding profiler bucket (which also
//! includes four GEMM projections).
use std::ffi::c_void;

use rocml_hip::{elapsed_ms, kernel_params, Device, DeviceBuffer, Event, LaunchConfig, Module};

fn load(hsaco: &[u8], name: &str) -> (Module, rocml_hip::Function) {
    let module = Module::load_from_bytes(hsaco).expect("module load failed");
    let function = module.get_function(name).expect("kernel lookup failed");
    (module, function)
}

#[test]
#[ignore]
fn conv_chunk_ornith_shape_timing() {
    let _device = Device::new(0).expect("failed to select device 0");
    let (_mc, chunk_fn) = load(
        rocml_kernels::GDN_CONV1D_CHUNK_F32_HSACO,
        rocml_kernels::GDN_CONV1D_CHUNK_F32_KERNEL,
    );
    let (_mcs, chunk_state_fn) = load(
        rocml_kernels::GDN_CONV1D_CHUNK_STATE_UPDATE_F32_HSACO,
        rocml_kernels::GDN_CONV1D_CHUNK_STATE_UPDATE_F32_KERNEL,
    );

    let channels: u32 = 8192;
    let kernel_size: u32 = 4;
    let chunk_len: u32 = 128;
    let hist_len = (kernel_size - 1) as usize;

    let x = vec![0.1f32; chunk_len as usize * channels as usize];
    let state = vec![0.05f32; channels as usize * hist_len];
    let weight = vec![0.2f32; channels as usize * kernel_size as usize];

    let mut buf_x = DeviceBuffer::<f32>::new(x.len()).unwrap();
    buf_x.copy_from_host(&x).unwrap();
    let mut buf_state = DeviceBuffer::<f32>::new(state.len()).unwrap();
    buf_state.copy_from_host(&state).unwrap();
    let mut buf_w = DeviceBuffer::<f32>::new(weight.len()).unwrap();
    buf_w.copy_from_host(&weight).unwrap();
    let buf_out = DeviceBuffer::<f32>::new(x.len()).unwrap();

    let x_ptr: *mut c_void = buf_x.device_ptr();
    let state_ptr: *mut c_void = buf_state.device_ptr();
    let w_ptr: *mut c_void = buf_w.device_ptr();
    let out_ptr: *mut c_void = buf_out.device_ptr();

    let block = 256u32;
    let main_cfg = LaunchConfig {
        grid: (channels.div_ceil(block), chunk_len, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let state_cfg = LaunchConfig {
        grid: (channels.div_ceil(block), 1, 1),
        block: (block, 1, 1),
        shared_mem_bytes: 0,
    };

    const ITERS: u32 = 100;
    let start = Event::new().unwrap();
    let stop = Event::new().unwrap();
    start.record(None).unwrap();
    for _ in 0..ITERS {
        let mut p1 = kernel_params!(
            x_ptr,
            state_ptr,
            w_ptr,
            out_ptr,
            channels,
            kernel_size,
            chunk_len
        );
        unsafe { chunk_fn.launch(&main_cfg, &mut p1, None) }.unwrap();
        let mut p2 = kernel_params!(x_ptr, state_ptr, channels, kernel_size, chunk_len);
        unsafe { chunk_state_fn.launch(&state_cfg, &mut p2, None) }.unwrap();
    }
    stop.record(None).unwrap();
    let ms = elapsed_ms(&start, &stop).unwrap();
    eprintln!(
        "NEW: {} iters, {:.3} ms total, {:.4} ms/iter (both kernels)",
        ITERS,
        ms,
        ms / ITERS as f64
    );
}

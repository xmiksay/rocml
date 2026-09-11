use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

// Build-script failures are reported by panicking with the full hipcc
// output — cargo surfaces the panic message directly as the build error,
// which is how a broken kernel gets a readable compiler error instead of a
// silent missing .hsaco discovered at include_bytes! time.
fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    let gfx_arch = env::var("ROCML_GFX_ARCH").unwrap_or_else(|_| "gfx1101".to_string());
    let out_dir = env::var("OUT_DIR").expect("cargo always sets OUT_DIR for build scripts");
    let hipcc = format!("{rocm_path}/bin/hipcc");
    let kernels_dir = Path::new("kernels");

    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
    println!("cargo:rerun-if-env-changed=ROCML_GFX_ARCH");

    let entries = fs::read_dir(kernels_dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", kernels_dir.display()));

    for entry in entries {
        let path = entry
            .unwrap_or_else(|e| panic!("failed to read a kernels/ directory entry: {e}"))
            .path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("hip") {
            continue;
        }
        compile_kernel(&hipcc, &gfx_arch, &out_dir, &path);
    }
}

fn compile_kernel(hipcc: &str, gfx_arch: &str, out_dir: &str, src: &Path) {
    let stem = src
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| panic!("non-UTF8 kernel source filename: {}", src.display()));
    let out_path = format!("{out_dir}/{stem}.hsaco");

    let output = Command::new(hipcc)
        .arg("--genco")
        .arg(format!("--offload-arch={gfx_arch}"))
        .arg("-O3")
        .arg("-o")
        .arg(&out_path)
        .arg(src)
        .output()
        .unwrap_or_else(|e| panic!("failed to spawn `{hipcc}` (is ROCM_PATH correct?): {e}"));

    if !output.status.success() {
        panic!(
            "hipcc failed compiling {}:\n--- stdout ---\n{}\n--- stderr ---\n{}",
            src.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
    }
}

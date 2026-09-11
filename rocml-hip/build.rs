use std::env;

// hipcc/rocm installs put libamdhip64.so under $ROCM_PATH/lib; there is no
// pkg-config coverage for HIP, so the search path has to be added by hand.
fn main() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
    println!("cargo:rustc-link-search=native={rocm_path}/lib");
    println!("cargo:rustc-link-lib=dylib=amdhip64");
    println!("cargo:rerun-if-env-changed=ROCM_PATH");
}

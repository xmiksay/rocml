// Shared device-side decode helper for the fused dequant-GEMV kernels.
// Kept in a header (not a .hip file — build.rs only compiles kernels/*.hip)
// so gemv_q4_k.hip and gemv_q5_k.hip don't duplicate the packed 6-bit
// scale/min unpack, which must match rocml-core's
// `quant::common::get_scale_min_k4` (ported verbatim from ggml's
// `get_scale_min_k4`) bit-for-bit.
#pragma once

// Unpacks the 6-bit (scale, min) pair for sub-block `j` from the 12-byte
// packed table shared by Q4_K/Q5_K blocks.
__device__ __forceinline__ void get_scale_min_k4(
    unsigned j, const unsigned char* q, unsigned char* sc, unsigned char* m) {
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

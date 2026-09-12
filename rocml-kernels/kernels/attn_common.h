// Shared device-side helper for the attention kernels (`attn_decode.hip`,
// `attn_prefill.hip`). Kept in a header (not a .hip file — build.rs only
// compiles kernels/*.hip) so the second kernel doesn't duplicate this.
#pragma once

// Butterfly (XOR) reduction across one 32-lane wavefront: after the full
// 5-step exchange every lane holds the total, so callers need no separate
// broadcast. gfx1101 is wave32, so 5 halving steps (16,8,4,2,1) cover the
// whole wavefront.
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, 32);
    }
    return v;
}

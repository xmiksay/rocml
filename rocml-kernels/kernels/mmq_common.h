// Shared tile geometry and fragment-load helper for the int8 MMQ-style
// batched prefill-path GEMM kernels (`gemm_xwt_quant_mmq_q{8_0,4_k,5_k,6_k}.hip`
// — issue #6's WMMA-pipeline round, lever 2, split per-family in the
// int8-MMQ-integration round to stay under the workspace's 400-line file
// cap once Q5_K/Q6_K were added). See `gemm_xwt_quant_mmq_q8_0.hip`'s module
// doc for the full design writeup (fragment layout, K-reduction structure,
// activation-quantizer contract) — this header only holds what every
// variant needs bit-for-bit identical: tile sizes, the int8 fragment
// loader, and the native Q8_0 block-byte constant every variant's staging
// loop clamps against.
//
// `#pragma once`, not compiled standalone — `rocml-kernels/build.rs` only
// compiles `kernels/*.hip`, so this header carries no ODR risk across the
// four `.hip` translation units that include it (each gets its own copy of
// every symbol below, exactly like the existing `quant_dequant_elem.h`/
// `quant_gemv_common.h` headers this crate already ships).
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstring>

#include "quant_gemv_common.h"

namespace {

typedef int int4x __attribute__((ext_vector_type(4)));
typedef int int8x __attribute__((ext_vector_type(8)));

constexpr unsigned TILE_ROWS = 128;
constexpr unsigned TILE_M = 64;
constexpr unsigned K_STAGE = 32; // == Q8_0's QK and the activation quantizer's block width
constexpr unsigned WARPS_M = TILE_ROWS / 16;
constexpr unsigned WARPS_N = 2;
constexpr unsigned SUBCOLS_PER_WARP = (TILE_M / 16) / WARPS_N;
constexpr unsigned WARPS_PER_BLOCK = WARPS_M * WARPS_N;
constexpr unsigned Q8_0_BLOCK_BYTES = 34; // 2-byte f16 scale + 32 int8 codes

// Reads 16 contiguous rows of a `[rows][K_STAGE]` row-major int8 LDS tile
// into a `v_wmma_i32_16x16x16_iu8` operand fragment: lane `L`'s fragment
// packs codes `[row_base + L%16][k_off, k_off+16)` 4-per-`int32` GPR
// (`GPR floor(k/4)`, byte `k%4`) — see `gemm_xwt_quant_mmq_q8_0.hip`'s
// module doc's fragment-layout paragraph.
__device__ __forceinline__ int4x load_row_frag_i8(
    const signed char* tile, unsigned row_base, unsigned k_off, unsigned lane) {
    unsigned row = row_base + lane % 16;
    const signed char* p = tile + (size_t)row * K_STAGE + k_off;
    int4x frag;
#pragma unroll
    for (unsigned w = 0; w < 4; ++w) {
        int packed;
        std::memcpy(&packed, p + w * 4, sizeof(int));
        frag[w] = packed;
    }
    return frag;
}

} // namespace

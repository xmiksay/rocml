// Gated Delta Net per-head gate scalars, one decode step: write strength
// beta = sigmoid(b), decay g = -exp(a_log) * softplus(a + dt_bias). Mirrors
// Crane's `compute_beta_g`/`softplus` bit-for-bit (naive `log(1+exp(x))`,
// no overflow guard) since this feeds a greedy-decode parity target where
// matching the reference's exact numerics matters more than robustness to
// extreme inputs. No block-size constraint (tiny, one thread per head).
#include <hip/hip_runtime.h>

extern "C" __global__ void gdn_gate_f32(
    const float* a_raw, const float* b_raw, const float* a_log, const float* dt_bias,
    float* beta_out, float* g_out, unsigned n) {
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    beta_out[i] = 1.0f / (1.0f + expf(-b_raw[i]));
    float softplus = logf(1.0f + expf(a_raw[i] + dt_bias[i]));
    g_out[i] = -expf(a_log[i]) * softplus;
}

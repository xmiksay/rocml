// Gated Delta Net per-head gate scalars, one decode step: write strength
// beta = sigmoid(b), decay g = a_neg_exp * softplus(a + dt_bias), where
// `a_neg_exp` (the GGUF `ssm_a` tensor) ALREADY stores `-exp(A_log)`:
// llama.cpp's converter pre-bakes the negation+exponential at conversion
// time (`data_torch = -torch.exp(data_torch)` for `.A_log`), matching
// llama.cpp's own runtime `gate = softplus * ssm_a`. Applying `-expf()`
// here a second time (the pre-fix bug, inherited from mirroring Crane's
// safetensors-path formula) collapses every head's decay multiplier to
// ~-1 instead of the true small magnitudes, i.e. `exp(g)` decays ~13-100x
// too fast and the recurrent state is wiped almost every step.
// Softplus stays the naive `log(1+exp(x))` (no overflow guard) to keep
// bit-parity with the CPU reference.
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
    g_out[i] = a_log[i] * softplus;
}

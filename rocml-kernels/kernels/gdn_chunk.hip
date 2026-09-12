// Gated Delta Net prefill-chunk causal-conv1d kernel: batched depthwise
// conv1d + SiLU over a whole T-token chunk in one launch instead of T serial
// decode-style launches, reproducing `causal_conv1d_decode_f32`'s per-step
// math exactly. The recurrence's chunk kernel used to live here too
// (`gdn_recurrence_chunk_f32`, a v1 pragmatic token-serial-inside-chunk
// loop); it has been replaced by the true chunkwise (blocked delta-rule)
// formulation, `kernels/gdn_chunkwise.hip` (issue #6's chunkwise rewrite) —
// see that file for the recurrence.
//
// **Not a recurrence** (issue #6 prefill-cleanup round): despite being
// implemented for a long time as a per-thread shift-register loop over
// `chunk_len` (mirroring `causal_conv1d_decode_f32`'s single-step update),
// this op has no actual sequential dependency between output tokens — a
// depthwise causal conv with `kernel_size`=4 only ever reads a 4-wide
// sliding window of *inputs*, never a previous *output*. The old kernel's
// grid was `channels.div_ceil(256)` blocks (32 for Ornith's 8192-channel
// `conv_dim`) regardless of `chunk_len`, i.e. at most 32 CUs of gfx1101's 60
// ever had a block scheduled — the other ~28 sat idle for the whole
// launch — while the 256-thread block that did run spent it walking a
// 128-iteration serial dependency chain that couldn't be scheduled around
// (each `t` needs the previous `t`'s shifted-in `hist` value in registers).
// Measured effect: ~30% FLOP roofline, 19-20% of prefill wall time on
// ornith-9b @ depth 2048 despite being nearly the smallest FLOP count in the
// profile.
//
// This kernel now parallelizes over `(channel, token)` directly instead of
// looping `t` per-thread: output token `t`'s window is `virt[t-hist_len ..
// t]`, where `virt` is `conv_state` (oldest-first, covering positions
// `-hist_len..-1`) concatenated with this chunk's own `x` (positions
// `0..chunk_len-1`) — exactly the same virtual sequence the old shift
// register walked, just read as a window instead of accumulated
// sequentially. `conv_state` is read-only here; grid is `[channels,
// chunk_len]` blocks (up to 4096 for Ornith's 128-token chunks vs. the old
// 32), restoring full-GPU occupancy and turning the serial chain into
// independent per-output-token work the scheduler can freely interleave to
// hide memory latency.
//
// State-carry is split into a second, separate kernel
// (`causal_conv1d_chunk_state_update_f32`) rather than folded back into this
// one: multiple `t`-blocks would otherwise race to read-then-overwrite the
// same `conv_state` row (every token in `[chunk_len-hist_len, chunk_len)`
// needs to read history that other blocks in the *same* launch may already
// have clobbered, with no ordering guarantee across blocks). Two
// stream-ordered launches side-step the race for free — HIP guarantees the
// first fully completes before the second's blocks start — at the cost of
// one extra launch per chunk per GDN layer; the second kernel's own grid is
// still just `channels.div_ceil(256)` (state is O(1) in `chunk_len`, so
// there's nothing to gain from more parallelism there), but it does O(1)
// work per channel instead of O(chunk_len), so its underutilization is
// immaterial in wall-clock terms.
#include <hip/hip_runtime.h>

// Bounds the local sliding-window register array below; GDN's conv kernel
// size is 4 in every model this codebase loads (window 4, per issue #6), so
// 8 leaves headroom without costing registers anywhere that matters.
#define MAX_CONV_HIST 8

// Batched causal depthwise conv1d + SiLU over a `chunk_len`-token chunk,
// parallel over both channel and token (see the module doc above).
// `x` is `[chunk_len, channels]`; `conv_state` (`[channels, kernel_size-1]`,
// oldest first) is read-only — `causal_conv1d_chunk_state_update_f32` below
// updates it in a follow-up launch. `out` is `[chunk_len, channels]`.
//
// Grid: `(channels.div_ceil(blockDim.x), chunk_len, 1)` — one block per
// (channel tile, output token). Block: `(blockDim.x, 1, 1)`, one thread per
// channel within the tile.
extern "C" __global__ void causal_conv1d_chunk_f32(
    const float* x, const float* conv_state, const float* weight, float* out, unsigned channels,
    unsigned kernel_size, unsigned chunk_len) {
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned t = blockIdx.y;
    if (c >= channels || t >= chunk_len) {
        return;
    }

    unsigned hist_len = kernel_size - 1;
    const float* w = weight + (size_t)c * kernel_size;
    const float* state = conv_state + (size_t)c * hist_len;

    float acc = 0.0f;
    for (unsigned k = 0; k < kernel_size; ++k) {
        // Tap `k` reads virtual position `t - hist_len + k`: negative
        // positions come from the carried-in history (oldest first, so
        // position `-hist_len+j` is `state[j]`), non-negative ones from this
        // chunk's own `x`.
        long pos = (long)t - (long)hist_len + (long)k;
        float v = (pos < 0) ? state[pos + (long)hist_len] : x[(size_t)pos * channels + c];
        acc += v * w[k];
    }

    float sig = 1.0f / (1.0f + expf(-acc));
    out[(size_t)t * channels + c] = acc * sig;
}

// Follow-up launch: writes the post-chunk `conv_state` (the last
// `kernel_size-1` positions of the same virtual sequence
// `causal_conv1d_chunk_f32` read from) — see the module doc above for why
// this must be a separate launch rather than folded into the main kernel.
// Grid: `(channels.div_ceil(blockDim.x), 1, 1)`. Block: `(blockDim.x, 1,
// 1)`.
extern "C" __global__ void causal_conv1d_chunk_state_update_f32(
    const float* x, float* conv_state, unsigned channels, unsigned kernel_size,
    unsigned chunk_len) {
    unsigned c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }

    unsigned hist_len = kernel_size - 1;
    float* state = conv_state + (size_t)c * hist_len;

    // Read every new state slot from the pre-update buffer/x before writing
    // any of them back — `state` and the read side alias the same memory
    // for the `pos < 0` case, so this ordering (read-all-then-write-all,
    // same trick the old single-launch kernel used for its shift register)
    // is what keeps it race-free within one thread.
    float new_state[MAX_CONV_HIST];
    for (unsigned j = 0; j < hist_len; ++j) {
        long pos = (long)chunk_len - (long)hist_len + (long)j;
        new_state[j] = (pos < 0) ? state[pos + (long)hist_len] : x[(size_t)pos * channels + c];
    }
    for (unsigned j = 0; j < hist_len; ++j) {
        state[j] = new_state[j];
    }
}

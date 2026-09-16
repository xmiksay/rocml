//! qwen35moe M4 lever 2: overlaps a decode-step cache miss's H2D copy with
//! the *previous* selected expert's `gemv_quant` compute, instead of paying
//! the copy inline before that expert's own reads can even be issued.
//! Off by default (`LoadOptions::moe_decode_overlap`/
//! `--moe-decode-overlap`) — see `.claude/CLAUDE.md` for the measured
//! status and `moe::moe_ffn_step`'s doc comment for where this plugs in
//! (only decode's real per-token call, gated on `row_capture.is_none() &&
//! expert_cache.overlap_enabled()` — never the `LayerCapture` diagnostic
//! per-row prefill path, which stays on `moe::moe_ffn_step`'s original
//! inline loop regardless of this flag).
//!
//! ## Ordering argument
//!
//! This codebase launches every kernel on HIP's legacy default (null)
//! stream, which implicitly synchronizes with an *ordinary* stream but not
//! with a `hipStreamNonBlocking` one (`rocml_hip::Stream::new_non_blocking`)
//! — the whole point of using one here, so an in-flight copy doesn't force
//! the default stream to stall waiting for it. That means every ordering
//! constraint between the two streams must be established explicitly via
//! events, never assumed:
//!
//! 1. Before starting expert `k+1`'s copy into a slot the LRU may be
//!    reusing, the copy stream waits (`hipStreamWaitEvent`, not a host
//!    block) on `ExpertCache`'s rolling `last_compute_done` event — recorded
//!    on the default stream right after the *previous* iteration's own
//!    reads were issued. Since the LRU always marks a key most-recently-used
//!    at the point it's reserved, and every slot the eviction here could
//!    possibly recycle was last touched at or before that point, waiting on
//!    the latest recorded `last_compute_done` is always a sufficient (if
//!    occasionally more conservative than strictly required) barrier
//!    against racing that slot's still-in-flight previous reader.
//! 2. Before reading a slot the copy stream just finished writing, the
//!    default stream waits (again via `hipStreamWaitEvent`, via
//!    `rocml_hip::wait_on_default_stream`) on `ExpertCache`'s `copy_done`
//!    event, recorded on the copy stream right after that slot's three H2D
//!    copies were enqueued.
//!
//! Both waits are GPU-side only — neither blocks the host thread, which is
//! the whole point of this lever (hiding the copy's own latency behind
//! real compute instead of a host-observed stall). `ExpertCache::new`
//! records `last_compute_done` once at construction time so the very first
//! `start_copy_async` call (before any real compute has run) has a valid,
//! already-fired event to wait on rather than an unrecorded one.
//!
//! ## What this does *not* pipeline
//!
//! Expert 0 of a token's top-k is always loaded via the existing,
//! synchronous `ExpertCache::ensure_loaded` — there is nothing before it in
//! this layer's own FFN step to overlap its copy with (the router's top-k
//! indices are only known on the host after a D2H read that itself
//! synchronizes with every prior default-stream kernel, including the
//! shared expert's own FFN, so that work has already completed by the time
//! any routed expert's identity is known). Only one expert is ever
//! prefetched ahead (`k+1` while `k` computes) — this design was not
//! extended to a deeper pipeline, since qwen35moe's `top_k`=8 already gives
//! 7 real overlap opportunities per layer and a deeper window would need
//! per-slot (not one rolling) events to stay correct once more than one
//! copy can be outstanding at a time.

use rocml_core::gguf::GgufFile;

use super::kernels_moe::MoeKernels;
use super::moe_cache::ExpertCache;
use super::moe_scratch::MoeScratch;
use crate::error::RocmlError;
use crate::forward::kernels::{offset, Kernels};
use crate::qwen35::config::MoeConfig;
use crate::qwen35::weights::MoeFfnWeights;

/// Resolves and range-checks every `idx_host` entry up front (identical
/// validation to `moe::moe_ffn_step`'s original inline loop) — the
/// pipelined loop below needs all `top_k` expert ids known before it starts
/// prefetching ahead, whereas the original loop resolved one at a time.
fn resolve_experts(idx_host: &[i32], expert_count: u32) -> Result<Vec<u32>, RocmlError> {
    idx_host
        .iter()
        .map(|&raw_idx| {
            let expert = u32::try_from(raw_idx).map_err(|_| {
                RocmlError::Config(format!(
                    "moe router produced an invalid expert index {raw_idx} (must be in \
                     [0, {expert_count}))"
                ))
            })?;
            if expert >= expert_count {
                return Err(RocmlError::Config(format!(
                    "moe router selected expert {expert}, out of range for expert_count \
                     {expert_count}"
                )));
            }
            Ok(expert)
        })
        .collect()
}

/// The pipelined counterpart of `moe::moe_ffn_step`'s routed-expert loop —
/// same math, same kernel calls, same accumulation into `scratch.accum`
/// (`weighted_accum`, never a plain store — top-k means multiple experts
/// contribute to the same output). Never called with a `MoeRowCapture`:
/// this function has no capture parameter at all, since `moe_ffn_step`
/// only reaches here when `row_capture.is_none()` (see this module's own
/// doc comment).
#[allow(clippy::too_many_arguments)]
pub(crate) fn routed_experts_overlap(
    kernels: &Kernels,
    moe_kernels: &MoeKernels,
    gguf: &GgufFile,
    moe: &MoeFfnWeights,
    moe_cfg: &MoeConfig,
    hidden: u32,
    scratch: &mut MoeScratch,
    layer_idx: u32,
    cache: &mut ExpertCache,
    idx_host: &[i32],
) -> Result<(), RocmlError> {
    let experts = resolve_experts(idx_host, moe_cfg.expert_count)?;
    let top_k = experts.len();

    // Expert 0 has nothing to overlap its (possible) copy with — see the
    // module doc's "What this does not pipeline" section.
    let mut current =
        cache.ensure_loaded(gguf, (layer_idx, experts[0]), &moe.gate, &moe.up, &moe.down)?;

    for k in 0..top_k {
        // Kick off expert k+1's copy (if it's a miss) now, so it runs on
        // the copy stream concurrently with expert k's compute below.
        let lookahead = if k + 1 < top_k {
            let key_next = (layer_idx, experts[k + 1]);
            let (slot, hit) = cache.reserve(key_next);
            if !hit {
                cache.start_copy_async(gguf, key_next, slot, &moe.gate, &moe.up, &moe.down)?;
            }
            Some((cache.slot_ptrs(slot), hit))
        } else {
            None
        };

        let (gate_ptr, up_ptr, down_ptr) = current;
        kernels.gemv_quant(
            moe.gate.dtype,
            gate_ptr,
            offset(&scratch.xn, 0),
            offset(&scratch.ffn_a, 0),
            moe_cfg.expert_ff_len,
            hidden,
        )?;
        kernels.gemv_quant(
            moe.up.dtype,
            up_ptr,
            offset(&scratch.xn, 0),
            offset(&scratch.ffn_b, 0),
            moe_cfg.expert_ff_len,
            hidden,
        )?;
        kernels.silu_mul(
            offset(&scratch.ffn_a, 0),
            offset(&scratch.ffn_b, 0),
            offset(&scratch.ffn_a, 0),
            moe_cfg.expert_ff_len,
        )?;
        kernels.gemv_quant(
            moe.down.dtype,
            down_ptr,
            offset(&scratch.ffn_a, 0),
            offset(&scratch.expert_out, 0),
            hidden,
            moe_cfg.expert_ff_len,
        )?;
        moe_kernels.weighted_accum(
            offset(&scratch.expert_out, 0),
            offset(&scratch.topk_weight, k),
            offset(&scratch.accum, 0),
            hidden,
        )?;
        // Marks "every default-stream kernel up to here has been issued" —
        // the barrier the *next* iteration's lookahead copy (if it evicts
        // this iteration's slot) waits on before overwriting it.
        cache.mark_compute_done()?;

        if let Some((ptrs, hit)) = lookahead {
            if !hit {
                let copy_done = cache.copy_done_event().ok_or_else(|| {
                    RocmlError::Config(
                        "moe_decode_overlap: copy_done_event missing on an overlap-enabled \
                         cache (internal bug)"
                            .into(),
                    )
                })?;
                rocml_hip::wait_on_default_stream(copy_done)?;
            }
            current = ptrs;
        }
    }
    Ok(())
}

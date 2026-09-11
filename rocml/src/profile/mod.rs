//! Per-op performance instrumentation: bytes moved, FLOPs, and wall time for
//! every kernel-launch group in the forward pass, tagged by layer/op-kind/
//! phase (prefill vs decode) and aggregated into a roofline-relative
//! [`report::Report`].
//!
//! **Threading model**: every forward-pass call site takes `Option<&Profiler>`
//! and routes through [`Profiler::scope`], which is a no-op when `None` (the
//! default — bench/generate/chat/serve never pass a profiler unless `--profile`
//! is given) and costs exactly one HIP event pair when `Some`. `Profiler` uses
//! `RefCell`/`Cell` internally so `Option<&Profiler>` (a `Copy` type) can be
//! passed by value into every nested call without the `&mut` reborrow
//! plumbing an `Option<&mut Profiler>` would need through ~10 call sites per
//! layer across two architectures — this is the "least invasive" choice the
//! issue asks for. HIP state is single-threaded in this codebase already
//! (see `rocml-hip`'s own doc comments), so the interior mutability is sound.
//!
//! **Overhead discipline**: `Event::record` is an async stream enqueue, not a
//! sync — recording start/stop pairs through an entire decode loop costs
//! nothing until [`Profiler::finish`] reads them back, and that happens once,
//! after the run, when the stream is already idle (so `hipEventSynchronize`
//! there is instant, not a stall). The one deliberate exception is a
//! **prefill** layer: per the issue's own granularity note, a token-serial
//! prefill loop times each layer as a single span (tag [`OpKind::Layer`])
//! instead of per-op, to keep event count bounded at long `--depth` (a naive
//! per-op prefill would create `tokens * layers * ~6` events — for
//! `--depth 2048` on a 48-layer model that's ~590k events). Decode, whose
//! token count is bounded by `--decode-tokens`, keeps full per-op granularity.
//! A `MAX_TIMED_EVENTS` cap is a second, independent safety net against any
//! run creating unbounded events regardless of phase.

mod cost;
mod report;

pub use cost::*;
pub use report::Report;

use std::cell::{Cell, RefCell};

use rocml_hip::{elapsed_ms, Event};

use crate::error::RocmlError;

/// Caps total timed event pairs per `Profiler` instance. Past this, `time`
/// still records bytes/flops (cheap, just a `Vec` push) but stops creating
/// HIP events, so a pathological run degrades to "no timing beyond this
/// point" rather than growing without bound.
const MAX_TIMED_EVENTS: usize = 100_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Phase {
    Prefill,
    Decode,
}

impl Phase {
    pub fn label(self) -> &'static str {
        match self {
            Phase::Prefill => "prefill",
            Phase::Decode => "decode",
        }
    }
}

/// What kind of work a timed span did. `Layer` is the prefill-only coarse
/// bucket (see module doc); every other variant is decode-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OpKind {
    Embed,
    Norm,
    Qkv,
    /// The fused `attn_decode` kernel (KV-cache append + flash-decoding
    /// online-softmax attention) — replaces the old separate `AttnScore`
    /// pass now that scores are never materialized. See
    /// `crate::forward::kernels::Kernels::attn_decode`.
    AttnDecode,
    /// Attention output projection (+ residual add) only — the weighted-V
    /// pass this used to include moved into `AttnDecode`.
    AttnOut,
    GdnConv,
    GdnRecur,
    GdnOut,
    FfnGateUp,
    FfnDown,
    LmHead,
    /// Whole-layer span (attention/GDN + FFN together) used for prefill's
    /// coarser granularity instead of the per-op breakdown decode gets.
    Layer,
}

impl OpKind {
    pub fn label(self) -> &'static str {
        match self {
            OpKind::Embed => "embed",
            OpKind::Norm => "norm",
            OpKind::Qkv => "qkv",
            OpKind::AttnDecode => "attn-decode",
            OpKind::AttnOut => "attn-out",
            OpKind::GdnConv => "gdn-conv",
            OpKind::GdnRecur => "gdn-recur",
            OpKind::GdnOut => "gdn-out",
            OpKind::FfnGateUp => "ffn-gate-up",
            OpKind::FfnDown => "ffn-down",
            OpKind::LmHead => "lm-head",
            OpKind::Layer => "layer",
        }
    }
}

/// One recorded span: analytically-computed bytes/flops (always present) and
/// an optional HIP event pair (absent once `MAX_TIMED_EVENTS` is hit).
struct Pending {
    layer: Option<u32>,
    op: OpKind,
    phase: Phase,
    bytes: u64,
    flops: u64,
    timing: Option<(Event, Event)>,
}

/// A run-scoped profiling session. Cheap to construct; `Option<&Profiler>`
/// is threaded through the forward pass (see module doc) and every call
/// site is a no-op unless a profiler is actually present.
pub struct Profiler {
    phase: Cell<Phase>,
    exhausted: Cell<bool>,
    pending: RefCell<Vec<Pending>>,
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            phase: Cell::new(Phase::Decode),
            exhausted: Cell::new(false),
            pending: RefCell::new(Vec::new()),
        }
    }

    /// Sets the phase every subsequent `time`/`scope` call is tagged with,
    /// until the next call. Callers switch this once per phase transition
    /// (prompt loop -> decode loop), not per token.
    pub fn set_phase(&self, phase: Phase) {
        self.phase.set(phase);
    }

    pub fn phase(&self) -> Phase {
        self.phase.get()
    }

    /// Runs `f`, recording its bytes/flops (and, unless the event cap has
    /// been hit, its wall time) as one span tagged `layer`/`op`/current
    /// phase. `bytes`/`flops` are the caller's analytical estimate for this
    /// span (see `cost` submodule) — never measured, so this adds no extra
    /// device work of its own beyond the two event records.
    pub fn time<F>(
        &self,
        layer: Option<u32>,
        op: OpKind,
        bytes: u64,
        flops: u64,
        f: F,
    ) -> Result<(), RocmlError>
    where
        F: FnOnce() -> Result<(), RocmlError>,
    {
        if self.exhausted.get() {
            f()?;
            self.pending.borrow_mut().push(Pending {
                layer,
                op,
                phase: self.phase(),
                bytes,
                flops,
                timing: None,
            });
            return Ok(());
        }

        let start = Event::new()?;
        start.record(None)?;
        f()?;
        let stop = Event::new()?;
        stop.record(None)?;

        let mut pending = self.pending.borrow_mut();
        pending.push(Pending {
            layer,
            op,
            phase: self.phase(),
            bytes,
            flops,
            timing: Some((start, stop)),
        });
        if pending.len() >= MAX_TIMED_EVENTS {
            self.exhausted.set(true);
        }
        Ok(())
    }

    /// Convenience for call sites: times `f` through `prof` if present,
    /// otherwise just runs it — the single branch every instrumented call
    /// site needs, keeping the "off" path a plain function call with no
    /// `Profiler` machinery touched at all.
    pub fn scope<F>(
        prof: Option<&Profiler>,
        layer: Option<u32>,
        op: OpKind,
        bytes: u64,
        flops: u64,
        f: F,
    ) -> Result<(), RocmlError>
    where
        F: FnOnce() -> Result<(), RocmlError>,
    {
        match prof {
            Some(p) => p.time(layer, op, bytes, flops, f),
            None => f(),
        }
    }

    /// Drains every recorded span into a [`Report`]. Each timed span's
    /// `hipEventSynchronize` runs here, not in the hot loop — by the time
    /// `finish` is called the stream is long idle (the run is over), so
    /// these syncs return immediately; see the module doc for why this
    /// ordering is what keeps profiling from perturbing what it measures.
    pub fn finish(&self) -> Result<Report, RocmlError> {
        let pending = self.pending.borrow_mut().drain(..).collect::<Vec<_>>();
        let mut records = Vec::with_capacity(pending.len());
        for p in pending {
            let elapsed_ms = match &p.timing {
                Some((start, stop)) => elapsed_ms(start, stop)?,
                None => 0.0,
            };
            records.push(report::OpRecord {
                layer: p.layer,
                op: p.op,
                phase: p.phase,
                bytes: p.bytes,
                flops: p.flops,
                elapsed_ms,
            });
        }
        Ok(Report::build(records))
    }
}

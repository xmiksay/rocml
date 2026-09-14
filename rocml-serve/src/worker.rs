//! The dedicated model-owning thread: `Model` holds live HIP state (device
//! context, streams, resident GPU buffers) that this codebase treats as not
//! Send-safe, so exactly one plain OS thread loads it once at startup and
//! keeps it for the server's whole lifetime. Every request becomes a `Job`
//! sent over a `std::sync::mpsc` channel; jobs are drained strictly
//! sequentially (the documented v1 concurrency model — no batching, no
//! interleaving). Each job streams its output back over its own
//! `tokio::sync::mpsc` channel, which is fine to construct and send from
//! this plain thread: tokio's channel senders don't require a runtime to
//! send on, only to receive-`.await` on the other end.
//!
//! The conversation-state snapshot store (issue #1, `rocml::snapshot`) lives
//! here too, alongside `Model` — restoring/capturing needs `&mut Model`, so
//! it can only ever happen on this same thread, between jobs.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use rocml::snapshot::turn::run_turn;
use rocml::snapshot::{KvConfigStamp, ModelStamp, SnapshotStore};
use rocml::{GenerateStats, LoadOptions, Model, SamplingParams};
use rocml_core::tokenizer::BpeTokenizer;
use tokio::sync::mpsc::UnboundedSender;

pub struct Job {
    pub prompt_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampling: SamplingParams,
    pub stop_strings: Vec<String>,
    pub respond_to: UnboundedSender<WorkerEvent>,
}

pub enum WorkerEvent {
    Chunk(String),
    Done(TurnStats),
    Error(String),
}

/// A completed turn's stats plus the one extra number the OpenAI usage
/// surface needs beyond what `GenerateStats` already tracks: how many
/// prompt tokens this request restored from a snapshot instead of
/// re-prefilling (`rocml::snapshot::turn::TurnOutcome::reused_prefix`,
/// `0` on a miss or when snapshots don't apply).
pub struct TurnStats {
    pub stats: GenerateStats,
    pub cached_tokens: usize,
}

/// `--snapshot-ram-mb`/`--snapshot-dir`/`--snapshot-disk-mb` — see
/// `rocml_serve::ServerConfig`'s doc comment.
pub struct SnapshotConfig {
    pub ram_mb: usize,
    pub dir: Option<PathBuf>,
    pub disk_mb: u64,
}

/// Spawns the worker thread and blocks (briefly) until the model has either
/// finished loading or failed to, so startup failures surface synchronously
/// from `main` instead of only showing up on the first request.
///
/// Returns the `JoinHandle` alongside the job sender: the production binary
/// never needs it (the process just runs until killed), but a caller that
/// *does* shut down cleanly (the test suite) must drop every clone of the
/// sender and then join this handle before letting the process exit —
/// otherwise this thread can still be mid-teardown of `Model`'s HIP state
/// (streams, device buffers) when the process's own exit path starts
/// tearing down the HIP driver's global state, which reliably aborts with
/// "pure virtual method called" rather than exiting cleanly.
pub fn spawn(
    model_path: PathBuf,
    load_opts: LoadOptions,
    tokenizer: Arc<BpeTokenizer>,
    snapshot_config: SnapshotConfig,
) -> Result<(Sender<Job>, std::thread::JoinHandle<()>), String> {
    let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    let handle = std::thread::Builder::new()
        .name("rocml-worker".to_string())
        .spawn(move || {
            worker_loop(
                model_path,
                load_opts,
                tokenizer,
                snapshot_config,
                job_rx,
                ready_tx,
            )
        })
        .map_err(|e| format!("failed to spawn worker thread: {e}"))?;

    ready_rx
        .recv()
        .map_err(|_| "worker thread exited during startup".to_string())??;
    Ok((job_tx, handle))
}

fn worker_loop(
    model_path: PathBuf,
    load_opts: LoadOptions,
    tokenizer: Arc<BpeTokenizer>,
    snapshot_config: SnapshotConfig,
    job_rx: Receiver<Job>,
    ready_tx: Sender<Result<(), String>>,
) {
    let mut model = match Model::load(&model_path, load_opts) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };
    let model_stamp = match ModelStamp::from_path(&model_path) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready_tx.send(Err(format!(
                "failed to stamp {}: {e}",
                model_path.display()
            )));
            return;
        }
    };
    let kv_config = KvConfigStamp {
        mode: load_opts.kv_cache,
        ctx: load_opts.ctx,
    };
    // `None` when both tiers are off: `run_turn` still pays the D2H capture
    // cost for a `Some` store even if nothing ends up stored, so a fully
    // disabled snapshot layer skips capture entirely rather than just
    // skipping the store (see `rocml_cli::common::SnapshotArgs::build_store`'s
    // doc comment, which this mirrors).
    let mut snapshot_store = if snapshot_config.ram_mb == 0 && snapshot_config.dir.is_none() {
        None
    } else {
        match SnapshotStore::new(
            snapshot_config.ram_mb * 1024 * 1024,
            snapshot_config.dir,
            snapshot_config.disk_mb * 1024 * 1024,
        ) {
            Ok(s) => Some(s),
            Err(e) => {
                let _ = ready_tx.send(Err(e.to_string()));
                return;
            }
        }
    };
    let _ = ready_tx.send(Ok(()));

    for job in job_rx.iter() {
        // `&mut Model` isn't `UnwindSafe` — deliberately: a panic mid-token
        // could leave the KV cache half-updated, which is exactly why the
        // panic arm below resets it before the next job runs rather than
        // trusting whatever state was left behind.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job(
                &mut model,
                &tokenizer,
                snapshot_store.as_mut(),
                &model_stamp,
                &kv_config,
                &job,
            )
        }));
        let event = match outcome {
            Ok(Ok(turn_stats)) => WorkerEvent::Done(turn_stats),
            Ok(Err(e)) => WorkerEvent::Error(e.to_string()),
            Err(_) => {
                let _ = model.reset();
                WorkerEvent::Error("internal error during generation".to_string())
            }
        };
        let _ = job.respond_to.send(event);
    }
}

/// Runs one job to completion via `rocml::snapshot::turn::run_turn`: a
/// snapshot restore on a prefix hit, resumed prefill of the remaining
/// suffix, sampled decode until `max_new_tokens`, eos, or an OpenAI `stop`
/// string, then an end-of-turn capture. Logs the hit/miss outcome so
/// snapshot effectiveness is visible in the server's normal logs.
fn run_job(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    snapshot_store: Option<&mut SnapshotStore>,
    model_stamp: &ModelStamp,
    kv_config: &KvConfigStamp,
    job: &Job,
) -> Result<TurnStats, rocml::RocmlError> {
    let outcome = run_turn(
        model,
        tokenizer,
        snapshot_store,
        model_stamp,
        kv_config,
        &job.prompt_ids,
        job.max_new_tokens,
        true,
        &job.sampling,
        &job.stop_strings,
        |chunk| {
            let _ = job.respond_to.send(WorkerEvent::Chunk(chunk.to_string()));
        },
    )?;
    if outcome.reused_prefix > 0 {
        tracing::info!(
            reused_tokens = outcome.reused_prefix,
            prompt_tokens = job.prompt_ids.len(),
            restore_ms = outcome.restore_seconds * 1000.0,
            capture_ms = outcome.capture_seconds * 1000.0,
            "snapshot hit"
        );
    } else {
        tracing::info!(
            prompt_tokens = job.prompt_ids.len(),
            capture_ms = outcome.capture_seconds * 1000.0,
            "snapshot miss"
        );
    }
    Ok(TurnStats {
        stats: outcome.stats,
        cached_tokens: outcome.reused_prefix as usize,
    })
}

#[cfg(test)]
mod tests {
    // `run_turn`'s stop-string matching (empty-string-never-matches,
    // substring-anywhere, first-of-several) is covered by
    // `rocml::snapshot::turn`'s own tests now that the loop lives there —
    // this module has no more logic of its own to unit test beyond what the
    // real-GPU `server_e2e` integration test already exercises end to end.
}

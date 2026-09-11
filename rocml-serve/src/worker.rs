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

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

use rocml::generate::generate_sampled_with_stop;
use rocml::{GenerateStats, Model, SamplingParams};
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
    Done(GenerateStats),
    Error(String),
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
    tokenizer: Arc<BpeTokenizer>,
) -> Result<(Sender<Job>, std::thread::JoinHandle<()>), String> {
    let (job_tx, job_rx) = std::sync::mpsc::channel::<Job>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();

    let handle = std::thread::Builder::new()
        .name("rocml-worker".to_string())
        .spawn(move || worker_loop(model_path, tokenizer, job_rx, ready_tx))
        .map_err(|e| format!("failed to spawn worker thread: {e}"))?;

    ready_rx
        .recv()
        .map_err(|_| "worker thread exited during startup".to_string())??;
    Ok((job_tx, handle))
}

fn worker_loop(
    model_path: PathBuf,
    tokenizer: Arc<BpeTokenizer>,
    job_rx: Receiver<Job>,
    ready_tx: Sender<Result<(), String>>,
) {
    let mut model = match Model::load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            let _ = ready_tx.send(Err(e.to_string()));
            return;
        }
    };
    let _ = ready_tx.send(Ok(()));

    for job in job_rx.iter() {
        // `&mut Model` isn't `UnwindSafe` — deliberately: a panic mid-token
        // could leave the KV cache half-updated, which is exactly why the
        // panic arm below resets it before the next job runs rather than
        // trusting whatever state was left behind.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_job(&mut model, &tokenizer, &job)
        }));
        let event = match outcome {
            Ok(Ok(stats)) => WorkerEvent::Done(stats),
            Ok(Err(e)) => WorkerEvent::Error(e.to_string()),
            Err(_) => {
                let _ = model.reset();
                WorkerEvent::Error("internal error during generation".to_string())
            }
        };
        let _ = job.respond_to.send(event);
    }
}

/// Runs one job to completion: a fresh sequence (`Model::reset`) prefilled
/// with `job.prompt_ids`, then sampled decode until `max_new_tokens`, eos,
/// or an OpenAI `stop` string appears in the accumulated decoded text.
fn run_job(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    job: &Job,
) -> Result<GenerateStats, rocml::RocmlError> {
    model.reset()?;
    let mut decoded_so_far = String::new();
    generate_sampled_with_stop(
        model,
        tokenizer,
        &job.prompt_ids,
        job.max_new_tokens,
        true,
        &job.sampling,
        |chunk| {
            let _ = job.respond_to.send(WorkerEvent::Chunk(chunk.to_string()));
            decoded_so_far.push_str(chunk);
            stop_matched(&decoded_so_far, &job.stop_strings)
        },
    )
}

/// An empty stop string never matches (it would trivially match any text
/// and halt generation on the first chunk) — `contains` alone can't
/// distinguish "not yet seen" from "seen the whole prompt so far", so an
/// empty string must be filtered explicitly rather than relying on it.
fn stop_matched(decoded_so_far: &str, stop_strings: &[String]) -> bool {
    stop_strings
        .iter()
        .any(|s| !s.is_empty() && decoded_so_far.contains(s.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_stop_strings_never_matches() {
        assert!(!stop_matched("hello world", &[]));
    }

    #[test]
    fn matches_substring_anywhere_in_accumulated_text() {
        let stops = vec!["STOP".to_string()];
        assert!(!stop_matched("hello wor", &stops));
        assert!(stop_matched("hello world STOP here", &stops));
    }

    #[test]
    fn empty_stop_string_is_ignored() {
        let stops = vec![String::new(), "END".to_string()];
        assert!(!stop_matched("anything at all", &stops));
        assert!(stop_matched("reached the END now", &stops));
    }

    #[test]
    fn matches_first_of_several_stop_strings() {
        let stops = vec!["```".to_string(), "\n\n".to_string()];
        assert!(stop_matched("some code```", &stops));
        assert!(stop_matched("line one\n\nline two", &stops));
        assert!(!stop_matched("no stop here", &stops));
    }
}

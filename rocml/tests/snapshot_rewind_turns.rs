//! `run_turn`'s tier policy end to end (the companion of
//! `snapshot_rewind.rs`, split out for the 400-line cap): GPU hits for both
//! rewind-slot shapes with a RAM budget too small to hold anything, a fall
//! back to the RAM tier once an unrelated turn has reset the GPU slots
//! (with output identical to a from-scratch run — the invariant that a
//! stale slot must never be served), and nothing at all with the store off.
//!
//! Real hardware + the real Qwen3.5-2B-Q8_0 checkpoint required; skips
//! itself if absent. Run via `make test-model` (`--release`).

use rocml::snapshot::turn::{run_turn, HitSource, TurnOutcome};
use rocml::snapshot::{KvConfigStamp, ModelStamp, SnapshotStore};
use rocml::{KvCacheMode, LoadOptions, Model, SamplingParams};
use rocml_core::testpaths::checkpoint;
use rocml_core::tokenizer::BpeTokenizer;

mod support;

use support::snapshot_check::synthetic_prompt;

const GGUF_REL: &str = "Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";

struct Turn {
    outcome: TurnOutcome,
    text: String,
}

#[allow(clippy::too_many_arguments)]
fn turn(
    model: &mut Model,
    tokenizer: &BpeTokenizer,
    store: Option<&mut SnapshotStore>,
    stamp: &ModelStamp,
    kv: &KvConfigStamp,
    prompt: &[u32],
    stable_boundary: Option<u32>,
) -> Turn {
    let mut text = String::new();
    let outcome = run_turn(
        model,
        tokenizer,
        store,
        stamp,
        kv,
        prompt,
        12,
        false,
        &SamplingParams::greedy(),
        &[],
        stable_boundary,
        |chunk| text.push_str(chunk),
    )
    .expect("run_turn failed");
    Turn { outcome, text }
}

#[test]
fn qwen35_gpu_rewind_run_turn_tiers() {
    let Some(path) = checkpoint(GGUF_REL) else {
        return;
    };
    let path = path.as_path();
    let ctx = 4096;
    let opts = LoadOptions::new(ctx).with_kv_cache(KvCacheMode::Fp16);
    let mut model = Model::load(path, opts).expect("Model::load failed");
    let tokenizer =
        BpeTokenizer::from_gguf(&rocml_core::gguf::GgufFile::open(path).expect("gguf open failed"))
            .expect("tokenizer load failed");
    let vocab_size = model.vocab_size();
    let stamp = ModelStamp::from_path(path).expect("stamp failed");
    let kv = KvConfigStamp {
        mode: KvCacheMode::Fp16,
        ctx,
    };
    let p1 = synthetic_prompt(64, vocab_size);
    let extra: Vec<u32> = synthetic_prompt(20, vocab_size)
        .into_iter()
        .map(|t| (t + 1) % vocab_size)
        .collect();

    // A 1 MiB budget can't hold even a 64-token snapshot of this model (its
    // GDN state alone is tens of MiB), so every hit below is the GPU's.
    let mut tiny = SnapshotStore::new(1024 * 1024, None, 0).expect("store failed");
    let t1 = turn(
        &mut model,
        &tokenizer,
        Some(&mut tiny),
        &stamp,
        &kv,
        &p1,
        None,
    );
    assert_eq!((t1.outcome.reused_prefix, t1.outcome.hit_source), (0, None));

    // Thinking-off shape: the next prompt extends prompt + reply verbatim.
    let mut p2 = p1.clone();
    p2.extend_from_slice(&t1.outcome.stats.generated_ids);
    p2.extend_from_slice(&extra);
    let t2 = turn(
        &mut model,
        &tokenizer,
        Some(&mut tiny),
        &stamp,
        &kv,
        &p2,
        None,
    );
    assert_eq!(t2.outcome.hit_source, Some(HitSource::Gpu));
    assert_eq!(
        t2.outcome.reused_prefix as usize,
        p1.len() + t1.outcome.stats.generated_ids.len()
    );
    assert_eq!(
        tiny.ram_used_bytes(),
        0,
        "nothing should have fit the tiny RAM budget"
    );

    // Thinking-on shape: turn 3 re-sends turn 2's prompt declaring a stable
    // boundary. Nothing can serve it yet (turns 1-2 saved no stable slot,
    // and the end-of-turn slot is longer than this prompt), so it misses...
    let p3 = p2.clone();
    let boundary = (p3.len() - 3) as u32;
    let t3 = turn(
        &mut model,
        &tokenizer,
        Some(&mut tiny),
        &stamp,
        &kv,
        &p3,
        Some(boundary),
    );
    assert_eq!(t3.outcome.hit_source, None);
    // ...a verbatim regenerate then hits the stable slot it just saved...
    let t3b = turn(
        &mut model,
        &tokenizer,
        Some(&mut tiny),
        &stamp,
        &kv,
        &p3,
        Some(boundary),
    );
    assert_eq!(t3b.outcome.hit_source, Some(HitSource::Gpu));
    assert_eq!(t3b.outcome.reused_prefix, boundary);
    // ...and so does turn 4, which keeps that prefix but diverges right
    // after it (a stripped think block).
    let mut p4 = p3[..boundary as usize].to_vec();
    p4.extend(p3[boundary as usize..].iter().map(|t| (t + 5) % vocab_size));
    p4.extend_from_slice(&extra);
    let t4 = turn(
        &mut model,
        &tokenizer,
        Some(&mut tiny),
        &stamp,
        &kv,
        &p4,
        None,
    );
    assert_eq!(t4.outcome.hit_source, Some(HitSource::Gpu));
    assert_eq!(t4.outcome.reused_prefix, boundary);

    // Fallback: with a real budget, an unrelated turn in between resets the
    // GPU slots, and the original conversation's stable snapshot must come
    // back from RAM — with output identical to a from-scratch run.
    let mut store = SnapshotStore::new(256 * 1024 * 1024, None, 0).expect("store failed");
    let pa = synthetic_prompt(80, vocab_size);
    let boundary_a = (pa.len() - 4) as u32;
    let ta = turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &stamp,
        &kv,
        &pa,
        Some(boundary_a),
    );
    assert_eq!(ta.outcome.hit_source, None);
    assert_eq!(store.ram_pinned_position(), Some(boundary_a));
    let pu: Vec<u32> = synthetic_prompt(50, vocab_size)
        .into_iter()
        .map(|t| (t + 11) % vocab_size)
        .collect();
    let tu = turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &stamp,
        &kv,
        &pu,
        None,
    );
    assert_eq!(tu.outcome.hit_source, None);
    let mut pa2 = pa[..boundary_a as usize].to_vec();
    pa2.extend(
        pa[boundary_a as usize..]
            .iter()
            .map(|t| (t + 5) % vocab_size),
    );
    pa2.extend_from_slice(&extra);
    let ta2 = turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &stamp,
        &kv,
        &pa2,
        None,
    );
    assert_eq!(
        ta2.outcome.hit_source,
        Some(HitSource::Ram),
        "the GPU slots belong to the unrelated turn now; RAM must serve this one"
    );
    assert_eq!(ta2.outcome.reused_prefix, boundary_a);
    let fresh = turn(&mut model, &tokenizer, None, &stamp, &kv, &pa2, None);
    assert_eq!(
        (fresh.outcome.reused_prefix, fresh.outcome.hit_source),
        (0, None)
    );
    assert_eq!(
        ta2.text, fresh.text,
        "RAM-restored turn must match a from-scratch turn"
    );

    // A stop-string ending leaves its last token un-forwarded; the
    // end-of-turn slot must still save (at the position the cache really
    // holds) so the next turn hits it. Stop on the reference run's first
    // character, so generation ends right after its first token.
    let mut ps = synthetic_prompt(90, vocab_size);
    ps.iter_mut().for_each(|t| *t = (*t + 13) % vocab_size);
    let reference = turn(&mut model, &tokenizer, None, &stamp, &kv, &ps, None);
    let first_char = reference.text.chars().next().expect("non-empty reply");
    let mut stopped_text = String::new();
    let stopped = run_turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &stamp,
        &kv,
        &ps,
        12,
        false,
        &SamplingParams::greedy(),
        &[first_char.to_string()],
        None,
        |chunk| stopped_text.push_str(chunk),
    )
    .expect("run_turn failed");
    let generated = &stopped.stats.generated_ids;
    assert!(
        !generated.is_empty() && generated.len() < reference.outcome.stats.generated_ids.len(),
        "the stop string should end generation early"
    );
    let mut ps2 = ps.clone();
    ps2.extend_from_slice(generated);
    ps2.extend_from_slice(&extra);
    let after_stop = turn(
        &mut model,
        &tokenizer,
        Some(&mut store),
        &stamp,
        &kv,
        &ps2,
        None,
    );
    assert_eq!(after_stop.outcome.hit_source, Some(HitSource::Gpu));
    assert_eq!(
        after_stop.outcome.reused_prefix as usize,
        ps.len() + generated.len() - 1,
        "end-of-turn slot must hold exactly the forwarded prefix"
    );

    // Store off means the GPU tier is off too: a verbatim repeat re-prefills.
    let off1 = turn(&mut model, &tokenizer, None, &stamp, &kv, &p2, None);
    let off2 = turn(&mut model, &tokenizer, None, &stamp, &kv, &p2, None);
    assert_eq!(off1.outcome.reused_prefix, 0);
    assert_eq!(off2.outcome.reused_prefix, 0);
    assert_eq!(off1.text, off2.text);
}

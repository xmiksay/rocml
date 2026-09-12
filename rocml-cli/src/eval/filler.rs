//! Deterministic filler-text generation for `long_context` scenarios: turns
//! a `u64` seed into reproducible prose from a small fixed wordlist, with
//! the scenario's needle spliced in at a given fractional position. This is
//! what lets `bench/eval/scenarios.json` store just the needle and a target
//! length instead of checking in kilobytes of filler per scenario.
//!
//! Uses a local xorshift64* PRNG rather than `rocml::sample::Rng`, whose
//! internals are private to that crate's `sample` module — deterministic
//! and cheap either way, no `rand` dependency.

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Xorshift's state must never be all-zero, or it stays zero forever.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_index(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

/// A small fixed vocabulary of common, semantically-neutral English words —
/// only used to pad out filler prose, never read for content.
const WORDLIST: &[&str] = &[
    "the",
    "quick",
    "brown",
    "fox",
    "jumps",
    "over",
    "lazy",
    "dog",
    "river",
    "runs",
    "through",
    "valley",
    "past",
    "old",
    "stone",
    "bridge",
    "toward",
    "distant",
    "mountains",
    "under",
    "clear",
    "morning",
    "sky",
    "travelers",
    "gather",
    "near",
    "market",
    "square",
    "exchange",
    "stories",
    "goods",
    "children",
    "play",
    "quiet",
    "garden",
    "while",
    "birds",
    "sing",
    "from",
    "tall",
    "trees",
    "shopkeepers",
    "open",
    "doors",
    "early",
    "prepare",
    "day",
    "ahead",
    "wind",
    "carries",
    "scent",
    "fresh",
    "bread",
    "down",
    "narrow",
    "streets",
    "workers",
    "walk",
    "toward",
    "factory",
    "gates",
    "carrying",
    "tools",
    "lunch",
    "wrapped",
    "cloth",
    "farmers",
    "load",
    "carts",
    "with",
    "vegetables",
    "grain",
    "head",
    "town",
    "before",
    "sun",
    "rises",
    "fully",
    "students",
    "hurry",
    "school",
    "books",
    "under",
    "arm",
    "teachers",
    "wait",
    "patiently",
    "front",
    "classroom",
    "doors",
    "librarians",
    "arrange",
    "shelves",
    "new",
    "arrivals",
    "readers",
    "quietly",
    "turn",
    "pages",
    "seeking",
    "answers",
    "old",
    "questions",
];

/// Deterministically produces `count` lowercase words from `seed` — the
/// same seed always yields the same sequence, so a scenario's filler is
/// reproducible without storing it.
pub fn generate_words(seed: u64, count: usize) -> Vec<String> {
    let mut rng = Rng::new(seed);
    (0..count)
        .map(|_| WORDLIST[rng.next_index(WORDLIST.len())].to_string())
        .collect()
}

/// Words per generated sentence — purely cosmetic (readable prose vs. one
/// giant run-on), doesn't affect scoring.
const WORDS_PER_SENTENCE: usize = 9;

/// Builds one `long_context` scenario's filler: `target_words` words
/// generated from `seed`, with `needle`'s own words spliced in intact at
/// `position_fraction` of the way through, then grouped into capitalized,
/// period-terminated sentences.
pub fn build_context(
    seed: u64,
    target_words: usize,
    needle: &str,
    position_fraction: f32,
) -> String {
    let mut words = generate_words(seed, target_words);
    let frac = position_fraction.clamp(0.0, 1.0) as f64;
    let idx = ((frac * words.len() as f64).round() as usize).min(words.len());
    let needle_words: Vec<String> = needle.split_whitespace().map(str::to_string).collect();
    words.splice(idx..idx, needle_words);
    to_sentences(&words)
}

/// Converts token count into a filler word budget: this BPE tokenizer
/// family runs roughly 1.3-1.5 tokens per common English word, so 0.7
/// words/token keeps the actual encoded length in the right ballpark
/// without needing a loaded tokenizer at scenario-authoring time. Approximate
/// by design — `long_context` scenarios only need "about 4000" / "about
/// 8000" tokens, not an exact count.
pub fn words_for_tokens(target_tokens: usize) -> usize {
    ((target_tokens as f64) * 0.7).round() as usize
}

fn to_sentences(words: &[String]) -> String {
    let mut out = String::new();
    for (i, chunk) in words.chunks(WORDS_PER_SENTENCE).enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(&capitalize(&chunk.join(" ")));
        out.push('.');
    }
    out
}

fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_is_deterministic() {
        assert_eq!(generate_words(42, 200), generate_words(42, 200));
    }

    #[test]
    fn different_seeds_diverge() {
        assert_ne!(generate_words(1, 200), generate_words(2, 200));
    }

    #[test]
    fn zero_seed_does_not_degenerate() {
        // The all-zero xorshift state would stay zero forever if not remapped.
        let words = generate_words(0, 50);
        assert!(words.iter().any(|w| w != &words[0]));
    }

    #[test]
    fn needle_appears_in_output() {
        let context = build_context(7, 300, "The secret code is XJ-4471.", 0.5);
        assert!(context.contains("XJ-4471"));
    }

    #[test]
    fn position_fraction_zero_puts_needle_near_the_start() {
        let context = build_context(7, 300, "NEEDLETOKEN", 0.0);
        let needle_pos = context.find("NEEDLETOKEN").expect("needle present");
        // Comfortably within the first sentence, regardless of capitalization.
        assert!(needle_pos < 20, "needle at {needle_pos}, expected near 0");
    }

    #[test]
    fn position_fraction_one_puts_needle_near_the_end() {
        let context = build_context(7, 300, "NEEDLETOKEN", 1.0);
        let needle_pos = context.find("NEEDLETOKEN").expect("needle present");
        assert!(
            needle_pos > context.len() - 20,
            "needle at {needle_pos} of {}, expected near the end",
            context.len()
        );
    }

    #[test]
    fn build_context_is_deterministic_for_same_inputs() {
        let a = build_context(99, 500, "fact one two three", 0.3);
        let b = build_context(99, 500, "fact one two three", 0.3);
        assert_eq!(a, b);
    }

    #[test]
    fn words_for_tokens_scales_down_from_token_budget() {
        assert!(words_for_tokens(4000) < 4000);
        assert!(words_for_tokens(8000) > words_for_tokens(4000));
    }
}

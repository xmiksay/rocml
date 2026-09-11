//! Splits input text on special/control tokens (`<|im_start|>` and
//! friends) before pre-tokenization and BPE ever see it, so they can never
//! be torn apart by a merge. Matching is a simple greedy longest-match scan
//! rather than Aho-Corasick: there are only a few dozen special tokens, so
//! an O(n * specials) scan is more than fast enough and needs no extra deps.

pub(super) enum Segment<'a> {
    Text(&'a str),
    Special(u32),
}

/// `specials` must be sorted longest-first so that one special token that's
/// a prefix of another (there are none in practice, but nothing enforces
/// it) still resolves to the longer, more specific match.
pub(super) fn split<'a>(text: &'a str, specials: &[(String, u32)]) -> Vec<Segment<'a>> {
    let mut segments = Vec::new();
    // `pos`: byte offset of the next unprocessed character.
    // `text_start`: byte offset where the pending `Text` segment began.
    let mut pos = 0usize;
    let mut text_start = 0usize;

    while pos < text.len() {
        let rest = &text[pos..];
        if let Some((token, id)) = specials.iter().find(|(t, _)| rest.starts_with(t.as_str())) {
            if text_start < pos {
                segments.push(Segment::Text(&text[text_start..pos]));
            }
            segments.push(Segment::Special(*id));
            pos += token.len();
            text_start = pos;
        } else {
            // Advance by one char (not one byte) to stay on a UTF-8
            // boundary; special tokens are themselves ASCII so this never
            // steps over a match.
            pos += rest.chars().next().map_or(1, char::len_utf8);
        }
    }
    if text_start < text.len() {
        segments.push(Segment::Text(&text[text_start..]));
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(segs: &[Segment<'_>]) -> Vec<Option<u32>> {
        segs.iter()
            .map(|s| match s {
                Segment::Text(_) => None,
                Segment::Special(id) => Some(*id),
            })
            .collect()
    }

    #[test]
    fn splits_around_a_special_token() {
        let specials = vec![("<|im_start|>".to_string(), 100u32)];
        let segs = split("hello <|im_start|>world", &specials);
        assert_eq!(ids(&segs), vec![None, Some(100), None]);
        let texts: Vec<&str> = segs
            .iter()
            .filter_map(|s| match s {
                Segment::Text(t) => Some(*t),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["hello ", "world"]);
    }

    #[test]
    fn text_with_no_special_tokens_is_one_segment() {
        let specials = vec![("<|im_start|>".to_string(), 100u32)];
        let segs = split("just plain text", &specials);
        assert_eq!(ids(&segs), vec![None]);
    }

    #[test]
    fn empty_input_yields_no_segments() {
        let specials = vec![("<|im_start|>".to_string(), 100u32)];
        assert!(split("", &specials).is_empty());
    }
}

//! Incremental scanner over decoded generation output, for a future server
//! to suppress `<think>`/`<tool_call>` tags in a streaming response instead
//! of relaying them raw.
//!
//! Deliberately simple: tool-call bodies are buffered whole (no partial
//! `ToolCallDelta`) and reused via [`super::parser`]'s block parser once the
//! closing tag arrives; plain text and thinking text stream out a chunk at
//! a time, holding back only the trailing bytes that might still complete a
//! recognized tag.

use super::error::ChatError;
use super::parser::parse_one_tool_call_block;
use super::types::ToolCall;

const OPEN_THINK: &str = "<think>";
const CLOSE_THINK: &str = "</think>";
const OPEN_TOOL: &str = "<tool_call>";
const CLOSE_TOOL: &str = "</tool_call>";

#[derive(Debug, Clone, PartialEq)]
pub enum ScanEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolCallStarted,
    ToolCallComplete(ToolCall),
}

#[derive(Debug, Clone, Copy, Default)]
enum Mode {
    #[default]
    Text,
    Thinking,
    ToolCall,
}

/// Feed decoded text pieces in; get scanner events out. See module docs.
#[derive(Debug, Default)]
pub struct StreamScanner {
    buffer: String,
    mode: Mode,
}

impl StreamScanner {
    pub fn new() -> Self {
        Self::default()
    }

    /// A scanner for output generated after a prompt whose
    /// `add_generation_prompt` tail already opened `<think>\n` (i.e.
    /// `RenderOpts.enable_thinking != Some(false)`, the template's default —
    /// see `render`'s doc comment). The opening tag itself never appears in
    /// the *generated* text in that case (it was part of the prompt), so a
    /// scanner starting in the default `Mode::Text` would treat the whole
    /// reasoning block, and even the literal `</think>` closing tag, as
    /// plain visible content instead of recognizing it as reasoning —
    /// `parser::extract_thinking` already handles this same "primed
    /// prompt" shape for the non-streaming case (see its own doc comment);
    /// this is the streaming equivalent.
    pub fn new_primed_for_thinking() -> Self {
        Self {
            buffer: String::new(),
            mode: Mode::Thinking,
        }
    }

    /// Consume one decoded chunk, returning the events it completed. A tag
    /// split across chunk boundaries is held in the internal buffer until
    /// enough of it has arrived to resolve.
    pub fn feed(&mut self, chunk: &str) -> Result<Vec<ScanEvent>, ChatError> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        loop {
            match self.mode {
                Mode::Text => {
                    let open_think = self.buffer.find(OPEN_THINK);
                    let open_tool = self.buffer.find(OPEN_TOOL);
                    match earliest(open_think, open_tool) {
                        Some((pos, is_think)) => {
                            emit_text(&mut events, self.buffer.drain(..pos).collect(), false);
                            let tag_len = if is_think {
                                OPEN_THINK.len()
                            } else {
                                OPEN_TOOL.len()
                            };
                            self.buffer.drain(..tag_len);
                            if is_think {
                                self.mode = Mode::Thinking;
                            } else {
                                self.mode = Mode::ToolCall;
                                events.push(ScanEvent::ToolCallStarted);
                            }
                        }
                        None => {
                            let flush_len = safe_flush_len(&self.buffer, &[OPEN_THINK, OPEN_TOOL]);
                            emit_text(&mut events, self.buffer.drain(..flush_len).collect(), false);
                            break;
                        }
                    }
                }
                Mode::Thinking => match self.buffer.find(CLOSE_THINK) {
                    Some(pos) => {
                        emit_text(&mut events, self.buffer.drain(..pos).collect(), true);
                        self.buffer.drain(..CLOSE_THINK.len());
                        self.mode = Mode::Text;
                    }
                    None => {
                        let flush_len = safe_flush_len(&self.buffer, &[CLOSE_THINK]);
                        emit_text(&mut events, self.buffer.drain(..flush_len).collect(), true);
                        break;
                    }
                },
                Mode::ToolCall => match self.buffer.find(CLOSE_TOOL) {
                    Some(pos) => {
                        let block: String = self.buffer.drain(..pos).collect();
                        self.buffer.drain(..CLOSE_TOOL.len());
                        events.push(ScanEvent::ToolCallComplete(parse_one_tool_call_block(
                            &block,
                        )?));
                        self.mode = Mode::Text;
                    }
                    None => break, // Buffer whole; no partial tool-call events.
                },
            }
        }
        Ok(events)
    }

    /// Call once the stream ends. Flushes any trailing plain text/thinking
    /// text that turned out not to be the start of a tag; a still-open
    /// `<tool_call>` at end of stream is a truncated generation and is
    /// dropped rather than guessed at.
    pub fn finish(mut self) -> Vec<ScanEvent> {
        let mut events = Vec::new();
        match self.mode {
            Mode::Text => emit_text(&mut events, std::mem::take(&mut self.buffer), false),
            Mode::Thinking => emit_text(&mut events, std::mem::take(&mut self.buffer), true),
            Mode::ToolCall => {}
        }
        events
    }
}

fn emit_text(events: &mut Vec<ScanEvent>, text: String, thinking: bool) {
    if text.is_empty() {
        return;
    }
    events.push(if thinking {
        ScanEvent::ThinkingDelta(text)
    } else {
        ScanEvent::TextDelta(text)
    });
}

/// Picks whichever of two optional positions comes first, tagging which one
/// it was (`true` = the `<think>` position).
fn earliest(think: Option<usize>, tool: Option<usize>) -> Option<(usize, bool)> {
    match (think, tool) {
        (Some(t), Some(c)) => Some(if t <= c { (t, true) } else { (c, false) }),
        (Some(t), None) => Some((t, true)),
        (None, Some(c)) => Some((c, false)),
        (None, None) => None,
    }
}

/// How much of `buffer` is safe to flush as plain text: everything except a
/// trailing suffix that could still grow into one of `tags`. Never splits a
/// UTF-8 character.
fn safe_flush_len(buffer: &str, tags: &[&str]) -> usize {
    let total = buffer.len();
    let max_candidate = tags
        .iter()
        .map(|t| t.len().saturating_sub(1))
        .max()
        .unwrap_or(0);
    for len in (1..=max_candidate.min(total)).rev() {
        let idx = total - len;
        if !buffer.is_char_boundary(idx) {
            continue;
        }
        let suffix = &buffer[idx..];
        if tags.iter().any(|t| t.starts_with(suffix)) {
            return idx;
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_text_passes_through() {
        let mut scanner = StreamScanner::new();
        let events = scanner.feed("hello world").unwrap();
        assert_eq!(
            events,
            vec![ScanEvent::TextDelta("hello world".to_string())]
        );
    }

    #[test]
    fn primed_scanner_treats_leading_text_as_thinking() {
        let mut scanner = StreamScanner::new_primed_for_thinking();
        let mut events = scanner.feed("still reasoning").unwrap();
        events.extend(scanner.feed("</think>answer").unwrap());
        assert_eq!(
            events,
            vec![
                ScanEvent::ThinkingDelta("still reasoning".to_string()),
                ScanEvent::TextDelta("answer".to_string()),
            ]
        );
    }

    #[test]
    fn think_tag_split_across_chunks_still_resolves() {
        let mut scanner = StreamScanner::new();
        let mut events = scanner.feed("<thi").unwrap();
        assert!(events.is_empty(), "must hold back a possible tag prefix");
        events.extend(scanner.feed("nk>reasoning</think>answer").unwrap());
        assert_eq!(
            events,
            vec![
                ScanEvent::ThinkingDelta("reasoning".to_string()),
                ScanEvent::TextDelta("answer".to_string())
            ]
        );
    }

    #[test]
    fn close_think_tag_split_across_chunks() {
        let mut scanner = StreamScanner::new();
        let mut events = scanner.feed("<think>partial</thi").unwrap();
        events.extend(scanner.feed("nk>rest").unwrap());
        assert_eq!(
            events,
            vec![
                ScanEvent::ThinkingDelta("partial".to_string()),
                ScanEvent::TextDelta("rest".to_string())
            ]
        );
    }

    #[test]
    fn tool_call_buffers_whole_and_completes_on_close_tag() {
        let mut scanner = StreamScanner::new();
        let mut events = scanner.feed("before<tool_call>\n<function=f>\n").unwrap();
        assert_eq!(
            events,
            vec![
                ScanEvent::TextDelta("before".to_string()),
                ScanEvent::ToolCallStarted
            ]
        );
        events.clear();
        events.extend(
            scanner
                .feed("<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>after")
                .unwrap(),
        );
        assert_eq!(events.len(), 2);
        assert!(matches!(events[0], ScanEvent::ToolCallComplete(_)));
        assert_eq!(events[1], ScanEvent::TextDelta("after".to_string()));
    }

    #[test]
    fn finish_flushes_trailing_text() {
        let mut scanner = StreamScanner::new();
        // "tail" is emitted immediately; only the possible-tag-prefix "<thi"
        // is held back in the buffer for `feed` to keep watching.
        let fed_events = scanner.feed("tail<thi").unwrap();
        assert_eq!(fed_events, vec![ScanEvent::TextDelta("tail".to_string())]);
        // "<thi" never completed into a tag, so `finish` flushes it as text.
        let events = scanner.finish();
        assert_eq!(events, vec![ScanEvent::TextDelta("<thi".to_string())]);
    }
}

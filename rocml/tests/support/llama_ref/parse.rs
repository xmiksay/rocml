//! Parser for `rocml-dump`'s plain-text tensor dump format (see the tool's
//! source, quoted in `docs/llama-diff.md`, for the writer side):
//!
//! ```text
//! #TENSOR <name> <ne0> <ne1> <ne2> <ne3>
//! <ne1*ne2*ne3 lines of ne0 whitespace-separated floats>
//! ```
//!
//! Deliberately a hand-rolled line scanner rather than a `regex`-based one
//! — the workspace has no `regex` dependency anywhere, and this format is
//! simple enough (one fixed-shape header line per tensor) not to need one.

use std::collections::BTreeMap;

/// One captured llama.cpp ggml tensor: `ne0` (features, ggml's fastest-
/// varying/innermost dimension) by `ne1*ne2*ne3` (tokens/planes),
/// flattened row-major. ggml's in-memory layout for a 2D `[n_embd,
/// n_tokens]` tensor already matches rocml's `CapturedTensor`
/// `[rows, cols]` convention with no transpose needed.
#[derive(Debug, Clone)]
pub struct RawTensor {
    pub ne: [i64; 4],
    pub values: Vec<f32>,
}

/// Canonical ggml node name (ggml auto-suffixes like `" (reshaped)"`/
/// `" (view)"` stripped, see [`canonicalize`]) -> tensor. A `BTreeMap` so a
/// later write for a name that already exists overwrites deterministically
/// — see [`canonicalize`]'s doc comment for why that's safe for every node
/// this module's caller (`convert`) actually reads.
pub type RawDump = BTreeMap<String, RawTensor>;

/// Drops any ggml auto-generated suffix (`" (reshaped)"`, `" (view)"`,
/// ...): ggml always appends these as `"<name> (<suffix>)"` to a source
/// tensor's derived name, so truncating at the first `" ("` recovers the
/// name the graph builder actually gave the node via its `cb(...)` call.
///
/// Caveat (harmless for every tensor `convert` maps today, see its own
/// doc comment): a suffixed variant is not always numerically or
/// shape-identical to its un-suffixed source (a `" (view)"` can be a
/// sub-tensor, unlike a whole-tensor `" (reshaped)"`), so this dedup keeps
/// whichever the dump's writer emitted last under a given canonical name.
fn canonicalize(name: &str) -> &str {
    match name.find(" (") {
        Some(i) => &name[..i],
        None => name,
    }
}

/// Parses `rocml-dump`'s output. Malformed input is a hard error (this is
/// diagnostic-tool output produced by a build this crate doesn't control,
/// not a format worth silently tolerating partial corruption of).
pub fn parse(text: &str) -> Result<RawDump, String> {
    let mut out = RawDump::new();
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        let Some(header) = line.strip_prefix("#TENSOR ") else {
            continue;
        };
        let tokens: Vec<&str> = header.split_whitespace().collect();
        if tokens.len() < 5 {
            return Err(format!("malformed #TENSOR header: {line:?}"));
        }
        let ne_start = tokens.len() - 4;
        let mut ne = [0i64; 4];
        for (i, tok) in tokens[ne_start..].iter().enumerate() {
            ne[i] = tok
                .parse()
                .map_err(|e| format!("bad ne[{i}] in {line:?}: {e}"))?;
        }
        let name = tokens[..ne_start].join(" ");
        let n_rows = (ne[1] * ne[2] * ne[3]).max(0) as usize;
        let cols = ne[0].max(0) as usize;
        let mut values = Vec::with_capacity(n_rows * cols);
        for _ in 0..n_rows {
            let row = lines
                .next()
                .ok_or_else(|| format!("truncated dump: expected a row for {name:?}"))?;
            let mut count = 0usize;
            for tok in row.split_whitespace() {
                values.push(
                    tok.parse::<f32>()
                        .map_err(|e| format!("bad float {tok:?} in {name:?}: {e}"))?,
                );
                count += 1;
            }
            if count != cols {
                return Err(format!(
                    "row width mismatch for {name:?}: expected {cols}, got {count}"
                ));
            }
        }
        out.insert(canonicalize(&name).to_string(), RawTensor { ne, values });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_simple_two_row_tensor() {
        let text = "#TENSOR attn_norm-0 3 2 1 1\n1 2 3\n4 5 6\n";
        let dump = parse(text).expect("parse failed");
        let t = dump.get("attn_norm-0").expect("missing tensor");
        assert_eq!(t.ne, [3, 2, 1, 1]);
        assert_eq!(t.values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn strips_ggml_auto_suffixes_and_keeps_the_last_write() {
        let text = "#TENSOR linear_attn_out-0 2 1 1 1\n1 2\n\
                     #TENSOR linear_attn_out-0 (reshaped) 2 1 1 1\n3 4\n";
        let dump = parse(text).expect("parse failed");
        assert_eq!(dump.len(), 1);
        assert_eq!(dump["linear_attn_out-0"].values, vec![3.0, 4.0]);
    }

    #[test]
    fn multi_plane_tensor_reads_ne1_times_ne2_times_ne3_rows() {
        // Real shape from a `beta` tensor: ne = [1, 2, 2, 1] (1 head-scalar
        // x 2 tokens x 2 seqs).
        let text = "#TENSOR beta-0 1 2 2 1\n1\n2\n3\n4\n";
        let dump = parse(text).expect("parse failed");
        assert_eq!(dump["beta-0"].values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rejects_a_short_row() {
        let text = "#TENSOR x-0 3 1 1 1\n1 2\n";
        assert!(parse(text).is_err());
    }

    #[test]
    fn rejects_a_missing_row() {
        let text = "#TENSOR x-0 3 2 1 1\n1 2 3\n";
        assert!(parse(text).is_err());
    }

    #[test]
    fn ignores_non_tensor_lines() {
        let text = "some log line\n#TENSOR x-0 1 1 1 1\n5\nanother log line\n";
        let dump = parse(text).expect("parse failed");
        assert_eq!(dump["x-0"].values, vec![5.0]);
    }
}

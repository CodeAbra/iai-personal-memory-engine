//! Query-time snippet extraction: given a hit file already known from a
//! search result, re-reads it and returns a bounded window of the lines
//! most relevant to the query. The on-disk index carries no offsets (see
//! `segment.rs`'s frozen v1 payload), so this is necessarily a fresh read
//! at query time, not an index lookup -- content may have changed since
//! the file was indexed; that staleness window is accepted, never solved.
//!
//! Line selection reuses the crate's own fuzzy `tokenize()` (the same
//! tokenizer that built the postings), so the highlighted line is
//! consistent with why the file ranked, rather than a stricter exact-token
//! match that could select a different line -- or none at all.

use std::collections::HashSet;
use std::fs;
use std::io;
use std::io::Read;

use super::tokenize::{feature_id, tokenize};

/// Per-hit line cap default. Distinct constant from `exec::tail`'s
/// byte-offset caps -- this one bounds a line-indexed window, not a
/// file-offset stream, so it is not shared with that module.
pub const SNIPPET_LINE_CAP: usize = 10;

/// Upper bound on bytes read per hit before line-selection runs. Every
/// hit file is re-read fresh at query time (no offset in the index), so
/// an unbounded read on a huge matching file would load it whole, once
/// per hit, on every search call that ranks it -- this caps that cost
/// regardless of file size.
pub(crate) const MAX_SNIPPET_READ_BYTES: u64 = 4 * 1024 * 1024;

/// Reads `path` and returns `(line_start_1based, line_end_1based,
/// snippet_text, elided_lines)`. `line_start`/`line_end` are both 0 on an
/// empty file (0-length range, never an error). `elided_lines` counts
/// query-matching lines that exist in the file but fall outside the
/// returned window -- 0 when every matching line is shown or nothing
/// matched at all. The read is capped at `MAX_SNIPPET_READ_BYTES`; a
/// file larger than that is line-selected from its truncated prefix
/// only, never read past the cap. A genuinely non-UTF-8 file returns
/// `InvalidData`; only an incomplete trailing byte sequence introduced
/// by the cap itself cutting mid-character is tolerated, by dropping
/// that incomplete tail.
pub fn extract_snippet(path: &str, query: &str, line_cap: usize) -> io::Result<(usize, usize, String, usize)> {
    let mut file = fs::File::open(path)?;
    let mut buf = Vec::new();
    let bytes_read = file.by_ref().take(MAX_SNIPPET_READ_BYTES).read_to_end(&mut buf)?;
    let truncated = bytes_read as u64 >= MAX_SNIPPET_READ_BYTES;
    let text = match std::str::from_utf8(&buf) {
        Ok(s) => s.to_string(),
        Err(e) if truncated && e.error_len().is_none() => {
            String::from_utf8_lossy(&buf[..e.valid_up_to()]).into_owned()
        }
        Err(e) => return Err(io::Error::new(io::ErrorKind::InvalidData, e)),
    };
    Ok(select_snippet(&text, query, line_cap))
}

fn line_feature_ids(line: &str) -> HashSet<u32> {
    tokenize(line).iter().map(|t| feature_id(t)).collect()
}

/// Pure line-selection core, no I/O -- the anchor line is the query's
/// highest-overlap line (lowest line number breaks a tie), and the
/// returned window is a CONTIGUOUS run of at most `line_cap` lines
/// starting at that anchor. A window is never widened to also cover a
/// second high-overlap line far away in the file -- the per-hit line cap
/// is a hard bound, not a target the window stretches to reach.
fn select_snippet(text: &str, query: &str, line_cap: usize) -> (usize, usize, String, usize) {
    let lines: Vec<&str> = text.lines().collect();
    let total = lines.len();
    if total == 0 {
        return (0, 0, String::new(), 0);
    }
    let cap = line_cap.max(1);

    let query_ids = line_feature_ids(query);

    let mut overlaps: Vec<usize> = Vec::with_capacity(total);
    let mut matching_indices: Vec<usize> = Vec::new();
    for line in &lines {
        let overlap = line_feature_ids(line).intersection(&query_ids).count();
        overlaps.push(overlap);
    }
    for (i, &overlap) in overlaps.iter().enumerate() {
        if overlap > 0 {
            matching_indices.push(i);
        }
    }

    let anchor = match matching_indices.first() {
        None => 0usize,
        Some(&first) => {
            let mut best = first;
            let mut best_overlap = overlaps[first];
            for &i in &matching_indices[1..] {
                if overlaps[i] > best_overlap {
                    best = i;
                    best_overlap = overlaps[i];
                }
            }
            best
        }
    };

    let start = anchor;
    let end = (start + cap).min(total); // exclusive, 0-based
    let window_matches = matching_indices
        .iter()
        .filter(|&&i| i >= start && i < end)
        .count();
    let elided_lines = matching_indices.len() - window_matches;

    let snippet_text = lines[start..end].join("\n");
    (start + 1, end, snippet_text, elided_lines)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_tokenizer_parity_selects_the_matching_line() {
        let text = "def unrelated():\n    pass\n\n\ndef get_db():\n    return conn\n";
        let (start, end, snippet, elided) = select_snippet(text, "getDB", 10);
        assert_eq!(start, 5);
        assert_eq!(end, 6);
        assert!(snippet.contains("get_db"));
        assert_eq!(elided, 0);
    }

    #[test]
    fn zero_overlap_falls_back_to_the_first_line_cap_lines() {
        let text = "alpha\nbeta\ngamma\ndelta\n";
        let (start, end, snippet, elided) = select_snippet(text, "zzz_no_match_zzz", 2);
        assert_eq!(start, 1);
        assert_eq!(end, 2);
        assert_eq!(snippet, "alpha\nbeta");
        assert_eq!(elided, 0);
    }

    #[test]
    fn empty_file_returns_a_zero_length_range_without_panicking() {
        let (start, end, snippet, elided) = select_snippet("", "anything", 10);
        assert_eq!(start, 0);
        assert_eq!(end, 0);
        assert_eq!(snippet, "");
        assert_eq!(elided, 0);
    }

    #[test]
    fn equal_overlap_ties_resolve_to_the_lower_line_number() {
        let text = "marker_token here\nother stuff\nmarker_token here too\n";
        let (start, end, _snippet, _elided) = select_snippet(text, "marker_token", 1);
        assert_eq!(start, 1);
        assert_eq!(end, 1);
    }

    #[test]
    fn window_never_exceeds_the_line_cap_even_when_matches_are_far_apart() {
        let mut text = String::new();
        for i in 0..500 {
            if i == 3 || i == 480 {
                text.push_str("marker_token_far_apart\n");
            } else {
                text.push_str("filler line\n");
            }
        }
        let (start, end, _snippet, elided) = select_snippet(&text, "marker_token_far_apart", 10);
        assert!(end - start + 1 <= 10);
        assert_eq!(start, 4); // 0-based index 3 -> 1-based line 4
        // The second far-apart match (index 480) falls outside the window.
        assert_eq!(elided, 1);
    }

    #[test]
    fn extract_snippet_reads_the_file_and_reports_the_matching_line() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("marker.py");
        std::fs::write(&path, "def zebracorn_finder():\n    return 42\n").unwrap();

        let (start, end, snippet, elided) =
            extract_snippet(path.to_str().unwrap(), "zebracorn", SNIPPET_LINE_CAP).unwrap();
        assert_eq!(start, 1);
        assert_eq!(end, 2); // both file lines fit under the default line cap
        assert!(snippet.contains("zebracorn_finder"));
        assert_eq!(elided, 0);
    }

    #[test]
    fn extract_snippet_propagates_io_error_for_a_missing_file() {
        let result = extract_snippet("/does/not/exist/at/all.py", "query", SNIPPET_LINE_CAP);
        assert!(result.is_err());
    }

    #[test]
    fn extract_snippet_bounds_the_read_on_a_file_larger_than_the_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("huge.py");
        let filler_line = "filler_line_padding_to_grow_the_file_size\n";
        // A marker line placed well past MAX_SNIPPET_READ_BYTES must never
        // be seen -- proving the read genuinely stops at the cap rather
        // than merely capping how much of an already-fully-read buffer is
        // scanned.
        let past_cap_lines = (MAX_SNIPPET_READ_BYTES as usize) / filler_line.len() + 1000;
        let mut content = filler_line.repeat(past_cap_lines);
        content.push_str("zzz_marker_past_the_cap_zzz\n");

        std::fs::write(&path, &content).unwrap();
        assert!(content.len() as u64 > MAX_SNIPPET_READ_BYTES);

        let (_start, _end, snippet, _elided) =
            extract_snippet(path.to_str().unwrap(), "zzz_marker_past_the_cap_zzz", SNIPPET_LINE_CAP)
                .unwrap();
        assert!(
            !snippet.contains("zzz_marker_past_the_cap_zzz"),
            "the marker line lives past the byte cap and must never be read"
        );
    }

    #[test]
    fn extract_snippet_still_errors_on_a_genuinely_non_utf8_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("binary.bin");
        // An interior invalid byte well before the cap must still error --
        // only a cap-induced incomplete trailing sequence is tolerated.
        std::fs::write(&path, [b'a', b'b', 0xff, b'c', b'd']).unwrap();

        let result = extract_snippet(path.to_str().unwrap(), "query", SNIPPET_LINE_CAP);
        assert!(result.is_err());
    }

    /// Pinned per the `FROZEN_GATE_THRESHOLD` discipline (fusion.rs) --
    /// update this value together with this test, never after the fact.
    /// The Python side (`tools/__init__.py`'s own `SNIPPET_LINE_CAP`)
    /// passes this same numeric value as the `line_cap` argument; the two
    /// constants are not shared across the language boundary and must be
    /// kept in sync by hand.
    #[test]
    fn snippet_line_cap_is_pinned() {
        assert_eq!(SNIPPET_LINE_CAP, 10);
    }
}

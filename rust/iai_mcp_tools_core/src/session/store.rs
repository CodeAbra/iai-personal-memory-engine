//! Own append-only per-session JSONL store under `{root}/sessions/<id>.jsonl`.

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

use thiserror::Error;

use super::SessionEvent;

#[derive(Debug, Error)]
pub enum SessionStoreError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Resolves the tools-subsystem root: `IAI_TOOLS_ROOT` env var if set and
/// non-empty, else `$HOME/.iai-tools` -- same semantics as
/// `exec::artifact::tools_root()`, kept as a sibling helper so this module
/// carries no error-type coupling to `exec`.
pub fn default_root() -> Result<PathBuf, SessionStoreError> {
    if let Ok(v) = std::env::var("IAI_TOOLS_ROOT") {
        if !v.is_empty() {
            return Ok(PathBuf::from(v));
        }
    }
    let home = std::env::var("HOME").map_err(|_| {
        SessionStoreError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "HOME not set",
        ))
    })?;
    Ok(PathBuf::from(home).join(".iai-tools"))
}

/// Single choke point for turning a raw session id into a store path.
/// Returns the id VERBATIM only when it already matches
/// `^[A-Za-z0-9_-]{1,64}$`; otherwise rejects. Never strips, truncates, or
/// substitutes -- a rewrite would let two distinct raw ids collapse onto one
/// file and break per-session privacy.
pub fn validate_session_id(id: &str) -> Option<&str> {
    if id.is_empty() || id.len() > 64 {
        return None;
    }
    if id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
    {
        Some(id)
    } else {
        None
    }
}

/// Builds `{root}/sessions/{id}.jsonl` only from a validated verbatim id.
/// `None` on an invalid id -- callers must treat that as "touch no file".
fn session_file_path(root: &Path, session_id: &str) -> Option<PathBuf> {
    let id = validate_session_id(session_id)?;
    Some(root.join("sessions").join(format!("{id}.jsonl")))
}

fn open_append_0600(path: &Path) -> std::io::Result<std::fs::File> {
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    opts.mode(0o600);
    opts.open(path)
}

fn count_lines(path: &Path) -> std::io::Result<u64> {
    if !path.exists() {
        return Ok(0);
    }
    let f = std::fs::File::open(path)?;
    Ok(BufReader::new(f).lines().count() as u64)
}

/// Advisory exclusive lock on the open file description. Must be held
/// across both seq computation and the write in `append_event` -- releasing
/// it between the two steps lets two concurrent appenders stamp the same
/// seq or interleave partial writes into one corrupt line. Released on
/// `File` drop.
#[cfg(unix)]
fn lock_exclusive(file: &std::fs::File) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    let fd = file.as_raw_fd();
    loop {
        // SAFETY: fd is a valid, open descriptor for the lifetime of this call.
        let ret = unsafe { libc::flock(fd, libc::LOCK_EX) };
        if ret == 0 {
            return Ok(());
        }
        let err = std::io::Error::last_os_error();
        if err.kind() != std::io::ErrorKind::Interrupted {
            return Err(err);
        }
    }
}

#[cfg(not(unix))]
fn lock_exclusive(_file: &std::fs::File) -> std::io::Result<()> {
    Ok(())
}

/// Appends one event to the validated session's file, stamping `seq` from
/// the file's current line count so append order is chronological order
/// with no clock dependency. An invalid `session_id` touches no file and
/// returns `Ok(())` so the caller can exit 0 with a clear message.
pub fn append_event(
    root: &Path,
    session_id: &str,
    event: &SessionEvent,
) -> Result<(), SessionStoreError> {
    let Some(path) = session_file_path(root, session_id) else {
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = open_append_0600(&path)?;
    // Lock spans seq computation AND the write -- see `lock_exclusive`.
    lock_exclusive(&file)?;
    let seq = count_lines(&path)?;
    let mut stamped = event.clone();
    stamped.seq = seq;
    let mut line = serde_json::to_string(&stamped)?;
    line.push('\n');
    file.write_all(line.as_bytes())?;
    Ok(())
}

/// One-linear-pass query filter: every field is optional and AND-combined.
/// `since`/`until` compare directly against the RFC3339 `ts` string --
/// lexicographic order matches chronological order for that fixed-width
/// format, so no date-parsing dependency is needed.
#[derive(Debug, Default, Clone, Copy)]
pub struct QueryFilter<'a> {
    pub substring: Option<&'a str>,
    pub category: Option<&'a str>,
    pub since: Option<&'a str>,
    pub until: Option<&'a str>,
}

/// Validates `session_id`, then linear-scans the ONE session file, applying
/// `filter` in a single pass. Results stay in file/append (timeline) order.
/// An invalid `session_id` or a missing file both yield an empty result,
/// never an error and never a read of any other file -- this is the only
/// read path over session data, so it is also the structural privacy
/// boundary: there is no way to reach a session other than the one named
/// by `session_id`.
pub fn query_events(
    root: &Path,
    session_id: &str,
    filter: QueryFilter<'_>,
) -> Result<Vec<SessionEvent>, SessionStoreError> {
    let Some(path) = session_file_path(root, session_id) else {
        return Ok(Vec::new());
    };
    if !path.exists() {
        return Ok(Vec::new());
    }
    let file = std::fs::File::open(&path)?;
    let mut reader = BufReader::new(file);
    let mut out = Vec::new();
    let sub_lower = filter.substring.map(|s| s.to_ascii_lowercase());
    let sub_escape_safe = filter
        .substring
        .is_some_and(|s| !s.bytes().any(|b| b == b'"' || b == b'\\' || b < 0x20));
    let mut raw = Vec::new();
    let mut line_no: usize = 0;
    loop {
        raw.clear();
        // read_until (not lines()) so a torn/non-UTF-8 line is skipped below
        // instead of erroring the whole read -- a real I/O fault still propagates.
        let read = reader.read_until(b'\n', &mut raw)?;
        if read == 0 {
            break;
        }
        let this_line_no = line_no;
        line_no += 1;
        if raw.last() == Some(&b'\n') {
            raw.pop();
        }
        if raw.last() == Some(&b'\r') {
            raw.pop();
        }
        let line = match std::str::from_utf8(&raw) {
            Ok(s) => s,
            Err(err) => {
                eprintln!(
                    "iai_tools_session: skipping invalid utf8 line {} in {}: {err}",
                    this_line_no + 1,
                    path.display()
                );
                continue;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        // event.data is the sole membership authority; this prefilter only skips
        // deserialize work. Escape-unsafe needles must fall through here, or a
        // needle hidden by JSON escaping would produce a false negative.
        if let Some(sl) = &sub_lower {
            if sub_escape_safe && !line.to_ascii_lowercase().contains(sl.as_str()) {
                continue;
            }
        }
        // One corrupt line (partial write, disk fault) must not brick the
        // rest of the session's history -- skip it, but never silently: a
        // stderr signal keeps this fail-loud rather than a quiet data drop.
        let event: SessionEvent = match serde_json::from_str(line) {
            Ok(e) => e,
            Err(err) => {
                eprintln!(
                    "iai_tools_session: skipping unparseable line {} in {}: {err}",
                    this_line_no + 1,
                    path.display()
                );
                continue;
            }
        };
        if let Some(cat) = filter.category {
            if event.category != cat {
                continue;
            }
        }
        if let Some(sl) = &sub_lower {
            if !event.data.to_ascii_lowercase().contains(sl.as_str()) {
                continue;
            }
        }
        if let Some(since) = filter.since {
            if event.ts.as_str() < since {
                continue;
            }
        }
        if let Some(until) = filter.until {
            if event.ts.as_str() > until {
                continue;
            }
        }
        out.push(event);
    }
    Ok(out)
}

/// Loads every event in the validated session's file, unfiltered -- the
/// snapshot-load path. Goes through the same choke point as `query_events`.
pub fn load_all(root: &Path, session_id: &str) -> Result<Vec<SessionEvent>, SessionStoreError> {
    query_events(root, session_id, QueryFilter::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(session_id: &str, ts: &str, category: &str, data: &str) -> SessionEvent {
        SessionEvent {
            seq: 0,
            session_id: session_id.to_string(),
            ts: ts.to_string(),
            category: category.to_string(),
            event_type: "Edit".to_string(),
            data: data.to_string(),
            priority: 2,
            source_hook: "PostToolUse".to_string(),
            data_hash: super::super::data_hash(data),
        }
    }

    #[test]
    fn substring_category_and_time_filters_return_expected_rows_in_timeline_order() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-filters";

        append_event(root, sid, &event(sid, "2026-01-01T00:00:00Z", "file", "src/Alpha.rs")).unwrap();
        append_event(root, sid, &event(sid, "2026-01-02T00:00:00Z", "error", "boom")).unwrap();
        append_event(root, sid, &event(sid, "2026-01-03T00:00:00Z", "file", "src/beta.rs")).unwrap();

        // Case-insensitive substring match.
        let by_substring = query_events(
            root,
            sid,
            QueryFilter {
                substring: Some("ALPHA"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(by_substring.len(), 1);
        assert_eq!(by_substring[0].data, "src/Alpha.rs");

        // Exact category match, timeline order preserved across two hits.
        let by_category = query_events(
            root,
            sid,
            QueryFilter {
                category: Some("file"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(by_category.len(), 2);
        assert_eq!(by_category[0].data, "src/Alpha.rs");
        assert_eq!(by_category[1].data, "src/beta.rs");

        // Time range excludes the first and last event.
        let by_time = query_events(
            root,
            sid,
            QueryFilter {
                since: Some("2026-01-02T00:00:00Z"),
                until: Some("2026-01-02T23:59:59Z"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(by_time.len(), 1);
        assert_eq!(by_time[0].data, "boom");
    }

    #[test]
    fn cross_session_query_never_returns_the_other_sessions_events() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        append_event(
            root,
            "sess-a",
            &event("sess-a", "2026-01-01T00:00:00Z", "file", "a-only.rs"),
        )
        .unwrap();
        append_event(
            root,
            "sess-b",
            &event("sess-b", "2026-01-01T00:00:00Z", "file", "b-only.rs"),
        )
        .unwrap();

        let a_events = query_events(root, "sess-a", QueryFilter::default()).unwrap();
        assert_eq!(a_events.len(), 1);
        assert!(a_events.iter().all(|e| e.session_id == "sess-a"));
        assert!(a_events.iter().all(|e| e.data != "b-only.rs"));

        let b_events = query_events(root, "sess-b", QueryFilter::default()).unwrap();
        assert_eq!(b_events.len(), 1);
        assert!(b_events.iter().all(|e| e.session_id == "sess-b"));
        assert!(b_events.iter().all(|e| e.data != "a-only.rs"));
    }

    #[test]
    fn double_quote_needle_still_matches_escape_unsafe_row() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-quote";

        append_event(root, sid, &event(sid, "2026-01-01T00:00:00Z", "note", "he said \"hi\" today")).unwrap();

        let found = query_events(
            root,
            sid,
            QueryFilter {
                substring: Some("\"hi\""),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].data, "he said \"hi\" today");
    }

    #[test]
    fn backslash_needle_still_matches_escape_unsafe_row() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-backslash";

        append_event(root, sid, &event(sid, "2026-01-01T00:00:00Z", "note", "a\\b path")).unwrap();

        let found = query_events(
            root,
            sid,
            QueryFilter {
                substring: Some("a\\b"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].data, "a\\b path");
    }

    #[test]
    fn escape_safe_absent_needle_returns_empty_with_no_error() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-absent";

        append_event(root, sid, &event(sid, "2026-01-01T00:00:00Z", "note", "src/alpha.rs")).unwrap();

        let found = query_events(
            root,
            sid,
            QueryFilter {
                substring: Some("zzz"),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn substring_query_ignores_matches_outside_the_data_field() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-authority";

        append_event(root, sid, &event(sid, "2026-01-01T00:00:00Z", "note", "src/alpha.rs")).unwrap();

        let found = query_events(
            root,
            sid,
            QueryFilter {
                substring: Some("PostToolUse"),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(found.is_empty());
    }

    #[test]
    fn concurrent_appends_to_same_session_never_interleave_or_duplicate_seq() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().to_path_buf();
        let sid = "sess-concurrent";
        const THREADS: usize = 8;
        const PER_THREAD: usize = 25;

        let handles: Vec<_> = (0..THREADS)
            .map(|t| {
                let root = root.clone();
                std::thread::spawn(move || {
                    for i in 0..PER_THREAD {
                        let ev = event(
                            sid,
                            "2026-01-01T00:00:00Z",
                            "note",
                            &format!("thread-{t}-item-{i}"),
                        );
                        append_event(&root, sid, &ev).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let path = root.join("sessions").join(format!("{sid}.jsonl"));
        let content = std::fs::read_to_string(&path).unwrap();
        let mut seqs: Vec<u64> = Vec::new();
        for line in content.lines() {
            let parsed: SessionEvent = serde_json::from_str(line)
                .unwrap_or_else(|e| panic!("line failed to parse as valid JSON: {e}: {line:?}"));
            seqs.push(parsed.seq);
        }

        let total = THREADS * PER_THREAD;
        assert_eq!(seqs.len(), total, "every appended line must survive as its own row");
        seqs.sort_unstable();
        let expected: Vec<u64> = (0..total as u64).collect();
        assert_eq!(seqs, expected, "seq values must be unique and contiguous 0..N, no duplicates");
    }

    #[test]
    fn query_events_skips_a_corrupt_middle_line_and_returns_the_surviving_valid_events() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-corrupt";

        append_event(root, sid, &event(sid, "2026-01-01T00:00:00Z", "note", "first")).unwrap();
        append_event(root, sid, &event(sid, "2026-01-02T00:00:00Z", "note", "second")).unwrap();
        append_event(root, sid, &event(sid, "2026-01-03T00:00:00Z", "note", "third")).unwrap();

        let path = root.join("sessions").join(format!("{sid}.jsonl"));
        let mut lines: Vec<String> = std::fs::read_to_string(&path)
            .unwrap()
            .lines()
            .map(|l| l.to_string())
            .collect();
        assert_eq!(lines.len(), 3);
        lines.insert(1, "{this is not valid json".to_string());
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let found = query_events(root, sid, QueryFilter::default()).unwrap();
        assert_eq!(found.len(), 3, "the one corrupt line must be skipped, not fail the whole read");
        assert_eq!(found[0].data, "first");
        assert_eq!(found[1].data, "second");
        assert_eq!(found[2].data, "third");
    }

    #[test]
    fn query_events_skips_an_invalid_utf8_line_and_returns_the_surviving_valid_events() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let sid = "sess-badutf8";

        append_event(root, sid, &event(sid, "2026-01-01T00:00:00Z", "note", "first")).unwrap();
        append_event(root, sid, &event(sid, "2026-01-02T00:00:00Z", "note", "second")).unwrap();

        let path = root.join("sessions").join(format!("{sid}.jsonl"));
        let bytes = std::fs::read(&path).unwrap();
        let first_newline = bytes.iter().position(|&b| b == b'\n').unwrap();
        let mut rebuilt = bytes[..=first_newline].to_vec();
        // A lone continuation byte is invalid UTF-8 on its own -- simulates a
        // crash-torn line that split a multi-byte codepoint.
        rebuilt.push(0xFF);
        rebuilt.push(b'\n');
        rebuilt.extend_from_slice(&bytes[first_newline + 1..]);
        std::fs::write(&path, &rebuilt).unwrap();

        let found = query_events(root, sid, QueryFilter::default()).unwrap();
        assert_eq!(found.len(), 2, "the invalid-utf8 line must be skipped, not fail the whole read");
        assert_eq!(found[0].data, "first");
        assert_eq!(found[1].data, "second");
    }

    #[test]
    fn traversal_shaped_session_id_resolves_to_no_file_ever_outside_sessions_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        assert_eq!(session_file_path(root, "../../etc/passwd"), None);
        assert_eq!(session_file_path(root, "foo/bar"), None);

        let queried = query_events(root, "../../etc/passwd", QueryFilter::default()).unwrap();
        assert!(queried.is_empty());

        // A valid id always resolves under `{root}/sessions/`.
        let valid = session_file_path(root, "sess-valid").unwrap();
        assert_eq!(valid.parent(), Some(root.join("sessions").as_path()));
    }
}

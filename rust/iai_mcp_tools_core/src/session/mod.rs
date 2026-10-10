//! Own append-only per-session event pipeline: capture -> store -> query ->
//! snapshot. Fully independent of the brain (no capture.py/Hippo/lilli/embedder
//! coupling anywhere in this module tree).

pub mod extract;
pub mod snapshot;
pub mod store;

use serde::{Deserialize, Serialize};

/// One captured event. `data_hash` is recorded metadata only in this slice --
/// append does not suppress or dedup on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub seq: u64,
    pub session_id: String,
    pub ts: String,
    pub category: String,
    #[serde(rename = "type")]
    pub event_type: String,
    pub data: String,
    pub priority: u8,
    pub source_hook: String,
    pub data_hash: String,
}

/// Current UTC time as RFC3339 (`YYYY-MM-DDTHH:MM:SSZ`), computed from
/// `SystemTime` with no external time crate.
fn rfc3339_now() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs() as i64;
    let days = secs.div_euclid(86400);
    let secs_of_day = secs.rem_euclid(86400);
    let (y, m, d) = civil_from_days(days);
    let hh = secs_of_day / 3600;
    let mm = (secs_of_day % 3600) / 60;
    let ss = secs_of_day % 60;
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Hinnant civil_from_days: days-since-unix-epoch -> proleptic-Gregorian (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

/// Lowercase hex CRC32 of `data` -- not a cryptographic hash, recorded metadata only.
fn data_hash(data: &str) -> String {
    format!("{:08x}", crc32fast::hash(data.as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_event_round_trips_through_append_query_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let hook = extract::HookInput {
            tool_name: Some("Edit".to_string()),
            file_path: Some("src/foo.rs".to_string()),
            session_id: Some("sess-abc123".to_string()),
            source_hook: Some("PostToolUse".to_string()),
            ..Default::default()
        };
        let event = extract::classify(&hook).expect("Edit hook classifies to a file event");
        assert_eq!(event.category, "file");

        store::append_event(root, "sess-abc123", &event).unwrap();

        let jsonl_path = root.join("sessions").join("sess-abc123.jsonl");
        assert!(jsonl_path.exists());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(&jsonl_path).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode, 0o600);
        }

        let found = store::query_events(
            root,
            "sess-abc123",
            store::QueryFilter {
                substring: Some("foo.rs"),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].category, "file");
        assert_eq!(found[0].seq, 0);

        let snap = snapshot::build_snapshot(&found);
        assert!(snap.contains("file"));
        assert!(snap.contains("--session sess-abc123 --category file"));
    }

    #[test]
    fn classify_returns_none_on_malformed_input_without_panicking() {
        let empty = extract::HookInput::default();
        assert!(extract::classify(&empty).is_none());

        let wrong_tool = extract::HookInput {
            tool_name: Some("Bash".to_string()),
            file_path: Some("src/foo.rs".to_string()),
            ..Default::default()
        };
        assert!(extract::classify(&wrong_tool).is_none());

        let empty_path = extract::HookInput {
            tool_name: Some("Edit".to_string()),
            file_path: Some(String::new()),
            ..Default::default()
        };
        assert!(extract::classify(&empty_path).is_none());
    }

    #[test]
    fn invalid_session_id_is_rejected_not_rewritten() {
        assert_eq!(store::validate_session_id("foo/bar"), None);
        assert_eq!(store::validate_session_id("foobar"), Some("foobar"));
        assert_eq!(store::validate_session_id(""), None);
        assert_eq!(store::validate_session_id(&"a".repeat(65)), None);
        assert_eq!(
            store::validate_session_id(&"a".repeat(64)),
            Some("a".repeat(64).as_str())
        );
    }

    #[test]
    fn distinct_raw_ids_can_never_collide_on_one_file() {
        // Traversal-shaped id vs. a plain id that looks similar once stripped:
        // `foo/bar` must be rejected outright, never coerced into `foobar`.
        assert_eq!(store::validate_session_id("foo/bar"), None);
        assert_eq!(store::validate_session_id("foobar"), Some("foobar"));

        // Two ids sharing a 64-char prefix but differing after it: the longer
        // one is over the length allowlist and is rejected, so it can never
        // land on the same file as the 64-char prefix id.
        let prefix = "b".repeat(64);
        let longer = format!("{prefix}c");
        assert_eq!(store::validate_session_id(&prefix), Some(prefix.as_str()));
        assert_eq!(store::validate_session_id(&longer), None);
    }

    #[test]
    fn invalid_session_id_touches_no_file_on_every_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();

        let event = SessionEvent {
            seq: 0,
            session_id: "foo/bar".to_string(),
            ts: rfc3339_now(),
            category: "file".to_string(),
            event_type: "Edit".to_string(),
            data: "x".to_string(),
            priority: 2,
            source_hook: "PostToolUse".to_string(),
            data_hash: data_hash("x"),
        };
        store::append_event(root, "foo/bar", &event).unwrap();
        assert!(!root.join("sessions").exists());

        let queried =
            store::query_events(root, "foo/bar", store::QueryFilter::default()).unwrap();
        assert!(queried.is_empty());

        let snapshot_events = store::load_all(root, "foo/bar").unwrap();
        assert!(snapshot_events.is_empty());
        assert!(!root.join("sessions").exists());
    }
}

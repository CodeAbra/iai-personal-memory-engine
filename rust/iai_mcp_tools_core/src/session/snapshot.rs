//! Pure post-compaction snapshot builder. No I/O and no session_id
//! validation here -- the caller has already loaded validated events
//! through the store's choke point, so this function structurally cannot
//! reach any session other than the one whose events it was handed.

use std::collections::BTreeMap;

use super::SessionEvent;

/// Cap on how many characters of a summarized item's `data` are shown
/// inline -- readability only. The follow-up query below it retrieves the
/// full, untruncated event, so this cap never limits what can be recovered.
const SUMMARY_ITEM_MAX_CHARS: usize = 80;

/// How many of a category's most recent items get an inline summary line.
const SUMMARY_RECENT_ITEMS: usize = 2;

/// Groups `events` by category and emits, for each non-empty category, one
/// summary section (count plus the most recent one or two items' `data`,
/// text-shortened for readability) plus exactly one runnable follow-up
/// query -- never a raw dump of the events themselves, and never a
/// byte-budget truncation of which events exist. Category order is
/// deterministic (sorted) so output is stable across calls on the same
/// input; within a category, the summarized items are the most recent
/// first, matching the timeline order the events arrived in.
pub fn build_snapshot(events: &[SessionEvent]) -> String {
    let mut by_category: BTreeMap<&str, Vec<&SessionEvent>> = BTreeMap::new();
    for event in events {
        by_category
            .entry(event.category.as_str())
            .or_default()
            .push(event);
    }
    let session_id = events.first().map(|e| e.session_id.as_str()).unwrap_or("");

    let mut out = String::new();
    for (category, evs) in &by_category {
        out.push_str(&format!("{category}: {} event(s)\n", evs.len()));
        for recent in evs.iter().rev().take(SUMMARY_RECENT_ITEMS) {
            out.push_str(&format!(
                "  - {}\n",
                shorten(&recent.data, SUMMARY_ITEM_MAX_CHARS)
            ));
        }
        out.push_str(&format!(
            "  iai_tools_session search --session {session_id} --category {category}\n"
        ));
    }
    out
}

/// Shortens `s` to at most `max_chars` characters for the human-readable
/// summary line only -- the emitted follow-up query is unaffected and
/// always retrieves the full stored event.
fn shorten(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max_chars).collect();
    format!("{truncated}...")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(session_id: &str, category: &str, data: &str) -> SessionEvent {
        SessionEvent {
            seq: 0,
            session_id: session_id.to_string(),
            ts: "2026-01-01T00:00:00Z".to_string(),
            category: category.to_string(),
            event_type: "Edit".to_string(),
            data: data.to_string(),
            priority: 2,
            source_hook: "PostToolUse".to_string(),
            data_hash: String::new(),
        }
    }

    #[test]
    fn one_section_per_non_empty_category_with_exact_follow_up_query_shape() {
        let events = vec![
            event("sess-golden", "file", "src/one.rs"),
            event("sess-golden", "error", "boom"),
            event("sess-golden", "file", "src/two.rs"),
        ];
        let snap = build_snapshot(&events);

        // One section per non-empty category, no section for an absent category.
        assert!(snap.contains("file: 2 event(s)"));
        assert!(snap.contains("error: 1 event(s)"));
        assert!(!snap.contains("decision:"));
        assert!(!snap.contains("task:"));

        // Exact runnable follow-up query shape, one per non-empty category.
        assert!(snap.contains("iai_tools_session search --session sess-golden --category file"));
        assert!(snap.contains("iai_tools_session search --session sess-golden --category error"));

        // Underlying data is never dropped: both file items are named, not just the count.
        assert!(snap.contains("src/one.rs"));
        assert!(snap.contains("src/two.rs"));
    }

    #[test]
    fn empty_events_produce_no_sections() {
        assert_eq!(build_snapshot(&[]), "");
    }

    #[test]
    fn long_item_data_is_shortened_only_in_the_summary_text() {
        let long_data = "x".repeat(200);
        let events = vec![event("sess-long", "file", &long_data)];
        let snap = build_snapshot(&events);

        assert!(snap.contains("..."));
        assert!(!snap.contains(&long_data));
        // The follow-up query targets the whole category, not a truncated
        // id -- running it retrieves the full untruncated event.
        assert!(snap.contains("iai_tools_session search --session sess-long --category file"));
    }
}

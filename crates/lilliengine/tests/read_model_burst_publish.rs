//! Regression fence (own test binary, own process): a burst of
//! single-row commits spaced under the publish-throttle interval must not
//! force a refreshing reader to pay a full-corpus `IdIndex::ensure_built`
//! rebuild when every demanded read-model component is already built and
//! incrementally maintained on the writer. Kept in its own file so this
//! test's `LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS` override is never raced
//! against `tests/conn.rs`'s own overrides of the same process-wide
//! `OnceLock` — cargo compiles each integration test file into its own
//! process, so a shared statically-cached env-derived value can only ever
//! reflect whichever test in a given process reads it first.

use lillibrain::Value;
use lilliengine::conn::Connection;
use tempfile::tempdir;

fn t(s: &str) -> Value {
    Value::Text(s.to_string())
}

const DDL: &str = "CREATE TABLE IF NOT EXISTS recs ( \
    vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
    pending INTEGER , created_at TEXT , payload TEXT )";

const ID_SQL: &str = "SELECT id FROM recs WHERE id IN (?, ?, ?)";

#[test]
fn id_index_survives_burst_of_sub_interval_spaced_single_row_commits() {
    // A throttle interval far longer than any realistic test-loop spacing:
    // a naive commit-density gate would suppress every publish in the burst
    // below, so this proves the fix bypasses the gate once every demanded
    // component is already built, rather than happening to sneak under a
    // permissive interval.
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "60000");

    let dir = tempdir().unwrap();
    let path = dir.path().join("burst.lilli").to_str().unwrap().to_string();
    let mut writer = Connection::open(&path, 384).unwrap();
    writer.execute(DDL, vec![]).unwrap();
    writer
        .execute("CREATE INDEX idx_recs_pending ON recs (pending)", vec![])
        .unwrap();

    writer.execute("BEGIN", vec![]).unwrap();
    for i in 0..800 {
        writer
            .execute(
                "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?)",
                vec![
                    t(&format!("id-{i:05}")),
                    Value::Int(1),
                    t(&format!("2026-01-{:02}T00:00:00", (i % 28) + 1)),
                    t(&format!("payload-{i}")),
                ],
            )
            .unwrap();
    }
    writer.execute("COMMIT", vec![]).unwrap();

    // Warm the writer's own id_caches entry before the burst -- the
    // production shape (a long-lived writer connection that has already run
    // an id lookup, e.g. the pending-embed reembed path's rowid probe).
    writer
        .execute(ID_SQL, vec![t("id-00000"), t("id-00001"), t("id-00002")])
        .unwrap();

    let mut ro = Connection::open_read_only(&path, 384).unwrap();
    ro.execute(ID_SQL, vec![t("id-00000"), t("id-00001"), t("id-00002")])
        .unwrap();

    // Burst: single-row INSERT + COMMIT in a tight loop, mirroring
    // drain_capture_backlog draining N queued captures in one wake-spool-sweep
    // pass -- each capture_turn -> insert_pending_row is its own transaction,
    // with no artificial delay between them.
    for i in 0..12 {
        writer.execute("BEGIN", vec![]).unwrap();
        writer
            .execute(
                "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?)",
                vec![
                    t(&format!("burst-{i:03}")),
                    Value::Int(1),
                    t("2026-02-01T00:00:00"),
                    t("p"),
                ],
            )
            .unwrap();
        writer.execute("COMMIT", vec![]).unwrap();
    }

    let advanced = ro.refresh_read_view().unwrap();
    assert!(
        advanced,
        "refresh_read_view reported no snapshot advance after the burst"
    );

    ro.reset_cells_visited_count();
    let mut cur = ro
        .execute(
            "SELECT id FROM recs WHERE id IN (?, ?, ?)",
            vec![t("id-00000"), t("burst-000"), t("burst-011")],
        )
        .unwrap();
    let rows = cur.fetchall();
    let cells = ro.cells_visited_count();
    assert_eq!(
        rows.len(),
        3,
        "completeness twin: pre-burst AND burst-committed rows (first and last \
         of the burst) must all be visible post-refresh"
    );
    assert!(
        cells <= 8,
        "post-burst-refresh id read cost {cells} cells -- a burst of commits spaced \
         under the publish-throttle interval forced a full-corpus IdIndex::ensure_built \
         rebuild instead of adopting the writer's already-built, incrementally-maintained \
         id index"
    );
}

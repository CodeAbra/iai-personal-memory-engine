//! Read-only snapshot refresh: `refresh_read_view` re-arms an open reader to
//! the newest committed state in place — the freshness path that replaces the
//! close + reopen (and its whole-table lazy index rebuild) the RO pool
//! otherwise pays after every commit.

use lillibrain::Value;
use lilliengine::conn::Connection;
use tempfile::tempdir;

fn setup_writer(path: &str) -> Connection {
    let mut conn = Connection::open(path, 0).unwrap();
    conn.execute("PRAGMA journal_mode=WAL", vec![]).unwrap();
    conn.execute("CREATE TABLE edges (src TEXT, dst TEXT, w INTEGER)", vec![])
        .unwrap();
    conn.execute("CREATE INDEX idx_edges_src ON edges (src)", vec![])
        .unwrap();
    for i in 0..60 {
        conn.execute(
            "INSERT INTO edges (src, dst, w) VALUES (?, ?, ?)",
            vec![
                Value::Text(format!("s{}", i % 10)),
                Value::Text(format!("d{i}")),
                Value::Int(i),
            ],
        )
        .unwrap();
    }
    conn.execute(
        "SELECT dst FROM edges WHERE src IN (?)",
        vec![Value::Text("s1".to_string())],
    )
    .unwrap();
    assert!(
        conn.col_index_ready("edges"),
        "writer probe must build the index"
    );
    conn
}

fn count_src(conn: &mut Connection, src: &str) -> usize {
    let mut cur = conn
        .execute(
            "SELECT dst FROM edges WHERE src IN (?)",
            vec![Value::Text(src.to_string())],
        )
        .unwrap();
    cur.fetchall().len()
}

fn insert_committed(writer: &mut Connection, src: &str, dst: &str) {
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO edges (src, dst, w) VALUES (?, ?, ?)",
            vec![
                Value::Text(src.to_string()),
                Value::Text(dst.to_string()),
                Value::Int(0),
            ],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
}

#[test]
fn refresh_moves_reader_to_newest_committed_snapshot() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);

    let mut reader = Connection::open_read_only(path, 0).unwrap();
    let before = count_src(&mut reader, "s1");

    insert_committed(&mut writer, "s1", "d-fresh");

    // Snapshot isolation until refresh: the reader's view must not move.
    assert_eq!(count_src(&mut reader, "s1"), before);

    assert!(
        reader.refresh_read_view().unwrap(),
        "snapshot must report moved"
    );
    assert_eq!(count_src(&mut reader, "s1"), before + 1);
}

#[test]
fn refresh_without_new_commits_is_a_noop() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();
    let _writer = setup_writer(path);

    let mut reader = Connection::open_read_only(path, 0).unwrap();
    let before = count_src(&mut reader, "s1");
    assert!(
        !reader.refresh_read_view().unwrap(),
        "no commit — no movement"
    );
    assert_eq!(count_src(&mut reader, "s1"), before);
}

#[test]
fn refresh_adopts_writer_published_pair_at_new_generation() {
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);

    // First reader misses (nothing published yet) and records demand.
    let mut reader = Connection::open_read_only(path, 0).unwrap();
    assert!(!reader.col_index_ready("edges"));

    // Demand is live, so this commit publishes the writer's built pair at
    // the post-commit generation.
    insert_committed(&mut writer, "s2", "d-pub");

    assert!(reader.refresh_read_view().unwrap());
    assert!(
        reader.col_index_ready("edges"),
        "refresh must adopt the published pair — no lazy rebuild"
    );
    // The adopted index must serve the refreshed data exactly.
    assert_eq!(count_src(&mut reader, "s2"), 7);
}

#[test]
fn refresh_sees_tables_created_after_open() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);

    let mut reader = Connection::open_read_only(path, 0).unwrap();
    assert!(reader.execute("SELECT w FROM latecomer", vec![]).is_err());

    writer
        .execute("CREATE TABLE latecomer (w INTEGER)", vec![])
        .unwrap();
    writer
        .execute("INSERT INTO latecomer (w) VALUES (?)", vec![Value::Int(7)])
        .unwrap();

    assert!(reader.refresh_read_view().unwrap());
    let mut cur = reader.execute("SELECT w FROM latecomer", vec![]).unwrap();
    assert_eq!(cur.fetchall().len(), 1);
}

#[test]
fn refresh_rejected_on_a_read_write_connection() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);
    assert!(writer.refresh_read_view().is_err());
}

#[test]
fn repeated_commit_refresh_cycles_stay_exact() {
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);
    let mut reader = Connection::open_read_only(path, 0).unwrap();

    let base = count_src(&mut reader, "s3");
    for round in 0..25 {
        insert_committed(&mut writer, "s3", &format!("d-round-{round}"));
        assert!(reader.refresh_read_view().unwrap());
        assert_eq!(
            count_src(&mut reader, "s3"),
            base + round + 1,
            "round {round}: refreshed reader must serve every committed row exactly once"
        );
    }
}

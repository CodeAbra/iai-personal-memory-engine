//! Writer-maintained read models: an insert-only writer must build (once)
//! and publish its col/id index pair + committed row count at commit, and a
//! refreshing reader must adopt them — never pay a per-query whole-table
//! scan. Without the ensure-build, a writer that only inserts has nothing
//! to publish and every reader refresh after every commit degrades every
//! indexed read (and bare COUNT) to a leaf-chain walk.

use lilliengine::conn::Connection;
use lillibrain::Value;
use tempfile::tempdir;

fn setup_writer(path: &str) -> Connection {
    let mut conn = Connection::open(path, 0).unwrap();
    conn.execute("PRAGMA journal_mode=WAL", vec![]).unwrap();
    conn.execute(
        "CREATE TABLE records (id TEXT, tier TEXT, body TEXT)",
        vec![],
    )
    .unwrap();
    conn.execute("CREATE INDEX idx_records_tier ON records (tier)", vec![])
        .unwrap();
    for i in 0..40 {
        conn.execute(
            "INSERT INTO records (id, tier, body) VALUES (?, ?, ?)",
            vec![
                Value::Text(format!("id-{i}")),
                Value::Text("episodic".to_string()),
                Value::Text(format!("row {i}")),
            ],
        )
        .unwrap();
    }
    conn
}

fn insert_committed(writer: &mut Connection, i: i64) {
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO records (id, tier, body) VALUES (?, ?, ?)",
            vec![
                Value::Text(format!("late-{i}")),
                Value::Text("episodic".to_string()),
                Value::Text(format!("late row {i}")),
            ],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
}

#[test]
fn insert_only_writer_publishes_on_commit_and_reader_adopts() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);
    assert!(
        !writer.col_index_ready("records"),
        "an insert-only writer starts with no built index"
    );

    // A reader open records adoption demand for the col-indexed table.
    let mut reader = Connection::open_read_only(path, 0).unwrap();

    // The commit epilogue must ensure-build and publish even though the
    // writer never ran an indexed read of its own.
    insert_committed(&mut writer, 1);
    assert!(
        writer.col_index_ready("records"),
        "commit with reader demand must ensure-build the writer's col index"
    );
    assert!(
        writer.id_index_ready("records"),
        "commit with reader demand must ensure-build the writer's id index"
    );

    // The refreshing reader adopts the published pair — ready BEFORE any
    // query runs on it.
    reader.refresh_read_view().unwrap();
    assert!(
        reader.col_index_ready("records"),
        "refreshed reader must adopt the writer-published col index"
    );
    assert!(
        reader.id_index_ready("records"),
        "refreshed reader must adopt the writer-published id index"
    );

    // The adopted read model is correct.
    let mut cur = reader
        .execute(
            "SELECT body FROM records WHERE id IN (?)",
            vec![Value::Text("late-1".to_string())],
        )
        .unwrap();
    assert_eq!(cur.fetchall().len(), 1);
}

#[test]
fn adopted_reader_serves_bare_count_without_a_walk() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);
    let mut reader = Connection::open_read_only(path, 0).unwrap();
    insert_committed(&mut writer, 1);
    reader.refresh_read_view().unwrap();

    reader.reset_cells_visited_count();
    let mut cur = reader.execute("SELECT COUNT(*) FROM records", vec![]).unwrap();
    let rows = cur.fetchall();
    assert_eq!(rows[0].get_index(0), Some(&Value::Int(41)));
    assert_eq!(
        reader.cells_visited_count(),
        0,
        "adopted row count must serve bare COUNT(*) without a leaf walk"
    );
}

#[test]
fn writer_without_reader_demand_skips_the_build() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);
    insert_committed(&mut writer, 1);
    assert!(
        !writer.col_index_ready("records"),
        "a one-shot writer with no reader traffic must not pay the build"
    );
}

#[test]
fn publish_read_models_primes_readers_before_any_commit() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);
    writer.publish_read_models().unwrap();
    assert!(writer.col_index_ready("records"));
    assert!(writer.id_index_ready("records"));

    let reader = Connection::open_read_only(path, 0).unwrap();
    assert!(
        reader.col_index_ready("records"),
        "a reader opened after publish_read_models must adopt at open"
    );
    assert!(reader.id_index_ready("records"));
}

#[test]
fn non_indexed_update_keeps_reader_indexes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);
    writer.publish_read_models().unwrap();
    let mut reader = Connection::open_read_only(path, 0).unwrap();
    assert!(reader.col_index_ready("records"));
    assert!(reader.id_index_ready("records"));

    // Reinforcement shape: UPDATE touching only non-indexed columns. The
    // write-generation must not move — the reader's refresh keeps its
    // adopted indexes instead of dropping and re-adopting per write.
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "UPDATE records SET body = ? WHERE id = ?",
            vec![
                Value::Text("reinforced".to_string()),
                Value::Text("id-3".to_string()),
            ],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    reader.refresh_read_view().unwrap();
    assert!(
        reader.col_index_ready("records"),
        "a non-indexed-column UPDATE must not invalidate reader col indexes"
    );
    assert!(
        reader.id_index_ready("records"),
        "a non-indexed-column UPDATE must not invalidate reader id indexes"
    );
    // The refreshed reader still sees the new payload (pager fence, not
    // generation, governs data freshness).
    let mut cur = reader
        .execute(
            "SELECT body FROM records WHERE id IN (?)",
            vec![Value::Text("id-3".to_string())],
        )
        .unwrap();
    let rows = cur.fetchall();
    assert_eq!(rows[0].get_index(0), Some(&Value::Text("reinforced".to_string())));

    // An INDEXED-column UPDATE must still fence: generation moves, the
    // reader drops and re-adopts the republished pair.
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "UPDATE records SET tier = ? WHERE id = ?",
            vec![
                Value::Text("semantic".to_string()),
                Value::Text("id-3".to_string()),
            ],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    reader.refresh_read_view().unwrap();
    let mut cur = reader
        .execute(
            "SELECT id FROM records WHERE tier IN (?)",
            vec![Value::Text("semantic".to_string())],
        )
        .unwrap();
    assert_eq!(cur.fetchall().len(), 1, "indexed update must stay visible to indexed reads");
}

#[test]
fn filtered_count_serves_cached_after_first_walk() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let path = path.to_str().unwrap();
    let mut writer = setup_writer(path);

    let sql = "SELECT COUNT(*) FROM records WHERE tier = 'episodic'";
    let mut cur = writer.execute(sql, vec![]).unwrap();
    let first = cur.fetchall();
    assert_eq!(first[0].get_index(0), Some(&Value::Int(40)));

    writer.reset_cells_visited_count();
    let mut cur = writer.execute(sql, vec![]).unwrap();
    let second = cur.fetchall();
    assert_eq!(second[0].get_index(0), Some(&Value::Int(40)));
    assert_eq!(
        writer.cells_visited_count(),
        0,
        "repeated filtered COUNT at an unchanged generation must serve cached"
    );

    // A committed write moves the generation: the cache self-fences and the
    // recount reflects the new row.
    insert_committed(&mut writer, 7);
    let mut cur = writer.execute(sql, vec![]).unwrap();
    let third = cur.fetchall();
    assert_eq!(third[0].get_index(0), Some(&Value::Int(41)));

    // A parameterized filtered COUNT must bypass the cache entirely.
    let mut cur = writer
        .execute(
            "SELECT COUNT(*) FROM records WHERE tier = ?",
            vec![Value::Text("episodic".to_string())],
        )
        .unwrap();
    assert_eq!(cur.fetchall()[0].get_index(0), Some(&Value::Int(41)));
}

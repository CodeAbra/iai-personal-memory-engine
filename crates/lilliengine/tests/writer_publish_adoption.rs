//! Demand-driven writer index publication.
//!
//! Under a write-heavy window the committed write-generation moves on every
//! commit, so each read-only open misses the generation-keyed built-index
//! cache and pays a whole-table lazy build — the reopen storm that turns
//! sub-second reads into multi-second reads. The fix: a read-only adoption
//! MISS records demand, and the writer publishes its live built pair on each
//! successful untainted commit while demand is fresh, so the next reader at
//! the new generation adopts instead of building.

use lillibrain::Value;
use lilliengine::conn::Connection;
use tempfile::tempdir;

fn setup_writer(path: &str) -> Connection {
    let mut conn = Connection::open(path, 0).unwrap();
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
    // An indexed probe pays the writer's one lazy build; incremental
    // maintenance keeps the pair current across clean commits from here on.
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

#[test]
fn reader_adopts_writer_published_indexes_after_commit() {
    // Publishes are throttled per table (a per-row DML stream must not pay a
    // copy-on-write per statement); this test drives back-to-back commits and
    // asserts adoption after each, so disable the spacing.
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let path = path.to_str().unwrap();

    let mut writer = setup_writer(path);

    // Reader #1 at a generation nobody has published: adoption misses, the
    // reader records demand and would pay the lazy build itself.
    let r1 = Connection::open_read_only(path, 0).unwrap();
    assert!(
        !r1.col_index_ready("edges"),
        "no publication exists yet — the first reader must miss"
    );
    drop(r1);

    // The writer commits fresh work; demand is live, so the commit publishes
    // its (still-valid, incrementally maintained) built pair at the NEW
    // generation.
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO edges (src, dst, w) VALUES (?, ?, ?)",
            vec![
                Value::Text("s1".to_string()),
                Value::Text("d-new".to_string()),
                Value::Int(999),
            ],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();

    // Reader #2 opens at the post-commit generation and adopts immediately —
    // no probe, no lazy build.
    let mut r2 = Connection::open_read_only(path, 0).unwrap();
    assert!(
        r2.col_index_ready("edges"),
        "reader at the post-commit generation must adopt the writer's pair"
    );

    // Adopted index answers correctly, including the row from the very
    // commit that published it.
    assert_eq!(
        count_src(&mut r2, "s1"),
        7,
        "6 seeded s1 rows + the fresh one"
    );
    assert_eq!(count_src(&mut r2, "s9"), 6);

    // The writer keeps mutating after publishing: copy-on-write must keep
    // the reader's adopted snapshot intact.
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "DELETE FROM edges WHERE src IN (?)",
            vec![Value::Text("s1".to_string())],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    assert_eq!(
        count_src(&mut r2, "s1"),
        7,
        "reader's committed snapshot must not see writes after its open"
    );

    // A fresh reader at the newest generation adopts the post-delete pair.
    let mut r3 = Connection::open_read_only(path, 0).unwrap();
    assert!(r3.col_index_ready("edges"));
    assert_eq!(
        count_src(&mut r3, "s1"),
        0,
        "post-delete adoption reflects the delete"
    );
    assert_eq!(count_src(&mut r3, "s9"), 6);
}

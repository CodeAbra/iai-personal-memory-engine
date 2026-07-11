//! Ordered-index fast paths: the MIN/MAX index-seek and the predicate-aware
//! ordered LIMIT-k walk.
//!
//! Each test asserts byte-identical results between a connection with the
//! target column indexed (fast path) and one without (scan path) — a
//! differential oracle catching divergence in value, ordering, tie handling,
//! NULL handling, or output shape.

use lilliengine::conn::Connection;
use lillibrain::Value;
use tempfile::tempdir;

const DDL_INDEXED: &str = "CREATE TABLE IF NOT EXISTS records ( \
    vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
    tombstoned_at TEXT , embedding_pending INTEGER , created_at TEXT , n INTEGER )";

const IDX_VEC_LABEL: &str = "CREATE INDEX IF NOT EXISTS idx_records_vec_label \
    ON records(vec_label)";
const IDX_CREATED_AT: &str = "CREATE INDEX IF NOT EXISTS idx_records_created_at \
    ON records(created_at)";

fn t(s: &str) -> Value {
    Value::Text(s.to_string())
}

/// Open two identically-populated connections: one with the declared indexes
/// (fast paths eligible) and one without (scan path). Rows: vec_label 1..=n
/// auto-injected; every third row tombstoned; every fifth row pending;
/// created_at ascending except a NULL every seventh row.
fn open_pair(n: i64) -> (tempfile::TempDir, Connection, Connection) {
    let dir = tempdir().unwrap();
    let mk = |name: &str, with_indexes: bool| -> Connection {
        let path = dir.path().join(name);
        let mut conn = Connection::open(path.to_str().unwrap(), 384).unwrap();
        conn.execute(DDL_INDEXED, vec![]).unwrap();
        if with_indexes {
            conn.execute(IDX_VEC_LABEL, vec![]).unwrap();
            conn.execute(IDX_CREATED_AT, vec![]).unwrap();
        }
        for i in 1..=n {
            let tomb = if i % 3 == 0 {
                t("2026-01-01T00:00:00+00:00")
            } else {
                Value::Null
            };
            let pending = Value::Int(i64::from(i % 5 == 0));
            let created = if i % 7 == 0 {
                Value::Null
            } else {
                t(&format!("2026-01-01T00:00:{:02}+00:00", i % 60))
            };
            conn.execute(
                "INSERT INTO records (id, tombstoned_at, embedding_pending, created_at, n) \
                 VALUES (?, ?, ?, ?, ?)",
                vec![
                    t(&format!("id-{i}")),
                    tomb,
                    pending,
                    created,
                    Value::Int(i * 10),
                ],
            )
            .unwrap();
        }
        conn
    };
    let indexed = mk("indexed.lilli", true);
    let scan = mk("scan.lilli", false);
    (dir, indexed, scan)
}

/// Fetch every result row of `sql` as a comparable Vec of (name, Value) rows.
fn all_rows(conn: &mut Connection, sql: &str, params: Vec<Value>) -> Vec<Vec<(String, Value)>> {
    let mut cur = conn.execute(sql, params).unwrap();
    let mut out = Vec::new();
    while let Some(row) = cur.fetchone() {
        let pairs: Vec<(String, Value)> = row
            .columns
            .iter()
            .cloned()
            .zip(row.values.iter().cloned())
            .collect();
        out.push(pairs);
    }
    out
}

/// Assert `sql` produces byte-identical rows on the indexed and scan
/// connections.
fn assert_differential(sql: &str, params: Vec<Value>, indexed: &mut Connection, scan: &mut Connection) {
    let fast = all_rows(indexed, sql, params.clone());
    let slow = all_rows(scan, sql, params);
    assert_eq!(fast, slow, "indexed vs scan diverged for: {sql}");
}

#[test]
fn min_max_with_predicate_match_scan_path() {
    let (_dir, mut indexed, mut scan) = open_pair(60);
    for sql in [
        "SELECT MIN(vec_label) FROM records \
         WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0",
        "SELECT MAX(vec_label) FROM records \
         WHERE tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0",
        "SELECT MIN(vec_label) FROM records",
        "SELECT MAX(vec_label) FROM records",
        "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL",
        "SELECT MIN(created_at) FROM records WHERE tombstoned_at IS NULL",
        "SELECT MIN(created_at) FROM records",
    ] {
        assert_differential(sql, vec![], &mut indexed, &mut scan);
    }
}

#[test]
fn min_max_all_filtered_returns_null_like_scan() {
    let (_dir, mut indexed, mut scan) = open_pair(20);
    for sql in [
        "SELECT MIN(vec_label) FROM records WHERE n > 100000",
        "SELECT MAX(vec_label) FROM records WHERE n > 100000",
        "SELECT MAX(created_at) FROM records WHERE n > 100000",
    ] {
        assert_differential(sql, vec![], &mut indexed, &mut scan);
    }
}

#[test]
fn min_max_on_empty_table_matches_scan() {
    let (_dir, mut indexed, mut scan) = open_pair(0);
    for sql in [
        "SELECT MIN(vec_label) FROM records",
        "SELECT MAX(created_at) FROM records",
    ] {
        assert_differential(sql, vec![], &mut indexed, &mut scan);
    }
}

#[test]
fn min_max_stays_fresh_across_writes() {
    let (_dir, mut indexed, mut scan) = open_pair(20);
    // Warm the ordered index first.
    assert_differential("SELECT MAX(vec_label) FROM records", vec![], &mut indexed, &mut scan);
    // Insert / delete / update on BOTH, then re-compare: the built index must
    // reflect maintained state, never a stale snapshot.
    for conn in [&mut indexed, &mut scan] {
        conn.execute(
            "INSERT INTO records (id, tombstoned_at, embedding_pending, created_at, n) \
             VALUES (?, ?, ?, ?, ?)",
            vec![t("id-new"), Value::Null, Value::Int(0), t("2026-02-01T00:00:00+00:00"), Value::Int(999)],
        )
        .unwrap();
        conn.execute("DELETE FROM records WHERE id = ?", vec![t("id-2")]).unwrap();
        conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            vec![t("2026-03-01T00:00:00+00:00"), t("id-1")],
        )
        .unwrap();
    }
    for sql in [
        "SELECT MIN(vec_label) FROM records WHERE tombstoned_at IS NULL",
        "SELECT MAX(vec_label) FROM records WHERE tombstoned_at IS NULL",
        "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL",
    ] {
        assert_differential(sql, vec![], &mut indexed, &mut scan);
    }
}

#[test]
fn ordered_walk_with_residual_predicate_matches_scan() {
    let (_dir, mut indexed, mut scan) = open_pair(60);
    // The boot-probe boundary shape: a lower bound on the ordered column PLUS
    // residual conjuncts on other columns — the strict lower-bound-only gate
    // declines this, so it exercises the predicate-aware walk.
    let sql = "SELECT vec_label, id FROM records \
               WHERE vec_label >= ? AND tombstoned_at IS NULL \
               AND COALESCE(embedding_pending, 0) = 0 \
               ORDER BY vec_label LIMIT 1";
    for bound in [0_i64, 1, 7, 30, 59, 60, 61] {
        assert_differential(sql, vec![Value::Int(bound)], &mut indexed, &mut scan);
    }
}

#[test]
fn ordered_walk_asc_desc_limits_offsets_match_scan() {
    let (_dir, mut indexed, mut scan) = open_pair(60);
    for sql in [
        "SELECT vec_label FROM records WHERE tombstoned_at IS NULL \
         ORDER BY vec_label LIMIT 5",
        "SELECT vec_label FROM records WHERE tombstoned_at IS NULL \
         ORDER BY vec_label DESC LIMIT 5",
        "SELECT vec_label FROM records ORDER BY vec_label LIMIT 100",
        "SELECT created_at, vec_label FROM records ORDER BY created_at LIMIT 9",
        "SELECT created_at, vec_label FROM records ORDER BY created_at DESC LIMIT 9",
        "SELECT vec_label FROM records WHERE embedding_pending = 1 \
         ORDER BY vec_label LIMIT 3",
        "SELECT vec_label FROM records WHERE n > 100000 ORDER BY vec_label LIMIT 3",
        "SELECT vec_label FROM records ORDER BY vec_label LIMIT 4 OFFSET 7",
        "SELECT vec_label FROM records WHERE tombstoned_at IS NULL \
         ORDER BY vec_label DESC LIMIT 4 OFFSET 3",
    ] {
        assert_differential(sql, vec![], &mut indexed, &mut scan);
    }
}

#[test]
fn ordered_walk_includes_null_ordered_values_like_scan() {
    // created_at carries NULLs (every seventh row): ascending order puts them
    // FIRST, descending LAST — the walk must cover the NULL bucket exactly as
    // the scan path's NULLs-first collation does.
    let (_dir, mut indexed, mut scan) = open_pair(30);
    for sql in [
        "SELECT created_at, vec_label FROM records ORDER BY created_at LIMIT 6",
        "SELECT created_at, vec_label FROM records ORDER BY created_at DESC LIMIT 40",
    ] {
        assert_differential(sql, vec![], &mut indexed, &mut scan);
    }
}

#[test]
fn ordered_walk_ties_keep_scan_order() {
    // created_at collides across rows (i % 60 wraps and repeats i and i+60
    // patterns are absent at n=50, but seconds repeat between differing rows
    // via the % 60 formula when i differs by 60 — instead force explicit
    // duplicates here).
    let (_dir, mut indexed, mut scan) = open_pair(0);
    for conn in [&mut indexed, &mut scan] {
        for i in 1..=12_i64 {
            conn.execute(
                "INSERT INTO records (id, tombstoned_at, embedding_pending, created_at, n) \
                 VALUES (?, NULL, 0, ?, ?)",
                vec![
                    t(&format!("id-{i}")),
                    t(&format!("2026-01-01T00:00:{:02}+00:00", i % 3)),
                    Value::Int(i),
                ],
            )
            .unwrap();
        }
    }
    for sql in [
        "SELECT created_at, id FROM records ORDER BY created_at LIMIT 5",
        "SELECT created_at, id FROM records ORDER BY created_at DESC LIMIT 5",
        "SELECT MIN(created_at) FROM records",
        "SELECT MAX(created_at) FROM records",
    ] {
        assert_differential(sql, vec![], &mut indexed, &mut scan);
    }
}

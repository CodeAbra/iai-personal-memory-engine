//! DB-API surface tests for the sqlite3-shaped Connection/Cursor/Row state
//! machine and the read-only raw-access adapter.
//!
//! Drives the six connection-tuning PRAGMAs the host issues at open time
//! (journal_mode=WAL, synchronous, foreign_keys, busy_timeout, cache_size,
//! query_only) through Connection.execute, asserting they are accepted; the DDL
//! + INSERT + SELECT + composite-key UPSERT path; PRAGMA table_info synthesized
//! from the catalog; and the raw adapter's read-only gate (a write is rejected,
//! a read delegates).

use lillibrain::Value;
use lilliengine::conn::{Connection, RawAction, RawConn};
use tempfile::tempdir;

const DDL_RECORDS: &str = "CREATE TABLE IF NOT EXISTS records ( \
    vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
    n INTEGER , label TEXT )";

const DDL_EDGES: &str = "CREATE TABLE IF NOT EXISTS edges ( \
    src TEXT , dst TEXT , edge_type TEXT , weight REAL , updated_at TEXT , \
    PRIMARY KEY ( src , dst , edge_type ) )";

const UPSERT: &str = "INSERT INTO edges (src, dst, edge_type, weight, updated_at) \
    VALUES (?, ?, ?, ?, ?) ON CONFLICT (src, dst, edge_type) \
    DO UPDATE SET weight = excluded.weight , updated_at = excluded.updated_at";

fn t(s: &str) -> Value {
    Value::Text(s.to_string())
}

fn open() -> (tempfile::TempDir, Connection) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let conn = Connection::open(path.to_str().unwrap(), 384).unwrap();
    (dir, conn)
}

/// Issue the six connection PRAGMAs the host fires at HippoDB.__init__.
fn issue_connection_pragmas(conn: &mut Connection) {
    conn.execute("PRAGMA journal_mode=WAL", vec![]).unwrap();
    conn.execute("PRAGMA synchronous=NORMAL", vec![]).unwrap();
    conn.execute("PRAGMA foreign_keys=ON", vec![]).unwrap();
    conn.execute("PRAGMA busy_timeout=2000", vec![]).unwrap();
    conn.execute("PRAGMA cache_size=-65536", vec![]).unwrap();
    conn.execute("PRAGMA query_only=OFF", vec![]).unwrap();
}

#[test]
fn six_connection_pragmas_accepted() {
    let (_dir, mut conn) = open();
    // None of the six raise; journal_mode reports back the WAL mode row.
    let mut cur = conn.execute("PRAGMA journal_mode=WAL", vec![]).unwrap();
    let row = cur.fetchone().unwrap();
    assert_eq!(row.get_name("journal_mode"), Some(&t("wal")));
    conn.execute("PRAGMA synchronous=NORMAL", vec![]).unwrap();
    conn.execute("PRAGMA foreign_keys=ON", vec![]).unwrap();
    conn.execute("PRAGMA busy_timeout=2000", vec![]).unwrap();
    conn.execute("PRAGMA cache_size=-65536", vec![]).unwrap();
    conn.execute("PRAGMA query_only=OFF", vec![]).unwrap();
}

#[test]
fn ddl_insert_select_roundtrip() {
    let (_dir, mut conn) = open();
    issue_connection_pragmas(&mut conn);
    conn.execute(DDL_RECORDS, vec![]).unwrap();

    let cur = conn
        .execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t("a"), Value::Int(10)],
        )
        .unwrap();
    // vec_label AUTOINCREMENT is injected and exposed as lastrowid; one row.
    assert_eq!(cur.lastrowid, Some(1));
    assert_eq!(cur.rowcount, 1);

    let cur2 = conn
        .execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t("b"), Value::Int(20)],
        )
        .unwrap();
    assert_eq!(cur2.lastrowid, Some(2));

    let mut sel = conn.execute("SELECT id, n FROM records", vec![]).unwrap();
    let desc = sel.description();
    assert_eq!(desc[0].name, "id");
    assert_eq!(desc[1].name, "n");
    let rows = sel.fetchall();
    assert_eq!(rows.len(), 2);
    // Dict-like (by name) and positional (by index) access both resolve.
    assert_eq!(rows[0].get_name("id"), Some(&t("a")));
    assert_eq!(rows[0].get_index(0), Some(&t("a")));
    assert_eq!(rows[0].get_index(1), Some(&Value::Int(10)));
    assert_eq!(rows[1].get_name("id"), Some(&t("b")));
}

#[test]
fn parse_cache_reuses_ast_across_distinct_params() {
    // The parsed-statement cache keys on the SQL text only; binds apply per call.
    // Running the SAME query string twice with DIFFERENT params must return the
    // distinct correct rows — proving the cached AST does not capture binds.
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute(
        "INSERT INTO records (id, n) VALUES (?, ?)",
        vec![t("alice"), Value::Int(10)],
    )
    .unwrap();
    conn.execute(
        "INSERT INTO records (id, n) VALUES (?, ?)",
        vec![t("bob"), Value::Int(20)],
    )
    .unwrap();

    const Q: &str = "SELECT id, n FROM records WHERE id = ?";

    // First execute: cold parse, inserts the AST into the cache.
    let mut s1 = conn.execute(Q, vec![t("alice")]).unwrap();
    let r1 = s1.fetchall();
    assert_eq!(r1.len(), 1);
    assert_eq!(r1[0].get_name("id"), Some(&t("alice")));
    assert_eq!(r1[0].get_name("n"), Some(&Value::Int(10)));

    // Second execute, identical SQL string, different bind: must hit the cached
    // AST yet resolve the OTHER row — distinct result from the cache-hit path.
    let mut s2 = conn.execute(Q, vec![t("bob")]).unwrap();
    let r2 = s2.fetchall();
    assert_eq!(r2.len(), 1);
    assert_eq!(r2[0].get_name("id"), Some(&t("bob")));
    assert_eq!(r2[0].get_name("n"), Some(&Value::Int(20)));

    // Third execute, back to the first bind: still correct after the round trip.
    let mut s3 = conn.execute(Q, vec![t("alice")]).unwrap();
    let r3 = s3.fetchall();
    assert_eq!(r3.len(), 1);
    assert_eq!(r3[0].get_name("id"), Some(&t("alice")));
}

#[test]
fn parse_cache_survives_writes_between_reads() {
    // A write between two identical reads must not corrupt the cached AST: the
    // second read sees the new row, proving the cache holds the parse, not stale
    // result rows.
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute(
        "INSERT INTO records (id, n) VALUES (?, ?)",
        vec![t("x"), Value::Int(1)],
    )
    .unwrap();

    const Q: &str = "SELECT id FROM records WHERE n = ?";
    let mut a = conn.execute(Q, vec![Value::Int(1)]).unwrap();
    assert_eq!(a.fetchall().len(), 1);

    // Insert a second row matching the same predicate value via a NEW id.
    conn.execute(
        "INSERT INTO records (id, n) VALUES (?, ?)",
        vec![t("y"), Value::Int(1)],
    )
    .unwrap();

    // Same query string (cache hit on the AST) must now return BOTH rows.
    let mut b = conn.execute(Q, vec![Value::Int(1)]).unwrap();
    assert_eq!(b.fetchall().len(), 2);
}

#[test]
fn fetchone_then_fetchall_share_offset() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    for i in 0..3 {
        conn.execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t(&format!("id-{i}")), Value::Int(i)],
        )
        .unwrap();
    }
    let mut sel = conn.execute("SELECT id FROM records", vec![]).unwrap();
    let first = sel.fetchone().unwrap();
    assert_eq!(first.get_name("id"), Some(&t("id-0")));
    // fetchall resumes from the shared offset: two rows remain.
    let rest = sel.fetchall();
    assert_eq!(rest.len(), 2);
    assert!(sel.fetchone().is_none());
}

#[test]
fn pragma_table_info_from_catalog() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    let mut cur = conn.execute("PRAGMA table_info(records)", vec![]).unwrap();
    let cols = cur.columns().to_vec();
    assert_eq!(cols, ["cid", "name", "type", "notnull", "dflt_value", "pk"]);
    let rows = cur.fetchall();
    assert_eq!(rows.len(), 4);
    // vec_label is the INTEGER PRIMARY KEY (pk set; notnull is 0 — sqlite3
    // reports an INTEGER PRIMARY KEY rowid alias as nullable in table_info).
    assert_eq!(rows[0].get_name("name"), Some(&t("vec_label")));
    assert_eq!(rows[0].get_name("type"), Some(&t("INTEGER")));
    assert_eq!(rows[0].get_name("pk"), Some(&Value::Int(1)));
    assert_eq!(rows[0].get_name("notnull"), Some(&Value::Int(0)));
    // id is NOT NULL but not the PK.
    assert_eq!(rows[1].get_name("name"), Some(&t("id")));
    assert_eq!(rows[1].get_name("pk"), Some(&Value::Int(0)));
}

#[test]
fn composite_key_upsert_in_place() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_EDGES, vec![]).unwrap();
    conn.execute(
        UPSERT,
        vec![t("a"), t("b"), t("rel"), Value::Float(1.0), t("t1")],
    )
    .unwrap();
    conn.execute(
        UPSERT,
        vec![t("a"), t("b"), t("rel"), Value::Float(9.0), t("t2")],
    )
    .unwrap();
    let mut sel = conn
        .execute("SELECT src, weight, updated_at FROM edges", vec![])
        .unwrap();
    let rows = sel.fetchall();
    assert_eq!(rows.len(), 1, "the composite key must not duplicate");
    assert_eq!(rows[0].get_name("weight"), Some(&Value::Float(9.0)));
    assert_eq!(rows[0].get_name("updated_at"), Some(&t("t2")));
}

#[test]
fn in_transaction_reflects_begin_commit() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    assert!(!conn.in_transaction());
    conn.execute("BEGIN", vec![]).unwrap();
    assert!(conn.in_transaction());
    conn.execute(
        "INSERT INTO records (id, n) VALUES (?, ?)",
        vec![t("a"), Value::Int(1)],
    )
    .unwrap();
    conn.execute("COMMIT", vec![]).unwrap();
    assert!(!conn.in_transaction());
    let mut sel = conn.execute("SELECT id FROM records", vec![]).unwrap();
    assert_eq!(sel.fetchall().len(), 1);
}

#[test]
fn executemany_batch_atomic() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    let cur = conn
        .executemany(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![
                vec![t("a"), Value::Int(1)],
                vec![t("b"), Value::Int(2)],
                vec![t("c"), Value::Int(3)],
            ],
        )
        .unwrap();
    assert_eq!(cur.rowcount, 3);
    let mut sel = conn
        .execute("SELECT vec_label FROM records", vec![])
        .unwrap();
    let rows = sel.fetchall();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].get_name("vec_label"), Some(&Value::Int(1)));
    assert_eq!(rows[2].get_name("vec_label"), Some(&Value::Int(3)));
}

#[test]
fn alter_rename_of_quoted_identifier_survives_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ren.lilli");
    let p = path.to_str().unwrap();
    {
        let mut conn = Connection::open(p, 384).unwrap();
        // A bracket-quoted table name, renamed via a bracket-quoted reference.
        conn.execute("CREATE TABLE [t] ( id TEXT , n INTEGER )", vec![])
            .unwrap();
        conn.execute(
            "INSERT INTO [t] (id, n) VALUES (?, ?)",
            vec![t("a"), Value::Int(1)],
        )
        .unwrap();
        conn.execute("ALTER TABLE [t] RENAME TO u", vec![]).unwrap();
        conn.close().unwrap();
    }
    // Reopen replays the CREATE + RENAME: the table is u and its data survives.
    let mut conn = Connection::open(p, 384).unwrap();
    let mut sel = conn.execute("SELECT id, n FROM u", vec![]).unwrap();
    let rows = sel.fetchall();
    assert_eq!(
        rows.len(),
        1,
        "the renamed table u is queryable after reopen"
    );
    assert_eq!(rows[0].get_name("id"), Some(&t("a")));
    // The stored DDL surfaced via sqlite_master names the new table u.
    let mut master = conn
        .execute("SELECT sql FROM sqlite_master WHERE name = 'u'", vec![])
        .unwrap();
    let mrows = master.fetchall();
    assert_eq!(mrows.len(), 1, "u appears in sqlite_master after reopen");
    let sql = match mrows[0].get_name("sql") {
        Some(Value::Text(s)) => s.clone(),
        other => panic!("stored DDL must be text, got {other:?}"),
    };
    let upper = sql.to_ascii_uppercase();
    assert!(
        upper.contains("TABLE U "),
        "the replayed DDL names the new table u, got {sql:?}"
    );
}

#[test]
fn reopen_replays_schema_and_data() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let p = path.to_str().unwrap();
    {
        let mut conn = Connection::open(p, 384).unwrap();
        conn.execute(DDL_RECORDS, vec![]).unwrap();
        conn.execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t("a"), Value::Int(7)],
        )
        .unwrap();
        conn.close().unwrap();
    }
    let mut conn = Connection::open(p, 384).unwrap();
    let mut sel = conn.execute("SELECT id, n FROM records", vec![]).unwrap();
    let rows = sel.fetchall();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_name("id"), Some(&t("a")));
    assert_eq!(rows[0].get_name("n"), Some(&Value::Int(7)));
    // The AUTOINCREMENT high-water mark resumes after reopen.
    let cur = conn
        .execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t("b"), Value::Int(8)],
        )
        .unwrap();
    assert_eq!(cur.lastrowid, Some(2));
}

#[test]
fn query_only_blocks_writes() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute("PRAGMA query_only=ON", vec![]).unwrap();
    let err = conn
        .execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t("a"), Value::Int(1)],
        )
        .unwrap_err();
    assert!(format!("{err}").contains("attempt to write a readonly database"));
    // A read still succeeds under query_only.
    conn.execute("SELECT id FROM records", vec![]).unwrap();
}

#[test]
fn query_only_off_cannot_flip_a_readonly_mount() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("ro.lilli");
    let p = path.to_str().unwrap();
    {
        let mut conn = Connection::open(p, 384).unwrap();
        conn.execute(DDL_RECORDS, vec![]).unwrap();
        conn.execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t("a"), Value::Int(1)],
        )
        .unwrap();
        conn.close().unwrap();
    }
    let mut ro = Connection::open_read_only(p, 384).unwrap();
    // Clearing the mutable query_only flag must NOT grant write access on a
    // read-only mount; the write still reports the read-only error.
    ro.execute("PRAGMA query_only=OFF", vec![]).unwrap();
    let err = ro
        .execute(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t("b"), Value::Int(2)],
        )
        .unwrap_err();
    assert!(
        format!("{err}").contains("attempt to write a readonly database"),
        "a write on a read-only mount must report the read-only error, got {err}"
    );
    // The read path still works on the read-only mount.
    let mut sel = ro.execute("SELECT id FROM records", vec![]).unwrap();
    assert_eq!(sel.fetchall().len(), 1);
}

#[test]
fn commit_or_rollback_with_no_active_transaction_errors() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    // No BEGIN is open: COMMIT / ROLLBACK report the no-transaction error rather
    // than driving an illegal pager call.
    let cerr = conn.execute("COMMIT", vec![]).unwrap_err();
    assert!(
        format!("{cerr}").contains("no transaction is active"),
        "COMMIT with no active transaction must report no-transaction, got {cerr}"
    );
    let rerr = conn.execute("ROLLBACK", vec![]).unwrap_err();
    assert!(
        format!("{rerr}").contains("no transaction is active"),
        "ROLLBACK with no active transaction must report no-transaction, got {rerr}"
    );
    // A normal BEGIN/COMMIT still works after the guarded rejections.
    conn.execute("BEGIN", vec![]).unwrap();
    conn.execute("COMMIT", vec![]).unwrap();
}

#[test]
fn sql_rollback_clears_conflict_cache_no_wrong_row_overwrite() {
    // The production UPSERT path drives its transaction with SQL strings: `_txn`
    // issues `execute("BEGIN")` and, on a failure, `execute("ROLLBACK")`. Each
    // suppressed-scope UPSERT inside the open transaction populates the
    // connection's conflict cache with (conflict-key → row-key) entries. If the
    // SQL-string ROLLBACK fails to clear that cache, the reverted rows leave stale
    // entries pointing at row-keys the rollback freed — and when the pager reuses
    // such a key for an unrelated row, a later UPSERT whose conflict tuple matches
    // the stale entry overwrites the WRONG row: silent data corruption on the
    // edges store.
    //
    // This drives that exact wrong-row shape deterministically and asserts the
    // engine never overwrites a row the conflict tuple does not actually match.
    let (_dir, mut conn) = open();
    conn.execute(DDL_EDGES, vec![]).unwrap();

    // Under an outer transaction, UPSERT a single edge. It lands at row-key 1 and
    // primes the conflict cache with (g,h,rel) -> 1.
    conn.execute("BEGIN", vec![]).unwrap();
    conn.execute(
        UPSERT,
        vec![t("g"), t("h"), t("rel"), Value::Float(1.0), t("t1")],
    )
    .unwrap();
    // Roll the transaction back via the SQL string the production `_txn` uses.
    // The tree is reverted (row-key 1 is freed) — but a stale cache would retain
    // (g,h,rel) -> 1.
    conn.execute("ROLLBACK", vec![]).unwrap();
    let mut sel = conn.execute("SELECT src FROM edges", vec![]).unwrap();
    assert_eq!(sel.fetchall().len(), 0, "the rolled-back row must be gone");

    // A committed INSERT of a DIFFERENT edge now reuses the freed row-key 1 (the
    // tree is empty, so next_key descends to 1 again).
    conn.execute(
        UPSERT,
        vec![t("x"), t("y"), t("rel"), Value::Float(7.0), t("keep")],
    )
    .unwrap();

    // UPSERT the key from the rolled-back batch. With a stale cache, its lookup
    // returns row-key 1, `tree.get(1)` finds the LIVE (x,y,rel) row, and the
    // engine wrongly treats it as a conflict-hit — overwriting (x,y,rel) with the
    // (g,h,rel) values. With the cache cleared on ROLLBACK, (g,h,rel) inserts
    // fresh and (x,y,rel) is left intact.
    conn.execute(
        UPSERT,
        vec![t("g"), t("h"), t("rel"), Value::Float(2.0), t("t2")],
    )
    .unwrap();

    let mut sel = conn
        .execute(
            "SELECT src, dst, edge_type, weight, updated_at FROM edges ORDER BY src, edge_type",
            vec![],
        )
        .unwrap();
    let rows = sel.fetchall();
    assert_eq!(
        rows.len(),
        2,
        "both edges must exist: the stale entry must not collapse them into one"
    );
    // The unrelated (x,y,rel) row must be untouched (weight 7.0, updated_at keep) —
    // a stale-cache wrong-row overwrite would have clobbered it with the (g,h,rel)
    // values (weight 2.0, updated_at t2).
    let xy = rows
        .iter()
        .find(|r| r.get_name("src") == Some(&t("x")))
        .expect("the committed (x,y,rel) row");
    assert_eq!(
        xy.get_name("weight"),
        Some(&Value::Float(7.0)),
        "the unrelated row was overwritten by a stale conflict entry — data corruption"
    );
    assert_eq!(xy.get_name("updated_at"), Some(&t("keep")));
    // And the re-UPSERTed (g,h,rel) row landed fresh with its own values.
    let gh = rows
        .iter()
        .find(|r| r.get_name("src") == Some(&t("g")))
        .expect("the fresh (g,h,rel) row");
    assert_eq!(gh.get_name("weight"), Some(&Value::Float(2.0)));
    assert_eq!(gh.get_name("updated_at"), Some(&t("t2")));
}

#[test]
fn col_index_maintained_incrementally_no_rebuild_on_steady_state_read() {
    // A write between two recalls must NOT force the next recall to full-scan
    // rebuild the col-index. The index is maintained incrementally on the write
    // path, so the second adjacency lookup resolves entirely from the index with
    // zero whole-tree scans — the steady-state recall-latency churn the
    // drop-and-rebuild caused is gone.
    let (_dir, mut conn) = open();
    conn.execute(DDL_EDGES, vec![]).unwrap();

    // Seed two edges so the col-index has content, then a recall builds the index.
    conn.execute(
        UPSERT,
        vec![t("a"), t("b"), t("rel"), Value::Float(1.0), t("t1")],
    )
    .unwrap();
    conn.execute(
        UPSERT,
        vec![t("c"), t("d"), t("rel"), Value::Float(1.0), t("t1")],
    )
    .unwrap();

    const ADJ: &str = "SELECT src , dst , weight FROM edges \
                       WHERE ( src IN ( 'a' , 'c' ) OR dst IN ( 'a' , 'c' ) )";

    // First recall builds the col-index (one scan is permitted on the cold build).
    let _ = conn.execute(ADJ, vec![]).unwrap().fetchall();

    // A coalesced write between recalls — the provenance/event shape that DROPPED
    // the col-index under the old drop-and-rebuild discipline. Now it is maintained
    // incrementally: the new edge is filed at its row-key, the index stays built.
    conn.execute(
        UPSERT,
        vec![t("a"), t("e"), t("rel"), Value::Float(2.0), t("t2")],
    )
    .unwrap();

    // The next recall must be SCAN-FREE: the index already reflects the new edge
    // without a rebuild. A non-zero scan count here is the churn regression.
    conn.reset_full_scan_count();
    let mut sel = conn.execute(ADJ, vec![]).unwrap();
    let rows = sel.fetchall();
    assert_eq!(
        conn.full_scan_count(),
        0,
        "a recall after a write must resolve from the incrementally-maintained \
         col-index with no full-scan rebuild"
    );
    // And it returns the correct rows including the edge written between recalls:
    // (a,b), (c,d), and the new (a,e) all carry `a`/`c` in src.
    let has_src = |s: &str| rows.iter().any(|r| r.get_name("src") == Some(&t(s)));
    assert!(has_src("a"), "the (a,*) edges must be present");
    assert!(has_src("c"), "the (c,d) edge must be present");
    assert_eq!(
        rows.len(),
        3,
        "the index must reflect the edge written between recalls (a,b)+(c,d)+(a,e)"
    );
}

#[test]
fn col_index_incremental_write_reflects_new_and_dropped_rows() {
    // The incremental col-index must track BOTH an insert (new posting) and a
    // delete (dropped posting) without a rebuild, so a recall after either write
    // returns the tree-accurate result. This guards the insert_row / remove_row
    // maintenance against drift from the committed tree.
    let (_dir, mut conn) = open();
    conn.execute(DDL_EDGES, vec![]).unwrap();
    conn.execute(
        UPSERT,
        vec![t("a"), t("b"), t("rel"), Value::Float(1.0), t("t1")],
    )
    .unwrap();

    const ADJ_A: &str = "SELECT src , dst FROM edges WHERE src IN ( 'a' )";
    // Build the index.
    assert_eq!(conn.execute(ADJ_A, vec![]).unwrap().fetchall().len(), 1);

    // Insert another (a,*) edge: the incremental posting must add it.
    conn.execute(
        UPSERT,
        vec![t("a"), t("c"), t("rel"), Value::Float(1.0), t("t1")],
    )
    .unwrap();
    conn.reset_full_scan_count();
    let after_insert = conn.execute(ADJ_A, vec![]).unwrap().fetchall();
    assert_eq!(
        conn.full_scan_count(),
        0,
        "the post-insert recall must be scan-free"
    );
    assert_eq!(
        after_insert.len(),
        2,
        "the incrementally-added (a,c) edge must appear"
    );

    // Delete one (a,*) edge: the incremental remove must drop its posting so the
    // freed key is never returned by a later probe.
    conn.execute("DELETE FROM edges WHERE dst = 'b'", vec![])
        .unwrap();
    conn.reset_full_scan_count();
    let after_delete = conn.execute(ADJ_A, vec![]).unwrap().fetchall();
    assert_eq!(
        conn.full_scan_count(),
        0,
        "the post-delete recall must be scan-free"
    );
    assert_eq!(
        after_delete.len(),
        1,
        "the deleted (a,b) edge must be gone from the index"
    );
    assert_eq!(after_delete[0].get_name("dst"), Some(&t("c")));
}

#[test]
fn sql_rollback_discards_incremental_col_index_entries_no_freed_key_hit() {
    // Rollback-corruption guard for the col-index: a write rolled back via the
    // SQL-string ROLLBACK (the production `_txn` path) must NOT leave the
    // rolled-back row's entries in the col-index. If it did, a later recall could
    // return a freed row-key the pager reused for an unrelated row — the same
    // wrong-row hazard the conflict-cache clear-on-rollback already prevents. After
    // the rollback the index must reflect the COMMITTED tree, as if the write never
    // happened.
    let (_dir, mut conn) = open();
    conn.execute(DDL_EDGES, vec![]).unwrap();

    // Commit one real edge and build the index over it.
    conn.execute(
        UPSERT,
        vec![t("x"), t("y"), t("rel"), Value::Float(7.0), t("keep")],
    )
    .unwrap();
    const ADJ_Z: &str = "SELECT src , dst , weight FROM edges WHERE src IN ( 'z' )";
    const ADJ_X: &str = "SELECT src , dst , weight FROM edges WHERE src IN ( 'x' )";
    // Build the col-index (no (z,*) row yet).
    assert_eq!(conn.execute(ADJ_Z, vec![]).unwrap().fetchall().len(), 0);

    // Under an outer transaction, insert a (z,*) edge — incrementally filed at its
    // row-key — then roll the transaction back via the SQL string.
    conn.execute("BEGIN", vec![]).unwrap();
    conn.execute(
        UPSERT,
        vec![t("z"), t("w"), t("rel"), Value::Float(9.0), t("t2")],
    )
    .unwrap();
    conn.execute("ROLLBACK", vec![]).unwrap();

    // The rolled-back (z,*) edge must be gone — both from the tree AND from the
    // col-index (no stale posting pointing at the freed key).
    let zrows = conn.execute(ADJ_Z, vec![]).unwrap().fetchall();
    assert_eq!(
        zrows.len(),
        0,
        "the rolled-back (z,w) edge must not survive in the col-index"
    );
    // And the committed (x,y) edge is still intact and untouched by the rollback.
    let mut xsel = conn.execute(ADJ_X, vec![]).unwrap();
    let xrows = xsel.fetchall();
    assert_eq!(xrows.len(), 1);
    assert_eq!(xrows[0].get_name("dst"), Some(&t("y")));
    assert_eq!(xrows[0].get_name("weight"), Some(&Value::Float(7.0)));
}

// ---------------------------------------------------------------------------
// RawConn read-only gate
// ---------------------------------------------------------------------------

#[test]
fn raw_conn_read_only_gate_blocks_writes_delegates_reads() {
    let mut ro = RawConn::new(true);
    // A write SQL is rejected at the adapter.
    assert_eq!(
        ro.gate("INSERT INTO records (id) VALUES ('a')"),
        RawAction::Reject
    );
    assert_eq!(ro.gate("UPDATE records SET n = 1"), RawAction::Reject);
    assert_eq!(ro.gate("DELETE FROM records"), RawAction::Reject);
    // A read delegates.
    match ro.gate("SELECT id FROM records") {
        RawAction::Delegate(s) => assert_eq!(s, "SELECT id FROM records"),
        other => panic!("expected delegate, got {other:?}"),
    }
    // PRAGMA query_only is intercepted at the adapter, not delegated.
    assert_eq!(ro.gate("PRAGMA query_only=OFF"), RawAction::Empty);
    // After flipping it off, a write delegates.
    match ro.gate("INSERT INTO records (id) VALUES ('a')") {
        RawAction::Delegate(_) => {}
        other => panic!("expected delegate after query_only=OFF, got {other:?}"),
    }
}

#[test]
fn raw_conn_read_write_delegates_writes() {
    let mut rw = RawConn::new(false);
    match rw.gate("INSERT INTO records (id) VALUES ('a')") {
        RawAction::Delegate(_) => {}
        other => panic!("expected delegate, got {other:?}"),
    }
    // Reading the query_only flag reports false.
    assert_eq!(
        rw.gate("PRAGMA query_only"),
        RawAction::ReportQueryOnly(false)
    );
}

#[test]
fn busy_timeout_round_trips() {
    let (_dir, mut conn) = open();
    // Assignment reports back the value; a subsequent read returns it.
    let mut cur = conn.execute("PRAGMA busy_timeout=2000", vec![]).unwrap();
    let row = cur.fetchone().unwrap();
    assert_eq!(row.get_name("busy_timeout"), Some(&Value::Int(2000)));
    let mut read = conn.execute("PRAGMA busy_timeout", vec![]).unwrap();
    let r = read.fetchone().unwrap();
    assert_eq!(r.get_name("busy_timeout"), Some(&Value::Int(2000)));
}

#[test]
fn sqlite_master_lists_tables() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute(DDL_EDGES, vec![]).unwrap();

    // SELECT name ... WHERE type='table' ORDER BY name → the two table names.
    let mut cur = conn
        .execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            vec![],
        )
        .unwrap();
    let rows = cur.fetchall();
    let names: Vec<Value> = rows
        .iter()
        .map(|r| r.get_name("name").cloned().unwrap())
        .collect();
    assert_eq!(names, vec![t("edges"), t("records")]);
}

#[test]
fn sqlite_master_full_row_shape() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();

    let mut cur = conn
        .execute("SELECT * FROM sqlite_master WHERE name='records'", vec![])
        .unwrap();
    let rows = cur.fetchall();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.get_name("type"), Some(&t("table")));
    assert_eq!(row.get_name("name"), Some(&t("records")));
    assert_eq!(row.get_name("tbl_name"), Some(&t("records")));
    // rootpage is a positive integer (the data tree's root page).
    match row.get_name("rootpage") {
        Some(Value::Int(p)) => assert!(*p > 0, "rootpage should be > 0, got {p}"),
        other => panic!("expected Int rootpage, got {other:?}"),
    }
    // sql is the stored CREATE TABLE DDL.
    match row.get_name("sql") {
        Some(Value::Text(s)) => assert!(s.to_uppercase().starts_with("CREATE TABLE")),
        other => panic!("expected Text sql, got {other:?}"),
    }
}

#[test]
fn sqlite_schema_alias_equivalent() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    let mut cur = conn
        .execute("SELECT name FROM sqlite_schema WHERE type='table'", vec![])
        .unwrap();
    let rows = cur.fetchall();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_name("name"), Some(&t("records")));
}

#[test]
fn sqlite_master_name_param_filter() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute(DDL_EDGES, vec![]).unwrap();
    // Parametrized WHERE name=? resolves the bind against the schema rows.
    let mut cur = conn
        .execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            vec![t("edges")],
        )
        .unwrap();
    let rows = cur.fetchall();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_name("name"), Some(&t("edges")));
}

#[test]
fn sqlite_master_count_star() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute(DDL_EDGES, vec![]).unwrap();
    let mut cur = conn
        .execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'",
            vec![],
        )
        .unwrap();
    let row = cur.fetchone().unwrap();
    assert_eq!(row.get_name("COUNT(*)"), Some(&Value::Int(2)));
}

#[test]
fn sqlite_master_rootpage_survives_reopen() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("t.lilli");
    let p = path.to_str().unwrap();
    {
        let mut conn = Connection::open(p, 384).unwrap();
        conn.execute(DDL_RECORDS, vec![]).unwrap();
    }
    // A reopen replays the DDL + root rows; sqlite_master still reports the table.
    let mut conn = Connection::open(p, 384).unwrap();
    let mut cur = conn
        .execute(
            "SELECT name, rootpage, sql FROM sqlite_master WHERE type='table'",
            vec![],
        )
        .unwrap();
    let rows = cur.fetchall();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_name("name"), Some(&t("records")));
    match rows[0].get_name("rootpage") {
        Some(Value::Int(pg)) => assert!(*pg > 0),
        other => panic!("expected rootpage Int, got {other:?}"),
    }
    match rows[0].get_name("sql") {
        Some(Value::Text(s)) => assert!(s.to_uppercase().contains("CREATE TABLE")),
        other => panic!("expected sql Text, got {other:?}"),
    }
}

#[test]
fn maintenance_statements_are_accepted_noops() {
    // The host's storage-maintenance pass issues VACUUM (and may issue ANALYZE /
    // REINDEX). This engine compacts in-place via its own pager freelist, so a
    // stdlib whole-file rewrite has no meaning here; the statements must be
    // accepted as no-ops rather than raising "unsupported statement type", and
    // they must leave the data untouched.
    let (_dir, mut conn) = open();
    issue_connection_pragmas(&mut conn);
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute(
        "INSERT INTO records (id, n) VALUES (?, ?)",
        vec![t("keep"), Value::Int(7)],
    )
    .unwrap();

    for stmt in ["VACUUM", "VACUUM;", "vacuum", "ANALYZE", "REINDEX"] {
        let mut cur = conn
            .execute(stmt, vec![])
            .unwrap_or_else(|e| panic!("{stmt} must be an accepted no-op, got {e}"));
        assert!(cur.fetchall().is_empty(), "{stmt} must yield no rows");
    }

    // The row survives every maintenance no-op byte-identically.
    let mut sel = conn.execute("SELECT id, n FROM records", vec![]).unwrap();
    let rows = sel.fetchall();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get_name("id"), Some(&t("keep")));
    assert_eq!(rows[0].get_name("n"), Some(&Value::Int(7)));
}

#[test]
fn owned_scope_error_leaves_no_open_transaction() {
    // An auto-commit single statement runs under an owned store transaction. When
    // it errors after the transaction is open (here: a DO UPDATE that violates a
    // NOT NULL constraint on the rewritten row), the owned transaction must be
    // rolled back, so the IMMEDIATELY following write succeeds. Without the
    // rollback the next write hits "a transaction is already open".
    let (_dir, mut conn) = open();
    issue_connection_pragmas(&mut conn);
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    // Seed one row so the UPSERT below conflicts on its primary key.
    conn.execute(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(1), t("a"), Value::Int(10)],
    )
    .unwrap();

    // DO UPDATE conflicting on vec_label 1, rewriting the NOT NULL `id` to NULL:
    // a constraint violation that occurs while the owned transaction is open.
    let err = conn
        .execute(
            "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
             ON CONFLICT (vec_label) DO UPDATE SET id = NULL",
            vec![Value::Int(1), t("a"), Value::Int(20)],
        )
        .unwrap_err();
    assert!(
        format!("{err}").contains("NOT NULL constraint failed"),
        "the DO UPDATE NOT NULL violation must surface, got: {err}"
    );

    // The leaked-transaction symptom: the next owned write must succeed.
    conn.execute(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(2), t("b"), Value::Int(30)],
    )
    .unwrap_or_else(|e| panic!("the next write must succeed (no leaked open txn), got: {e}"));

    // The failed UPSERT left the seed row untouched; the follow-up row landed.
    let mut sel = conn
        .execute("SELECT vec_label, id FROM records", vec![])
        .unwrap();
    let rows = sel.fetchall();
    assert_eq!(rows.len(), 2);
}

#[test]
fn suppressed_scope_error_does_not_roll_back_outer_batch() {
    // A write inside an explicit BEGIN batch runs under a suppressed scope. When it
    // errors, the per-statement rollback guard must NOT fire — the outer
    // transaction stays open so the caller decides whether to commit or roll back.
    // Here the caller recovers: it issues a valid write after the failed one and
    // commits; both the pre-error row and the recovery row survive.
    let (_dir, mut conn) = open();
    issue_connection_pragmas(&mut conn);
    conn.execute(DDL_RECORDS, vec![]).unwrap();

    conn.execute("BEGIN", vec![]).unwrap();
    conn.execute(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(1), t("a"), Value::Int(10)],
    )
    .unwrap();
    // A NOT NULL violation inside the open batch: errors, but must not abort the
    // outer transaction.
    let err = conn
        .execute(
            "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
            vec![Value::Int(2), Value::Null, Value::Int(20)],
        )
        .unwrap_err();
    assert!(format!("{err}").contains("NOT NULL constraint failed"));

    // The outer transaction is still open: a follow-up write lands and commit
    // persists both it and the pre-error row.
    conn.execute(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(3), t("c"), Value::Int(30)],
    )
    .unwrap_or_else(|e| panic!("the suppressed-scope error must not abort the batch, got: {e}"));
    conn.execute("COMMIT", vec![]).unwrap();

    let mut sel = conn
        .execute("SELECT vec_label FROM records", vec![])
        .unwrap();
    let rows = sel.fetchall();
    assert_eq!(
        rows.len(),
        2,
        "the pre-error row and the recovery row both survive (the batch was not aborted)"
    );
}

#[test]
fn conflict_key_rewrite_reinserts_old_key_through_connection() {
    // End-to-end through the connection (the path the host drives): a DO UPDATE
    // that rewrites the conflict-key column re-keys conflict detection, so a later
    // insert of the OLD key value inserts fresh rather than falsely conflicting and
    // overwriting the moved row.
    let (_dir, mut conn) = open();
    conn.execute("CREATE TABLE keyed (k INTEGER, label TEXT)", vec![])
        .unwrap();
    let up = "INSERT INTO keyed (k, label) VALUES (?, ?) \
              ON CONFLICT (k) DO UPDATE SET label = excluded.label";
    conn.execute(up, vec![Value::Int(1), t("first")]).unwrap();
    conn.execute(
        "INSERT INTO keyed (k, label) VALUES (?, ?) \
         ON CONFLICT (k) DO UPDATE SET k = 2, label = excluded.label",
        vec![Value::Int(1), t("moved")],
    )
    .unwrap();
    conn.execute(up, vec![Value::Int(1), t("oldkey")]).unwrap();

    let mut cur = conn.execute("SELECT k, label FROM keyed", vec![]).unwrap();
    let rows = cur.fetchall();
    assert_eq!(
        rows.len(),
        2,
        "the old conflict key inserts fresh after the re-key (moved row + fresh old-key row)"
    );
}

#[test]
fn pragma_index_list_from_catalog() {
    let (_dir, mut conn) = open();
    conn.execute(DDL_RECORDS, vec![]).unwrap();
    conn.execute("CREATE INDEX idx_n ON records ( n )", vec![])
        .unwrap();
    conn.execute("CREATE UNIQUE INDEX idx_id ON records ( id )", vec![])
        .unwrap();
    conn.execute(
        "CREATE INDEX idx_part ON records ( label ) WHERE label = 'x'",
        vec![],
    )
    .unwrap();

    let mut cur = conn.execute("PRAGMA index_list(records)", vec![]).unwrap();
    let cols = cur.columns().to_vec();
    assert_eq!(cols, ["seq", "name", "unique", "origin", "partial"]);
    let rows = cur.fetchall();
    assert_eq!(rows.len(), 3, "all three created indexes are listed");
    // Newest first, matching sqlite3's ordering for created indexes.
    assert_eq!(rows[0].get_name("name"), Some(&t("idx_part")));
    assert_eq!(rows[0].get_name("seq"), Some(&Value::Int(0)));
    assert_eq!(rows[0].get_name("unique"), Some(&Value::Int(0)));
    assert_eq!(rows[0].get_name("partial"), Some(&Value::Int(1)));
    assert_eq!(rows[1].get_name("name"), Some(&t("idx_id")));
    assert_eq!(rows[1].get_name("unique"), Some(&Value::Int(1)));
    assert_eq!(rows[1].get_name("partial"), Some(&Value::Int(0)));
    assert_eq!(rows[2].get_name("name"), Some(&t("idx_n")));
    assert_eq!(rows[2].get_name("unique"), Some(&Value::Int(0)));
    for row in &rows {
        assert_eq!(row.get_name("origin"), Some(&t("c")));
    }

    // Unknown table: empty result, matching sqlite3.
    let mut cur = conn.execute("PRAGMA index_list(nosuch)", vec![]).unwrap();
    assert!(cur.fetchall().is_empty());
}

// ---------------------------------------------------------------------------
// Reused `col_generation` fence, component-wise publish, ordered-index
// maintenance gated on the ordered column actually being touched.
// ---------------------------------------------------------------------------

const DDL_RECS_ORDERED: &str = "CREATE TABLE IF NOT EXISTS recs ( \
    vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
    pending INTEGER , created_at TEXT , payload TEXT )";

fn open_recs_ordered() -> (tempfile::TempDir, Connection, String) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("recs.lilli").to_str().unwrap().to_string();
    let mut conn = Connection::open(&path, 384).unwrap();
    conn.execute(DDL_RECS_ORDERED, vec![]).unwrap();
    conn.execute("CREATE INDEX idx_recs_pending ON recs (pending)", vec![])
        .unwrap();
    conn.execute("CREATE INDEX idx_recs_created_at ON recs (created_at)", vec![])
        .unwrap();
    (dir, conn, path)
}

fn seed_recs(conn: &mut Connection, n: usize) {
    conn.execute("BEGIN", vec![]).unwrap();
    for i in 0..n {
        conn.execute(
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
    conn.execute("COMMIT", vec![]).unwrap();
}

fn warm_ordered_and_id(conn: &mut Connection) {
    // ORDER BY ... LIMIT is the sanctioned fast path that fixes a target
    // ordered column via `catalog_ordered_index_column`; a bare probe never
    // speculatively creates one.
    conn.execute("SELECT id FROM recs ORDER BY created_at DESC LIMIT 5", vec![])
        .unwrap();
    conn.execute(
        "SELECT id FROM recs WHERE id = ?",
        vec![t("id-00000")],
    )
    .unwrap();
    assert!(
        conn.ordered_index_is_built("recs"),
        "the ORDER BY probe must build the ordered index"
    );
}

/// (i) Each commit route advances `col_generation`: single-row INSERT, a
/// top-level executemany bulk INSERT, and ON CONFLICT DO UPDATE (`merge_insert`
/// shape). Asserted indirectly via reader adoption at the post-commit
/// generation — the airtight fence a mismatch would break.
#[test]
fn every_commit_route_advances_generation_and_reader_adopts() {
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let (_dir, mut writer, path) = open_recs_ordered();
    seed_recs(&mut writer, 20);
    warm_ordered_and_id(&mut writer);
    // Publication is demand-driven: an initial reader open records demand for
    // `recs` so the writer's later commits actually publish (mirrors
    // writer_publish_adoption.rs's r1-miss-then-drop pattern).
    drop(Connection::open_read_only(&path, 384).unwrap());

    // Route 1: single-row INSERT (inside an explicit transaction, mirroring
    // the production single-row commit route).
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?)",
            vec![t("single-row"), Value::Int(1), t("2026-02-01T00:00:00"), t("p")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    let r1 = Connection::open_read_only(&path, 384).unwrap();
    assert!(
        r1.id_index_ready("recs") || r1.col_index_ready("recs"),
        "a reader opened after the single-row commit must adopt a published component"
    );
    drop(r1);

    // Route 2: top-level executemany bulk INSERT, no open transaction — the
    // reproduced P3 gap this plan closes.
    let bulk_ids: Vec<Vec<Value>> = (0..30)
        .map(|i| {
            vec![
                t(&format!("bulk-{i:05}")),
                Value::Int(1),
                t("2026-02-02T00:00:00"),
                t("bp"),
            ]
        })
        .collect();
    writer
        .executemany(
            "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?)",
            bulk_ids,
        )
        .unwrap();
    let r2 = Connection::open_read_only(&path, 384).unwrap();
    let mut r2 = r2;
    let mut cur = r2
        .execute("SELECT id FROM recs WHERE id = ?", vec![t("bulk-00000")])
        .unwrap();
    assert_eq!(
        cur.fetchall().len(),
        1,
        "a bulk-executemany-inserted row must be visible to a reader opened after the batch \
         (the generation must have advanced, or a pre-existing stale cache would omit it)"
    );

    // Route 3: ON CONFLICT DO UPDATE ("merge_insert" shape).
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?) \
             ON CONFLICT (id) DO UPDATE SET pending = excluded.pending",
            vec![t("single-row"), Value::Int(0), t("2026-02-01T00:00:00"), t("p")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    let mut r3 = Connection::open_read_only(&path, 384).unwrap();
    let mut cur = r3
        .execute(
            "SELECT pending FROM recs WHERE id = ?",
            vec![t("single-row")],
        )
        .unwrap();
    assert_eq!(
        cur.fetchall()[0].get_index(0),
        Some(&Value::Int(0)),
        "the DO UPDATE's committed value must be visible under the new generation"
    );
}

/// (ii) Component-wise publish: an id-only build publishes id even when col
/// is unbuilt; a later col-only publish at the SAME generation must never
/// clobber an already-published, already-built ordered component
/// (the merge-or-keep invariant).
#[test]
fn component_wise_publish_never_clobbers_a_built_ordered_slot() {
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let (_dir, mut writer, path) = open_recs_ordered();
    seed_recs(&mut writer, 15);
    warm_ordered_and_id(&mut writer);
    drop(Connection::open_read_only(&path, 384).unwrap());

    // Commit publishes col + id + ordered together (all built on the writer).
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?)",
            vec![t("cw-1"), Value::Int(1), t("2026-03-01T00:00:00"), t("p")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();

    let r1 = Connection::open_read_only(&path, 384).unwrap();
    assert!(
        r1.ordered_index_is_built("recs"),
        "a reader at this generation must adopt the writer's built ordered component"
    );
    drop(r1);

    // An UPDATE that touches ONLY the col-indexed `pending` column (not
    // `created_at`) must not drop the writer's own ordered cache (the T2
    // fix), and its publish must not clobber the ordered slot already
    // published at this same generation for a LATER reader that opens before
    // any generation-moving write. Since `pending` IS col-indexed, this
    // UPDATE itself advances the generation — so the fresh publish under the
    // NEW generation must STILL carry ordered (proving merge-or-keep, since
    // the ordered component was never rebuilt by this specific commit's
    // per-row loop path, only carried over from the writer's still-built
    // cache).
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "UPDATE recs SET pending = 0 WHERE id = ?",
            vec![t("cw-1")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();

    let r2 = Connection::open_read_only(&path, 384).unwrap();
    assert!(
        r2.ordered_index_is_built("recs"),
        "a reader at the post-UPDATE generation must still adopt a built ordered component \
         — a col-only (or id-only) publish must never clobber the ordered slot"
    );
}

/// Two DIFFERENT columns built for the ordered slot at the SAME
/// generation must both actually reach the published cache -- merge-or-keep
/// must compare column identity, not just `is_built()`. Without the column
/// check, "both built" reads as "unchanged" and the second publish never
/// reaches the cache at all, so a reader refreshing to that generation pays
/// a full rebuild for the second column even though the writer built and
/// tried to publish it.
#[test]
fn ordered_merge_or_keep_compares_column_identity_not_just_built() {
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let (_dir, mut writer, path) = open_recs_ordered();
    seed_recs(&mut writer, 20);

    // First reader demands `created_at`; the writer's next commit builds and
    // publishes it.
    {
        let mut r = Connection::open_read_only(&path, 384).unwrap();
        r.execute("SELECT id FROM recs ORDER BY created_at DESC LIMIT 5", vec![])
            .unwrap();
    }
    writer.execute("BEGIN", vec![]).unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    let r1 = Connection::open_read_only(&path, 384).unwrap();
    assert!(
        r1.ordered_index_is_built("recs"),
        "a reader at this generation must adopt the writer's published `created_at` ordered index"
    );
    drop(r1);

    // Second reader demands a DIFFERENT declared ordered column (`pending`).
    // The writer's next commit is an EMPTY transaction -- it touches no
    // col-indexed column of `recs`, so `recs`'s col_generation does not move
    // and the publish key stays IDENTICAL to the first publish above, while
    // the writer's demand-driven warmup switches its local ordered_caches
    // entry to `pending` and tries to publish it.
    {
        let mut r = Connection::open_read_only(&path, 384).unwrap();
        r.execute("SELECT id FROM recs ORDER BY pending DESC LIMIT 5", vec![])
            .unwrap();
    }
    writer.execute("BEGIN", vec![]).unwrap();
    writer.execute("COMMIT", vec![]).unwrap();

    // A FRESH reader's first `pending`-ordered read must be served by
    // adoption (near-zero cost), not a local rebuild -- proving the second
    // column's publish actually reached the cache.
    let mut r2 = Connection::open_read_only(&path, 384).unwrap();
    r2.reset_cells_visited_count();
    let mut cur = r2
        .execute("SELECT id FROM recs ORDER BY pending DESC LIMIT 5", vec![])
        .unwrap();
    let rows = cur.fetchall();
    let cells = r2.cells_visited_count();
    assert_eq!(rows.len(), 5);
    assert!(
        cells <= 4,
        "post-publish `pending`-ordered read cost {cells} cells -- the second column's \
         publish at the same generation as `created_at`'s was silently dropped by \
         merge-or-keep treating \"both built\" as \"unchanged\""
    );
}

/// (iii) Bounded writer cost: publishing the ordered component after a
/// single-row commit on a warmed store does not scan O(corpus) — the ordered
/// map is shared by refcount (Arc/COW), not copied, and per-row maintenance
/// touches only the changed row.
#[test]
fn ordered_publish_and_maintenance_stay_bounded_on_the_writer() {
    let (_dir, mut writer, _path) = open_recs_ordered();
    seed_recs(&mut writer, 500);
    warm_ordered_and_id(&mut writer);

    writer.reset_cells_visited_count();
    writer.reset_full_scan_count();
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?)",
            vec![t("bounded-1"), Value::Int(1), t("2026-04-01T00:00:00"), t("p")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    assert_eq!(
        writer.full_scan_count(),
        0,
        "a single-row INSERT on a warmed store must not trigger a whole-tree rescan \
         while publishing/maintaining the ordered component"
    );

    // An UPDATE that touches the ordered column re-files ONE row, not the
    // whole index.
    writer.reset_cells_visited_count();
    writer.reset_full_scan_count();
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "UPDATE recs SET created_at = ? WHERE id = ?",
            vec![t("2099-01-01T00:00:00"), t("bounded-1")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    assert_eq!(
        writer.full_scan_count(),
        0,
        "an UPDATE resolved via the id fast path must not full-scan even when it \
         touches the ordered column"
    );
}

/// COMPLETENESS TWIN (correctness, distinct from cost): a row committed
/// immediately before a refresh is returned by the post-refresh id AND
/// ordered reads, for BOTH the single-row and the executemany-bulk UPDATE
/// route that mirrors a real embedding-pending-flag write.
#[test]
fn completeness_twin_single_row_and_bulk_update_after_refresh() {
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let (_dir, mut writer, path) = open_recs_ordered();
    seed_recs(&mut writer, 200);
    warm_ordered_and_id(&mut writer);

    let mut ro = Connection::open_read_only(&path, 384).unwrap();
    ro.execute("SELECT id FROM recs ORDER BY created_at DESC LIMIT 5", vec![])
        .unwrap();
    ro.execute("SELECT id FROM recs WHERE id = ?", vec![t("id-00000")])
        .unwrap();

    // Single-row route: an UPDATE that does NOT touch the ordered column.
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "UPDATE recs SET pending = 0 WHERE id = ?",
            vec![t("id-00001")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();
    let advanced = ro.refresh_read_view().unwrap();
    assert!(advanced, "the pager snapshot must advance after a committed write");
    let mut cur = ro
        .execute("SELECT pending FROM recs WHERE id = ?", vec![t("id-00001")])
        .unwrap();
    assert_eq!(
        cur.fetchall()[0].get_index(0),
        Some(&Value::Int(0)),
        "the single-row UPDATE's committed value must be visible post-refresh"
    );
    let mut cur = ro
        .execute("SELECT id FROM recs ORDER BY created_at DESC LIMIT 5", vec![])
        .unwrap();
    assert_eq!(cur.fetchall().len(), 5, "ordered read must still return rows post-refresh");

    // Bulk route: an executemany UPDATE mirroring reembed_pending_rows.
    let bulk_updates: Vec<Vec<Value>> = (2..12).map(|i| vec![Value::Int(0), t(&format!("id-{i:05}"))]).collect();
    writer
        .executemany("UPDATE recs SET pending = ? WHERE id = ?", bulk_updates)
        .unwrap();
    let advanced = ro.refresh_read_view().unwrap();
    assert!(advanced, "the pager snapshot must advance after the bulk commit");
    let mut cur = ro
        .execute(
            "SELECT pending FROM recs WHERE id = ?",
            vec![t("id-00005")],
        )
        .unwrap();
    assert_eq!(
        cur.fetchall()[0].get_index(0),
        Some(&Value::Int(0)),
        "a row committed via the executemany-bulk UPDATE route must be visible post-refresh \
         (the completeness twin the plan's threat model requires)"
    );
}

/// Schema-identity de-adopt: an ordered index built for a column that is no
/// longer a declared non-partial index on the table must never be adopted —
/// mirrors the col-side `same_column_set` gate.
#[test]
fn ordered_index_not_adopted_after_its_declared_index_is_dropped() {
    std::env::set_var("LILLI_INDEX_PUBLISH_MIN_INTERVAL_MS", "0");
    let (_dir, mut writer, path) = open_recs_ordered();
    seed_recs(&mut writer, 10);
    warm_ordered_and_id(&mut writer);
    drop(Connection::open_read_only(&path, 384).unwrap());

    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO recs (id, pending, created_at, payload) VALUES (?, ?, ?, ?)",
            vec![t("si-1"), Value::Int(1), t("2026-05-01T00:00:00"), t("p")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();

    let r1 = Connection::open_read_only(&path, 384).unwrap();
    assert!(r1.ordered_index_is_built("recs"));
    drop(r1);

    // DROP INDEX is unsupported by this SQL subset, and ALTER TABLE DROP
    // COLUMN does not retract the CREATE INDEX declaration it leaves behind
    // (a pre-existing engine limitation, not introduced by this plan) — so
    // neither reaches the schema-identity discriminator. DROP + recreate the
    // table under the SAME name WITHOUT the ordered column's declared index
    // is the reachable equivalent: the catalog's per-table index-column set
    // is rebuilt from scratch, genuinely no longer declaring `created_at`.
    writer.execute("DROP TABLE recs", vec![]).unwrap();
    writer
        .execute(
            "CREATE TABLE recs ( vec_label INTEGER PRIMARY KEY AUTOINCREMENT , \
             id TEXT NOT NULL UNIQUE , pending INTEGER , payload TEXT )",
            vec![],
        )
        .unwrap();
    writer
        .execute("CREATE INDEX idx_recs_pending ON recs (pending)", vec![])
        .unwrap();
    writer.execute("BEGIN", vec![]).unwrap();
    writer
        .execute(
            "INSERT INTO recs (id, pending, payload) VALUES (?, ?, ?)",
            vec![t("si-2"), Value::Int(1), t("p")],
        )
        .unwrap();
    writer.execute("COMMIT", vec![]).unwrap();

    let r2 = Connection::open_read_only(&path, 384).unwrap();
    assert!(
        !r2.ordered_index_is_built("recs"),
        "a schema change that no longer declares the ordered column as a \
         non-partial index must de-adopt any published ordered component \
         built for that column"
    );
}

//! Write-path behavior tests: INSERT (vec_label AUTOINCREMENT injection, next-key
//! via max_key, OR IGNORE / OR REPLACE), ON CONFLICT DO UPDATE / NOTHING with
//! `excluded.col` and NULL-PK-never-conflicts, UPDATE / DELETE with the
//! `id = '<literal>'` point-lookup fast path, and executemany batch atomicity with
//! the per-statement transaction suppressed for the batch duration.
//!
//! The expected results encode stdlib sqlite3 semantics: the next B-tree key is
//! one past the maximum (derived in O(tree-height), never a per-row full scan —
//! the quadratic-insert regression guard), a NULL conflict key never collides,
//! and a partial batch never reaches the store.

use std::collections::BTreeMap;

use lilliengine::catalog::Catalog;
use lilliengine::exec::{
    execute_delete, execute_insert, execute_insert_many, execute_select, execute_update,
    ConflictIndex, IdIndex, TxnScope, WriteOutcome,
};
use lilliengine::EngineError;
use lilliengine::meta::MetaTable;
use lilliengine::parser::parse;
use lilliengine::plan::plan_select;
use lilliengine::ast::Stmt;
use lillibrain::{Store, Value};
use tempfile::tempdir;

// A records-shaped table: integer AUTOINCREMENT vec_label PK + a UNIQUE text id.
const DDL_RECORDS: &str = "CREATE TABLE IF NOT EXISTS records ( \
    vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
    n INTEGER , label TEXT )";

// An edges-shaped table with a composite primary key, the primary-key UPSERT
// target (HippoMergeInsert drives ON CONFLICT against this shape).
const DDL_EDGES: &str = "CREATE TABLE IF NOT EXISTS edges ( \
    src TEXT , dst TEXT , edge_type TEXT , weight REAL , updated_at TEXT , \
    PRIMARY KEY ( src , dst , edge_type ) )";

/// A store fixture over one or more tables, returning everything the write and
/// read executors need plus the durable meta handle for vec_label injection.
struct Fixture {
    _dir: tempfile::TempDir,
    store: Store,
    meta: MetaTable,
    catalog: Catalog,
    root_map: BTreeMap<String, u32>,
}

impl Fixture {
    fn new(ddls: &[&str]) -> Fixture {
        let dir = tempdir().unwrap();
        let path = dir.path().join("t.lilli");
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut catalog = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        for ddl in ddls {
            meta.create_table(&store, &mut catalog, &mut root_map, ddl)
                .unwrap();
        }
        Fixture { _dir: dir, store, meta, catalog, root_map }
    }

    /// Parse + execute an INSERT, returning the lastrowid (if injected).
    fn insert(&self, sql: &str, params: Vec<Value>) -> Option<i64> {
        self.insert_idx(sql, params, None)
    }

    fn insert_idx(&self, sql: &str, params: Vec<Value>, idx: Option<&mut IdIndex>) -> Option<i64> {
        let stmt = match parse(sql).unwrap() {
            Stmt::Insert(i) => i,
            _ => panic!("not an insert"),
        };
        execute_insert(
            &stmt,
            &self.store,
            &self.meta,
            &self.catalog,
            &self.root_map,
            params,
            None,
            TxnScope::Own,
            idx,
            None,
            None,
            None,
        )
        .unwrap()
        .lastrowid
    }

    /// Parse + execute an INSERT routed through a caller-owned conflict index
    /// (the bulk UPSERT fast path), returning the lastrowid (if injected).
    fn insert_cidx(
        &self,
        sql: &str,
        params: Vec<Value>,
        cidx: &mut ConflictIndex,
    ) -> Option<i64> {
        let stmt = match parse(sql).unwrap() {
            Stmt::Insert(i) => i,
            _ => panic!("not an insert"),
        };
        execute_insert(
            &stmt,
            &self.store,
            &self.meta,
            &self.catalog,
            &self.root_map,
            params,
            None,
            TxnScope::Own,
            None,
            Some(cidx),
            None,
            None,
        )
        .unwrap()
        .lastrowid
    }

    /// Like [`insert_cidx`] but surfaces the raw `Result` so a test can assert a
    /// constraint rejection (e.g. a UNIQUE collision) instead of unwrapping.
    fn try_insert_cidx(
        &self,
        sql: &str,
        params: Vec<Value>,
        cidx: &mut ConflictIndex,
    ) -> Result<WriteOutcome, EngineError> {
        let stmt = match parse(sql).unwrap() {
            Stmt::Insert(i) => i,
            _ => panic!("not an insert"),
        };
        execute_insert(
            &stmt,
            &self.store,
            &self.meta,
            &self.catalog,
            &self.root_map,
            params,
            None,
            TxnScope::Own,
            None,
            Some(cidx),
            None,
            None,
        )
    }

    fn update(&self, sql: &str, params: Vec<Value>, idx: Option<&mut IdIndex>) -> i64 {
        let stmt = match parse(sql).unwrap() {
            Stmt::Update(u) => u,
            _ => panic!("not an update"),
        };
        execute_update(
            &stmt,
            &self.store,
            &self.meta,
            &self.catalog,
            &self.root_map,
            params,
            None,
            TxnScope::Own,
            idx,
            None,
            None,
            None,
        )
        .unwrap()
        .rowcount
    }

    fn delete(&self, sql: &str, params: Vec<Value>) -> i64 {
        let stmt = match parse(sql).unwrap() {
            Stmt::Delete(d) => d,
            _ => panic!("not a delete"),
        };
        execute_delete(
            &stmt,
            &self.store,
            &self.meta,
            &self.catalog,
            &self.root_map,
            params,
            None,
            TxnScope::Own,
            None,
            None,
            None,
            None,
        )
        .unwrap()
        .rowcount
    }

    /// Read every row of a table as `(col → value)` maps in key order.
    fn rows(&self, table: &str) -> Vec<Vec<(String, Value)>> {
        let stmt = parse(&format!("SELECT * FROM {table}")).unwrap();
        let select = match stmt {
            Stmt::Select(s) => s,
            _ => panic!("not a select"),
        };
        let plan = plan_select(&select, &self.catalog, vec![], None).unwrap();
        let rs = execute_select(&plan, &self.store, &self.catalog, &self.root_map, None, None, None).unwrap();
        rs.rows
    }

    /// Build a table's id-index from the current tree (one scan), so a later
    /// `WHERE id = '<lit>'` point lookup runs scan-free.
    fn build_id_index(&self, table: &str) -> IdIndex {
        let root = *self.root_map.get(table).unwrap();
        let col_names = self.catalog.column_names(table).unwrap();
        let mut idx = IdIndex::new();
        idx.ensure_built(&self.store.tree(root), &col_names).unwrap();
        idx
    }

    fn cell(row: &[(String, Value)], col: &str) -> Value {
        row.iter().find(|(c, _)| c == col).map(|(_, v)| v.clone()).unwrap_or(Value::Null)
    }
}

fn t(s: &str) -> Value {
    Value::Text(s.to_string())
}

// ---------------------------------------------------------------------------
// INSERT — vec_label injection, max_key next-key, OR IGNORE / OR REPLACE
// ---------------------------------------------------------------------------

#[test]
fn insert_injects_and_exposes_vec_label() {
    let f = Fixture::new(&[DDL_RECORDS]);
    let id1 = f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("a"), Value::Int(10)]);
    let id2 = f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("b"), Value::Int(20)]);
    // The injected vec_label is the monotonic HWM exposed as lastrowid.
    assert_eq!(id1, Some(1));
    assert_eq!(id2, Some(2));
    let rows = f.rows("records");
    assert_eq!(rows.len(), 2);
    assert_eq!(Fixture::cell(&rows[0], "vec_label"), Value::Int(1));
    assert_eq!(Fixture::cell(&rows[0], "id"), t("a"));
    assert_eq!(Fixture::cell(&rows[1], "vec_label"), Value::Int(2));
}

#[test]
fn insert_explicit_vec_label_is_not_reinjected() {
    let f = Fixture::new(&[DDL_RECORDS]);
    let id = f.insert(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(42), t("a"), Value::Int(1)],
    );
    // Caller supplied vec_label — no injection, so no lastrowid is reported.
    assert_eq!(id, None);
    let rows = f.rows("records");
    assert_eq!(Fixture::cell(&rows[0], "vec_label"), Value::Int(42));
}

#[test]
fn insert_next_key_via_max_key_no_full_scan_per_row() {
    // The quadratic-insert regression guard: a bulk insert loop derives every
    // next key from Tree::max_key (O(tree-height)), so the store's full-scan
    // counter stays at zero across the whole loop. A per-row full scan (the
    // O(rows^2) killer) would lift it.
    let f = Fixture::new(&[DDL_RECORDS]);
    f.store.reset_full_scan_count();
    for i in 0..2000 {
        f.insert(
            "INSERT INTO records (id, n) VALUES (?, ?)",
            vec![t(&format!("id-{i}")), Value::Int(i)],
        );
    }
    assert_eq!(
        f.store.full_scan_count(),
        0,
        "insert path must not full_scan per row (the O(rows^2) regression)"
    );
    // And the rows really landed with monotonic keys.
    let rows = f.rows("records");
    assert_eq!(rows.len(), 2000);
    assert_eq!(Fixture::cell(rows.last().unwrap(), "vec_label"), Value::Int(2000));
}

#[test]
fn insert_or_ignore_skips_collision() {
    // OR IGNORE detects a conflict on the catalog primary key (vec_label here).
    // A second row with the same explicit vec_label is skipped.
    let f = Fixture::new(&[DDL_RECORDS]);
    f.insert(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(7), t("a"), Value::Int(1)],
    );
    f.insert(
        "INSERT OR IGNORE INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(7), t("b"), Value::Int(2)],
    );
    let rows = f.rows("records");
    assert_eq!(rows.len(), 1);
    assert_eq!(Fixture::cell(&rows[0], "id"), t("a")); // original kept
    assert_eq!(Fixture::cell(&rows[0], "n"), Value::Int(1));
}

#[test]
fn insert_or_replace_overwrites_collision() {
    let f = Fixture::new(&[DDL_RECORDS]);
    f.insert(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(7), t("a"), Value::Int(1)],
    );
    f.insert(
        "INSERT OR REPLACE INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(7), t("b"), Value::Int(2)],
    );
    let rows = f.rows("records");
    assert_eq!(rows.len(), 1);
    assert_eq!(Fixture::cell(&rows[0], "id"), t("b")); // overwritten in place
    assert_eq!(Fixture::cell(&rows[0], "n"), Value::Int(2));
}

#[test]
fn insert_bind_count_mismatch_too_few() {
    let f = Fixture::new(&[DDL_RECORDS]);
    let stmt = match parse("INSERT INTO records (id, n) VALUES (?, ?)").unwrap() {
        Stmt::Insert(i) => i,
        _ => unreachable!(),
    };
    let err = execute_insert(
        &stmt,
        &f.store,
        &f.meta,
        &f.catalog,
        &f.root_map,
        vec![t("a")], // one value for two placeholders
        None,
        TxnScope::Own,
        None,
        None,
        None,
        None,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("Incorrect number of bindings supplied"));
}

#[test]
fn insert_bind_count_mismatch_too_many() {
    let f = Fixture::new(&[DDL_RECORDS]);
    let stmt = match parse("INSERT INTO records (id, n) VALUES (?, ?)").unwrap() {
        Stmt::Insert(i) => i,
        _ => unreachable!(),
    };
    let err = execute_insert(
        &stmt,
        &f.store,
        &f.meta,
        &f.catalog,
        &f.root_map,
        vec![t("a"), Value::Int(1), Value::Int(2)], // surplus
        None,
        TxnScope::Own,
        None,
        None,
        None,
        None,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("Incorrect number of bindings supplied"));
}

// ---------------------------------------------------------------------------
// ON CONFLICT / merge_insert — composite-PK edges UPSERT
// ---------------------------------------------------------------------------

const UPSERT: &str = "INSERT INTO edges (src, dst, edge_type, weight, updated_at) \
    VALUES (?, ?, ?, ?, ?) ON CONFLICT (src, dst, edge_type) \
    DO UPDATE SET weight = excluded.weight , updated_at = excluded.updated_at";

#[test]
fn upsert_inserts_then_updates_in_place() {
    let f = Fixture::new(&[DDL_EDGES]);
    // First write inserts the edge.
    f.insert(
        UPSERT,
        vec![t("a"), t("b"), t("rel"), Value::Float(1.0), t("t1")],
    );
    // Second write on the same composite key updates weight/updated_at in place.
    f.insert(
        UPSERT,
        vec![t("a"), t("b"), t("rel"), Value::Float(9.0), t("t2")],
    );
    let rows = f.rows("edges");
    assert_eq!(rows.len(), 1, "the composite key must not duplicate");
    assert_eq!(Fixture::cell(&rows[0], "weight"), Value::Float(9.0)); // excluded.weight
    assert_eq!(Fixture::cell(&rows[0], "updated_at"), t("t2")); // excluded.updated_at
}

#[test]
fn upsert_distinct_keys_each_insert() {
    let f = Fixture::new(&[DDL_EDGES]);
    f.insert(UPSERT, vec![t("a"), t("b"), t("rel"), Value::Float(1.0), t("t1")]);
    f.insert(UPSERT, vec![t("a"), t("b"), t("other"), Value::Float(2.0), t("t1")]);
    f.insert(UPSERT, vec![t("x"), t("y"), t("rel"), Value::Float(3.0), t("t1")]);
    assert_eq!(f.rows("edges").len(), 3);
}

#[test]
fn upsert_do_nothing_keeps_existing() {
    // An all-columns-are-keys table (DO NOTHING form) skips the duplicate but
    // keeps the rest of the batch.
    const DDL_TAGS: &str =
        "CREATE TABLE tags ( a TEXT , b TEXT , PRIMARY KEY ( a , b ) )";
    const DO_NOTHING: &str =
        "INSERT INTO tags (a, b) VALUES (?, ?) ON CONFLICT (a, b) DO NOTHING";
    let f = Fixture::new(&[DDL_TAGS]);
    f.insert(DO_NOTHING, vec![t("x"), t("y")]);
    f.insert(DO_NOTHING, vec![t("x"), t("y")]); // duplicate → skipped
    f.insert(DO_NOTHING, vec![t("x"), t("z")]); // distinct → inserted
    assert_eq!(f.rows("tags").len(), 2);
}

#[test]
fn upsert_null_conflict_key_never_conflicts() {
    // A conflict key column that holds NULL never matches (sqlite3 uniqueness
    // treats NULL != NULL), so two rows sharing a NULL in a conflict column both
    // insert. The table leaves the conflict columns nullable (no PRIMARY KEY,
    // which would force NOT NULL) and names them in an explicit ON CONFLICT.
    const DDL_LINKS: &str =
        "CREATE TABLE links ( src TEXT , dst TEXT , edge_type TEXT , weight REAL )";
    const UPSERT_LINKS: &str = "INSERT INTO links (src, dst, edge_type, weight) \
        VALUES (?, ?, ?, ?) ON CONFLICT (src, dst, edge_type) \
        DO UPDATE SET weight = excluded.weight";
    let f = Fixture::new(&[DDL_LINKS]);
    f.insert(UPSERT_LINKS, vec![t("a"), t("b"), Value::Null, Value::Float(1.0)]);
    f.insert(UPSERT_LINKS, vec![t("a"), t("b"), Value::Null, Value::Float(2.0)]);
    assert_eq!(f.rows("links").len(), 2, "a NULL conflict key always inserts");
}

#[test]
fn or_ignore_uses_catalog_pk_as_conflict_key() {
    // OR IGNORE with no explicit ON CONFLICT clause falls back to the catalog
    // primary key for conflict detection.
    let f = Fixture::new(&[DDL_EDGES]);
    f.insert(
        "INSERT INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)",
        vec![t("a"), t("b"), t("rel"), Value::Float(1.0)],
    );
    f.insert(
        "INSERT OR IGNORE INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)",
        vec![t("a"), t("b"), t("rel"), Value::Float(2.0)],
    );
    let rows = f.rows("edges");
    assert_eq!(rows.len(), 1);
    assert_eq!(Fixture::cell(&rows[0], "weight"), Value::Float(1.0)); // original kept
}

// ---------------------------------------------------------------------------
// ON CONFLICT with the conflict-key index (the bulk-UPSERT fast path)
// ---------------------------------------------------------------------------

#[test]
fn indexed_upsert_matches_unindexed_result() {
    // The indexed conflict path must produce the byte-identical result the
    // unindexed scan path does: same insert-then-update-in-place behavior on the
    // composite key, same final row set.
    let f = Fixture::new(&[DDL_EDGES]);
    let mut cidx = ConflictIndex::default();
    f.insert_cidx(UPSERT, vec![t("a"), t("b"), t("rel"), Value::Float(1.0), t("t1")], &mut cidx);
    // Same key → update weight/updated_at in place via the O(1) index probe.
    f.insert_cidx(UPSERT, vec![t("a"), t("b"), t("rel"), Value::Float(9.0), t("t2")], &mut cidx);
    // Distinct keys → fresh inserts.
    f.insert_cidx(UPSERT, vec![t("a"), t("b"), t("other"), Value::Float(2.0), t("t1")], &mut cidx);
    f.insert_cidx(UPSERT, vec![t("x"), t("y"), t("rel"), Value::Float(3.0), t("t1")], &mut cidx);

    let rows = f.rows("edges");
    assert_eq!(rows.len(), 3, "the composite key must not duplicate under the index");
    let rel = rows
        .iter()
        .find(|r| Fixture::cell(r, "src") == t("a") && Fixture::cell(r, "edge_type") == t("rel"))
        .expect("the (a,b,rel) edge");
    assert_eq!(Fixture::cell(rel, "weight"), Value::Float(9.0)); // updated in place
    assert_eq!(Fixture::cell(rel, "updated_at"), t("t2"));
}

#[test]
fn indexed_bulk_upsert_does_not_full_scan_per_row() {
    // The A4 durability gate at unit scale: a bulk UPSERT through the conflict
    // index issues at most ONE whole-table scan (the lazy index build), never one
    // per row. A per-row scan is the O(rows^2) regression; the counter would then
    // climb with the row count instead of staying at the single build scan.
    let f = Fixture::new(&[DDL_EDGES]);
    let mut cidx = ConflictIndex::default();
    // Seed one row so the lazy build has something to scan.
    f.insert_cidx(UPSERT, vec![t("a"), t("b"), t("rel"), Value::Float(0.0), t("t0")], &mut cidx);

    f.store.reset_full_scan_count();
    // 500 UPSERTs: a fan of repeated keys (updates) and fresh keys (inserts).
    for i in 0..500 {
        let src = format!("n{}", i % 16);
        let dst = format!("n{}", i % 8);
        f.insert_cidx(
            UPSERT,
            vec![t(&src), t(&dst), t("rel"), Value::Float(i as f64), t("t")],
            &mut cidx,
        );
    }
    let scans = f.store.full_scan_count();
    assert!(
        scans <= 1,
        "indexed bulk UPSERT issued {scans} full scans over 500 rows (must be <= 1: the lazy \
         build only); a per-row scan is the O(rows^2) conflict regression"
    );
    // Correctness still holds: 16*8 distinct (src,dst) combinations at most.
    assert!(f.rows("edges").len() <= 16 * 8 + 1);
}

#[test]
fn indexed_upsert_null_conflict_key_still_inserts() {
    // A NULL in a conflict column never matches under the index either (the
    // serialized conflict key is None, so the row is not indexed and always
    // inserts) — the same sqlite3 NULL-uniqueness semantics as the scan path.
    const DDL_LINKS: &str =
        "CREATE TABLE links ( src TEXT , dst TEXT , edge_type TEXT , weight REAL )";
    const UPSERT_LINKS: &str = "INSERT INTO links (src, dst, edge_type, weight) \
        VALUES (?, ?, ?, ?) ON CONFLICT (src, dst, edge_type) \
        DO UPDATE SET weight = excluded.weight";
    let f = Fixture::new(&[DDL_LINKS]);
    let mut cidx = ConflictIndex::default();
    f.insert_cidx(UPSERT_LINKS, vec![t("a"), t("b"), Value::Null, Value::Float(1.0)], &mut cidx);
    f.insert_cidx(UPSERT_LINKS, vec![t("a"), t("b"), Value::Null, Value::Float(2.0)], &mut cidx);
    assert_eq!(f.rows("links").len(), 2, "a NULL conflict key always inserts under the index");
}

#[test]
fn indexed_upsert_after_delete_reinserts() {
    // Deleting a conflicting row invalidates the index; a later UPSERT of the same
    // key must insert fresh (not update a row that no longer exists). This proves
    // the delete-invalidation keeps the index consistent with the tree.
    let f = Fixture::new(&[DDL_EDGES]);
    let mut cidx = ConflictIndex::default();
    f.insert_cidx(UPSERT, vec![t("a"), t("b"), t("rel"), Value::Float(1.0), t("t1")], &mut cidx);
    // Delete the row through the same conflict cache (invalidates it).
    {
        let stmt = match parse("DELETE FROM edges WHERE src = 'a'").unwrap() {
            Stmt::Delete(d) => d,
            _ => unreachable!(),
        };
        execute_delete(
            &stmt, &f.store, &f.meta, &f.catalog, &f.root_map, vec![], None, TxnScope::Own, None,
            Some(&mut cidx), None, None,
        )
        .unwrap();
    }
    assert_eq!(f.rows("edges").len(), 0);
    // Re-UPSERT the same key: must insert fresh, leaving exactly one row.
    f.insert_cidx(UPSERT, vec![t("a"), t("b"), t("rel"), Value::Float(5.0), t("t2")], &mut cidx);
    let rows = f.rows("edges");
    assert_eq!(rows.len(), 1, "the re-UPSERT after delete must insert, not vanish");
    assert_eq!(Fixture::cell(&rows[0], "weight"), Value::Float(5.0));
}

// ---------------------------------------------------------------------------
// UPDATE / DELETE + id='lit' fast path + executemany atomicity
// ---------------------------------------------------------------------------

#[test]
fn update_full_scan_where() {
    let f = Fixture::new(&[DDL_RECORDS]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("a"), Value::Int(1)]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("b"), Value::Int(1)]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("c"), Value::Int(5)]);
    let n = f.update("UPDATE records SET label = ? WHERE n = ?", vec![t("hit"), Value::Int(1)], None);
    assert_eq!(n, 2);
    let rows = f.rows("records");
    assert_eq!(Fixture::cell(&rows[0], "label"), t("hit"));
    assert_eq!(Fixture::cell(&rows[1], "label"), t("hit"));
    assert_eq!(Fixture::cell(&rows[2], "label"), Value::Null);
}

#[test]
fn update_id_literal_fast_path_no_full_scan() {
    // WHERE id = '<literal>' with a built id-index resolves by a point lookup —
    // the merge_insert provenance UPDATE path. Once the index is built, the
    // store's full-scan counter stays at zero for the UPDATE.
    let f = Fixture::new(&[DDL_RECORDS]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("a"), Value::Int(1)]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("b"), Value::Int(2)]);
    let mut idx = f.build_id_index("records");

    f.store.reset_full_scan_count();
    let n = f.update("UPDATE records SET n = ? WHERE id = 'b'", vec![Value::Int(99)], Some(&mut idx));
    assert_eq!(n, 1);
    assert_eq!(f.store.full_scan_count(), 0, "id='lit' UPDATE must be a point lookup");
    let rows = f.rows("records");
    assert_eq!(Fixture::cell(&rows[1], "n"), Value::Int(99));
}

#[test]
fn update_id_literal_absent_skips_scan() {
    let f = Fixture::new(&[DDL_RECORDS]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("a"), Value::Int(1)]);
    let mut idx = f.build_id_index("records");
    f.store.reset_full_scan_count();
    let n = f.update("UPDATE records SET n = ? WHERE id = 'missing'", vec![Value::Int(99)], Some(&mut idx));
    assert_eq!(n, 0);
    assert_eq!(f.store.full_scan_count(), 0, "an absent id skips the scan entirely");
}

#[test]
fn delete_where_and_clear_all() {
    let f = Fixture::new(&[DDL_RECORDS]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("a"), Value::Int(1)]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("b"), Value::Int(2)]);
    f.insert("INSERT INTO records (id, n) VALUES (?, ?)", vec![t("c"), Value::Int(3)]);
    // WHERE deletes the matching row.
    let n = f.delete("DELETE FROM records WHERE n = ?", vec![Value::Int(2)]);
    assert_eq!(n, 1);
    assert_eq!(f.rows("records").len(), 2);
    // No WHERE clears the whole table.
    let n = f.delete("DELETE FROM records", vec![]);
    assert_eq!(n, 2);
    assert_eq!(f.rows("records").len(), 0);
}

#[test]
fn executemany_is_atomic_and_suppresses_inner_txn() {
    // A batch of INSERTs wraps the whole sequence in ONE store transaction; the
    // inner per-statement begin_write is suppressed so no nested begin is opened
    // (which the pager would reject). The batch commits as one unit and every row
    // lands with its own monotonic vec_label.
    let f = Fixture::new(&[DDL_RECORDS]);
    let stmt = match parse("INSERT INTO records (id, n) VALUES (?, ?)").unwrap() {
        Stmt::Insert(i) => i,
        _ => unreachable!(),
    };
    let seq = vec![
        vec![t("a"), Value::Int(1)],
        vec![t("b"), Value::Int(2)],
        vec![t("c"), Value::Int(3)],
    ];
    let outcome = execute_insert_many(
        &stmt, &f.store, &f.meta, &f.catalog, &f.root_map, seq, None, None, None, None,
    )
    .unwrap();
    assert_eq!(outcome.rowcount, 3);
    let rows = f.rows("records");
    assert_eq!(rows.len(), 3);
    assert_eq!(Fixture::cell(&rows[0], "vec_label"), Value::Int(1));
    assert_eq!(Fixture::cell(&rows[2], "vec_label"), Value::Int(3));
}

#[test]
fn executemany_mid_batch_error_rolls_back_whole_batch() {
    // A NOT NULL violation mid-batch must roll the entire batch back: none of the
    // rows (including the ones before the failure) survive.
    let f = Fixture::new(&[DDL_RECORDS]);
    let stmt = match parse("INSERT INTO records (id, n) VALUES (?, ?)").unwrap() {
        Stmt::Insert(i) => i,
        _ => unreachable!(),
    };
    let seq = vec![
        vec![t("a"), Value::Int(1)],
        vec![Value::Null, Value::Int(2)], // id is NOT NULL → fails mid-batch
        vec![t("c"), Value::Int(3)],
    ];
    let err = execute_insert_many(
        &stmt, &f.store, &f.meta, &f.catalog, &f.root_map, seq, None, None, None, None,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("NOT NULL constraint failed"));
    assert_eq!(f.rows("records").len(), 0, "a partial batch must not survive");
}

#[test]
fn conflict_update_rewriting_id_keeps_id_index_current() {
    // The conflict-hit DO UPDATE branch must reconcile the id-index when the SET
    // rewrites the `id` column: drop the old id → row-key entry and record the new
    // id at the same key, so a later `WHERE id = '<new>'` point lookup still finds
    // the row (and the stale `<old>` no longer resolves). Production UPSERTs touch
    // only non-key columns, so this is latent today — but the id-index invariant
    // must hold for any caller that rewrites a key column.
    let f = Fixture::new(&[DDL_RECORDS]);
    // Build the (empty) id-index first so per-row insert maintenance records ids
    // (an unbuilt index defers to a later ensure_built rebuild from the tree).
    let mut idx = f.build_id_index("records");
    let mut cidx = ConflictIndex::default();

    // Seed one row at vec_label 7 with id 'old'. The id-index records 'old' -> key.
    {
        let stmt = match parse("INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?)").unwrap() {
            Stmt::Insert(i) => i,
            _ => unreachable!(),
        };
        execute_insert(
            &stmt, &f.store, &f.meta, &f.catalog, &f.root_map,
            vec![Value::Int(7), t("old"), Value::Int(1)],
            None, TxnScope::Own, Some(&mut idx), Some(&mut cidx), None, None,
        )
        .unwrap();
    }
    let row_key = idx.lookup("old").expect("the seeded id is indexed");

    // UPSERT conflicting on vec_label 7 with DO UPDATE SET id = 'new'. The conflict
    // key (vec_label) is stable; the rewrite changes the row's `id`.
    {
        let stmt = match parse(
            "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
             ON CONFLICT (vec_label) DO UPDATE SET id = excluded.id , n = excluded.n",
        )
        .unwrap()
        {
            Stmt::Insert(i) => i,
            _ => unreachable!(),
        };
        execute_insert(
            &stmt, &f.store, &f.meta, &f.catalog, &f.root_map,
            vec![Value::Int(7), t("new"), Value::Int(2)],
            None, TxnScope::Own, Some(&mut idx), Some(&mut cidx), None, None,
        )
        .unwrap();
    }

    // The id-index dropped the stale 'old' and recorded 'new' at the same key.
    assert!(idx.lookup("old").is_none(), "the stale id must be removed from the index");
    assert_eq!(idx.lookup("new"), Some(row_key), "the new id must map to the same row-key");

    // The row was updated in place (no duplicate) with the new id and n.
    let rows = f.rows("records");
    assert_eq!(rows.len(), 1);
    assert_eq!(Fixture::cell(&rows[0], "id"), t("new"));
    assert_eq!(Fixture::cell(&rows[0], "n"), Value::Int(2));
}

#[test]
fn conflict_index_reconciled_on_key_change() {
    // A DO UPDATE that rewrites a conflict-key column must re-key the conflict
    // index: the old key value is dropped (a later insert of it inserts fresh, no
    // false conflict) and the new key value is filed (a later insert of it
    // conflicts). Without the re-file the index still points the old key at the
    // rewritten row, so an insert of the old value falsely conflicts and an insert
    // of the new value misses the conflict.
    let f = Fixture::new(&[DDL_RECORDS]);
    let mut cidx = ConflictIndex::default();

    // Build the conflict index over the empty records table (keyed on vec_label),
    // then seed one row at vec_label 1.
    f.insert_cidx(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
         ON CONFLICT (vec_label) DO UPDATE SET n = excluded.n",
        vec![Value::Int(1), t("a"), Value::Int(10)],
        &mut cidx,
    );

    // DO UPDATE conflicting on vec_label 1, rewriting the conflict-key column to a
    // new literal value 2 (a key-column rewrite — the attempted row still keys on
    // 1 to find the conflict, the SET moves the stored key to 2).
    f.insert_cidx(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
         ON CONFLICT (vec_label) DO UPDATE SET vec_label = 2 , n = excluded.n",
        vec![Value::Int(1), t("a"), Value::Int(20)],
        &mut cidx,
    );
    // The single row now carries vec_label 2.
    {
        let rows = f.rows("records");
        assert_eq!(rows.len(), 1, "the rewrite updated in place, no duplicate");
        assert_eq!(Fixture::cell(&rows[0], "vec_label"), Value::Int(2));
    }

    // Insert the OLD key value (1): it must insert fresh — the stale entry was
    // dropped, so there is no false conflict.
    f.insert_cidx(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
         ON CONFLICT (vec_label) DO UPDATE SET n = excluded.n",
        vec![Value::Int(1), t("b"), Value::Int(30)],
        &mut cidx,
    );
    assert_eq!(
        f.rows("records").len(),
        2,
        "the old conflict key must insert fresh after the re-key (no false conflict)"
    );

    // Insert the NEW key value (2): it must conflict with the rewritten row and
    // update it in place — the index filed the new key.
    f.insert_cidx(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
         ON CONFLICT (vec_label) DO UPDATE SET n = excluded.n",
        vec![Value::Int(2), t("c"), Value::Int(99)],
        &mut cidx,
    );
    let rows = f.rows("records");
    assert_eq!(rows.len(), 2, "the new conflict key must update in place, not insert");
    let updated = rows
        .iter()
        .find(|r| Fixture::cell(r, "vec_label") == Value::Int(2))
        .expect("the row at vec_label 2");
    assert_eq!(Fixture::cell(updated, "n"), Value::Int(99), "the new key conflicted and updated");
}

#[test]
fn do_update_onto_third_row_key_raises_unique() {
    // A DO UPDATE that rewrites a conflict-key column onto a value ALREADY held by
    // a THIRD existing row is a UNIQUE violation: stdlib sqlite3 raises
    // "UNIQUE constraint failed: <table>.<col>" rather than silently letting two
    // rows share the key. The engine must reject it the same way, before the
    // rewrite lands, leaving both existing rows untouched.
    let f = Fixture::new(&[DDL_RECORDS]);
    let mut cidx = ConflictIndex::default();

    // Seed two rows on distinct conflict keys (vec_label 1 and vec_label 3),
    // building the conflict index over them.
    f.insert_cidx(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
         ON CONFLICT (vec_label) DO UPDATE SET n = excluded.n",
        vec![Value::Int(1), t("a"), Value::Int(10)],
        &mut cidx,
    );
    f.insert_cidx(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
         ON CONFLICT (vec_label) DO UPDATE SET n = excluded.n",
        vec![Value::Int(3), t("c"), Value::Int(30)],
        &mut cidx,
    );

    // DO UPDATE conflicting on vec_label 1, rewriting its conflict key to 3 — the
    // value already owned by the other row. This must raise the UNIQUE error.
    let err = f
        .try_insert_cidx(
            "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
             ON CONFLICT (vec_label) DO UPDATE SET vec_label = 3 , n = excluded.n",
            vec![Value::Int(1), t("a"), Value::Int(20)],
            &mut cidx,
        )
        .expect_err("rewriting a conflict key onto a third row must raise UNIQUE");
    assert!(
        format!("{err}").contains("UNIQUE constraint failed: records.vec_label"),
        "expected the sqlite3-shaped UNIQUE error, got {err}"
    );

    // Both original rows survive unchanged — no duplicate key was created.
    let rows = f.rows("records");
    assert_eq!(rows.len(), 2, "the rejected rewrite must leave both rows intact");
    let r1 = rows
        .iter()
        .find(|r| Fixture::cell(r, "vec_label") == Value::Int(1))
        .expect("the row at vec_label 1 still exists");
    assert_eq!(Fixture::cell(r1, "id"), t("a"));
    assert_eq!(Fixture::cell(r1, "n"), Value::Int(10), "row 1 was not rewritten");
    let r3 = rows
        .iter()
        .find(|r| Fixture::cell(r, "vec_label") == Value::Int(3))
        .expect("the row at vec_label 3 still exists");
    assert_eq!(Fixture::cell(r3, "id"), t("c"));
    assert_eq!(Fixture::cell(r3, "n"), Value::Int(30), "row 3 was not overwritten");
}

#[test]
fn or_replace_keeps_conflict_index_consistent() {
    // OR REPLACE matches on the conflict-key columns, so the replacement carries
    // the same conflict-key value that matched (the key is structurally stable on
    // this path). The conflict index must stay consistent through the OR REPLACE
    // re-file: the replaced row is still found as a conflict on its key, and a
    // later insert of the same key updates in place (no duplicate).
    let f = Fixture::new(&[DDL_RECORDS]);
    let mut cidx = ConflictIndex::default();

    // Seed the index + one row at vec_label 1.
    f.insert_cidx(
        "INSERT OR REPLACE INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(1), t("a"), Value::Int(10)],
        &mut cidx,
    );
    // OR REPLACE conflicting on vec_label 1: overwrites the row in place at the
    // same conflict key.
    f.insert_cidx(
        "INSERT OR REPLACE INTO records (vec_label, id, n) VALUES (?, ?, ?)",
        vec![Value::Int(1), t("b"), Value::Int(20)],
        &mut cidx,
    );
    {
        let rows = f.rows("records");
        assert_eq!(rows.len(), 1, "OR REPLACE overwrites in place, no duplicate");
        assert_eq!(Fixture::cell(&rows[0], "id"), t("b"));
        assert_eq!(Fixture::cell(&rows[0], "n"), Value::Int(20));
    }
    // The conflict index still files vec_label 1 at the replaced row: a later
    // UPSERT on key 1 conflicts and updates in place (not a fresh insert).
    f.insert_cidx(
        "INSERT INTO records (vec_label, id, n) VALUES (?, ?, ?) \
         ON CONFLICT (vec_label) DO UPDATE SET n = excluded.n",
        vec![Value::Int(1), t("c"), Value::Int(99)],
        &mut cidx,
    );
    let rows = f.rows("records");
    assert_eq!(rows.len(), 1, "the conflict index still resolves key 1: update in place");
    assert_eq!(Fixture::cell(&rows[0], "n"), Value::Int(99));
}

#[test]
fn executemany_rollback_invalidates_id_index() {
    // A built id-index incrementally records each inner insert's `id → row-key`.
    // When the batch rolls back on a failing tail row, those speculative ids must
    // be dropped: a later `WHERE id = '<rolled-back id>'` point lookup must find no
    // row. Without invalidating the id-index, the stale entry resolves to a key the
    // rollback reverted (wrong-row / phantom hit).
    let f = Fixture::new(&[DDL_RECORDS]);
    // Build the (empty) id-index so the inner inserts maintain it per row.
    let mut idx = f.build_id_index("records");

    let stmt = match parse("INSERT INTO records (id, n) VALUES (?, ?)").unwrap() {
        Stmt::Insert(i) => i,
        _ => unreachable!(),
    };
    let seq = vec![
        vec![t("alpha"), Value::Int(1)],
        vec![t("beta"), Value::Int(2)],
        vec![Value::Null, Value::Int(3)], // id is NOT NULL → fails on the tail
    ];
    let err = execute_insert_many(
        &stmt, &f.store, &f.meta, &f.catalog, &f.root_map, seq,
        Some(&mut idx), None, None, None,
    )
    .unwrap_err();
    assert!(format!("{err}").contains("NOT NULL constraint failed"));

    // The whole batch rolled back: no rows survive.
    assert_eq!(f.rows("records").len(), 0, "a partial batch must not survive");
    // The id-index must not hold a stale entry for a rolled-back id.
    assert!(
        idx.lookup("alpha").is_none(),
        "a rolled-back id must be dropped from the id-index"
    );
    assert!(idx.lookup("beta").is_none(), "a rolled-back id must be dropped from the id-index");
}

#[test]
fn conflict_rekey_through_both_indexes_inserts_old_key_fresh() {
    // The conn-shape regression: a table whose B-tree key is independent of the
    // conflict-key column, driven through BOTH the id-index and the conflict-index
    // (the connection always supplies both). After a DO UPDATE rewrites the
    // conflict-key column, inserting the OLD key value must insert fresh — the
    // re-key removed the stale entry, so it does not falsely conflict.
    const DDL_KEYED: &str = "CREATE TABLE keyed ( k INTEGER , label TEXT )";
    let f = Fixture::new(&[DDL_KEYED]);
    let mut cidx = ConflictIndex::default();
    let mut idx = IdIndex::new();
    let run = |sql: &str, params: Vec<Value>, idx: &mut IdIndex, cidx: &mut ConflictIndex| {
        let stmt = match parse(sql).unwrap() {
            Stmt::Insert(i) => i,
            _ => unreachable!(),
        };
        execute_insert(
            &stmt, &f.store, &f.meta, &f.catalog, &f.root_map, params, None,
            TxnScope::Own, Some(idx), Some(cidx), None, None,
        )
        .unwrap();
    };
    let up = "INSERT INTO keyed (k, label) VALUES (?, ?) \
              ON CONFLICT (k) DO UPDATE SET label = excluded.label";
    run(up, vec![Value::Int(1), t("first")], &mut idx, &mut cidx);
    // Rewrite the conflict key k: 1 -> 2.
    run(
        "INSERT INTO keyed (k, label) VALUES (?, ?) \
         ON CONFLICT (k) DO UPDATE SET k = 2, label = excluded.label",
        vec![Value::Int(1), t("moved")],
        &mut idx, &mut cidx,
    );
    // Insert the OLD key value 1 again: must insert fresh, not conflict.
    run(up, vec![Value::Int(1), t("oldkey")], &mut idx, &mut cidx);

    let rows = f.rows("keyed");
    assert_eq!(rows.len(), 2, "the old conflict key inserts fresh after the re-key");
    let mut ks: Vec<i64> = rows
        .iter()
        .map(|r| match Fixture::cell(r, "k") {
            Value::Int(i) => i,
            other => panic!("unexpected k {other:?}"),
        })
        .collect();
    ks.sort_unstable();
    assert_eq!(ks, vec![1, 2], "both the moved row (k=2) and the fresh old-key row (k=1) survive");
}

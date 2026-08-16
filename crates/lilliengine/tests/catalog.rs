//! Catalog parity tests: affinity derivation, PK detection (single + composite),
//! column constraints, ALTER/DROP/CREATE INDEX, and PRAGMA table_info shape,
//! exercised against the six Hippo table DDLs.

use lillibrain::Value;
use lilliengine::catalog::{coerce_affinity, render_float_text, Catalog, ColumnMeta};
use lilliengine::parser::parse;

/// Parse one DDL string and apply it to the catalog.
fn apply(cat: &mut Catalog, sql: &str) {
    let stmt = parse(sql).expect("parse DDL");
    cat.apply_ddl(&stmt).expect("apply_ddl");
}

fn col(cat: &Catalog, table: &str, name: &str) -> ColumnMeta {
    cat.get(table)
        .expect("table present")
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("column {name} missing from {table}"))
        .clone()
}

// The six Hippo DDLs, verbatim from src/iai_mcp/hippo/_table.py.
const DDL_RECORDS: &str = "\
CREATE TABLE IF NOT EXISTS records (
    vec_label       INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    tier            TEXT NOT NULL,
    literal_surface TEXT,
    embedding       BLOB NOT NULL,
    structure_hv    BLOB,
    centrality      REAL,
    detail_level    INTEGER,
    created_at      TEXT NOT NULL,
    schema_version  INTEGER DEFAULT 1,
    valence         REAL DEFAULT 0.0,
    hv_tier              TEXT NOT NULL DEFAULT 'bsc',
    structure_hv_payload BLOB NOT NULL DEFAULT x'',
    embedding_pending    INTEGER NOT NULL DEFAULT 0
)";

const DDL_EDGES: &str = "\
CREATE TABLE IF NOT EXISTS edges (
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.0,
    updated_at  TEXT,
    PRIMARY KEY (src, dst, edge_type)
)";

const DDL_EVENTS: &str = "\
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    severity        TEXT,
    ts              TEXT NOT NULL,
    data_json       TEXT
)";

const DDL_BUDGET: &str = "\
CREATE TABLE IF NOT EXISTS budget_ledger (
    date        TEXT,
    usd_spent   REAL,
    kind        TEXT,
    ts          TEXT
)";

const DDL_RATELIMIT: &str = "\
CREATE TABLE IF NOT EXISTS ratelimit_ledger (
    ts          TEXT,
    status_code INTEGER,
    endpoint    TEXT
)";

const DDL_HIPPO_META: &str = "\
CREATE TABLE IF NOT EXISTS _hippo_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)";

const DDL_RECORD_TAGS: &str = "\
CREATE TABLE IF NOT EXISTS record_tags (
    record_id TEXT NOT NULL,
    tag       TEXT NOT NULL,
    UNIQUE(record_id, tag)
)";

#[test]
fn records_columns_affinity_and_pk() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_RECORDS);

    let names = cat.column_names("records").unwrap();
    assert_eq!(names.first().map(String::as_str), Some("vec_label"));
    assert_eq!(names.len(), 14);

    // INTEGER PRIMARY KEY AUTOINCREMENT. A column-level PRIMARY KEY marks the
    // pk flag; nullability tracks only an explicit NOT NULL (reference parity:
    // the table-level PK path is where NOT NULL is implied).
    let vl = col(&cat, "records", "vec_label");
    assert_eq!(vl.affinity, "INTEGER");
    assert!(vl.primary_key);

    // TEXT NOT NULL UNIQUE.
    let id = col(&cat, "records", "id");
    assert_eq!(id.affinity, "TEXT");
    assert!(!id.nullable);
    assert!(!id.primary_key);

    // BLOB NOT NULL.
    let emb = col(&cat, "records", "embedding");
    assert_eq!(emb.affinity, "BLOB");
    assert!(!emb.nullable);

    // REAL.
    assert_eq!(col(&cat, "records", "centrality").affinity, "REAL");
    // INTEGER nullable.
    assert_eq!(col(&cat, "records", "detail_level").affinity, "INTEGER");

    // DEFAULT values.
    assert_eq!(
        col(&cat, "records", "schema_version").default,
        Some(Value::Int(1))
    );
    assert_eq!(
        col(&cat, "records", "valence").default,
        Some(Value::Float(0.0))
    );
    assert_eq!(
        col(&cat, "records", "hv_tier").default,
        Some(Value::Text("bsc".to_string()))
    );
    // x'' empty-blob default.
    assert_eq!(
        col(&cat, "records", "structure_hv_payload").default,
        Some(Value::Blob(Vec::new()))
    );
    assert_eq!(
        col(&cat, "records", "embedding_pending").default,
        Some(Value::Int(0))
    );
}

#[test]
fn edges_composite_primary_key() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_EDGES);

    for pk in ["src", "dst", "edge_type"] {
        let c = col(&cat, "edges", pk);
        assert!(c.primary_key, "{pk} should be part of the composite PK");
        assert!(!c.nullable, "{pk} PRIMARY KEY implies NOT NULL");
    }
    // weight is not a PK column.
    assert!(!col(&cat, "edges", "weight").primary_key);
    assert_eq!(col(&cat, "edges", "weight").affinity, "REAL");
    assert_eq!(
        col(&cat, "edges", "weight").default,
        Some(Value::Float(0.0))
    );

    // primary_key_columns reports the composite in declaration order.
    assert_eq!(
        cat.primary_key_columns("edges").unwrap(),
        vec![
            "src".to_string(),
            "dst".to_string(),
            "edge_type".to_string()
        ]
    );
}

#[test]
fn events_single_text_primary_key() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_EVENTS);
    let id = col(&cat, "events", "id");
    assert!(id.primary_key);
    assert_eq!(cat.primary_key_columns("events").unwrap(), vec!["id"]);
}

#[test]
fn all_six_hippo_schemas_register() {
    let mut cat = Catalog::new();
    for ddl in [
        DDL_RECORDS,
        DDL_EDGES,
        DDL_EVENTS,
        DDL_BUDGET,
        DDL_RATELIMIT,
        DDL_HIPPO_META,
    ] {
        apply(&mut cat, ddl);
    }
    // budget_ledger has no PK.
    assert!(cat.primary_key_columns("budget_ledger").unwrap().is_empty());
    assert_eq!(col(&cat, "budget_ledger", "usd_spent").affinity, "REAL");
    assert_eq!(
        col(&cat, "ratelimit_ledger", "status_code").affinity,
        "INTEGER"
    );
    // _hippo_meta key TEXT PRIMARY KEY.
    assert!(col(&cat, "_hippo_meta", "key").primary_key);
}

/// Count the enforceable key-sets: every tracked UNIQUE/PK key-set except a lone
/// AUTOINCREMENT `vec_label` PK, which the engine injects fresh per row and can
/// never collide (mirrors exec's auto-injected-only filter for the common INSERT
/// that does not name `vec_label`).
fn enforceable_keysets(keysets: &[Vec<String>]) -> usize {
    keysets
        .iter()
        .filter(|ks| !(ks.len() == 1 && ks[0].eq_ignore_ascii_case("vec_label")))
        .count()
}

/// The plain-INSERT UNIQUE/PK enforcement path reuses ONE conflict index per
/// INSERT and records no key-set identity on it, so it can enforce at most one
/// key-set. This locks the schema-level precondition: every canonical table
/// presents at most one enforceable (non-auto-injected) key-set. Adding a second
/// UNIQUE / PRIMARY KEY to any of these turns this RED — the signal to generalize
/// the conflict cache to a per-key-set map before the engine can enforce it.
#[test]
fn canonical_tables_present_at_most_one_enforceable_keyset() {
    for ddl in [
        DDL_RECORDS,
        DDL_EDGES,
        DDL_EVENTS,
        DDL_BUDGET,
        DDL_RATELIMIT,
        DDL_HIPPO_META,
        DDL_RECORD_TAGS,
    ] {
        let mut cat = Catalog::new();
        apply(&mut cat, ddl);
        let table = cat.table_names()[0].clone();
        let keysets = cat.unique_keysets(&table).unwrap();
        assert!(
            enforceable_keysets(&keysets) <= 1,
            "table {table} presents {} enforceable key-sets {keysets:?}; the single-conflict-index \
             plain-INSERT path enforces only one — generalize the cache first",
            enforceable_keysets(&keysets),
        );
    }
}

/// Positive control for the invariant above: a PRIMARY KEY plus a separate
/// column-level UNIQUE surfaces two enforceable key-sets, so the assertion
/// would fire — proving it is not vacuously green.
#[test]
fn two_enforceable_keysets_are_visible() {
    let mut cat = Catalog::new();
    apply(
        &mut cat,
        "CREATE TABLE two_keys ( a INTEGER PRIMARY KEY , b TEXT UNIQUE )",
    );
    let keysets = cat.unique_keysets("two_keys").unwrap();
    assert_eq!(
        enforceable_keysets(&keysets),
        2,
        "PK + separate UNIQUE must surface two enforceable key-sets, got {keysets:?}"
    );
}

#[test]
fn affinity_map_matches_sqlite_rules() {
    let mut cat = Catalog::new();
    apply(
        &mut cat,
        "CREATE TABLE t (\
            a INT, b INTEGER, c REAL, d FLOAT, e DOUBLE, f BLOB, \
            g NUMERIC, h DECIMAL, i BOOLEAN, j DATE, k DATETIME, \
            l TEXT, m CHAR, n CLOB, o VARCHAR, p MADEUP)",
    );
    let want = [
        ("a", "INTEGER"),
        ("b", "INTEGER"),
        ("c", "REAL"),
        ("d", "REAL"),
        ("e", "REAL"),
        ("f", "BLOB"),
        ("g", "NUMERIC"),
        ("h", "NUMERIC"),
        ("i", "NUMERIC"),
        ("j", "NUMERIC"),
        ("k", "NUMERIC"),
        ("l", "TEXT"),
        ("m", "TEXT"),
        ("n", "TEXT"),
        ("o", "TEXT"),
        ("p", "TEXT"), // unknown -> TEXT
    ];
    for (name, aff) in want {
        assert_eq!(col(&cat, "t", name).affinity, aff, "affinity for {name}");
    }
}

#[test]
fn alter_add_drop_rename() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_HIPPO_META);
    apply(&mut cat, "ALTER TABLE _hippo_meta ADD COLUMN extra INTEGER");
    assert_eq!(col(&cat, "_hippo_meta", "extra").affinity, "INTEGER");

    apply(&mut cat, "ALTER TABLE _hippo_meta DROP COLUMN extra");
    assert!(cat
        .get("_hippo_meta")
        .unwrap()
        .iter()
        .all(|c| c.name != "extra"));

    apply(&mut cat, "ALTER TABLE _hippo_meta RENAME TO kv");
    assert!(cat.get("_hippo_meta").is_err());
    assert!(cat.get("kv").is_ok());
}

#[test]
fn drop_table_if_exists() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_BUDGET);
    apply(&mut cat, "DROP TABLE budget_ledger");
    assert!(cat.get("budget_ledger").is_err());
    // IF EXISTS on an absent table is a no-op (must not panic / error).
    apply(&mut cat, "DROP TABLE IF EXISTS budget_ledger");
}

#[test]
fn drop_table_missing_without_if_exists_errors() {
    let mut cat = Catalog::new();
    let stmt = parse("DROP TABLE nope").unwrap();
    assert!(cat.apply_ddl(&stmt).is_err());
}

#[test]
fn create_index_records_but_no_scan_path() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_RECORDS);
    // CREATE INDEX is accepted; it must not alter the column set.
    let before = cat.column_names("records").unwrap();
    apply(&mut cat, "CREATE INDEX IF NOT EXISTS idx_r ON records(id)");
    let after = cat.column_names("records").unwrap();
    assert_eq!(before, after);
}

#[test]
fn create_unique_index_records_correct_metadata() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_RECORDS);
    apply(&mut cat, "CREATE UNIQUE INDEX ix ON records(id)");
    // The UNIQUE keyword must be consumed so the index records its real name,
    // table, and column — not the keywords INDEX / ON.
    let idx_row = cat
        .sqlite_master_rows()
        .into_iter()
        .find(|r| r.object_type == "index")
        .expect("the unique index is recorded in the catalog");
    assert_eq!(
        idx_row.name, "ix",
        "index name is the actual name, not a keyword"
    );
    assert_eq!(
        idx_row.tbl_name, "records",
        "index table is records, not a keyword"
    );
    assert_eq!(
        idx_row.sql,
        Value::Text("CREATE UNIQUE INDEX ix ON records (id)".to_string()),
        "the index metadata names the right column and keeps the UNIQUE keyword"
    );
}

#[test]
fn alter_rename_of_quoted_identifier_rewrites_stored_ddl() {
    let mut cat = Catalog::new();
    // A bracket-quoted table name. The lexer unwraps `[t]` -> `t`, so the catalog
    // tracks it under `t`, but the stored DDL still carries the quoted form.
    apply(&mut cat, "CREATE TABLE [t] ( id TEXT , n INTEGER )");
    apply(&mut cat, "ALTER TABLE [t] RENAME TO u");

    assert!(cat.get("t").is_err(), "the old name is gone");
    assert!(cat.get("u").is_ok(), "the table is now u");

    // The stored DDL must name the new table (vs reverting to the old quoted name
    // on a reopen that replays this DDL).
    let table_row = cat
        .sqlite_master_rows()
        .into_iter()
        .find(|r| r.object_type == "table" && r.name == "u")
        .expect("the renamed table appears as u");
    let sql = match table_row.sql {
        Value::Text(s) => s,
        other => panic!("stored DDL must be text, got {other:?}"),
    };
    let upper = sql.to_ascii_uppercase();
    assert!(
        upper.contains("TABLE U ") || upper.contains("TABLE U("),
        "the rewritten DDL must name the new table u, got {sql:?}"
    );
    assert!(
        !upper.contains("[T]"),
        "the rewritten DDL must not retain the old quoted name, got {sql:?}"
    );
}

#[test]
fn pragma_table_info_shape() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_EDGES);
    let info = cat.table_info("edges").unwrap();
    // (cid, name, type, notnull, dflt_value, pk)
    assert_eq!(info.len(), 5);
    let (cid, name, ty, notnull, _dflt, pk) = &info[0];
    assert_eq!(*cid, 0);
    assert_eq!(name, "src");
    assert_eq!(ty, "TEXT");
    assert_eq!(*notnull, 1);
    assert_eq!(*pk, 1);
    // weight: notnull, not pk.
    let weight = info.iter().find(|r| r.1 == "weight").unwrap();
    assert_eq!(weight.3, 1);
    assert_eq!(weight.5, 0);
}

#[test]
fn coerce_real_into_text_keeps_decimal_point_like_sqlite3() {
    // sqlite3 stores a whole-valued REAL coerced into a TEXT column with its
    // trailing ".0" ("1.0" -> "1.0", not "1"); Rust's default {} formatting drops
    // it. The coercion must keep the decimal point so a TEXT-affinity column round
    // trips the same text sqlite3 produces.
    assert_eq!(
        coerce_affinity(Value::Float(1.0), "TEXT"),
        Value::Text("1.0".into())
    );
    assert_eq!(
        coerce_affinity(Value::Float(100.0), "TEXT"),
        Value::Text("100.0".into())
    );
    assert_eq!(
        coerce_affinity(Value::Float(0.5), "TEXT"),
        Value::Text("0.5".into())
    );
    assert_eq!(
        coerce_affinity(Value::Float(0.9), "TEXT"),
        Value::Text("0.9".into())
    );
    assert_eq!(
        coerce_affinity(Value::Float(-1.5), "TEXT"),
        Value::Text("-1.5".into())
    );
}

#[test]
fn render_float_text_matches_sqlite3_decimal_forms() {
    // Across the bounded decimal range the REAL columns (score / weight / valence /
    // centrality / usd_spent) actually carry, the rendered text is byte-identical
    // to sqlite3 / Python str(): a whole value keeps its trailing ".0", a
    // fractional value renders its shortest round-tripping digits.
    assert_eq!(render_float_text(0.0), "0.0");
    assert_eq!(render_float_text(1.0), "1.0");
    assert_eq!(render_float_text(9.0), "9.0");
    assert_eq!(render_float_text(0.1), "0.1");
    assert_eq!(render_float_text(0.7), "0.7");
    assert_eq!(render_float_text(0.9), "0.9");
    assert_eq!(render_float_text(0.123456789), "0.123456789");
    assert_eq!(render_float_text(123.456), "123.456");
    assert_eq!(render_float_text(1000.5), "1000.5");
    assert_eq!(render_float_text(-1.5), "-1.5");
}

#[test]
fn sqlite_master_rows_carry_sql_and_rootpage() {
    let mut cat = Catalog::new();
    apply(&mut cat, "CREATE TABLE records ( id TEXT , n INTEGER )");
    cat.set_rootpage("records", 4);

    let rows = cat.sqlite_master_rows();
    assert_eq!(rows.len(), 1);
    let r = &rows[0];
    assert_eq!(r.object_type, "table");
    assert_eq!(r.name, "records");
    assert_eq!(r.tbl_name, "records");
    assert_eq!(r.rootpage, 4);
    match &r.sql {
        Value::Text(s) => assert!(s.to_uppercase().starts_with("CREATE TABLE")),
        other => panic!("expected Text sql, got {other:?}"),
    }
}

#[test]
fn sqlite_master_rename_rewrites_ddl_table_name() {
    let mut cat = Catalog::new();
    apply(&mut cat, "CREATE TABLE old_t ( id TEXT , n INTEGER )");
    cat.set_rootpage("old_t", 4);
    apply(&mut cat, "ALTER TABLE old_t RENAME TO new_t");

    let rows = cat.sqlite_master_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].name, "new_t");
    assert_eq!(rows[0].rootpage, 4);
    match &rows[0].sql {
        Value::Text(s) => {
            assert!(s.contains("new_t"), "DDL should name new_t: {s}");
            assert!(!s.contains("old_t"), "DDL should not name old_t: {s}");
        }
        other => panic!("expected Text sql, got {other:?}"),
    }
}

#[test]
fn sqlite_master_drop_removes_row() {
    let mut cat = Catalog::new();
    apply(&mut cat, "CREATE TABLE records ( id TEXT )");
    apply(&mut cat, "CREATE TABLE edges ( src TEXT )");
    apply(&mut cat, "DROP TABLE records");
    let names: Vec<String> = cat
        .sqlite_master_rows()
        .into_iter()
        .map(|r| r.name)
        .collect();
    assert_eq!(names, vec!["edges".to_string()]);
}

#[test]
fn sqlite_master_includes_index_rows() {
    let mut cat = Catalog::new();
    apply(
        &mut cat,
        "CREATE TABLE records ( id TEXT , embedding_pending INTEGER )",
    );
    apply(
        &mut cat,
        "CREATE INDEX idx_records_pending ON records ( embedding_pending )",
    );
    let rows = cat.sqlite_master_rows();
    let idx: Vec<_> = rows.iter().filter(|r| r.object_type == "index").collect();
    assert_eq!(idx.len(), 1);
    assert_eq!(idx[0].name, "idx_records_pending");
    assert_eq!(idx[0].tbl_name, "records");
}

#[test]
fn alter_add_duplicate_column_rejected_with_sqlite_error_text() {
    let mut cat = Catalog::new();
    apply(&mut cat, "CREATE TABLE t ( id TEXT , n INTEGER )");
    apply(&mut cat, "ALTER TABLE t ADD COLUMN role TEXT");
    let stmt = parse("ALTER TABLE t ADD COLUMN role TEXT").expect("parse DDL");
    let err = cat.apply_ddl(&stmt).unwrap_err();
    assert_eq!(err.to_string(), "duplicate column name: role");
    let names = cat.column_names("t").expect("table present");
    assert_eq!(
        names.iter().filter(|n| n.as_str() == "role").count(),
        1,
        "a rejected repeat ADD COLUMN must not widen the catalog"
    );
}

#[test]
fn alter_add_duplicate_column_is_case_insensitive() {
    let mut cat = Catalog::new();
    apply(&mut cat, "CREATE TABLE t ( id TEXT )");
    apply(&mut cat, "ALTER TABLE t ADD COLUMN role TEXT");
    let stmt = parse("ALTER TABLE t ADD COLUMN ROLE TEXT").expect("parse DDL");
    let err = cat.apply_ddl(&stmt).unwrap_err();
    assert!(
        err.to_string().starts_with("duplicate column name"),
        "column-name comparison is case-insensitive, got {err}"
    );
}

#[test]
fn alter_add_duplicate_of_create_time_column_rejected() {
    let mut cat = Catalog::new();
    apply(&mut cat, "CREATE TABLE t ( id TEXT , n INTEGER )");
    let stmt = parse("ALTER TABLE t ADD COLUMN n INTEGER").expect("parse DDL");
    let err = cat.apply_ddl(&stmt).unwrap_err();
    assert_eq!(err.to_string(), "duplicate column name: n");
}

#[test]
fn create_table_duplicate_column_rejected() {
    let mut cat = Catalog::new();
    let stmt = parse("CREATE TABLE t ( id TEXT , id INTEGER )").expect("parse DDL");
    let err = cat.apply_ddl(&stmt).unwrap_err();
    assert_eq!(err.to_string(), "duplicate column name: id");
}

// ---------------------------------------------------------------------------
// unique_keysets — column-level UNIQUE tracking + PK/UNIQUE key-set exposure
// ---------------------------------------------------------------------------

#[test]
fn unique_column_constraint_is_tracked() {
    // The column-level UNIQUE keyword must no longer be silently discarded.
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_RECORDS);
    let id = col(&cat, "records", "id");
    assert!(id.unique, "records.id UNIQUE must be tracked on ColumnMeta");
    let vec_label = col(&cat, "records", "vec_label");
    assert!(
        !vec_label.unique,
        "vec_label carries no column-level UNIQUE (it is the PK)"
    );
}

/// Order-insensitive comparison of two key-set lists.
fn assert_keysets_eq(mut got: Vec<Vec<String>>, mut want: Vec<Vec<String>>) {
    got.sort();
    want.sort();
    assert_eq!(got, want, "unique_keysets mismatch");
}

#[test]
fn unique_keysets_records_yields_pk_and_unique_column() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_RECORDS);
    let keysets = cat.unique_keysets("records").unwrap();
    assert_keysets_eq(
        keysets,
        vec![vec!["vec_label".to_string()], vec!["id".to_string()]],
    );
}

#[test]
fn unique_keysets_edges_yields_composite_pk() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_EDGES);
    let keysets = cat.unique_keysets("edges").unwrap();
    assert_keysets_eq(
        keysets,
        vec![vec![
            "src".to_string(),
            "dst".to_string(),
            "edge_type".to_string(),
        ]],
    );
}

#[test]
fn unique_keysets_events_yields_single_pk_column() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_EVENTS);
    let keysets = cat.unique_keysets("events").unwrap();
    assert_keysets_eq(keysets, vec![vec!["id".to_string()]]);
}

#[test]
fn unique_keysets_hippo_meta_yields_single_pk_column() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_HIPPO_META);
    let keysets = cat.unique_keysets("_hippo_meta").unwrap();
    assert_keysets_eq(keysets, vec![vec!["key".to_string()]]);
}

#[test]
fn unique_keysets_table_with_no_pk_or_unique_is_empty() {
    let mut cat = Catalog::new();
    apply(&mut cat, DDL_BUDGET);
    let keysets = cat.unique_keysets("budget_ledger").unwrap();
    assert!(
        keysets.is_empty(),
        "a table with neither PK nor UNIQUE must yield no key-sets, got {keysets:?}"
    );
}

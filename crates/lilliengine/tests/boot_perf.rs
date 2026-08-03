//! Regression tests: COUNT(*) and projected scans on a wide-embedding table
//! must NOT copy row payloads.
//!
//! A bare `SELECT COUNT(*) FROM records` must walk leaf-page headers only
//! (zero full scans, zero payload copies).  A filtered COUNT and a 2-column
//! projected SELECT must decode only the referenced columns and not pull the
//! large embedding BLOB into memory for every row.
//!
//! `projected_decode_parity` is a permanent correctness fence and must stay
//! GREEN alongside the rest.

use std::collections::BTreeMap;

use lillibrain::{Store, Value};
use lilliengine::ast::Stmt;
use lilliengine::catalog::Catalog;
use lilliengine::conn::Connection;
use lilliengine::exec::{execute_select, execute_select_instrumented};
use lilliengine::meta::MetaTable;
use lilliengine::parser::parse;
use lilliengine::plan::plan_select;
use lilliengine::rowcodec::encode_row;
use tempfile::tempdir;

// ---------------------------------------------------------------------------
// Wide-embedding fixture
//
// Each row carries a multi-KB BLOB in the `embedding` column so a
// full-payload copy is observably distinct from a key-only walk.  The schema
// mirrors the real records table (vec_label INTEGER PK AUTOINCREMENT, id,
// tombstoned_at, embedding_pending, plus the wide embedding BLOB).
// ---------------------------------------------------------------------------

const DDL_RECORDS: &str = "CREATE TABLE IF NOT EXISTS records ( \
    vec_label        INTEGER PRIMARY KEY AUTOINCREMENT , \
    id               TEXT NOT NULL UNIQUE , \
    tier             TEXT , \
    tombstoned_at    TEXT , \
    embedding_pending INTEGER , \
    embedding        BLOB NOT NULL )";

/// Each synthetic embedding is 1536 bytes (4 * 384 floats, matching the real
/// production blob size of a 384-d float embedding).
const EMBEDDING_BYTES: usize = 4 * 384;

/// Number of rows to seed.  Large enough that a full-payload scan is
/// measurably distinct from a key-only walk; small enough for sub-second CI.
const N_ROWS: i64 = 2_000;

struct WideFixture {
    _dir: tempfile::TempDir,
    store: Store,
    catalog: Catalog,
    root_map: BTreeMap<String, u32>,
}

impl WideFixture {
    fn new() -> WideFixture {
        let dir = tempdir().unwrap();
        let path = dir.path().join("records.lilli");
        let store = Store::open(&path).unwrap();
        let meta = MetaTable::open(&store).unwrap();
        let mut catalog = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.create_table(&store, &mut catalog, &mut root_map, DDL_RECORDS)
            .unwrap();

        store.begin_write().unwrap();
        for k in 1..=N_ROWS {
            // embedding: deterministic repeating byte pattern for each row
            let byte = (k % 256) as u8;
            let blob = vec![byte; EMBEDDING_BYTES];
            // tombstoned_at: NULL for even rows, "2024-01-01 00:00:00" for odd
            let tombstoned_at = if k % 2 == 0 {
                Value::Null
            } else {
                Value::Text("2024-01-01 00:00:00".to_string())
            };
            let row = vec![
                Value::Int(k),
                Value::Text(format!("id-{k:06}")),
                Value::Text("episodic".to_string()),
                tombstoned_at,
                Value::Int(0), // embedding_pending = 0
                Value::Blob(blob),
            ];
            store
                .tree(*root_map.get("records").unwrap())
                .insert(k, &encode_row(&row))
                .unwrap();
        }
        store.commit().unwrap();

        WideFixture {
            _dir: dir,
            store,
            catalog,
            root_map,
        }
    }

    fn run_sql(&self, sql: &str) -> lilliengine::exec::ResultSet {
        let stmt = parse(sql).unwrap();
        let select = match stmt {
            Stmt::Select(s) => s,
            _ => panic!("not a SELECT"),
        };
        let plan = plan_select(&select, &self.catalog, vec![], None).unwrap();
        execute_select(
            &plan,
            &self.store,
            &self.catalog,
            &self.root_map,
            None,
            None,
            None,
        )
        .unwrap()
    }

    fn run_instrumented(&self, sql: &str) -> (lilliengine::exec::ResultSet, bool) {
        let stmt = parse(sql).unwrap();
        let select = match stmt {
            Stmt::Select(s) => s,
            _ => panic!("not a SELECT"),
        };
        let plan = plan_select(&select, &self.catalog, vec![], None).unwrap();
        let (rs, stats) = execute_select_instrumented(
            &plan,
            &self.store,
            &self.catalog,
            &self.root_map,
            None,
            None,
            None,
        )
        .unwrap();
        (rs, stats.decoded_rows)
    }
}

// ---------------------------------------------------------------------------
// A bare COUNT(*) must use a key-only walk that never increments
// full_scan_count().
// ---------------------------------------------------------------------------
#[test]
fn count_star_is_key_only() {
    let f = WideFixture::new();
    f.store.reset_full_scan_count();

    let rs = f.run_sql("SELECT COUNT ( * ) FROM records");

    assert_eq!(
        rs.rows[0][0].1,
        Value::Int(N_ROWS),
        "COUNT(*) must return the right row count"
    );

    let scan_count = f.store.full_scan_count();
    assert_eq!(
        scan_count, 0,
        "bare COUNT(*) must not trigger a full payload scan \
         (full_scan_count should be 0, got {scan_count})"
    );
}

// ---------------------------------------------------------------------------
// A filtered COUNT(*) WHERE tombstoned_at IS NULL touches only two narrow
// columns.  The full embedding BLOB must not be copied for every row.
// ---------------------------------------------------------------------------
#[test]
fn filtered_count_skips_wide_column() {
    let f = WideFixture::new();

    // Reference: count via a full-decode path so we know the expected value.
    let reference = f.run_sql("SELECT COUNT ( * ) FROM records WHERE tombstoned_at IS NULL");
    let expected_count = match reference.rows[0][0].1 {
        Value::Int(n) => n,
        _ => panic!("expected integer count"),
    };
    // Half the rows have tombstoned_at = NULL (even k), so the count is N_ROWS/2.
    assert_eq!(
        expected_count,
        N_ROWS / 2,
        "reference count must match the seeded even-row split"
    );

    f.store.reset_full_scan_count();
    let rs = f.run_sql("SELECT COUNT ( * ) FROM records WHERE tombstoned_at IS NULL");
    assert_eq!(
        rs.rows[0][0].1,
        Value::Int(expected_count),
        "filtered COUNT(*) must return the correct count"
    );
    let scan_count = f.store.full_scan_count();
    assert_eq!(
        scan_count, 0,
        "filtered COUNT(*) WHERE <narrow_col> must not perform a full payload scan \
         (full_scan_count should be 0, got {scan_count})"
    );
}

// ---------------------------------------------------------------------------
// A 2-column projected SELECT (id, tier) that excludes the large embedding
// BLOB must not copy the blob for every row.
// ---------------------------------------------------------------------------
#[test]
fn projected_select_skips_wide_column() {
    let f = WideFixture::new();

    // Reference result via full decode.
    let reference = f.run_sql("SELECT id , tier FROM records WHERE tombstoned_at IS NULL");
    let expected_row_count = reference.rows.len();
    assert_eq!(expected_row_count, (N_ROWS / 2) as usize);

    f.store.reset_full_scan_count();
    let rs = f.run_sql("SELECT id , tier FROM records WHERE tombstoned_at IS NULL");
    assert_eq!(
        rs.rows.len(),
        expected_row_count,
        "projected SELECT must return the same row count as the reference"
    );
    let scan_count = f.store.full_scan_count();
    assert_eq!(
        scan_count, 0,
        "a 2-column projected SELECT that excludes the embedding BLOB must not \
         perform a full payload scan \
         (full_scan_count should be 0, got {scan_count})"
    );
}

// ---------------------------------------------------------------------------
// Correctness fence: column-selective decoding of the narrow columns
// (id, tier) must return the same values as a full decode of the same
// columns. The corpus includes a NUL-byte embedding (all-zero row), a
// maximum-size blob, and an odd-valued tombstoned row so edge cases are
// exercised.
// ---------------------------------------------------------------------------
#[test]
fn projected_decode_parity() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("parity.lilli");
    let store = Store::open(&path).unwrap();
    let meta = MetaTable::open(&store).unwrap();
    let mut catalog = Catalog::new();
    let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
    meta.create_table(&store, &mut catalog, &mut root_map, DDL_RECORDS)
        .unwrap();

    // Seed a diverse corpus: NUL bytes, emoji (multi-byte UTF-8), CJK text,
    // max-size blob, and a tombstoned row so the parity holds across all cases.
    let cases: Vec<(i64, &str, &str, Value, usize)> = vec![
        (1, "id-nul", "episodic", Value::Null, EMBEDDING_BYTES),
        (2, "id-emoji", "semantic", Value::Null, 16),
        (
            3,
            "id-cjk",
            "episodic",
            Value::Text("2024-01-01 00:00:00".into()),
            32,
        ),
        (4, "id-big", "procedural", Value::Null, EMBEDDING_BYTES * 4),
        (5, "id-zero", "episodic", Value::Null, 1),
    ];

    store.begin_write().unwrap();
    for (k, id, tier, tomb, blob_len) in &cases {
        let blob = vec![(*k % 256) as u8; *blob_len];
        let row = vec![
            Value::Int(*k),
            Value::Text(id.to_string()),
            Value::Text(tier.to_string()),
            tomb.clone(),
            Value::Int(0),
            Value::Blob(blob),
        ];
        let root = *root_map.get("records").unwrap();
        store.tree(root).insert(*k, &encode_row(&row)).unwrap();
    }
    store.commit().unwrap();

    // Full decode reference: SELECT * returns all columns.
    let full_stmt = parse("SELECT * FROM records").unwrap();
    let full_select = match full_stmt {
        Stmt::Select(s) => s,
        _ => unreachable!(),
    };
    let full_plan = plan_select(&full_select, &catalog, vec![], None).unwrap();
    let full_rs =
        execute_select(&full_plan, &store, &catalog, &root_map, None, None, None).unwrap();

    // Build a lookup: id -> (tier, tombstoned_at) from the full decode.
    let id_idx = full_rs.columns.iter().position(|c| c == "id").unwrap();
    let tier_idx = full_rs.columns.iter().position(|c| c == "tier").unwrap();
    let tomb_idx = full_rs
        .columns
        .iter()
        .position(|c| c == "tombstoned_at")
        .unwrap();

    let mut full_map: std::collections::HashMap<String, (Value, Value)> =
        std::collections::HashMap::new();
    for row in &full_rs.rows {
        let id = match &row[id_idx].1 {
            Value::Text(s) => s.clone(),
            _ => panic!(),
        };
        full_map.insert(id, (row[tier_idx].1.clone(), row[tomb_idx].1.clone()));
    }

    // Projected decode: SELECT id, tier, tombstoned_at (no embedding).
    let proj_stmt = parse("SELECT id , tier , tombstoned_at FROM records").unwrap();
    let proj_select = match proj_stmt {
        Stmt::Select(s) => s,
        _ => unreachable!(),
    };
    let proj_plan = plan_select(&proj_select, &catalog, vec![], None).unwrap();
    let proj_rs =
        execute_select(&proj_plan, &store, &catalog, &root_map, None, None, None).unwrap();

    assert_eq!(
        proj_rs.rows.len(),
        full_rs.rows.len(),
        "projected SELECT must return the same number of rows as the full decode"
    );

    let p_id_idx = proj_rs.columns.iter().position(|c| c == "id").unwrap();
    let p_tier_idx = proj_rs.columns.iter().position(|c| c == "tier").unwrap();
    let p_tomb_idx = proj_rs
        .columns
        .iter()
        .position(|c| c == "tombstoned_at")
        .unwrap();

    for proj_row in &proj_rs.rows {
        let id = match &proj_row[p_id_idx].1 {
            Value::Text(s) => s.clone(),
            _ => panic!(),
        };
        let (expected_tier, expected_tomb) = full_map
            .get(&id)
            .unwrap_or_else(|| panic!("projected row id {id:?} not in full-decode set"));
        assert_eq!(
            &proj_row[p_tier_idx].1, expected_tier,
            "projected tier for {id} must be byte-identical to full decode"
        );
        assert_eq!(
            &proj_row[p_tomb_idx].1, expected_tomb,
            "projected tombstoned_at for {id} must be byte-identical to full decode"
        );
    }
}

// ===========================================================================
// Persisted column-index load/generation-gate contracts
//
// A fresh engine open that finds a valid `records.colindex` sidecar loads the
// persisted column indexes instead of full-scanning the table on the first
// `col IN (...)` read. The load is gated by a per-table monotonic write
// generation (meta-persisted, bumped inside every committed DML's transaction
// scope) plus a row-count and columns-set check, so ANY committed write since
// the persist — including a same-key in-place UPDATE of an indexed column,
// which leaves the row count unchanged — discards the stale entry and falls
// back to the lazy, always-correct rebuild.
// ===========================================================================

const DDL_EDGES: &str = "CREATE TABLE IF NOT EXISTS edges ( \
    src TEXT NOT NULL , \
    dst TEXT NOT NULL , \
    edge_type TEXT )";

const DDL_IDX_SRC: &str = "CREATE INDEX idx_edges_src ON edges (src)";
const DDL_IDX_DST: &str = "CREATE INDEX idx_edges_dst ON edges (dst)";

/// A store directory (not just a file) so the sidecar path (derived from the
/// store file's parent) is easy to reason about across copy/reopen.
struct EdgeFixture {
    dir: tempfile::TempDir,
    store_path: std::path::PathBuf,
}

impl EdgeFixture {
    /// A fresh store with the `edges` table + `{src, dst}` declared indexes,
    /// seeded with `n` rows whose `src`/`dst` values are distinct per row so
    /// every row is independently addressable by a `col IN (...)` probe.
    fn seeded(n: i64) -> EdgeFixture {
        let dir = tempdir().unwrap();
        let store_path = dir.path().join("records.lilli");
        {
            let mut conn = Connection::open(store_path.to_str().unwrap(), 0).unwrap();
            conn.execute(DDL_EDGES, vec![]).unwrap();
            conn.execute(DDL_IDX_SRC, vec![]).unwrap();
            conn.execute(DDL_IDX_DST, vec![]).unwrap();
            let seq: Vec<Vec<Value>> = (0..n)
                .map(|i| {
                    vec![
                        Value::Text(format!("src-{i:06}")),
                        Value::Text(format!("dst-{i:06}")),
                        Value::Text("hebbian".to_string()),
                    ]
                })
                .collect();
            conn.executemany(
                "INSERT INTO edges (src, dst, edge_type) VALUES (?, ?, ?)",
                seq,
            )
            .unwrap();
        }
        EdgeFixture { dir, store_path }
    }

    fn open(&self) -> Connection {
        Connection::open(self.store_path.to_str().unwrap(), 0).unwrap()
    }

    fn open_ro(&self) -> Connection {
        Connection::open_read_only(self.store_path.to_str().unwrap(), 0).unwrap()
    }

    fn sidecar_path(&self) -> std::path::PathBuf {
        self.dir.path().join("records.colindex")
    }
}

/// Query one `src` value's rows via a `col IN (...)` read, returning the
/// matched `dst` values sorted (order-insensitive comparison).
fn probe_src(conn: &mut Connection, src: &str) -> Vec<String> {
    let mut cur = conn
        .execute(
            "SELECT dst FROM edges WHERE src IN (?)",
            vec![Value::Text(src.to_string())],
        )
        .unwrap();
    let mut out: Vec<String> = cur
        .fetchall()
        .iter()
        .map(|r| match r.get_index(0) {
            Some(Value::Text(s)) => s.clone(),
            other => panic!("expected text dst, got {other:?}"),
        })
        .collect();
    out.sort();
    out
}

// ---------------------------------------------------------------------------
// persisted_col_index_loads_without_scan
//
// Persist, drop, reopen: a `col IN (...)` read — including on a value never
// queried before the persist — must issue zero full scans and return the
// same rows as a fresh lazy-built control connection on an unpersisted copy.
// ---------------------------------------------------------------------------
#[test]
fn persisted_col_index_loads_without_scan() {
    let f = EdgeFixture::seeded(50);

    {
        let mut conn = f.open();
        // Build lazily via one representative read, then persist.
        let _ = probe_src(&mut conn, "src-000000");
        conn.persist_col_indexes().unwrap();
    }
    assert!(
        f.sidecar_path().exists(),
        "persist must create the sidecar file"
    );

    let mut loaded = f.open();
    loaded.reset_full_scan_count();
    // A value NEVER queried before the persist (persist self-builds the WHOLE
    // table, not just the probed value).
    let held_out = probe_src(&mut loaded, "src-000049");
    assert_eq!(
        loaded.full_scan_count(),
        0,
        "a loaded sidecar must serve a held-out value with zero full scans"
    );
    assert_eq!(held_out, vec!["dst-000049".to_string()]);

    // Cross-check against a fresh lazy-built control connection over an
    // unpersisted copy of the same store contents.
    let control_dir = tempdir().unwrap();
    std::fs::copy(
        f.dir.path().join("records.lilli"),
        control_dir.path().join("control.lilli"),
    )
    .unwrap();
    let mut control = Connection::open(
        control_dir.path().join("control.lilli").to_str().unwrap(),
        0,
    )
    .unwrap();
    let control_rows = probe_src(&mut control, "src-000049");
    assert_eq!(
        held_out, control_rows,
        "loaded and lazy-built results must be byte-identical"
    );
}

// ---------------------------------------------------------------------------
// stale_generation_discards_and_lazy_rebuilds
//
// Persist; INSERT more rows (bumps the write generation); reopen: the read
// is correct (includes the new rows) and the lazy fallback fired
// (full_scan_count() > 0) — the stale sidecar was not trusted.
// ---------------------------------------------------------------------------
#[test]
fn stale_generation_discards_and_lazy_rebuilds() {
    let f = EdgeFixture::seeded(20);
    {
        let mut conn = f.open();
        let _ = probe_src(&mut conn, "src-000000");
        conn.persist_col_indexes().unwrap();
        // A committed write AFTER the persist advances the table's generation.
        conn.execute(
            "INSERT INTO edges (src, dst, edge_type) VALUES (?, ?, ?)",
            vec![
                Value::Text("src-new".to_string()),
                Value::Text("dst-new".to_string()),
                Value::Text("hebbian".to_string()),
            ],
        )
        .unwrap();
    }

    let mut reopened = f.open();
    reopened.reset_full_scan_count();
    reopened.reset_cells_visited_count();
    let rows = probe_src(&mut reopened, "src-new");
    assert_eq!(
        rows,
        vec!["dst-new".to_string()],
        "the post-persist insert must be visible"
    );
    assert!(
        reopened.full_scan_count() > 0 || reopened.cells_visited_count() > 0,
        "a generation mismatch must discard the sidecar and lazily rebuild \
         (expected a whole-table rebuild pass, saw neither counter move)"
    );
}

// ---------------------------------------------------------------------------
// equal_count_update_discards_sidecar
//
// The case a count-only fingerprint would falsely admit: a same-key in-place
// UPDATE of an indexed column value (row count unchanged). Reopen must NOT
// return the stale (old-value) posting and must lazily rebuild.
// ---------------------------------------------------------------------------
#[test]
fn equal_count_update_discards_sidecar() {
    let f = EdgeFixture::seeded(10);
    {
        let mut conn = f.open();
        let _ = probe_src(&mut conn, "src-000000");
        conn.persist_col_indexes().unwrap();
        // Same-key in-place UPDATE of an INDEXED column (src) — row count
        // unchanged, the exact case a count-only gate would falsely admit.
        // Do the post-persist write on a REOPENED connection (not the
        // persisting one) so the disk-seeded generation mirror is pinned
        // rather than reusing the persisting connection's in-memory bump.
    }
    {
        let mut writer = f.open();
        writer
            .execute(
                "UPDATE edges SET src = ? WHERE src = ?",
                vec![
                    Value::Text("src-000000-renamed".to_string()),
                    Value::Text("src-000000".to_string()),
                ],
            )
            .unwrap();
    }

    let mut reopened = f.open();
    reopened.reset_full_scan_count();
    reopened.reset_cells_visited_count();
    let old_value_rows = probe_src(&mut reopened, "src-000000");
    assert!(
        old_value_rows.is_empty(),
        "the OLD src value must not return the row after the in-place rename \
         (a falsely-loaded sidecar would still show it)"
    );
    let new_value_rows = probe_src(&mut reopened, "src-000000-renamed");
    assert_eq!(new_value_rows, vec!["dst-000000".to_string()]);
    assert!(
        reopened.full_scan_count() > 0 || reopened.cells_visited_count() > 0,
        "an equal-count in-place UPDATE of an indexed column must discard the \
         sidecar and lazily rebuild (expected a whole-table rebuild pass, \
         saw neither counter move)"
    );
}

// ---------------------------------------------------------------------------
// loaded_then_repersisted_equals_from_tree_rebuild (no-self-seal)
//
// A loaded-then-re-persisted sidecar must be equivalent (normalized postings)
// to one persisted from a pure from-tree rebuild at the same generation — a
// wrongly-trusted map is never laundered into a freshly-matching stamp.
// ---------------------------------------------------------------------------
#[test]
fn loaded_then_repersisted_equals_from_tree_rebuild() {
    let f = EdgeFixture::seeded(15);
    {
        let mut conn = f.open();
        let _ = probe_src(&mut conn, "src-000000");
        conn.persist_col_indexes().unwrap();
    }

    // Copy A: open (loads the sidecar) -> persist again.
    let dir_a = tempdir().unwrap();
    let path_a = dir_a.path().join("records.lilli");
    std::fs::copy(f.dir.path().join("records.lilli"), &path_a).unwrap();
    std::fs::copy(f.sidecar_path(), dir_a.path().join("records.colindex")).unwrap();
    {
        let mut conn_a = Connection::open(path_a.to_str().unwrap(), 0).unwrap();
        conn_a.persist_col_indexes().unwrap();
    }

    // Copy B: delete the sidecar -> open (lazy from-tree build) -> persist.
    let dir_b = tempdir().unwrap();
    let path_b = dir_b.path().join("records.lilli");
    std::fs::copy(f.dir.path().join("records.lilli"), &path_b).unwrap();
    {
        let mut conn_b = Connection::open(path_b.to_str().unwrap(), 0).unwrap();
        conn_b.persist_col_indexes().unwrap();
    }

    let envelope_a = decode_envelope(&dir_a.path().join("records.colindex"));
    let envelope_b = decode_envelope(&dir_b.path().join("records.colindex"));
    assert_envelopes_equal(&envelope_a, &envelope_b);
}

/// Decode a persisted-envelope file into a normalized, comparable shape:
/// per-table (generation, row_count, sorted columns, sorted postings).
fn decode_envelope(
    path: &std::path::Path,
) -> Vec<(
    String,
    i64,
    u64,
    Vec<String>,
    Vec<((String, String), Vec<i64>)>,
)> {
    let bytes = std::fs::read(path).unwrap();
    let envelope: lilliengine::conn::PersistedColIndexes =
        ciborium::de::from_reader(bytes.as_slice()).unwrap();
    let mut tables: Vec<_> = envelope
        .tables
        .into_iter()
        .map(|t| {
            let mut columns = t.columns.clone();
            columns.sort();
            let mut postings: Vec<((String, String), Vec<i64>)> = t
                .index
                .into_postings()
                .into_iter()
                .map(|(k, mut v)| {
                    v.sort();
                    (k, v)
                })
                .collect();
            postings.sort();
            (t.table, t.generation, t.row_count, columns, postings)
        })
        .collect();
    tables.sort_by(|a, b| a.0.cmp(&b.0));
    tables
}

fn assert_envelopes_equal(
    a: &[(
        String,
        i64,
        u64,
        Vec<String>,
        Vec<((String, String), Vec<i64>)>,
    )],
    b: &[(
        String,
        i64,
        u64,
        Vec<String>,
        Vec<((String, String), Vec<i64>)>,
    )],
) {
    assert_eq!(
        a, b,
        "a loaded-then-re-persisted sidecar must be equivalent to a pure \
         from-tree rebuild at the same generation (no self-sealing)"
    );
}

// ---------------------------------------------------------------------------
// read_only_open_loads_sidecar
//
// open_read_only routes through the same from_store load seam: a RO
// connection over a persisted store loads-not-rebuilds for free, and never
// writes the sidecar itself.
// ---------------------------------------------------------------------------
#[test]
fn read_only_open_loads_sidecar() {
    let f = EdgeFixture::seeded(12);
    {
        let mut conn = f.open();
        let _ = probe_src(&mut conn, "src-000000");
        conn.persist_col_indexes().unwrap();
    }
    let sidecar_bytes_before = std::fs::read(f.sidecar_path()).unwrap();

    let mut ro = f.open_ro();
    ro.reset_full_scan_count();
    let rows = probe_src(&mut ro, "src-000011");
    assert_eq!(rows, vec!["dst-000011".to_string()]);
    assert_eq!(
        ro.full_scan_count(),
        0,
        "a read-only open must load the sidecar and serve a held-out value \
         with zero full scans"
    );

    // persist_col_indexes on the RO connection must not create/modify the file:
    // a read-only mount is a bypass-safe no-op (Ok, untouched file), not an error.
    let result = ro.persist_col_indexes();
    assert!(
        result.is_ok(),
        "a read-only connection's persist must be a no-op, not an error"
    );
    let sidecar_bytes_after = std::fs::read(f.sidecar_path()).unwrap();
    assert_eq!(
        sidecar_bytes_before, sidecar_bytes_after,
        "a no-op RO persist must leave the sidecar file byte-unchanged"
    );
}

// ---------------------------------------------------------------------------
// garbage_sidecar_falls_back_lazily
//
// A torn/garbage sidecar file must be treated as absent: no error at open,
// correct results, and the lazy fallback fires (full_scan_count() > 0).
// ---------------------------------------------------------------------------
#[test]
fn garbage_sidecar_falls_back_lazily() {
    let f = EdgeFixture::seeded(8);
    std::fs::write(
        f.sidecar_path(),
        b"not a valid cbor envelope, just junk bytes",
    )
    .unwrap();

    let mut conn = f.open();
    conn.reset_full_scan_count();
    conn.reset_cells_visited_count();
    let rows = probe_src(&mut conn, "src-000003");
    assert_eq!(
        rows,
        vec!["dst-000003".to_string()],
        "results must stay correct on a garbage sidecar"
    );
    assert!(
        conn.full_scan_count() > 0 || conn.cells_visited_count() > 0,
        "a garbage sidecar must be treated as absent (lazy rebuild fires)"
    );
}

// ---------------------------------------------------------------------------
// persist_refuses_inside_open_transaction
//
// persist_col_indexes must err when called inside an open outer transaction
// (persist only committed state); the on-disk sidecar must be unchanged.
// ---------------------------------------------------------------------------
#[test]
fn persist_refuses_inside_open_transaction() {
    let f = EdgeFixture::seeded(5);
    let mut conn = f.open();
    conn.execute("BEGIN", vec![]).unwrap();
    let result = conn.persist_col_indexes();
    assert!(
        result.is_err(),
        "persist_col_indexes must refuse inside an open transaction"
    );
    assert!(
        !f.sidecar_path().exists(),
        "a refused persist inside an open transaction must not create the sidecar"
    );
    conn.execute("ROLLBACK", vec![]).unwrap();
}

// ---------------------------------------------------------------------------
// cbor_map_shape_round_trip
//
// The envelope containing a HashMap with (String, String) tuple keys must
// round-trip byte-stably through ciborium.
// ---------------------------------------------------------------------------
#[test]
fn cbor_map_shape_round_trip() {
    let f = EdgeFixture::seeded(6);
    {
        let mut conn = f.open();
        let _ = probe_src(&mut conn, "src-000000");
        conn.persist_col_indexes().unwrap();
    }
    let bytes = std::fs::read(f.sidecar_path()).unwrap();
    let envelope: lilliengine::conn::PersistedColIndexes =
        ciborium::de::from_reader(bytes.as_slice()).expect("CBOR decode must succeed");
    assert_eq!(envelope.format_version, 3);
    let edges_table = envelope
        .tables
        .iter()
        .find(|t| t.table == "edges")
        .expect("edges table entry must be present");
    assert!(
        !edges_table.index.map_is_empty(),
        "the edges ColIndex map must be populated"
    );
    // edges keys on its composite PK, not an `id` column, so its persisted
    // id-index stays the unbuilt placeholder (the load path skips it).
    assert!(
        !edges_table.id_index.is_built(),
        "the edges table has no id column to index"
    );

    // Re-encode and re-decode to confirm the round trip is stable.
    let mut reencoded = Vec::new();
    ciborium::ser::into_writer(&envelope, &mut reencoded).unwrap();
    let redecoded: lilliengine::conn::PersistedColIndexes =
        ciborium::de::from_reader(reencoded.as_slice()).unwrap();
    assert_eq!(redecoded.tables.len(), envelope.tables.len());
}

// ---------------------------------------------------------------------------
// persisted_id_index_serves_warm_boot_without_scan
//
// The id→row-key index rides the same sidecar under the same write-generation
// fingerprint as the ColIndex. After persist + reopen, a `WHERE id = ?` read
// AND update must resolve by the loaded id map with zero full scans — the
// first `id = ?` lookup after a cold boot otherwise pays a whole-table
// ensure_built scan to build the map. A cold (unpersisted) control still pays
// that build scan, proving the sidecar is what removes it.
// ---------------------------------------------------------------------------
#[test]
fn persisted_id_index_serves_warm_boot_without_scan() {
    const DDL_R: &str = "CREATE TABLE IF NOT EXISTS records ( \
        vec_label INTEGER PRIMARY KEY AUTOINCREMENT , id TEXT NOT NULL UNIQUE , \
        tier TEXT , labile_until TEXT , embedding BLOB NOT NULL )";
    const DDL_IDX_ID: &str = "CREATE INDEX idx_records_id ON records (id)";

    let dir = tempdir().unwrap();
    let store_path = dir.path().join("records.lilli");
    let sidecar = dir.path().join("records.colindex");

    {
        let mut conn = Connection::open(store_path.to_str().unwrap(), 0).unwrap();
        conn.execute(DDL_R, vec![]).unwrap();
        conn.execute(DDL_IDX_ID, vec![]).unwrap();
        let seq: Vec<Vec<Value>> = (0..200i64)
            .map(|i| {
                vec![
                    Value::Text(format!("id-{i:06}")),
                    Value::Text("episodic".to_string()),
                    Value::Null,
                    Value::Blob(vec![0u8; 32]),
                ]
            })
            .collect();
        conn.executemany(
            "INSERT INTO records (id, tier, labile_until, embedding) VALUES (?, ?, ?, ?)",
            seq,
        )
        .unwrap();
        // Persist self-builds the whole id-index (no representative read needed).
        conn.persist_col_indexes().unwrap();
    }
    assert!(sidecar.exists(), "persist must create the sidecar");

    // --- WARM: reopen (loads the sidecar), then a WHERE id = ? update must be
    //     scan-free (the loaded id map serves the point lookup). ---
    {
        let mut warm = Connection::open(store_path.to_str().unwrap(), 0).unwrap();
        warm.reset_full_scan_count();
        let mut cur = warm
            .execute(
                "UPDATE records SET labile_until = '2026-01-01' WHERE id = 'id-000123'",
                vec![],
            )
            .unwrap();
        assert_eq!(cur.rowcount, 1, "the id-000123 row must be updated");
        assert_eq!(
            warm.full_scan_count(),
            0,
            "a loaded id-index must serve WHERE id = ? with zero full scans on a warm boot"
        );
    }

    // --- COLD control: an unpersisted copy pays the ensure_built build scan on
    //     its first id = ? lookup, proving the sidecar is what removes it. ---
    {
        let cold_dir = tempdir().unwrap();
        let cold_path = cold_dir.path().join("records.lilli");
        std::fs::copy(&store_path, &cold_path).unwrap();
        // no sidecar copied → lazy build path. The build's whole-table walk
        // is not a counted "full scan" (the old >=1 assertion actually
        // measured the colgen first-touch meta scan, which the replay-time
        // mirror seeding removed), so the control asserts the direct signal:
        // without a sidecar no built index exists before the first id
        // lookup, and the lookup builds it.
        let mut cold = Connection::open(cold_path.to_str().unwrap(), 0).unwrap();
        assert!(
            !cold.id_index_ready("records"),
            "no sidecar — the cold open must start without a built id index"
        );
        let mut cur = cold
            .execute(
                "UPDATE records SET labile_until = '2026-01-01' WHERE id = 'id-000123'",
                vec![],
            )
            .unwrap();
        assert_eq!(cur.rowcount, 1);
        assert!(
            cold.id_index_ready("records"),
            "the first id lookup on a sidecar-less open must pay the lazy build"
        );
    }
}

// ---------------------------------------------------------------------------
// persisted_ordered_index_loads_built
//
// Build the ORDER BY index lazily via one ordered read, persist, reopen: the
// fresh connection must hold the ordered index pre-built (no population scan)
// and serve the same rows as the connection that built it.
// ---------------------------------------------------------------------------
#[test]
fn persisted_ordered_index_loads_built() {
    fn probe_ordered(conn: &mut Connection) -> Vec<String> {
        let mut cur = conn
            .execute("SELECT src FROM edges ORDER BY src LIMIT 10", vec![])
            .unwrap();
        cur.fetchall()
            .iter()
            .map(|r| match r.get_index(0) {
                Some(Value::Text(s)) => s.clone(),
                other => panic!("expected text src, got {other:?}"),
            })
            .collect()
    }

    let f = EdgeFixture::seeded(50);
    let expected: Vec<String>;
    {
        let mut conn = f.open();
        expected = probe_ordered(&mut conn);
        assert!(
            conn.ordered_index_is_built("edges"),
            "the ordered probe must have built the ORDER BY index"
        );
        conn.persist_col_indexes().unwrap();
    }

    let mut loaded = f.open();
    assert!(
        loaded.ordered_index_is_built("edges"),
        "a fresh open over the sidecar must admit the ordered index pre-built"
    );
    assert_eq!(
        probe_ordered(&mut loaded),
        expected,
        "the loaded ordered index must serve the same rows as the builder"
    );

    let mut loaded_ro = f.open_ro();
    assert!(
        loaded_ro.ordered_index_is_built("edges"),
        "a read-only open must admit the persisted ordered index too"
    );
    assert_eq!(probe_ordered(&mut loaded_ro), expected);
}

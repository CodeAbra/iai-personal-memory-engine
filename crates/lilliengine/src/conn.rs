//! The sqlite3-shaped DB-API state machine over the storage engine.
//!
//! Reconciles two transaction models, mirroring the reference shim:
//!
//! - Outer model (the host): explicit `BEGIN` / `COMMIT` / `ROLLBACK` SQL
//!   strings, reflected by [`Connection::in_transaction`].
//! - Inner model (the executor): each single INSERT/UPDATE/DELETE opens its own
//!   store `begin_write()` / `commit()`. When an outer transaction is open, that
//!   inner per-statement transaction is suppressed ([`TxnScope::Suppressed`]) so
//!   all DML accumulates in the one outer transaction.
//!
//! Pure Rust — no pyo3 here. The pyo3 surface (`iai_mcp_native.engine`) wraps a
//! [`Connection`] behind a Mutex in `py.rs`. The six connection-tuning PRAGMAs
//! the host issues at open time are accepted-and-ignored; `PRAGMA table_info(t)`
//! is synthesized from the catalog; `PRAGMA query_only=ON` flips a read-only
//! guard that rejects writes with the stdlib readonly error.

use std::collections::{BTreeMap, HashMap};

use lillibrain::{Store, Value};

use crate::ast::Stmt;
use crate::catalog::Catalog;
use crate::error::{EngineError, Result};
use crate::exec::{
    execute_delete, execute_insert, execute_insert_many, execute_select, execute_update,
    ColIndex, ConflictIndex, IdIndex, OrderedColIndex,
    TxnScope, WriteOutcome,
};
use crate::meta::MetaTable;
use crate::parser::parse;
use crate::plan::{plan_select, ScanType};

/// The stdlib error text raised when a write is attempted under read-only mode.
pub const READONLY_WRITE_ERR: &str = "attempt to write a readonly database";

/// Upper bound on distinct SQL strings held in the parsed-statement cache.
///
/// The hot recall path issues a small fixed set of distinct query strings
/// (pending-marker probes, the `id IN (...)` batch fetch, the community-gate
/// reads), so the live working set is far below this cap. The bound exists only
/// to keep the cache from growing without limit under an adversarial caller that
/// varies its SQL text — on overflow the cache is dropped and refilled from the
/// current working set, never grows past the cap.
const PARSE_CACHE_CAP: usize = 256;

/// The on-disk column-index sidecar's file name, alongside the store file.
const COL_INDEX_SIDECAR_NAME: &str = "records.colindex";

/// The persisted-envelope codec version. A load with a mismatched version is
/// treated as absent (structurally invalidates old files across a codec change).
const COL_INDEX_FORMAT_VERSION: u32 = 2;

/// The on-disk sidecar: one committed, self-built [`ColIndex`] snapshot per
/// catalog-indexed table, each stamped with the fingerprint the load gate
/// checks before trusting it.
///
/// Written only by [`Connection::persist_col_indexes`] on the read-write path
/// (the SLEEP maintenance cadence); read by every fresh `open` / `open_read_only`
/// inside `from_store`. Never trusted on its own — the load gate in `from_store`
/// requires every one of a table's stamped fingerprint fields to match the
/// store's current committed state before the loaded map replaces the lazy,
/// always-correct `ensure_built` path for that table.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistedColIndexes {
    /// The envelope codec version. Bumped on any incompatible shape change.
    pub format_version: u32,
    /// One entry per catalog-indexed table that was built at persist time.
    pub tables: Vec<PersistedTableIndex>,
}

/// One table's persisted [`ColIndex`] plus the fingerprint the load gate
/// checks: the table's committed write-generation (catches every committed
/// DML, including a count-preserving in-place UPDATE), its row count (a
/// scan-free defense-in-depth cross-check), and its indexed-columns set (a DDL
/// drift detector). ALL three, plus the envelope's `format_version`, must match
/// before the loaded map is trusted; any mismatch discards this table's entry
/// and the pre-existing lazy `ensure_built` path serves the table instead.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PersistedTableIndex {
    /// The table name.
    pub table: String,
    /// The indexed-columns set at persist time (order-insensitive compare).
    pub columns: Vec<String>,
    /// The table's committed write-generation at persist time.
    pub generation: i64,
    /// The table's row count at persist time (`Tree::count_cells`, scan-free).
    pub row_count: u64,
    /// The built column index.
    pub index: ColIndex,
    /// The built id→row-key index, persisted under the SAME fingerprint so a warm
    /// boot loads it instead of paying the first `WHERE id = ?` lookup's build
    /// scan. Empty/unbuilt for tables without an `id` column; the load path only
    /// admits a built one into `id_caches`.
    #[serde(default)]
    pub id_index: IdIndex,
    /// The built ORDER BY index, persisted under the SAME fingerprint so a cold
    /// process serves its first ordered scan from the sidecar instead of paying
    /// the population scan. Unbuilt when the writer never ran an ordered query
    /// on the table; the load path only admits a built one into `ordered_caches`.
    #[serde(default)]
    pub ordered_index: OrderedColIndex,
}

/// A transferable snapshot of built index pairs (Arc-backed — cloning copies
/// no postings), each entry stamped with the committed write-generation it
/// was exported at. Produced by [`Connection::export_built_indexes`], consumed
/// by [`Connection::adopt_built_indexes`] on a freshly-opened reader.
#[derive(Debug, Clone, Default)]
pub struct IndexSnapshot {
    tables: Vec<(String, i64, ColIndex, IdIndex, OrderedColIndex)>,
}

/// A single result column's description, mirroring the DB-API `description`
/// 7-tuple (only the name slot is populated; the rest are NULL).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDesc {
    /// The column name.
    pub name: String,
}

/// One materialized result row: the output column values in column order.
///
/// Carries the column-name slice alongside the values so the pyo3 layer can
/// present both dict-like (by name) and positional (by index) access, matching
/// `sqlite3.Row`.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    /// The column values in output order.
    pub values: Vec<Value>,
    /// The output column names, shared with the cursor description.
    pub columns: std::sync::Arc<Vec<String>>,
}

impl Row {
    /// The value at a positional index, or `None` when out of range.
    pub fn get_index(&self, i: usize) -> Option<&Value> {
        self.values.get(i)
    }

    /// The value for a column name, or `None` when the name is absent.
    pub fn get_name(&self, name: &str) -> Option<&Value> {
        self.columns
            .iter()
            .position(|c| c == name)
            .and_then(|i| self.values.get(i))
    }
}

/// A materialized result set with a position-offset fetch API.
///
/// The rows are captured at execution; `fetchone`/`fetchmany`/`fetchall` and
/// iteration advance a shared offset rather than re-running any query — the
/// contract the streaming recall scan depends on.
#[derive(Debug, Clone)]
pub struct CursorState {
    columns: std::sync::Arc<Vec<String>>,
    rows: Vec<Row>,
    pos: usize,
    /// The injected AUTOINCREMENT label exposed as `lastrowid`, or `None`.
    pub lastrowid: Option<i64>,
    /// The affected-row count for a write statement (`-1` for a SELECT, matching
    /// the stdlib default for a statement that returns rows).
    pub rowcount: i64,
}

impl CursorState {
    /// An empty result set (no rows, no columns).
    pub fn empty() -> CursorState {
        CursorState {
            columns: std::sync::Arc::new(Vec::new()),
            rows: Vec::new(),
            pos: 0,
            lastrowid: None,
            rowcount: -1,
        }
    }

    /// Build a result set from column names and value rows.
    fn from_rows(columns: Vec<String>, value_rows: Vec<Vec<Value>>) -> CursorState {
        let cols = std::sync::Arc::new(columns);
        let rows = value_rows
            .into_iter()
            .map(|values| Row {
                values,
                columns: std::sync::Arc::clone(&cols),
            })
            .collect();
        CursorState {
            columns: cols,
            rows,
            pos: 0,
            lastrowid: None,
            rowcount: -1,
        }
    }

    /// A single-row result reporting a `query_only` flag as an integer, matching
    /// the `PRAGMA query_only` read form.
    pub fn query_only_report(value: bool) -> CursorState {
        CursorState::from_rows(
            vec!["query_only".to_string()],
            vec![vec![Value::Int(i64::from(value))]],
        )
    }

    /// The output column names in result order.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// The DB-API description: one [`ColumnDesc`] per output column.
    pub fn description(&self) -> Vec<ColumnDesc> {
        self.columns
            .iter()
            .map(|name| ColumnDesc { name: name.clone() })
            .collect()
    }

    /// Return the next row, or `None` when exhausted.
    pub fn fetchone(&mut self) -> Option<Row> {
        if self.pos >= self.rows.len() {
            return None;
        }
        let row = self.rows[self.pos].clone();
        self.pos += 1;
        Some(row)
    }

    /// Return all remaining rows, advancing the offset to the end.
    pub fn fetchall(&mut self) -> Vec<Row> {
        let out = self.rows[self.pos..].to_vec();
        self.pos = self.rows.len();
        out
    }

    /// Return up to `size` rows, advancing the offset by that count.
    pub fn fetchmany(&mut self, size: usize) -> Vec<Row> {
        let end = (self.pos + size).min(self.rows.len());
        let out = self.rows[self.pos..end].to_vec();
        self.pos = end;
        out
    }
}

/// The sqlite3-shaped connection over the storage engine.
///
/// Owns the [`Store`], the durable [`MetaTable`], the live [`Catalog`], and the
/// `table → root_page` map. Tracks outer-transaction state and the per-table
/// `id → row-key` index for the `WHERE id = '<lit>'` UPDATE/DELETE fast path.
pub struct Connection {
    store: Store,
    meta: MetaTable,
    catalog: Catalog,
    root_map: BTreeMap<String, u32>,
    in_transaction: bool,
    /// Set when a mutating statement (INSERT/UPDATE/DELETE) returns an error while
    /// an outer transaction is open, and cleared on every COMMIT or ROLLBACK.
    ///
    /// A non-id-column UPDATE leaves the row at the same B-tree key under the same
    /// id, so the id→key map remains valid across a successful commit.  The commit
    /// path therefore keeps the id/col/conflict caches warm — dropping them
    /// unconditionally on every commit was an over-defensive guard.
    ///
    /// The guard was correct for one specific case: a mutating statement that errors
    /// inside an open transaction may have run the in-memory cache ahead of the
    /// B-tree (inserting a speculative id→key entry for a row that was never durably
    /// written).  If the host swallows that error and then commits the outer
    /// transaction, a warm cache would expose a stale entry.  This flag scopes the
    /// defensive clear to exactly that scenario: it is set on inner DML error and
    /// checked at commit — caches are cleared only when the flag is true, then the
    /// flag is reset.  ROLLBACK always clears unconditionally (reverted rows must
    /// never survive in the cache regardless of error state).
    txn_tainted: bool,
    query_only: bool,
    /// True when the connection was opened against a read-only mount. Unlike the
    /// mutable `query_only` flag, this is fixed at open time and `PRAGMA
    /// query_only=OFF` cannot clear it — a write on a read-only mount always
    /// fails with the read-only error, matching a database opened mode=ro.
    is_readonly_mount: bool,
    /// The `PRAGMA busy_timeout` value in milliseconds, round-tripped so a read
    /// after `PRAGMA busy_timeout=N` reports `N` (the host sets and reads it back
    /// to confirm the lock-wait window). The engine's same-process open model
    /// never blocks, so this is a reported value, not a wait the engine honors.
    busy_timeout: i64,
    id_caches: HashMap<String, IdIndex>,
    conflict_caches: HashMap<String, ConflictIndex>,
    /// Per-table non-unique `column-value → row-keys` indexes for the
    /// `WHERE col IN (...)` recall edge-adjacency lookup. Built lazily and dropped
    /// on any write to the table (recall is read-only, so they hold across recalls
    /// and rebuild only after a write batch).
    col_caches: HashMap<String, ColIndex>,
    /// Per-table BTreeMap ordered index for `ORDER BY <col> DESC LIMIT k` queries.
    /// Keyed by table name; the index column is determined by the catalog's declared
    /// indexes for that table.  Lifecycle mirrors `col_caches`: warm across untainted
    /// commits, cleared on rollback and tainted commit, dropped on executemany and
    /// rename/drop.
    ordered_caches: HashMap<String, OrderedColIndex>,
    /// Bare `COUNT(*)` cache: `table → (committed write-generation, count)`.
    ///
    /// A bare count walks the whole leaf sibling chain (`count_cells`), which
    /// on a production-scale table costs hundreds of milliseconds — and the
    /// host's read paths issue it per query (an emptiness probe ahead of every
    /// similarity search). The committed write-generation is the same airtight
    /// staleness fence the column-index sidecar trusts: every committed DML on
    /// a col-indexed table bumps it, so a cached count keyed by the generation
    /// it was observed at is exact for as long as the generation still reads
    /// the same. Only col-indexed tables are cached (others never bump the
    /// fence), and the cache is neither read-populated nor consulted for a
    /// mutated state inside an open transaction (see the Select arm), so a
    /// rolled-back in-transaction count can never be served later.
    bare_count_cache: HashMap<String, (i64, i64)>,
    /// Parsed-statement cache: trimmed SQL string → its AST.
    ///
    /// The AST is a pure function of the SQL text — `tokenize` + recursive
    /// descent read no catalog or transaction state — so a cached `Stmt` is
    /// reused across executes regardless of binds or schema. Params bind onto the
    /// reused AST per call (in `plan_select` / the executors), and the plan is
    /// rebuilt per call, so caching the AST alone changes no result, only the
    /// per-execute lex+parse cost. Bounded by [`PARSE_CACHE_CAP`].
    parse_cache: HashMap<String, Stmt>,
}

impl Connection {
    /// The column for the ordered (BTreeMap) index for `table`, selected to serve
    /// the query's ORDER BY column or a MIN/MAX aggregate column.
    ///
    /// Returns the ordered column ONLY when `query_order_col` names a
    /// catalog-declared non-partial index on `table`.  Returns `None` when the
    /// query has no ordered target (`query_order_col` is `None`) or when the
    /// column is not a non-partial indexed column.
    ///
    /// Keying the ordered index to the actual target column is what prevents a
    /// no-ORDER-BY read (e.g. `SELECT * FROM events`) from creating an ordered
    /// cache on the alphabetically-first indexed column (e.g. `kind`) that would
    /// then shadow the `ts` index a later `ORDER BY ts DESC` recall needs.  For
    /// the events table an `ORDER BY ts` query resolves to `ts`; a no-ORDER-BY
    /// query creates no ordered cache at all.
    ///
    /// No affinity or nullability restriction applies: the ordered index files
    /// EVERY row (including NULLs) by its collation tag, so an ordered walk is
    /// total across storage classes and the executor's shape gates alone decide
    /// whether a query is index-servable.
    fn catalog_ordered_index_column(&self, table: &str, query_order_col: Option<&str>) -> Option<String> {
        // No ordered target => no ordered index to serve; never create one
        // speculatively.
        let target = query_order_col?;
        let cols = self.catalog.indexed_columns(table);
        cols.into_iter()
            .find(|(col, is_partial)| !is_partial && col == target)
            .map(|(col, _)| col)
    }

    /// The columns to hash-index for a table, derived from the schema's recorded
    /// `CREATE INDEX` declarations.
    ///
    /// Every single-column non-partial declared index contributes its column.
    /// Partial indexes are skipped, except `embedding_pending`: its partial
    /// predicate (`WHERE embedding_pending=1`) is satisfied implicitly by an
    /// equality probe (`col = 1`), so it is retained as a plain equality index
    /// matching the previous hardcoded behavior. Any other partial-flagged column
    /// is omitted; queries on those columns fall through to a full scan (correct,
    /// no stale rows).
    ///
    /// Returns an empty `Vec` for tables with no declared indexes.
    fn catalog_col_index_columns(&self, table: &str) -> Vec<String> {
        catalog_col_index_columns(&self.catalog, table)
    }

    /// Open (or create) the engine database at `path`.
    ///
    /// Opens the store, formats the reserved meta page on a fresh file, and
    /// replays the persisted DDL + root mappings so the catalog and `root_map`
    /// are ready. `embed_dim` is accepted for parity with the reference open
    /// signature; the engine derives its column shapes from the replayed DDL.
    pub fn open(path: &str, _embed_dim: i64) -> Result<Connection> {
        let store = Store::open(path).map_err(open_err)?;
        Self::from_store(store, false)
    }

    /// Open an existing store for lock-free read-only access.
    ///
    /// The store is opened without taking the exclusive advisory lock and the
    /// connection's read-only guard is armed, so a separate reader process (the
    /// ANN-index rebuild worker, a daemon-down recall reader) can read committed
    /// rows while a writer process holds the store. The caller must read a
    /// checkpointed store (the maintenance pass checkpoints the WAL before
    /// spawning the reader).
    pub fn open_read_only(path: &str, _embed_dim: i64) -> Result<Connection> {
        let store = Store::open_read_only(path).map_err(open_err)?;
        Self::from_store(store, true)
    }

    fn from_store(store: Store, read_only: bool) -> Result<Connection> {
        let meta = MetaTable::open(&store)?;
        let mut catalog = Catalog::new();
        let mut root_map: BTreeMap<String, u32> = BTreeMap::new();
        meta.replay(&store, &mut catalog, &mut root_map)?;
        // Load the on-disk column-index sidecar under the all-must-match
        // fingerprint gate (section 3 of the design). ANY failure — absent
        // file, I/O error, decode error, format_version mismatch, or a
        // per-table fingerprint mismatch — leaves that table (or the whole
        // set) out of `col_caches` / `id_caches`, and the pre-existing lazy
        // `ensure_built` path serves it unchanged. Covers BOTH `open` and
        // `open_read_only` (both route through `from_store`), so the daemon-down
        // RO recall reader loads-not-rebuilds for free. The id-index rides the
        // same gate, so a warm boot serves `WHERE id = ?` with no build scan.
        let (mut col_caches, mut id_caches, ordered_caches) =
            load_col_indexes(&store, &meta, &catalog, &root_map);
        // A read-write connection maintains its indexes per committed write, so
        // it must own private postings: materialize the copy-on-write clone HERE,
        // once, on the open (boot) path — never lazily on the first awake-path
        // write. Read-only connections keep sharing the decoded snapshot and
        // never pay any copy.
        if !read_only {
            for idx in col_caches.values_mut() {
                idx.unshare();
            }
            for idx in id_caches.values_mut() {
                idx.unshare();
            }
        }
        // Read-only open: for any col-indexed table the sidecar gate did NOT
        // admit, adopt a same-generation pair from the process-wide
        // built-index cache (published by an earlier reader that paid the
        // lazy build), so a burst of reader opens at one write-generation
        // pays at most ONE build per table. Column-set mismatch declines the
        // adoption (the lazy path stays the always-correct fallback).
        if read_only {
            let cache = built_index_cache();
            for table in root_map.keys() {
                if col_caches.contains_key(table) {
                    continue;
                }
                let cols = catalog_col_index_columns(&catalog, table);
                if cols.is_empty() {
                    continue;
                }
                let Ok(generation) = meta.col_generation(&store, table) else {
                    continue;
                };
                let key = (store.path().to_path_buf(), table.clone(), generation);
                if let Ok(guard) = cache.lock() {
                    if let Some((cidx, iidx)) = guard.get(&key) {
                        if !same_column_set(&cols, cidx.columns()) {
                            continue;
                        }
                        col_caches.insert(table.clone(), cidx.clone());
                        if iidx.is_built() {
                            id_caches.insert(table.clone(), iidx.clone());
                        }
                    }
                }
            }
        }
        Ok(Connection {
            store,
            meta,
            catalog,
            root_map,
            in_transaction: false,
            txn_tainted: false,
            query_only: read_only,
            is_readonly_mount: read_only,
            busy_timeout: 0,
            id_caches,
            conflict_caches: HashMap::new(),
            col_caches,
            ordered_caches,
            bare_count_cache: HashMap::new(),
            parse_cache: HashMap::new(),
        })
    }

    /// Publish this read-only connection's built index pair for `table` to
    /// the process-wide built-index cache, keyed by the connection's
    /// snapshot write-generation, so sibling readers at the same generation
    /// adopt it instead of paying their own lazy build. No-op when the
    /// table's column index is not built or the generation cannot be read;
    /// an already-published key is left untouched.
    fn maybe_publish_built_indexes(&self, table: &str) {
        let Some(cidx) = self.col_caches.get(table) else {
            return;
        };
        if !cidx.is_built() {
            return;
        }
        let Ok(generation) = self.meta.col_generation(&self.store, table) else {
            return;
        };
        let key = (
            self.store.path().to_path_buf(),
            table.to_string(),
            generation,
        );
        let cache = built_index_cache();
        if let Ok(mut guard) = cache.lock() {
            if guard.contains_key(&key) {
                return;
            }
            if guard.len() >= BUILT_INDEX_CACHE_CAP {
                guard.clear();
            }
            let iidx = self
                .id_caches
                .get(table)
                .cloned()
                .unwrap_or_default();
            guard.insert(key, (cidx.clone(), iidx));
        }
    }

    /// Export this connection's BUILT index pairs as a transferable snapshot
    /// (Arc clones — no postings copy), each stamped with the table's
    /// committed write-generation, so a freshly-opened read-only connection
    /// can adopt them instead of paying its own lazy whole-table builds.
    ///
    /// Returns an EMPTY snapshot while a transaction is open: the in-memory
    /// indexes may then include uncommitted rows, which must never leak into
    /// another connection's view.
    pub fn export_built_indexes(&self) -> IndexSnapshot {
        let mut tables = Vec::new();
        if self.in_transaction {
            return IndexSnapshot { tables };
        }
        for (table, cidx) in &self.col_caches {
            if !cidx.is_built() {
                continue;
            }
            let Ok(generation) = self.meta.col_generation(&self.store, table) else {
                continue;
            };
            let iidx = self
                .id_caches
                .get(table)
                .cloned()
                .unwrap_or_default();
            // The ordered index rides the same generation stamp: rebuilt from
            // scratch by every fresh reader otherwise (its population scan is
            // the dominant reopen cost after the col/id indexes are adopted).
            let oidx = self
                .ordered_caches
                .get(table)
                .cloned()
                .unwrap_or_default();
            tables.push((table.clone(), generation, cidx.clone(), iidx, oidx));
        }
        IndexSnapshot { tables }
    }

    /// Adopt index pairs from `snapshot` for every table whose stamped
    /// write-generation matches THIS connection's committed generation and
    /// whose column set matches the current catalog — the same airtight
    /// fence the sidecar load gate uses, applied per table. A mismatched
    /// table is skipped (the lazy build stays the always-correct fallback);
    /// an already-built local index is never overwritten.
    pub fn adopt_built_indexes(&mut self, snapshot: &IndexSnapshot) {
        for (table, generation, cidx, iidx, oidx) in &snapshot.tables {
            let already_built = self
                .col_caches
                .get(table)
                .map(|c| c.is_built())
                .unwrap_or(false);
            if already_built {
                continue;
            }
            let cols = catalog_col_index_columns(&self.catalog, table);
            if cols.is_empty() || !same_column_set(&cols, cidx.columns()) {
                continue;
            }
            let Ok(current) = self.meta.col_generation(&self.store, table) else {
                continue;
            };
            if current != *generation {
                continue;
            }
            self.col_caches.insert(table.clone(), cidx.clone());
            if iidx.is_built() && !self
                .id_caches
                .get(table)
                .map(|i| i.is_built())
                .unwrap_or(false)
            {
                self.id_caches.insert(table.clone(), iidx.clone());
            }
            if oidx.is_built() && !self
                .ordered_caches
                .get(table)
                .map(|o| o.is_built())
                .unwrap_or(false)
            {
                self.ordered_caches.insert(table.clone(), oidx.clone());
            }
        }
    }

    /// True when `table`'s ORDER BY index is resident and built — exposed so
    /// integration tests can assert a sidecar load or a snapshot adoption
    /// admitted it without paying the population scan.
    pub fn ordered_index_is_built(&self, table: &str) -> bool {
        self.ordered_caches
            .get(table)
            .map(|o| o.is_built())
            .unwrap_or(false)
    }

    /// Parse `sql` into its AST, reusing a cached parse when one exists.
    ///
    /// The AST depends only on the SQL text, so a hit returns a clone of the
    /// cached `Stmt` — correct for any binds and any catalog state. A miss parses
    /// once and inserts. The cache only ever holds successful parses; a parse
    /// error propagates without poisoning the cache. On reaching the size cap the
    /// cache is dropped wholesale before the new entry lands, keeping resident
    /// memory bounded.
    fn parse_cached(&mut self, sql: &str) -> Result<Stmt> {
        if let Some(stmt) = self.parse_cache.get(sql) {
            return Ok(stmt.clone());
        }
        let stmt = parse(sql)?;
        if self.parse_cache.len() >= PARSE_CACHE_CAP {
            self.parse_cache.clear();
        }
        self.parse_cache.insert(sql.to_string(), stmt.clone());
        Ok(stmt)
    }

    /// True when an outer `BEGIN` is open and not yet committed/rolled back.
    pub fn in_transaction(&self) -> bool {
        self.in_transaction
    }

    /// The number of whole-tree scans the underlying store has issued since the
    /// last reset. A bulk write loop that keeps this flat is sub-quadratic; one
    /// that grows it per row is the O(rows^2) regression. Exposed for the
    /// scaling gate.
    pub fn full_scan_count(&self) -> u64 {
        self.store.full_scan_count()
    }

    /// Reset the underlying store's whole-tree-scan counter to zero.
    pub fn reset_full_scan_count(&self) {
        self.store.reset_full_scan_count();
    }

    /// The number of leaf cells the underlying store's streaming scans have
    /// visited since the last reset. An index-served query (early-stopping at
    /// LIMIT) keeps this near the LIMIT; a linear column-selective sweep drives
    /// it toward the corpus size. Exposed for the recall-cost gate.
    pub fn cells_visited_count(&self) -> u64 {
        self.store.cells_visited_count()
    }

    /// Reset the underlying store's cells-visited counter to zero.
    pub fn reset_cells_visited_count(&self) {
        self.store.reset_cells_visited_count();
    }

    /// Set `table`'s vec_label high-water mark to an absolute `value`, persisting
    /// it so the next auto-assigned label is `value + 1`. Used after a verbatim
    /// bulk copy that wrote explicit `vec_label` values, so a subsequent
    /// auto-insert cannot collide with a copied row. Monotonic: a value not
    /// greater than the current mark is a no-op.
    pub fn set_vec_label_hwm(&self, table: &str, value: i64) -> Result<()> {
        self.meta.set_vec_label_hwm(&self.store, table, value)
    }

    /// Build every catalog-indexed table's column index and atomically persist
    /// the whole set to the on-disk sidecar, stamped with each table's
    /// committed write-generation. Called by the maintenance/SLEEP cadence — a
    /// boot accelerator, never on the write-time critical path.
    ///
    /// Refuses inside an open outer transaction (persists only committed,
    /// untainted state). A read-only-mount connection never persists (returns `Ok`
    /// without touching the sidecar file): the daemon-down RO recall reader
    /// loads-not-rebuilds via `from_store` but must never write into a store
    /// directory it does not own.
    ///
    /// For each table with declared indexed columns and a live root, the
    /// entry is self-built (via `ensure_built`, using the table's own
    /// `col_caches` entry so an already-warm in-process map is reused rather
    /// than rescanned) before being stamped and serialized — the host never
    /// has to issue a representative read first. The stamp is the table's
    /// CURRENT committed write-generation (`meta.col_generation`, seeded from
    /// disk on first touch) plus its scan-free `count_cells()`. Serialization
    /// is CBOR (`ciborium`) to a `.tmp` sibling, then an atomic `fs::rename`
    /// onto the final sidecar path — a crash mid-save leaves the previous
    /// good sidecar intact (mirrors `_save_index_atomic`).
    pub fn persist_col_indexes(&mut self) -> Result<()> {
        if self.in_transaction {
            return Err(EngineError::parse(
                "persist_col_indexes: cannot persist while a transaction is open".to_string(),
            ));
        }
        if self.is_readonly_mount {
            return Ok(());
        }

        let mut tables: Vec<PersistedTableIndex> = Vec::new();
        // Collect the target table list before mutably borrowing `col_caches`
        // (disjoint-field discipline: `root_map`/`catalog` immutable, `col_caches`
        // mutable, never aliased in the same statement).
        let targets: Vec<(String, u32)> = self
            .root_map
            .iter()
            .map(|(t, r)| (t.clone(), *r))
            .collect();
        for (table, root) in targets {
            let cols = self.catalog_col_index_columns(&table);
            if cols.is_empty() {
                continue;
            }
            let col_names = match self.catalog.column_names(&table) {
                Ok(names) => names,
                Err(_) => continue,
            };
            let tree = self.store.tree(root);
            let col_snapshot = {
                let entry = self
                    .col_caches
                    .entry(table.clone())
                    .or_insert_with(|| ColIndex::new(cols.clone()));
                entry.ensure_built(&tree, &col_names)?;
                entry.clone()
            };
            // Build+snapshot the id→key index too, but only for a table that
            // actually has an `id` column (edges keys on its composite PK, not an
            // `id`). Reuse the already-warm `id_caches` entry so a fresh scan is
            // avoided when a prior lookup warmed it. Persisting it under the SAME
            // fingerprint means a warm boot loads the id map instead of paying the
            // first `WHERE id = ?` lookup's build scan.
            let id_snapshot = if col_names.iter().any(|c| c == "id") {
                let id_entry = self.id_caches.entry(table.clone()).or_default();
                id_entry.ensure_built(&tree, &col_names)?;
                id_entry.clone()
            } else {
                IdIndex::new()
            };
            // The ordered index is persisted only when the writer already
            // built one (it exists per ORDER BY usage, never force-built) —
            // an unbuilt default round-trips as "nothing to adopt".
            let ordered_snapshot = self
                .ordered_caches
                .get(&table)
                .filter(|o| o.is_built())
                .cloned()
                .unwrap_or_default();
            let generation = self.meta.col_generation(&self.store, &table)?;
            let row_count = tree.count_cells().map_err(open_err)?;
            tables.push(PersistedTableIndex {
                table,
                columns: cols,
                generation,
                row_count,
                index: col_snapshot,
                id_index: id_snapshot,
                ordered_index: ordered_snapshot,
            });
        }

        let envelope = PersistedColIndexes {
            format_version: COL_INDEX_FORMAT_VERSION,
            tables,
        };
        let sidecar = col_index_sidecar_path(self.store.path());
        let tmp = sidecar.with_extension("colindex.tmp");
        let mut bytes: Vec<u8> = Vec::new();
        ciborium::ser::into_writer(&envelope, &mut bytes).map_err(|e| {
            EngineError::parse(format!("persist_col_indexes: encode error: {e}"))
        })?;
        std::fs::write(&tmp, &bytes).map_err(|e| {
            EngineError::parse(format!("persist_col_indexes: write error: {e}"))
        })?;
        std::fs::rename(&tmp, &sidecar).map_err(|e| {
            EngineError::parse(format!("persist_col_indexes: rename error: {e}"))
        })?;
        Ok(())
    }

    /// The transaction scope a single statement runs under: its own store
    /// transaction when no outer transaction is open, else suppressed so the DML
    /// accumulates in the outer transaction.
    fn scope(&self) -> TxnScope {
        if self.in_transaction {
            TxnScope::Suppressed
        } else {
            TxnScope::Own
        }
    }

    /// Execute one SQL statement under positional `?` binds and return a cursor.
    pub fn execute(&mut self, sql: &str, params: Vec<Value>) -> Result<CursorState> {
        self.execute_inner(sql, params, None)
    }

    /// Execute one SQL statement under named `:name` binds and return a cursor.
    pub fn execute_named(
        &mut self,
        sql: &str,
        named: HashMap<String, Value>,
    ) -> Result<CursorState> {
        self.execute_inner(sql, Vec::new(), Some(named))
    }

    fn execute_inner(
        &mut self,
        sql: &str,
        params: Vec<Value>,
        named: Option<HashMap<String, Value>>,
    ) -> Result<CursorState> {
        let trimmed = sql.trim();
        let upper = trimmed.to_ascii_uppercase();

        // Re-validate the read-only snapshot fence for EVERY statement on a
        // read-only mount, unconditionally — not only on a page-cache miss.
        // A page already resident from an earlier read on this connection is
        // otherwise served straight from the pager's cache, silently bypassing
        // the fence and serving stale pre-checkpoint content instead of
        // reporting the invalidation. Cheap: a file-length stat plus a 32-byte
        // WAL-sidecar header read, no lock.
        if self.is_readonly_mount {
            self.store.pager().revalidate_ro_snapshot_fence().map_err(open_err)?;
        }

        // Read-only guard: reject writes on a read-only mount (immutable) or while
        // PRAGMA query_only=ON (mutable). query_only=OFF clears only the mutable
        // flag, never the mount's read-only state.
        if (self.is_readonly_mount || self.query_only) && is_write_sql(&upper) {
            return Err(EngineError::parse(READONLY_WRITE_ERR.to_string()));
        }

        // --- Transaction control ---
        if matches!(
            upper.as_str(),
            "BEGIN" | "BEGIN DEFERRED" | "BEGIN IMMEDIATE" | "BEGIN EXCLUSIVE"
        ) {
            if self.in_transaction {
                return Err(EngineError::parse(
                    "cannot start a transaction within a transaction".to_string(),
                ));
            }
            self.store.begin_write().map_err(open_err)?;
            self.in_transaction = true;
            self.txn_tainted = false;
            return Ok(CursorState::empty());
        }
        if upper == "COMMIT" {
            if !self.in_transaction {
                return Err(EngineError::parse(
                    "cannot commit - no transaction is active".to_string(),
                ));
            }
            self.store.commit().map_err(open_err)?;
            self.in_transaction = false;
            // A non-id-column UPDATE leaves the row at the same B-tree key under
            // the same id, so the id→key map (and the col/conflict caches) stay
            // valid across a successful, untainted commit.  Only clear when the
            // transaction was tainted — i.e. a mutating statement inside it
            // returned an error that the caller may have swallowed, leaving the
            // in-memory caches ahead of the durably committed tree.
            if self.txn_tainted {
                self.conflict_caches.clear();
                self.id_caches.clear();
                self.col_caches.clear();
                self.ordered_caches.clear();
                self.bare_count_cache.clear();
            }
            self.txn_tainted = false;
            return Ok(CursorState::empty());
        }
        if upper == "ROLLBACK" {
            if !self.in_transaction {
                return Err(EngineError::parse(
                    "cannot rollback - no transaction is active".to_string(),
                ));
            }
            // Capture the rollback result but clear the speculative caches
            // REGARDLESS of it: the conflict / id / col caches accumulated entries
            // for rows the rollback just reverted, and they must not survive even if
            // the store rollback itself errors. A surviving cache would leave stale
            // conflict-key → row-key (or id → row-key) entries pointing at reverted
            // rows — a false conflict / wrong-row overwrite on the next batch. The
            // caches are speculative, so dropping them on a failed rollback is
            // strictly safer (the next probe rebuilds from the committed tree). The
            // production UPSERT path drives its transaction with this SQL string
            // (its transaction wrapper issues ROLLBACK on failure).
            let result = self.store.rollback();
            self.in_transaction = false;
            self.txn_tainted = false;
            self.conflict_caches.clear();
            self.id_caches.clear();
            self.col_caches.clear();
            self.ordered_caches.clear();
            self.bare_count_cache.clear();
            result.map_err(open_err)?;
            return Ok(CursorState::empty());
        }

        // --- Maintenance statements: accepted as no-ops ---
        // VACUUM/ANALYZE/REINDEX are whole-file rewrite / statistics-rebuild
        // operations defined for the stdlib B-tree file format. This engine has
        // its own pager with an in-place freelist, so a SQLite-style rewrite has
        // no meaning here; the host's maintenance pass issues `VACUUM` and only
        // needs it not to error. Accept and ignore — the on-disk store is already
        // compacted incrementally by the pager.
        let maint_head = upper
            .trim_end_matches(';')
            .trim_end()
            .split_whitespace()
            .next()
            .unwrap_or("");
        if matches!(maint_head, "VACUUM" | "ANALYZE" | "REINDEX") {
            return Ok(CursorState::empty());
        }

        // --- PRAGMA handling (the six connection PRAGMAs + table_info) ---
        if upper.starts_with("PRAGMA ") {
            return self.execute_pragma(trimmed, &upper);
        }

        // --- DDL: CREATE TABLE / ALTER TABLE / DROP TABLE ---
        if upper.starts_with("CREATE TABLE") {
            self.run_create_table(trimmed)?;
            return Ok(CursorState::empty());
        }
        if upper.starts_with("CREATE INDEX") || upper.starts_with("CREATE UNIQUE INDEX") {
            // Recorded in the catalog; persisted as a DDL meta row. No scan path.
            self.run_ddl_passthrough(trimmed)?;
            return Ok(CursorState::empty());
        }
        if upper.starts_with("ALTER TABLE") || upper.starts_with("DROP TABLE") {
            self.run_ddl_passthrough(trimmed)?;
            return Ok(CursorState::empty());
        }

        // --- DML / SELECT through the parser + executor ---
        let stmt = self.parse_cached(trimmed)?;
        match stmt {
            Stmt::Select(select) => {
                let plan = plan_select(&select, &self.catalog, params, named)?;
                // Supply the per-table id-index so a `WHERE id = ?` point lookup
                // resolves O(log N) instead of a whole-table scan. Also supply the
                // ColIndex for any table that has declared single-column secondary
                // indexes (derived from the catalog), so `col IN (...)` and
                // `col = literal` lookups resolve without a full-table scan.
                // Both are lazily built on first use. The catalog-derived column
                // set is computed before any mutable borrow so the immutable
                // catalog reference does not alias the mutable cache fields.
                // Derive the catalog-driven indexed-column set before taking any
                // mutable borrow, so the immutable catalog borrow does not alias
                // the mutable cache borrows below (disjoint field access).
                let col_cols = self.catalog_col_index_columns(&plan.table);
                // Bare-COUNT(*) generation-fenced cache: this exact shape is
                // the host's per-query emptiness probe, and its uncached cost
                // is a whole leaf-chain walk. Cached only for col-indexed
                // tables (their committed write-generation is the airtight
                // fence) and never consulted-or-populated inside an open
                // transaction (an in-transaction count could otherwise be
                // rolled back yet later served for a re-reached generation).
                // The served row is constructed exactly as the executor's
                // bare-count fast path constructs it, so the result is
                // byte-identical.
                let bare_count_shape = plan.scan_type == ScanType::Full
                    && plan.aggregate.as_deref() == Some("count")
                    && plan.where_clause.is_none()
                    && plan.group_by_col.is_none()
                    && plan.order_by.is_none()
                    && plan.select_exprs.is_empty()
                    && !col_cols.is_empty();
                if bare_count_shape && !self.in_transaction {
                    if let (Some(&(cached_gen, cached_count)), Ok(current_gen)) = (
                        self.bare_count_cache.get(&plan.table),
                        self.meta.col_generation(&self.store, &plan.table),
                    ) {
                        if cached_gen == current_gen {
                            let label = plan
                                .count_alias
                                .clone()
                                .unwrap_or_else(|| "COUNT(*)".to_string());
                            let mut cur = CursorState::from_rows(
                                vec![label],
                                vec![vec![Value::Int(cached_count)]],
                            );
                            cur.rowcount = -1;
                            return Ok(cur);
                        }
                    }
                }
                // Derive the ordered-index column before mutable borrows (same
                // disjoint-field discipline).  The ordered target is the ORDER BY
                // column (only when a LIMIT makes an early-stopped walk usable) or
                // a MIN/MAX aggregate's column — both resolve to the same
                // catalog-declared index pick.
                let order_col = if plan.limit.is_some() || plan.limit_is_param {
                    plan.order_by.as_ref().map(|k| k.col.as_str())
                } else {
                    None
                };
                let minmax_col = match plan.aggregate.as_deref() {
                    Some("min") | Some("max") => plan.aggregate_column.as_deref(),
                    _ => None,
                };
                let ordered_col =
                    self.catalog_ordered_index_column(&plan.table, order_col.or(minmax_col));
                let id_ref = self.id_caches.entry(plan.table.clone()).or_default();
                let col_ref = if col_cols.is_empty() {
                    None
                } else {
                    Some(
                        self.col_caches
                            .entry(plan.table.clone())
                            .or_insert_with(|| ColIndex::new(col_cols)),
                    )
                };
                let ord_ref = ordered_col.map(|col| {
                    // Re-key the ordered cache to the ORDER BY column this query
                    // serves.  A prior query on a DIFFERENT ordered column would
                    // have cached an index over the wrong column; replace it so a
                    // `ts` ORDER BY always gets a `ts` ordered index regardless of
                    // any earlier non-`ts` ordered read on the same connection.
                    let entry = self
                        .ordered_caches
                        .entry(plan.table.clone())
                        .or_insert_with(|| OrderedColIndex::new(col.clone()));
                    if entry.column() != col {
                        *entry = OrderedColIndex::new(col);
                    }
                    entry
                });
                let rs = execute_select(
                    &plan, &self.store, &self.catalog, &self.root_map,
                    Some(id_ref), col_ref, ord_ref,
                )?;
                // Populate the bare-count cache from the freshly computed
                // result, keyed by the generation observed AFTER the walk (any
                // committed write between the two reads simply re-keys the
                // next probe to a miss — err toward recount, never staleness).
                if bare_count_shape && !self.in_transaction {
                    if let (Some(row), Ok(current_gen)) = (
                        rs.rows.first(),
                        self.meta.col_generation(&self.store, &plan.table),
                    ) {
                        if let Some((_, Value::Int(count))) = row.first() {
                            self.bare_count_cache
                                .insert(plan.table.clone(), (current_gen, *count));
                        }
                    }
                }
                // Read-only connections publish a freshly lazy-built index
                // pair to the process-wide cache so sibling readers at the
                // same write-generation install it instead of rebuilding.
                if self.is_readonly_mount && !self.in_transaction {
                    self.maybe_publish_built_indexes(&plan.table);
                }
                let mut cur = CursorState::from_rows(rs.columns, into_value_rows(rs.rows));
                cur.rowcount = -1;
                Ok(cur)
            }
            Stmt::Insert(insert) => {
                let table = insert.table.clone();
                let scope = self.scope();
                // The id-index, conflict-index, col-index, and ordered-index are
                // distinct fields, so the borrow checker permits simultaneous
                // mutable borrows (disjoint field access). Derive column sets
                // before any mutable borrow (same pattern as the SELECT path).
                let insert_cols = self.catalog_col_index_columns(&table);
                let result = {
                    let idx = self.id_caches.entry(table.clone()).or_default();
                    let cidx = self.conflict_caches.entry(table.clone()).or_default();
                    let colx = if insert_cols.is_empty() {
                        None
                    } else {
                        Some(
                            self.col_caches
                                .entry(table.clone())
                                .or_insert_with(|| ColIndex::new(insert_cols)),
                        )
                    };
                    // Maintain an already-warm ordered index for this table, but do
                    // not create a new entry here — the SELECT path creates entries
                    // keyed to the exact ORDER BY column, so write paths maintain
                    // whatever is already cached without risking a column mismatch.
                    let ordx = self.ordered_caches.get_mut(&table);
                    execute_insert(
                        &insert,
                        &self.store,
                        &self.meta,
                        &self.catalog,
                        &self.root_map,
                        params,
                        named,
                        scope,
                        Some(idx),
                        Some(cidx),
                        colx,
                        ordx,
                    )
                };
                match result {
                    Ok(outcome) => Ok(write_cursor(outcome)),
                    Err(e) => {
                        if self.in_transaction {
                            self.txn_tainted = true;
                        }
                        Err(e)
                    }
                }
            }
            Stmt::Update(update) => {
                let table = update.table.clone();
                let scope = self.scope();
                // Derive the catalog-driven indexed-column set before any mutable
                // borrow (same disjoint-field discipline as the SELECT/INSERT
                // paths), so a table WITH declared indexes always gets a col_caches
                // entry (creating an unbuilt one if absent) — `col_index.is_some()`
                // is the write-generation bump's signal for "this table has an
                // on-disk sidecar entry that may need invalidating", and that signal
                // must not depend on whether an earlier read happened to warm the
                // cache first. An unbuilt entry is a no-op for insert_row/remove_row,
                // so this changes no behavior beyond making the signal reliable.
                let update_cols = self.catalog_col_index_columns(&table);
                let result = {
                    let idx = self.id_caches.entry(table.clone()).or_default();
                    let cidx = self.conflict_caches.get_mut(&table);
                    let colx = if update_cols.is_empty() {
                        None
                    } else {
                        Some(
                            self.col_caches
                                .entry(table.clone())
                                .or_insert_with(|| ColIndex::new(update_cols)),
                        )
                    };
                    let ordx = self.ordered_caches.get_mut(&table);
                    execute_update(
                        &update,
                        &self.store,
                        &self.meta,
                        &self.catalog,
                        &self.root_map,
                        params,
                        named,
                        scope,
                        Some(idx),
                        cidx,
                        colx,
                        ordx,
                    )
                };
                match result {
                    Ok(outcome) => Ok(write_cursor(outcome)),
                    Err(e) => {
                        if self.in_transaction {
                            self.txn_tainted = true;
                        }
                        Err(e)
                    }
                }
            }
            Stmt::Delete(delete) => {
                let table = delete.table.clone();
                let scope = self.scope();
                // Same discipline as the UPDATE arm above: a table with declared
                // indexes always gets a col_caches entry so `col_index.is_some()`
                // reliably signals "this table may have a persisted sidecar entry
                // to invalidate," independent of whether the cache happens to be
                // warm yet.
                let delete_cols = self.catalog_col_index_columns(&table);
                let result = {
                    let idx = self.id_caches.entry(table.clone()).or_default();
                    let cidx = self.conflict_caches.get_mut(&table);
                    let colx = if delete_cols.is_empty() {
                        None
                    } else {
                        Some(
                            self.col_caches
                                .entry(table.clone())
                                .or_insert_with(|| ColIndex::new(delete_cols)),
                        )
                    };
                    let ordx = self.ordered_caches.get_mut(&table);
                    execute_delete(
                        &delete,
                        &self.store,
                        &self.meta,
                        &self.catalog,
                        &self.root_map,
                        params,
                        named,
                        scope,
                        Some(idx),
                        cidx,
                        colx,
                        ordx,
                    )
                };
                match result {
                    Ok(outcome) => Ok(write_cursor(outcome)),
                    Err(e) => {
                        if self.in_transaction {
                            self.txn_tainted = true;
                        }
                        Err(e)
                    }
                }
            }
            Stmt::Ddl(_) => {
                // A DDL/PRAGMA that fell through the prefix checks: apply to the
                // catalog so the in-memory schema stays consistent.
                self.catalog.apply_ddl(&stmt)?;
                Ok(CursorState::empty())
            }
        }
    }

    /// Execute `sql` once per parameter row, atomically.
    ///
    /// INSERTs route through the batch executor (one store transaction, inner
    /// per-statement transactions suppressed). Other statements fall back to a
    /// per-row execute under one outer transaction so the batch is still atomic.
    pub fn executemany(
        &mut self,
        sql: &str,
        seq: Vec<Vec<Value>>,
    ) -> Result<CursorState> {
        if seq.is_empty() {
            return Ok(CursorState::empty());
        }
        let trimmed = sql.trim();
        let upper = trimmed.to_ascii_uppercase();
        if self.query_only && is_write_sql(&upper) {
            return Err(EngineError::parse(READONLY_WRITE_ERR.to_string()));
        }

        // An INSERT batch with no outer transaction routes through the atomic
        // batch executor (one store transaction, inner per-statement
        // transactions suppressed). When an outer BEGIN is already open, the
        // batch executor's own begin_write would nest illegally — so each row
        // runs as a single suppressed-scope INSERT inside the open transaction.
        if upper.starts_with("INSERT") && !self.in_transaction {
            let stmt = self.parse_cached(trimmed)?;
            let Stmt::Insert(insert) = stmt else {
                return Err(EngineError::parse(
                    "executemany: expected an INSERT statement".to_string(),
                ));
            };
            let table = insert.table.clone();
            // Drop any built id-index and col-index for this table before the
            // batch: the batch executor rolls back internally on a mid-batch error,
            // so rather than unwind per-row index mutations we invalidate up front
            // and let the next read rebuild from the committed tree. The bulk corpus
            // load runs before any recall, so each index rebuilds exactly once at
            // first recall, not per insert — and dropping up front keeps the
            // rollback-corruption invariant (a reverted batch leaves no stale
            // entries) without per-row unwind bookkeeping.
            self.id_caches.remove(&table);
            self.col_caches.remove(&table);
            self.ordered_caches.remove(&table);
            self.bare_count_cache.remove(&table);
            let outcome = {
                let cidx = self.conflict_caches.entry(table.clone()).or_default();
                execute_insert_many(
                    &insert,
                    &self.store,
                    &self.meta,
                    &self.catalog,
                    &self.root_map,
                    seq,
                    None,
                    Some(cidx),
                    None,
                    None,
                )?
            };
            return Ok(write_cursor(outcome));
        }

        // Per-row batch: wrap in one outer transaction (unless one is already
        // open), run each row as a suppressed-scope statement.
        let opened = !self.in_transaction;
        if opened {
            self.store.begin_write().map_err(open_err)?;
            self.in_transaction = true;
        }
        let mut total: i64 = 0;
        for params in seq {
            match self.execute_inner(trimmed, params, None) {
                Ok(cur) => {
                    if cur.rowcount > 0 {
                        total += cur.rowcount;
                    }
                }
                Err(e) => {
                    if opened {
                        let _ = self.store.rollback();
                        self.in_transaction = false;
                        self.txn_tainted = false;
                        // Per-row index mutations under the rolled-back batch are
                        // now stale; drop them so the next lookup rebuilds from the tree.
                        self.conflict_caches.clear();
                        self.id_caches.clear();
                        self.col_caches.clear();
                        self.ordered_caches.clear();
                        self.bare_count_cache.clear();
                    } else {
                        // Inside a caller-owned outer transaction: mark it tainted
                        // so that if the caller commits, caches are cleared.
                        self.txn_tainted = true;
                    }
                    return Err(e);
                }
            }
        }
        if opened {
            self.store.commit().map_err(open_err)?;
            self.in_transaction = false;
            // Clear caches only when tainted — a clean (untainted) batch commit
            // leaves the id→key map valid (no reverted-row entries).
            if self.txn_tainted {
                self.conflict_caches.clear();
                self.id_caches.clear();
                self.col_caches.clear();
                self.ordered_caches.clear();
                self.bare_count_cache.clear();
            }
            self.txn_tainted = false;
        }
        let mut cur = CursorState::empty();
        cur.rowcount = total;
        Ok(cur)
    }

    /// Commit the open write transaction, if any.
    pub fn commit(&mut self) -> Result<()> {
        if self.in_transaction {
            self.store.commit().map_err(open_err)?;
            self.in_transaction = false;
        }
        // Clear caches only when the transaction was tainted by a swallowed inner
        // DML error (the speculative-entry hazard the original blanket clear
        // defended).  A clean commit leaves the id→key map valid.
        if self.txn_tainted {
            self.conflict_caches.clear();
            self.id_caches.clear();
            self.col_caches.clear();
            self.ordered_caches.clear();
            self.bare_count_cache.clear();
        }
        self.txn_tainted = false;
        Ok(())
    }

    /// Roll back the open write transaction, if any.
    pub fn rollback(&mut self) -> Result<()> {
        // Clear speculative caches UNCONDITIONALLY — reverted rows must never
        // survive in the cache regardless of error state or taint.  Clear even
        // if the store rollback itself errors (clearing on a failed rollback is
        // strictly safer than leaving stale reverted-row entries).
        let result = if self.in_transaction {
            let r = self.store.rollback();
            self.in_transaction = false;
            r
        } else {
            Ok(())
        };
        self.txn_tainted = false;
        self.conflict_caches.clear();
        self.id_caches.clear();
        self.col_caches.clear();
        self.ordered_caches.clear();
        self.bare_count_cache.clear();
        result.map_err(open_err)
    }

    /// True while the backing store file handle is still held. Goes `false` once
    /// the file/lock has been released; a released connection serves no further
    /// I/O. The pyo3 surface reads this to skip handing a released connection
    /// back out of the borrow registry.
    pub fn is_open(&self) -> bool {
        self.store.is_open()
    }

    /// The always-safe part of close: discard an uncommitted transaction (roll
    /// it back, never silently commit) and flush committed data durably. Does
    /// NOT touch the file handle or the advisory lock, so it is safe to run while
    /// other holders of the shared connection are still live. Idempotent.
    pub fn flush_only(&mut self) -> Result<()> {
        if self.in_transaction {
            let _ = self.store.rollback();
            self.in_transaction = false;
        }
        self.store.flush().map_err(open_err)?;
        Ok(())
    }

    /// Release the store file and its advisory single-writer lock. Run ONLY by
    /// the last holder of the shared connection — releasing it while a borrower,
    /// cursor, or raw adapter is still live would tear the file out from under
    /// them. Idempotent: a second release on an already-released file is a no-op
    /// (the underlying file guard is already cleared).
    pub fn release_file(&mut self) -> Result<()> {
        self.store.unlock().map_err(open_err)?;
        Ok(())
    }

    /// Close the connection: flush committed data, discard any open transaction,
    /// then release the file and advisory lock. Matches stdlib `close()` — an
    /// uncommitted transaction is rolled back, never silently committed. The
    /// pyo3 surface drives the two halves separately so the file/lock release is
    /// deferred to the last holder; this combined form is the single-holder path.
    pub fn close(&mut self) -> Result<()> {
        self.flush_only()?;
        // Release the store file lock so a fresh open on the same path in the
        // same process (the migrate -> verify -> recall-probe sequence) does not
        // block on a lock this closed handle would otherwise still hold until GC.
        self.release_file()?;
        Ok(())
    }

    /// PRAGMA dispatch: the six connection PRAGMAs the host issues at open time
    /// are accepted (with their real side effects where one exists), and
    /// `PRAGMA table_info(t)` is synthesized from the catalog.
    fn execute_pragma(&mut self, raw: &str, upper: &str) -> Result<CursorState> {
        // PRAGMA journal_mode[=WAL]: switch to write-ahead mode on assignment.
        if upper.starts_with("PRAGMA JOURNAL_MODE") {
            if upper.contains('=') {
                let rhs = pragma_rhs(upper);
                if rhs == "WAL" {
                    self.store.enable_wal_mode().map_err(open_err)?;
                }
            }
            return Ok(CursorState::from_rows(
                vec!["journal_mode".to_string()],
                vec![vec![Value::Text("wal".to_string())]],
            ));
        }
        // PRAGMA foreign_keys (read query): report enabled.
        if upper == "PRAGMA FOREIGN_KEYS" {
            return Ok(CursorState::from_rows(
                vec!["foreign_keys".to_string()],
                vec![vec![Value::Int(1)]],
            ));
        }
        // PRAGMA cache_size=<n>: bound the pager's resident page cache. A
        // negative value is KiB of memory, a positive value is a page count.
        if upper.starts_with("PRAGMA CACHE_SIZE") {
            if upper.contains('=') {
                let rhs = pragma_rhs(upper);
                if let Ok(value) = rhs.parse::<i64>() {
                    let pages = if value < 0 {
                        // Negative cache_size is a byte budget in KiB; convert to
                        // a page count at the engine's real page size so the
                        // configured memory budget is honoured (a previous /4096
                        // assumed a 4 KiB page and silently doubled the budget at
                        // the actual 8 KiB page size).
                        let page_size = lillibrain::consts::PAGE_SIZE as u64;
                        Some(((value.unsigned_abs() * 1024) / page_size).max(1) as usize)
                    } else if value > 0 {
                        Some(value as usize)
                    } else {
                        None
                    };
                    self.store.set_cache_pages(pages);
                }
            }
            return Ok(CursorState::empty());
        }
        // PRAGMA wal_checkpoint: merge committed WAL frames into the main file.
        if upper.starts_with("PRAGMA WAL_CHECKPOINT") {
            self.store.checkpoint().map_err(open_err)?;
            return Ok(CursorState::empty());
        }
        // PRAGMA query_only[=ON|OFF]: flip the read-only guard.
        if upper.starts_with("PRAGMA QUERY_ONLY") {
            if upper.contains('=') {
                let rhs = pragma_rhs(upper);
                self.query_only = matches!(rhs.as_str(), "ON" | "1" | "TRUE");
                return Ok(CursorState::empty());
            }
            return Ok(CursorState::from_rows(
                vec!["query_only".to_string()],
                vec![vec![Value::Int(i64::from(self.query_only))]],
            ));
        }
        // PRAGMA index_list(t): synthesize from the catalog, sqlite3-shaped.
        if upper.starts_with("PRAGMA INDEX_LIST") {
            let table = pragma_paren_arg(raw);
            let rows = self
                .catalog
                .index_list(&table)
                .into_iter()
                .map(|(seq, name, unique, origin, partial)| {
                    vec![
                        Value::Int(seq),
                        Value::Text(name),
                        Value::Int(i64::from(unique)),
                        Value::Text(origin),
                        Value::Int(i64::from(partial)),
                    ]
                })
                .collect();
            return Ok(CursorState::from_rows(
                vec![
                    "seq".to_string(),
                    "name".to_string(),
                    "unique".to_string(),
                    "origin".to_string(),
                    "partial".to_string(),
                ],
                rows,
            ));
        }
        // PRAGMA table_info(t): synthesize from the catalog.
        if upper.starts_with("PRAGMA TABLE_INFO") {
            let table = pragma_paren_arg(raw);
            let rows = self
                .catalog
                .table_info(&table)
                .unwrap_or_default()
                .into_iter()
                .map(|(cid, name, type_, notnull, dflt, pk)| {
                    vec![
                        Value::Int(cid),
                        Value::Text(name),
                        Value::Text(type_),
                        Value::Int(notnull),
                        dflt,
                        Value::Int(pk),
                    ]
                })
                .collect();
            return Ok(CursorState::from_rows(
                vec![
                    "cid".to_string(),
                    "name".to_string(),
                    "type".to_string(),
                    "notnull".to_string(),
                    "dflt_value".to_string(),
                    "pk".to_string(),
                ],
                rows,
            ));
        }
        // PRAGMA busy_timeout[=N]: round-trip the value. An assignment stores N
        // and returns it; a read returns the current value. The host sets this at
        // open time and reads it back to confirm the configured lock-wait window.
        if upper.starts_with("PRAGMA BUSY_TIMEOUT") {
            if upper.contains('=') {
                let rhs = pragma_rhs(upper);
                if let Ok(value) = rhs.parse::<i64>() {
                    self.busy_timeout = value;
                }
            }
            return Ok(CursorState::from_rows(
                vec!["busy_timeout".to_string()],
                vec![vec![Value::Int(self.busy_timeout)]],
            ));
        }
        // PRAGMA synchronous and any other tuning PRAGMA: accept as a no-op.
        // These carry no engine-meaningful side effect here.
        Ok(CursorState::empty())
    }

    /// Register a new table: allocate its root, persist the DDL + root mapping,
    /// apply it to the live catalog. Idempotent per table name. Under an outer
    /// transaction the meta writes are suppressed so they land in that
    /// transaction (a nested store `begin_write` would be rejected).
    fn run_create_table(&mut self, ddl: &str) -> Result<()> {
        let scope = self.scope();
        self.meta.create_table_scoped(
            &self.store,
            &mut self.catalog,
            &mut self.root_map,
            ddl,
            scope,
        )?;
        Ok(())
    }

    /// Apply a non-CREATE-TABLE DDL (ALTER / DROP / CREATE INDEX) to the live
    /// catalog and persist it as a DDL meta row so a reopen replays it. A RENAME
    /// re-keys the `root_map`; a DROP frees the mapping.
    fn run_ddl_passthrough(&mut self, ddl: &str) -> Result<()> {
        let stmt = parse(ddl)?;
        let Stmt::Ddl(_) = &stmt else {
            return Err(EngineError::parse(format!(
                "expected a DDL statement, got: {ddl:?}"
            )));
        };

        // Capture a RENAME's old/new names before the catalog re-keys.
        let rename = rename_names(ddl);
        let drop_name = drop_table_name(ddl);
        let scope = self.scope();

        self.catalog.apply_ddl(&stmt)?;
        self.meta.record_ddl(&self.store, ddl, scope)?;

        if let Some((old, new)) = rename {
            if let Some(root) = self.root_map.remove(&old) {
                self.root_map.insert(new.clone(), root);
                self.meta.record_root(&self.store, &new, root, scope)?;
            }
            // Drop the per-table caches keyed under the old name.
            self.id_caches.remove(&old);
            self.col_caches.remove(&old);
            self.conflict_caches.remove(&old);
            self.ordered_caches.remove(&old);
            self.bare_count_cache.remove(&old);
        }
        if let Some(name) = drop_name {
            self.root_map.remove(&name);
            self.id_caches.remove(&name);
            self.col_caches.remove(&name);
            self.conflict_caches.remove(&name);
            self.ordered_caches.remove(&name);
            self.bare_count_cache.remove(&name);
        }
        Ok(())
    }

}

/// The read-only auto-commit adapter's pre-delegation decision.
///
/// A [`RawConn`] adapter mirrors the reference's `LilliBrainRawConn`: it
/// intercepts `PRAGMA query_only` at its own layer (never touching the shared
/// connection's `query_only` state) and blocks write SQL when read-only. The
/// host wrapper owns the shared [`Connection`] handle and the delegation; this
/// type holds only the adapter's read-only flag and the gate logic, so the
/// blocking/intercept policy lives in one place and stays testable without a
/// live store.
pub struct RawConn {
    read_only: bool,
}

/// What a [`RawConn`] does with a statement before (or instead of) delegating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RawAction {
    /// Delegate the (trimmed) statement to the backing connection's `execute`.
    Delegate(String),
    /// Answer at the adapter with an empty cursor (e.g. `PRAGMA query_only=ON`).
    Empty,
    /// Answer at the adapter with the current `query_only` value as a 1-row set.
    ReportQueryOnly(bool),
    /// Reject the statement as a write against a read-only adapter.
    Reject,
}

impl RawConn {
    /// A new raw adapter with the given read-only constraint.
    pub fn new(read_only: bool) -> RawConn {
        RawConn { read_only }
    }

    /// True when the adapter enforces read-only access.
    pub fn read_only(&self) -> bool {
        self.read_only
    }

    /// Decide what to do with `sql`. `PRAGMA query_only` is intercepted here (it
    /// flips this adapter's own flag, never the shared connection's); write SQL
    /// is rejected when the adapter is read-only; everything else delegates.
    pub fn gate(&mut self, sql: &str) -> RawAction {
        let trimmed = sql.trim();
        let upper = trimmed.to_ascii_uppercase();
        if upper.starts_with("PRAGMA QUERY_ONLY") {
            if upper.contains('=') {
                let rhs = pragma_rhs(&upper);
                self.read_only = matches!(rhs.as_str(), "ON" | "1" | "TRUE");
                return RawAction::Empty;
            }
            return RawAction::ReportQueryOnly(self.read_only);
        }
        if self.read_only && is_write_sql(&upper) {
            return RawAction::Reject;
        }
        RawAction::Delegate(trimmed.to_string())
    }
}

/// Materialize a write outcome onto a cursor: an empty result set carrying the
/// injected `lastrowid` and the affected `rowcount`.
fn write_cursor(outcome: WriteOutcome) -> CursorState {
    let mut cur = CursorState::empty();
    cur.lastrowid = outcome.lastrowid;
    cur.rowcount = outcome.rowcount;
    cur
}

/// Convert executor output rows (ordered `(name, value)` pairs) into bare value
/// rows in column order.
fn into_value_rows(rows: Vec<crate::exec::OutRow>) -> Vec<Vec<Value>> {
    rows.into_iter()
        .map(|r| r.into_iter().map(|(_, v)| v).collect())
        .collect()
}

/// True when an upper-cased SQL string begins a mutating statement.
fn is_write_sql(upper: &str) -> bool {
    upper.starts_with("INSERT ")
        || upper.starts_with("UPDATE ")
        || upper.starts_with("DELETE ")
        || upper.starts_with("CREATE ")
        || upper.starts_with("ALTER ")
        || upper.starts_with("DROP ")
        || upper == "INSERT"
        || upper == "UPDATE"
        || upper == "DELETE"
}

/// The right-hand side of a `PRAGMA name=value` after the `=`, upper-cased and
/// stripped of a trailing `;` and surrounding whitespace.
fn pragma_rhs(upper: &str) -> String {
    upper
        .split_once('=')
        .map(|(_, rhs)| rhs.trim().trim_end_matches(';').trim().to_string())
        .unwrap_or_default()
}

/// The argument inside `PRAGMA name(arg)`, with quotes/brackets stripped.
fn pragma_paren_arg(raw: &str) -> String {
    if let (Some(open), Some(close)) = (raw.find('('), raw.rfind(')')) {
        if close > open {
            return raw[open + 1..close]
                .trim()
                .trim_matches(|c| c == '"' || c == '\'' || c == '[' || c == ']')
                .to_string();
        }
    }
    String::new()
}

/// Extract `(old, new)` from an `ALTER TABLE <old> RENAME TO <new>` string.
fn rename_names(ddl: &str) -> Option<(String, String)> {
    let toks: Vec<String> = ddl.split_whitespace().map(|s| s.to_string()).collect();
    let upper: Vec<String> = toks.iter().map(|s| s.to_ascii_uppercase()).collect();
    if upper.first().map(String::as_str) != Some("ALTER") {
        return None;
    }
    let rename_pos = upper.iter().position(|t| t == "RENAME")?;
    if upper.get(rename_pos + 1).map(String::as_str) != Some("TO") {
        return None;
    }
    let old = toks.get(2)?.trim_matches(|c| c == '[' || c == ']' || c == '"').to_string();
    let new = toks
        .get(rename_pos + 2)?
        .trim_matches(|c| c == '[' || c == ']' || c == '"' || c == ';')
        .to_string();
    Some((old, new))
}

/// Extract the table name from a `DROP TABLE [IF EXISTS] <name>` string.
fn drop_table_name(ddl: &str) -> Option<String> {
    let toks: Vec<String> = ddl.split_whitespace().map(|s| s.to_string()).collect();
    let upper: Vec<String> = toks.iter().map(|s| s.to_ascii_uppercase()).collect();
    if upper.first().map(String::as_str) != Some("DROP")
        || upper.get(1).map(String::as_str) != Some("TABLE")
    {
        return None;
    }
    let mut i = 2;
    if upper.get(i).map(String::as_str) == Some("IF")
        && upper.get(i + 1).map(String::as_str) == Some("EXISTS")
    {
        i += 2;
    }
    toks.get(i)
        .map(|s| s.trim_matches(|c| c == '[' || c == ']' || c == '"' || c == ';').to_string())
}

/// Adapt a storage-layer error into the engine error taxonomy.
fn open_err(e: lillibrain::StoreError) -> EngineError {
    EngineError::parse(format!("engine connection: storage error: {e}"))
}

/// The columns to hash-index for `table`, derived from `catalog`'s recorded
/// `CREATE INDEX` declarations.
///
/// Every single-column non-partial declared index contributes its column.
/// Partial indexes are skipped, except `embedding_pending`: its partial
/// predicate (`WHERE embedding_pending=1`) is satisfied implicitly by an
/// equality probe (`col = 1`), so it is retained as a plain equality index
/// matching the previous hardcoded behavior. Any other partial-flagged column
/// is omitted; queries on those columns fall through to a full scan (correct,
/// no stale rows).
///
/// A free function (not a `Connection` method) so both `from_store` — which
/// runs before the `Connection` struct is constructed — and
/// `Connection::persist_col_indexes` can derive the same set. Returns an empty
/// `Vec` for tables with no declared indexes.
fn catalog_col_index_columns(catalog: &Catalog, table: &str) -> Vec<String> {
    catalog
        .indexed_columns(table)
        .into_iter()
        .filter_map(|(col, is_partial)| {
            if !is_partial || col == "embedding_pending" {
                Some(col)
            } else {
                None
            }
        })
        .collect()
}

/// Order-insensitive equality over two indexed-columns sets (a DDL-drift check:
/// the persisted entry's columns must equal the CURRENT catalog-derived set).
fn same_column_set(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort();
    b_sorted.sort();
    a_sorted == b_sorted
}

/// The on-disk sidecar path for a store opened at `store_path`: a sibling file
/// (`records.colindex` / `.tmp`) beside the store's own file, mirroring where
/// `records.hnsw` lives alongside `brain.sqlite3` in the host storage layer.
fn col_index_sidecar_path(store_path: &std::path::Path) -> std::path::PathBuf {
    let parent = store_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    parent.join(COL_INDEX_SIDECAR_NAME)
}

/// Attempt to load the on-disk column-index sidecar and gate each table's
/// entry against its CURRENT committed fingerprint.
///
/// On ANY failure to read/decode the sidecar file, or a `format_version`
/// mismatch, returns an empty map — unchanged (bypass-safe) behavior: the
/// caller's `col_caches` stays `HashMap::new()` and the existing lazy
/// `ensure_built` path serves every table. Per table entry, ALL of the
/// following must hold or that table's entry is skipped (never the whole
/// file): the table has a root in `root_map`; the persisted `columns` set
/// equals the table's CURRENT catalog-derived indexed-columns set
/// (order-insensitive — a DDL-drift detector); the table's CURRENT committed
/// write-generation equals the persisted `generation` (the airtight staleness
/// fence — catches every committed DML including a count-preserving in-place
/// UPDATE); and the table's CURRENT `count_cells()` equals the persisted
/// `row_count` (a scan-free defense-in-depth cross-check). The load itself
/// issues only generation reads and `count_cells()` — never a `full_scan()`.
fn load_col_indexes(
    store: &Store,
    meta: &MetaTable,
    catalog: &Catalog,
    root_map: &BTreeMap<String, u32>,
) -> (
    HashMap<String, ColIndex>,
    HashMap<String, IdIndex>,
    HashMap<String, OrderedColIndex>,
) {
    let empty = || (HashMap::new(), HashMap::new(), HashMap::new());
    let sidecar = col_index_sidecar_path(store.path());
    let Some(envelope) = cached_sidecar_envelope(&sidecar) else {
        return empty();
    };
    if envelope.format_version != COL_INDEX_FORMAT_VERSION {
        return empty();
    }
    let mut loaded: HashMap<String, ColIndex> = HashMap::new();
    let mut loaded_ids: HashMap<String, IdIndex> = HashMap::new();
    let mut loaded_ordered: HashMap<String, OrderedColIndex> = HashMap::new();
    for entry in &envelope.tables {
        let Some(&root) = root_map.get(&entry.table) else {
            continue;
        };
        let current_columns = catalog_col_index_columns(catalog, &entry.table);
        if !same_column_set(&current_columns, &entry.columns) {
            continue;
        }
        let current_generation = match meta.col_generation(store, &entry.table) {
            Ok(g) => g,
            Err(_) => continue,
        };
        if current_generation != entry.generation {
            continue;
        }
        let current_count = match gated_count_cells(store, root, &entry.table, current_generation)
        {
            Ok(c) => c,
            Err(_) => continue,
        };
        if current_count != entry.row_count {
            continue;
        }
        // The id-index rides the SAME fingerprint gate: only a built one (a table
        // with an `id` column that was populated at persist time) is admitted, so
        // a warm boot serves `WHERE id = ?` from the loaded map with no build scan.
        // An unbuilt placeholder (id-less table) is skipped — that table's lazy
        // path is unaffected.
        if entry.id_index.is_built() {
            loaded_ids.insert(entry.table.clone(), entry.id_index.clone());
        }
        // The ordered index rides the SAME fingerprint gate; only a built one
        // (the writer had run an ordered query on the table) is admitted, so a
        // cold process serves its first ORDER BY from the sidecar scan-free.
        if entry.ordered_index.is_built() {
            loaded_ordered.insert(entry.table.clone(), entry.ordered_index.clone());
        }
        loaded.insert(entry.table.clone(), entry.index.clone());
    }
    (loaded, loaded_ids, loaded_ordered)
}

/// How many distinct sidecar paths the process-wide decoded-envelope cache
/// retains before it is dropped wholesale (the same crude-but-bounded policy
/// the parse cache uses). A long-lived daemon touches one store; a test run
/// touches thousands of tmp stores — the cap keeps the latter from pinning
/// thousands of decoded envelopes in resident memory.
const SIDECAR_CACHE_CAP: usize = 4;

/// Entry cap for the load-gate row-count memo (small keys; a generous cap).
const GATE_COUNT_MEMO_CAP: usize = 64;

/// Entry cap for the built-index cache (each value is Arc-backed, so the
/// entries are cheap references into shared postings, but the cap still
/// bounds how many generations/tables a churny test run can pin).
const BUILT_INDEX_CACHE_CAP: usize = 32;

/// Process-wide cache of LAZILY-BUILT read-only index pairs, keyed by
/// `(store path, table, committed write-generation)`.
///
/// When the persisted sidecar's fingerprint no longer matches (any committed
/// write after the last persist), every fresh read-only connection — each
/// reader-pool slot, recycled per staleness generation — used to pay its own
/// whole-table lazy index build on the first indexed query, ON the awake
/// read path. The write-generation is the same airtight fence the sidecar
/// gate trusts, so an index built against a snapshot at generation G is
/// byte-exact for every other read-only snapshot at G: the FIRST reader to
/// build publishes here, and every later same-generation open installs the
/// shared (copy-on-write) pair instead of rebuilding. Read-only connections
/// never mutate their indexes, so the shared postings are never copied.
///
/// Publication happens only from read-only connections (never mid-write,
/// never inside a transaction), so an uncommitted state can never be cached.
static BUILT_INDEX_CACHE: std::sync::OnceLock<
    std::sync::Mutex<HashMap<(std::path::PathBuf, String, i64), (ColIndex, IdIndex)>>,
> = std::sync::OnceLock::new();

fn built_index_cache(
) -> &'static std::sync::Mutex<HashMap<(std::path::PathBuf, String, i64), (ColIndex, IdIndex)>> {
    BUILT_INDEX_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Process-wide memo for the load gate's row-count cross-check, keyed by
/// `(store path, table, committed write-generation)`.
///
/// `count_cells()` walks the whole leaf sibling chain, which on a
/// production-scale store costs hundreds of milliseconds per open — and every
/// read-only open (each reader-pool slot, each snapshot-scan connection)
/// re-ran it for every sidecar table during the boot window, serializing
/// foreground reads behind it. The write-generation is the airtight staleness
/// fence (every committed DML on a col-indexed table bumps it), so a row count
/// observed at generation G is valid for as long as the table's generation
/// still reads G: the memo re-keys itself out of existence on any committed
/// write. The count remains exactly the same defense-in-depth cross-check —
/// only its recomputation frequency changes from per-open to per-generation.
static GATE_COUNT_MEMO: std::sync::OnceLock<
    std::sync::Mutex<HashMap<(std::path::PathBuf, String, i64), u64>>,
> = std::sync::OnceLock::new();

/// The row count for the load gate: memo hit on `(path, table, generation)`,
/// else one real `count_cells()` walk whose result is memoized.
fn gated_count_cells(
    store: &Store,
    root: u32,
    table: &str,
    generation: i64,
) -> lillibrain::Result<u64> {
    let memo = GATE_COUNT_MEMO.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let key = (store.path().to_path_buf(), table.to_string(), generation);
    if let Ok(guard) = memo.lock() {
        if let Some(&count) = guard.get(&key) {
            return Ok(count);
        }
    }
    let count = store.tree(root).count_cells()?;
    if let Ok(mut guard) = memo.lock() {
        if guard.len() >= GATE_COUNT_MEMO_CAP && !guard.contains_key(&key) {
            guard.clear();
        }
        guard.insert(key, count);
    }
    Ok(count)
}

/// One cached decoded sidecar envelope, keyed by the file identity observed at
/// decode time. The persist path replaces the sidecar atomically (tmp +
/// rename), so a changed `(len, mtime)` pair is the staleness signal that
/// forces a re-decode.
struct SidecarCacheEntry {
    len: u64,
    mtime: std::time::SystemTime,
    envelope: std::sync::Arc<PersistedColIndexes>,
}

/// The process-wide decoded-envelope cache.
///
/// Decoding a production-scale sidecar is the dominant cost of an engine open
/// (hundreds of milliseconds for a tens-of-MB envelope). Without this cache,
/// EVERY read-only open — each reader-pool slot, each snapshot-scan
/// connection — re-reads and re-decodes the same bytes, and a concurrent
/// foreground read serializes behind that decode during the boot window. One
/// decode per sidecar identity per process keeps index loading an amortized
/// maintenance cost instead of a per-connection awake-path tax. Every open
/// still runs the full per-table fingerprint gate against ITS OWN store view;
/// only the byte-decode is shared.
static SIDECAR_DECODE_CACHE: std::sync::OnceLock<
    std::sync::Mutex<HashMap<std::path::PathBuf, SidecarCacheEntry>>,
> = std::sync::OnceLock::new();

/// The decoded sidecar envelope for `sidecar`, served from the process-wide
/// cache when the file identity `(len, mtime)` matches, else freshly read and
/// decoded (outside the cache lock, so a concurrent open never waits behind a
/// long decode; a racing duplicate decode is benign — last insert wins).
/// Returns `None` when the file is absent, unreadable, or undecodable —
/// exactly the cases the uncached load treated as an absent sidecar.
fn cached_sidecar_envelope(
    sidecar: &std::path::Path,
) -> Option<std::sync::Arc<PersistedColIndexes>> {
    let meta = std::fs::metadata(sidecar).ok()?;
    let len = meta.len();
    let mtime = meta.modified().ok()?;
    let cache = SIDECAR_DECODE_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()));

    if let Ok(guard) = cache.lock() {
        if let Some(entry) = guard.get(sidecar) {
            if entry.len == len && entry.mtime == mtime {
                return Some(std::sync::Arc::clone(&entry.envelope));
            }
        }
    }

    // Miss or stale: read + decode with no lock held.
    let bytes = std::fs::read(sidecar).ok()?;
    let envelope: PersistedColIndexes = ciborium::de::from_reader(bytes.as_slice()).ok()?;
    let envelope = std::sync::Arc::new(envelope);

    if let Ok(mut guard) = cache.lock() {
        if guard.len() >= SIDECAR_CACHE_CAP && !guard.contains_key(sidecar) {
            guard.clear();
        }
        guard.insert(
            sidecar.to_path_buf(),
            SidecarCacheEntry {
                len,
                mtime,
                envelope: std::sync::Arc::clone(&envelope),
            },
        );
    }
    Some(envelope)
}

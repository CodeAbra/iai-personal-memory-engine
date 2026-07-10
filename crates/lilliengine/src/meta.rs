//! The durable schema meta-table, pinned at the reserved page-3 B-tree root.
//!
//! The storage layer reserves page 3 at file creation; this module owns its
//! content. It records, in insertion order, four row kinds keyed by a
//! monotonic integer:
//!
//!   `("ddl",    seq,        ddl_text)`  — the CREATE/ALTER text, replayed on open
//!   `("root",   table_name, root_page)` — a table → B-tree root page mapping
//!   `("hwm",    table_name, value)`     — the per-table vec_label high-water mark
//!   `("colgen", table_name, value)`     — the per-table committed write-generation
//!
//! On reopen, [`MetaTable::replay`] reads the page-3 tree, applies every DDL row
//! through [`Catalog::apply_ddl`] in sequence order, and rebuilds the
//! `table → root_page` map. The persisted high-water marks let a reopened store
//! resume AUTOINCREMENT label assignment without scanning the data trees. The
//! persisted write-generation lets a reopened store decide whether an on-disk
//! column-index sidecar still matches the table's committed content without a
//! scan (see [`MetaTable::col_generation`]).
//!
//! This is the engine's own meta layout — it is written and read by this engine
//! only. It is not a byte-compatible reader of any externally produced meta
//! page; a store created elsewhere is migrated, not reopened in place.

use std::cell::RefCell;
use std::collections::BTreeMap;

use lillibrain::btree::cursor::init_root_leaf;
use lillibrain::btree::page::page_kind;
use lillibrain::{decode_record, encode_record, Store, Value};

use crate::catalog::Catalog;
use crate::error::{EngineError, Result};
use crate::exec::TxnScope;
use crate::lexer::{tokenize, Token};

/// The pinned meta-table root page. One past the data root reservation, so it is
/// findable with no catalog and no prior allocation.
pub const META_TABLE_PAGE: u32 = 3;

/// A handle to the durable schema meta-table.
///
/// Carries the in-memory mirror of each table's vec_label high-water mark and
/// the meta-tree key of its `hwm` row, so a label bump overwrites the existing
/// row in place rather than appending a new one each time. Carries the same
/// shape for the per-table write-generation counter (`colgen`/`colgen_key`).
pub struct MetaTable {
    hwm: RefCell<BTreeMap<String, i64>>,
    hwm_key: RefCell<BTreeMap<String, i64>>,
    colgen: RefCell<BTreeMap<String, i64>>,
    colgen_key: RefCell<BTreeMap<String, i64>>,
}

impl MetaTable {
    /// Open the meta-table over `store`, initializing page 3 as an empty leaf on
    /// a fresh file (where it is still a zeroed reservation).
    pub fn open(store: &Store) -> Result<MetaTable> {
        let pager = store.pager();
        let page = pager.read_page(META_TABLE_PAGE).map_err(store_err)?;
        if page_kind(&page).is_err() {
            // Fresh reservation (a zeroed page): format it as an empty leaf.
            store.begin_write().map_err(store_err)?;
            init_root_leaf(pager, META_TABLE_PAGE).map_err(store_err)?;
            store.commit().map_err(store_err)?;
        }
        Ok(MetaTable {
            hwm: RefCell::new(BTreeMap::new()),
            hwm_key: RefCell::new(BTreeMap::new()),
            colgen: RefCell::new(BTreeMap::new()),
            colgen_key: RefCell::new(BTreeMap::new()),
        })
    }

    /// The next free integer key in the meta tree.
    fn next_key(&self, store: &Store) -> Result<i64> {
        let max = store
            .tree(META_TABLE_PAGE)
            .max_key()
            .map_err(store_err)?;
        Ok(max.map_or(1, |k| k + 1))
    }

    /// Decode every meta row in key-ascending order.
    fn read_rows(&self, store: &Store) -> Result<Vec<Vec<Value>>> {
        let raw = store.tree(META_TABLE_PAGE).full_scan().map_err(store_err)?;
        let mut out = Vec::with_capacity(raw.len());
        for (_key, payload) in raw {
            let (values, _) = decode_record(&payload, 0).map_err(store_err)?;
            out.push(values);
        }
        Ok(out)
    }

    /// Append one record-encoded row under a fresh key. `scope` chooses whether
    /// this opens its own store transaction (the single-statement path) or writes
    /// inside the caller's already-open transaction (under an outer `BEGIN`, where
    /// a nested `begin_write` would be rejected by the pager).
    fn append_row(&self, store: &Store, values: &[Value], scope: TxnScope) -> Result<i64> {
        let key = self.next_key(store)?;
        let payload = encode_record(values);
        if scope == TxnScope::Own {
            store.begin_write().map_err(store_err)?;
        }
        store
            .tree(META_TABLE_PAGE)
            .insert(key, &payload)
            .map_err(store_err)?;
        if scope == TxnScope::Own {
            store.commit().map_err(store_err)?;
        }
        Ok(key)
    }

    /// Register a new table: allocate its data-tree root, persist the DDL text +
    /// root mapping, apply the DDL to the live catalog, and record the root in
    /// `root_map`. Returns the allocated root page.
    ///
    /// Idempotent per table name: re-registering an already-present table is a
    /// no-op that returns the existing root page (no second tree is allocated).
    pub fn create_table(
        &self,
        store: &Store,
        catalog: &mut Catalog,
        root_map: &mut BTreeMap<String, u32>,
        ddl: &str,
    ) -> Result<u32> {
        self.create_table_scoped(store, catalog, root_map, ddl, TxnScope::Own)
    }

    /// Register a new table under an explicit transaction scope. When the caller
    /// already holds an outer `BEGIN`, `scope` is `Suppressed` so the persisted
    /// meta rows land inside that transaction (the pager rejects a nested
    /// `begin_write`); when no outer transaction is open, `Own` wraps each meta
    /// write in its own transaction.
    pub fn create_table_scoped(
        &self,
        store: &Store,
        catalog: &mut Catalog,
        root_map: &mut BTreeMap<String, u32>,
        ddl: &str,
        scope: TxnScope,
    ) -> Result<u32> {
        let table_name = extract_table_name(ddl)?;
        if let Some(existing) = root_map.get(&table_name) {
            // A second CREATE TABLE of an existing table errors like sqlite3 unless
            // IF NOT EXISTS is given, in which case it is the existing-table no-op.
            if ddl_has_if_not_exists(ddl)? {
                return Ok(*existing);
            }
            return Err(EngineError::parse(format!(
                "table {table_name} already exists"
            )));
        }

        // Sequence number = count of DDL rows already recorded (insertion order).
        let rows = self.read_rows(store)?;
        let seq = rows
            .iter()
            .filter(|r| r.first() == Some(&Value::Text("ddl".to_string())))
            .count() as i64;

        // Allocate the data-tree root via the storage layer.
        let root_page = store.create_tree().map_err(store_err)?;

        // Persist the DDL text and the table → root mapping.
        self.append_row(
            store,
            &[
                Value::Text("ddl".to_string()),
                Value::Int(seq),
                Value::Text(ddl.to_string()),
            ],
            scope,
        )?;
        self.append_row(
            store,
            &[
                Value::Text("root".to_string()),
                Value::Text(table_name.clone()),
                Value::Int(i64::from(root_page)),
            ],
            scope,
        )?;

        // Apply to the live catalog and record the root mapping.
        let stmt = crate::parser::parse(ddl)?;
        catalog.apply_ddl(&stmt)?;
        catalog.set_rootpage(&table_name, i64::from(root_page));
        root_map.insert(table_name, root_page);
        Ok(root_page)
    }

    /// Persist a DDL text row (ALTER / DROP / CREATE INDEX) so a reopen replays
    /// it through the catalog. The sequence number is the count of DDL rows
    /// already recorded, preserving insertion order. `scope` routes the write
    /// inside an outer transaction when one is open.
    pub fn record_ddl(&self, store: &Store, ddl: &str, scope: TxnScope) -> Result<()> {
        let rows = self.read_rows(store)?;
        let seq = rows
            .iter()
            .filter(|r| r.first() == Some(&Value::Text("ddl".to_string())))
            .count() as i64;
        self.append_row(
            store,
            &[
                Value::Text("ddl".to_string()),
                Value::Int(seq),
                Value::Text(ddl.to_string()),
            ],
            scope,
        )?;
        Ok(())
    }

    /// Persist a `table → root_page` mapping row so a reopen restores it (used by
    /// a RENAME that moves the mapping to the new name). `scope` routes the write
    /// inside an outer transaction when one is open.
    pub fn record_root(
        &self,
        store: &Store,
        table: &str,
        root_page: u32,
        scope: TxnScope,
    ) -> Result<()> {
        self.append_row(
            store,
            &[
                Value::Text("root".to_string()),
                Value::Text(table.to_string()),
                Value::Int(i64::from(root_page)),
            ],
            scope,
        )?;
        Ok(())
    }

    /// Rebuild the catalog and `root_map` from the persisted meta rows.
    ///
    /// DDL rows replay through [`Catalog::apply_ddl`] in ascending sequence
    /// order; root rows populate `root_map`; hwm rows seed the in-memory
    /// high-water marks. A root entry whose table name is absent from the rebuilt
    /// catalog (a stale mapping) is dropped.
    pub fn replay(
        &self,
        store: &Store,
        catalog: &mut Catalog,
        root_map: &mut BTreeMap<String, u32>,
    ) -> Result<()> {
        let raw = store.tree(META_TABLE_PAGE).full_scan().map_err(store_err)?;

        let mut ddl_rows: Vec<(i64, String)> = Vec::new();
        let mut root_rows: Vec<(String, u32)> = Vec::new();

        for (key, payload) in raw {
            let (values, _) = decode_record(&payload, 0).map_err(store_err)?;
            match row_kind(&values) {
                Some("ddl") if values.len() >= 3 => {
                    let seq = as_int(&values[1]).unwrap_or(0);
                    if let Value::Text(text) = &values[2] {
                        ddl_rows.push((seq, text.clone()));
                    }
                }
                Some("root") if values.len() >= 3 => {
                    if let (Value::Text(table), Some(root)) =
                        (&values[1], as_int(&values[2]))
                    {
                        root_rows.push((table.clone(), root as u32));
                    }
                }
                Some("hwm") if values.len() >= 3 => {
                    if let (Value::Text(table), Some(v)) = (&values[1], as_int(&values[2])) {
                        self.hwm.borrow_mut().insert(table.clone(), v);
                        self.hwm_key.borrow_mut().insert(table.clone(), key);
                    }
                }
                _ => {}
            }
        }

        ddl_rows.sort_by_key(|(seq, _)| *seq);
        for (_seq, text) in ddl_rows {
            let stmt = crate::parser::parse(&text)?;
            if let Err(e) = catalog.apply_ddl(&stmt) {
                // A store may carry a repeated ADD COLUMN meta row (the
                // duplicate guard fires at live-DDL time, not here);
                // propagating would make such a store unopenable, so replay
                // treats the repeat as a no-op.
                if e.to_string().starts_with("duplicate column name") {
                    continue;
                }
                return Err(e);
            }
        }

        for (table, root) in root_rows {
            root_map.insert(table.clone(), root);
            catalog.set_rootpage(&table, i64::from(root));
        }
        // Drop any root mapping whose table did not survive replay (e.g. dropped).
        let stale: Vec<String> = root_map
            .keys()
            .filter(|name| catalog.get(name).is_err())
            .cloned()
            .collect();
        for name in stale {
            root_map.remove(&name);
        }
        Ok(())
    }

    /// Return the next vec_label for `table`, bumping and persisting its
    /// high-water mark in its own write transaction. Monotonic across the life of
    /// the store, including a close / reopen (the mark is written to the meta
    /// table immediately).
    pub fn next_vec_label(&self, store: &Store, table: &str) -> Result<i64> {
        self.bump_vec_label(store, table, true)
    }

    /// Return the next vec_label for `table` without opening a write
    /// transaction, for use when the caller already holds one open (an
    /// `executemany` batch or an outer `BEGIN`). The high-water-mark row write
    /// lands inside the caller's transaction so the whole batch — data rows and
    /// label bumps — commits atomically and the pager is never double-opened.
    pub fn next_vec_label_in_txn(&self, store: &Store, table: &str) -> Result<i64> {
        self.bump_vec_label(store, table, false)
    }

    /// Bump and persist `table`'s vec_label high-water mark. `own_txn` chooses
    /// whether this method opens its own write transaction (the single-statement
    /// path) or writes inside the caller's already-open transaction (the batch
    /// path), mirroring the reference's per-statement-vs-suppressed txn model.
    fn bump_vec_label(&self, store: &Store, table: &str, own_txn: bool) -> Result<i64> {
        let current = *self.hwm.borrow().get(table).unwrap_or(&0);
        let next = current + 1;
        let existing_key = self.hwm_key.borrow().get(table).copied();

        let payload = encode_record(&[
            Value::Text("hwm".to_string()),
            Value::Text(table.to_string()),
            Value::Int(next),
        ]);
        let key = match existing_key {
            Some(k) => k,
            None => self.next_key(store)?,
        };
        if own_txn {
            store.begin_write().map_err(store_err)?;
        }
        store
            .tree(META_TABLE_PAGE)
            .insert(key, &payload)
            .map_err(store_err)?;
        if own_txn {
            store.commit().map_err(store_err)?;
        }

        self.hwm.borrow_mut().insert(table.to_string(), next);
        self.hwm_key.borrow_mut().insert(table.to_string(), key);
        Ok(next)
    }

    /// Set `table`'s vec_label high-water mark to an absolute `value`, persisting
    /// it to the meta table so it survives a close / reopen. Unlike
    /// `next_vec_label`, this does not increment — it overwrites the mark with the
    /// supplied value, for use after a verbatim bulk copy that wrote explicit
    /// `vec_label` values: the mark is set to the max label copied so the next
    /// auto-assigned label is `value + 1` and cannot collide with a copied row.
    ///
    /// Opens its own write transaction. The mark is monotonic guard: a `value`
    /// that is not greater than the current mark is a no-op, so calling this with
    /// a stale max can never lower a higher persisted mark.
    pub fn set_vec_label_hwm(&self, store: &Store, table: &str, value: i64) -> Result<()> {
        // Load any persisted mark so a fresh handle that has not yet seen this
        // table does not overwrite a higher on-disk value with a lower one.
        if !self.hwm.borrow().contains_key(table) {
            self.load_hwm_from_disk(store, table)?;
        }
        let current = *self.hwm.borrow().get(table).unwrap_or(&0);
        if value <= current {
            return Ok(());
        }
        let existing_key = self.hwm_key.borrow().get(table).copied();
        let payload = encode_record(&[
            Value::Text("hwm".to_string()),
            Value::Text(table.to_string()),
            Value::Int(value),
        ]);
        let key = match existing_key {
            Some(k) => k,
            None => self.next_key(store)?,
        };
        store.begin_write().map_err(store_err)?;
        store
            .tree(META_TABLE_PAGE)
            .insert(key, &payload)
            .map_err(store_err)?;
        store.commit().map_err(store_err)?;

        self.hwm.borrow_mut().insert(table.to_string(), value);
        self.hwm_key.borrow_mut().insert(table.to_string(), key);
        Ok(())
    }

    /// Seed the in-memory `hwm` / `hwm_key` mirrors for `table` from the persisted
    /// meta rows, so a handle that has not yet inserted into `table` (e.g. one that
    /// opened a store written by the verbatim migrator) sees the on-disk mark.
    fn load_hwm_from_disk(&self, store: &Store, table: &str) -> Result<()> {
        let raw = store.tree(META_TABLE_PAGE).full_scan().map_err(store_err)?;
        for (key, payload) in raw {
            let (values, _) = decode_record(&payload, 0).map_err(store_err)?;
            if values.len() >= 3
                && row_kind(&values) == Some("hwm")
                && matches!(&values[1], Value::Text(t) if t == table)
            {
                if let Some(v) = as_int(&values[2]) {
                    self.hwm.borrow_mut().insert(table.to_string(), v);
                    self.hwm_key.borrow_mut().insert(table.to_string(), key);
                }
            }
        }
        Ok(())
    }

    /// Bump and persist `table`'s committed write-generation, INSIDE the
    /// caller's already-open store transaction, so the generation and the DML
    /// it fences commit or roll back together atomically.
    ///
    /// This is the airtight staleness fence for the on-disk column-index
    /// sidecar (see `conn::Connection::persist_col_indexes` / the load gate in
    /// `from_store`): a committed write to `table` — INSERT, UPDATE, DELETE, an
    /// `executemany` batch — must advance this counter before its transaction
    /// commits, so a mismatch on the next open discards a sidecar that a
    /// count-only fingerprint would falsely admit (e.g. a same-key in-place
    /// UPDATE of an indexed column, which leaves the row count unchanged).
    ///
    /// Mirrors [`Self::bump_vec_label`]'s in-txn write, but never opens or
    /// closes a transaction itself — the caller (already inside its DML's
    /// transaction scope) owns the commit/rollback boundary. A rolled-back
    /// bump leaves the in-memory mirror ABOVE the last committed on-disk
    /// value; that errs safe, because a later persist stamps the higher
    /// in-memory value while a subsequent open reads back the LOWER value the
    /// store actually committed — a mismatch, so a spurious (but correct)
    /// lazy rebuild follows. The mirror must never sit BELOW the committed
    /// value, so a table not yet seen by this handle is seeded from disk
    /// first (mirroring [`Self::load_hwm_from_disk`]).
    pub fn bump_col_generation(&self, store: &Store, table: &str) -> Result<i64> {
        if !self.colgen.borrow().contains_key(table) {
            self.load_col_generation_from_disk(store, table)?;
        }
        let current = *self.colgen.borrow().get(table).unwrap_or(&0);
        let next = current + 1;
        let existing_key = self.colgen_key.borrow().get(table).copied();

        let payload = encode_record(&[
            Value::Text("colgen".to_string()),
            Value::Text(table.to_string()),
            Value::Int(next),
        ]);
        let key = match existing_key {
            Some(k) => k,
            None => self.next_key(store)?,
        };
        store
            .tree(META_TABLE_PAGE)
            .insert(key, &payload)
            .map_err(store_err)?;

        self.colgen.borrow_mut().insert(table.to_string(), next);
        self.colgen_key.borrow_mut().insert(table.to_string(), key);
        Ok(next)
    }

    /// `table`'s last-seen committed write-generation (0 for a table never
    /// bumped), seeding the in-memory mirror from the persisted meta rows on
    /// first touch so a fresh handle sees the on-disk value rather than 0.
    pub fn col_generation(&self, store: &Store, table: &str) -> Result<i64> {
        if !self.colgen.borrow().contains_key(table) {
            self.load_col_generation_from_disk(store, table)?;
        }
        Ok(*self.colgen.borrow().get(table).unwrap_or(&0))
    }

    /// Seed the in-memory `colgen` / `colgen_key` mirrors for `table` from the
    /// persisted meta rows, mirroring [`Self::load_hwm_from_disk`].
    fn load_col_generation_from_disk(&self, store: &Store, table: &str) -> Result<()> {
        let raw = store.tree(META_TABLE_PAGE).full_scan().map_err(store_err)?;
        let mut found = false;
        for (key, payload) in raw {
            let (values, _) = decode_record(&payload, 0).map_err(store_err)?;
            if values.len() >= 3
                && row_kind(&values) == Some("colgen")
                && matches!(&values[1], Value::Text(t) if t == table)
            {
                if let Some(v) = as_int(&values[2]) {
                    self.colgen.borrow_mut().insert(table.to_string(), v);
                    self.colgen_key.borrow_mut().insert(table.to_string(), key);
                    found = true;
                }
            }
        }
        // Seed a 0 default when the table has no persisted colgen row yet, so the
        // in-memory mirror's `contains_key` becomes true after this single scan.
        // The callers only invoke this when the key is absent; without the default
        // a never-bumped table re-runs the whole-meta scan on EVERY write, turning
        // a one-shot boot seed into a per-write meta full-scan tax.
        if !found {
            self.colgen.borrow_mut().insert(table.to_string(), 0);
        }
        Ok(())
    }
}

/// The first cell of a meta row as a kind tag, when it is text.
fn row_kind(values: &[Value]) -> Option<&str> {
    match values.first() {
        Some(Value::Text(s)) => Some(s.as_str()),
        _ => None,
    }
}

/// Read a cell as an i64 when it is an integer.
fn as_int(value: &Value) -> Option<i64> {
    match value {
        Value::Int(v) => Some(*v),
        _ => None,
    }
}

/// Extract the table name from a CREATE TABLE DDL, honoring `IF NOT EXISTS` and
/// a bracket-quoted name (the lexer already unwraps `[name]` to a bare ident).
fn extract_table_name(ddl: &str) -> Result<String> {
    let tokens = tokenize(ddl)?;
    let mut i = 0;
    let val = |t: &Token| match t {
        Token::Keyword(s) | Token::Ident(s) => s.clone(),
        _ => String::new(),
    };
    let up = |t: &Token| val(t).to_ascii_uppercase();

    if i >= tokens.len() || up(&tokens[i]) != "CREATE" {
        return Err(EngineError::parse(format!(
            "cannot extract table name from DDL: {ddl:?}"
        )));
    }
    i += 1;
    if i >= tokens.len() || up(&tokens[i]) != "TABLE" {
        return Err(EngineError::parse(format!(
            "cannot extract table name from DDL: {ddl:?}"
        )));
    }
    i += 1;
    if i + 2 < tokens.len()
        && up(&tokens[i]) == "IF"
        && up(&tokens[i + 1]) == "NOT"
        && up(&tokens[i + 2]) == "EXISTS"
    {
        i += 3;
    }
    let name = tokens.get(i).map(val).unwrap_or_default();
    if name.is_empty() {
        return Err(EngineError::parse(format!(
            "cannot extract table name from DDL: {ddl:?}"
        )));
    }
    Ok(name)
}

/// True when a CREATE TABLE DDL carries the `IF NOT EXISTS` clause, tokenized so
/// quoting and whitespace do not change the answer.
///
/// The clause is detected only at its grammatical position — the tokens
/// immediately after `CREATE [TEMP|TEMPORARY] TABLE`, before the table name — so a
/// table name or column literally spelled `if`/`not`/`exists`, or a CHECK
/// expression containing those words, cannot false-trigger.
fn ddl_has_if_not_exists(ddl: &str) -> Result<bool> {
    let tokens = tokenize(ddl)?;
    let up = |t: &Token| match t {
        Token::Keyword(s) | Token::Ident(s) => s.to_ascii_uppercase(),
        _ => String::new(),
    };
    let mut i = 0;
    if i >= tokens.len() || up(&tokens[i]) != "CREATE" {
        return Ok(false);
    }
    i += 1;
    // Skip an optional TEMP / TEMPORARY qualifier before TABLE.
    if i < tokens.len() && matches!(up(&tokens[i]).as_str(), "TEMP" | "TEMPORARY") {
        i += 1;
    }
    if i >= tokens.len() || up(&tokens[i]) != "TABLE" {
        return Ok(false);
    }
    i += 1;
    Ok(i + 2 < tokens.len()
        && up(&tokens[i]) == "IF"
        && up(&tokens[i + 1]) == "NOT"
        && up(&tokens[i + 2]) == "EXISTS")
}

/// Adapt a storage-layer error into the engine error taxonomy.
fn store_err(e: lillibrain::StoreError) -> EngineError {
    EngineError::parse(format!("meta-table storage error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::ddl_has_if_not_exists;

    #[test]
    fn if_not_exists_detected_only_at_the_anchored_clause_position() {
        // The real clause, with and without a TEMP/TEMPORARY qualifier.
        assert!(ddl_has_if_not_exists("CREATE TABLE IF NOT EXISTS t ( id TEXT )").unwrap());
        assert!(
            ddl_has_if_not_exists("create table if not exists t ( id TEXT )").unwrap(),
            "the clause is case-insensitive"
        );
        assert!(
            ddl_has_if_not_exists("CREATE TEMP TABLE IF NOT EXISTS t ( id TEXT )").unwrap()
        );
        assert!(
            ddl_has_if_not_exists("CREATE TEMPORARY TABLE IF NOT EXISTS t ( id TEXT )").unwrap()
        );

        // A plain CREATE TABLE has no clause.
        assert!(!ddl_has_if_not_exists("CREATE TABLE t ( id TEXT )").unwrap());

        // A contiguous IF NOT EXISTS run that does NOT sit in the CREATE TABLE
        // anchor slot must not be read as the table clause. A whole-DDL window scan
        // would false-trigger on this CREATE INDEX form (the run is present, just
        // not after CREATE TABLE); the anchored detector requires the run
        // immediately after CREATE [TEMP] TABLE and before the name, so it returns
        // false here.
        assert!(
            !ddl_has_if_not_exists("CREATE INDEX IF NOT EXISTS ix ON t ( c )").unwrap(),
            "an IF NOT EXISTS run outside the CREATE TABLE anchor must not false-trigger"
        );
        // The same run under a different DDL verb is likewise ignored.
        assert!(
            !ddl_has_if_not_exists("DROP TABLE IF NOT EXISTS t").unwrap(),
            "the clause is detected only after CREATE TABLE, not any DDL verb"
        );
    }
}

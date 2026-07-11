//! In-memory schema catalog: column metadata and table registry.
//!
//! Populated by DDL statements (CREATE TABLE, ALTER TABLE, DROP TABLE, CREATE
//! INDEX). CREATE INDEX is accepted and recorded as index metadata but does not
//! create a scan path — the planner always emits a full scan in this engine.
//!
//! Type affinity follows the sqlite3 rules: a declared column type maps to one
//! of INTEGER / REAL / BLOB / NUMERIC / TEXT by a fixed keyword map, falling
//! back to TEXT for an unknown type. The map is exact parity with the reference
//! catalog so column storage classes agree across the two storage drivers.

use std::collections::BTreeMap;

use lillibrain::Value;

use crate::ast::{DdlStmt, Stmt};
use crate::error::{EngineError, Result};
use crate::lexer::{tokenize, Token};

/// Metadata for one column in the schema catalog.
#[derive(Debug, Clone, PartialEq)]
pub struct ColumnMeta {
    /// The column name in declaration order.
    pub name: String,
    /// The sqlite3 affinity: `INTEGER`, `TEXT`, `REAL`, `BLOB`, or `NUMERIC`.
    pub affinity: String,
    /// True unless the column is `NOT NULL` or part of a PRIMARY KEY.
    pub nullable: bool,
    /// True when the column participates in the table's PRIMARY KEY.
    pub primary_key: bool,
    /// The column `DEFAULT`, or `None` when no default was declared. A declared
    /// `DEFAULT NULL` is carried as `Some(Value::Null)`, distinct from `None`.
    pub default: Option<Value>,
}

impl ColumnMeta {
    /// A column with the given name and affinity, nullable, no PK, no default.
    fn new(name: impl Into<String>, affinity: impl Into<String>) -> Self {
        ColumnMeta {
            name: name.into(),
            affinity: affinity.into(),
            nullable: true,
            primary_key: false,
            default: None,
        }
    }
}

/// Map a declared type token to its sqlite3 affinity.
///
/// Tries a direct lookup against the keyword map first, then a substring match
/// for compound type names (e.g. `VARYING CHARACTER`), then falls back to TEXT.
fn affinity_for(type_token: &str) -> String {
    // The keyword → affinity pairs, in the reference's iteration order so a
    // substring match resolves identically.
    const MAP: &[(&str, &str)] = &[
        ("INTEGER", "INTEGER"),
        ("INT", "INTEGER"),
        ("REAL", "REAL"),
        ("FLOAT", "REAL"),
        ("DOUBLE", "REAL"),
        ("BLOB", "BLOB"),
        ("NONE", "BLOB"),
        ("NUMERIC", "NUMERIC"),
        ("DECIMAL", "NUMERIC"),
        ("BOOLEAN", "NUMERIC"),
        ("DATE", "NUMERIC"),
        ("DATETIME", "NUMERIC"),
        ("TEXT", "TEXT"),
        ("CHAR", "TEXT"),
        ("CLOB", "TEXT"),
        ("VARCHAR", "TEXT"),
    ];
    let upper = type_token.to_ascii_uppercase();
    for (kw, aff) in MAP {
        if *kw == upper {
            return (*aff).to_string();
        }
    }
    for (kw, aff) in MAP {
        if upper.contains(kw) {
            return (*aff).to_string();
        }
    }
    "TEXT".to_string()
}

/// Coerce a value to a column affinity, following the sqlite3 storage-class
/// rules: an INTEGER-affinity column stores a whole-valued REAL as an integer
/// and a numeric-looking TEXT as a number; a REAL-affinity column stores an
/// integer as a real; a TEXT-affinity column stores a number as its text form;
/// a BLOB-affinity column applies no coercion. NULL is never coerced.
///
/// This is the single coercion the row executor reuses so a stored cell's
/// storage class agrees with stdlib sqlite3 for the same column type.
pub fn coerce_affinity(value: Value, affinity: &str) -> Value {
    match (affinity, &value) {
        (_, Value::Null) | (_, Value::Blob(_)) => value,
        ("BLOB", _) => value,
        ("INTEGER" | "NUMERIC", Value::Float(f)) if f.fract() == 0.0 && f.is_finite() => {
            Value::Int(*f as i64)
        }
        ("INTEGER" | "NUMERIC", Value::Text(s)) => coerce_numeric_text(s, affinity),
        ("INTEGER", _) => value,
        ("REAL", Value::Int(i)) => Value::Float(*i as f64),
        ("REAL", Value::Text(s)) => match s.parse::<f64>() {
            Ok(f) => Value::Float(f),
            Err(_) => value,
        },
        ("REAL", _) => value,
        ("TEXT", Value::Int(i)) => Value::Text(i.to_string()),
        ("TEXT", Value::Float(f)) => Value::Text(render_float_text(*f)),
        ("TEXT", _) => value,
        _ => value,
    }
}

/// Render a float to the text form sqlite3 stores when coercing a REAL into a
/// TEXT column (and the form Python's `str(float)` produces).
///
/// sqlite3 renders a whole-valued REAL with a trailing `.0` (`1.0` -> `"1.0"`,
/// not `"1"`), unlike Rust's default `{}` formatting which drops it. Rust's
/// shortest round-tripping decimal (Ryū) equals sqlite3 / Python `str(float)`
/// byte-for-byte across the bounded decimal range these REAL columns carry
/// (scores, weights), and Rust emits no scientific notation for `f64` — so the
/// only normalization needed is guaranteeing the decimal point sqlite3 always
/// keeps on a whole value.
pub fn render_float_text(f: f64) -> String {
    if f.is_nan() {
        // sqlite3 stores a NaN REAL as NULL; the coercion layer never reaches a
        // NaN through a normal bind, but render a stable token rather than the
        // platform-specific "NaN" if one ever arrives.
        return "0.0".to_string();
    }
    if f.is_infinite() {
        return if f < 0.0 { "-Inf".to_string() } else { "Inf".to_string() };
    }
    let s = format!("{f}");
    if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Coerce a numeric-looking text cell for an INTEGER / NUMERIC affinity: an
/// integer string becomes Int, a real string becomes Float (or Int when whole
/// for INTEGER affinity), a non-numeric string is left as text.
fn coerce_numeric_text(s: &str, affinity: &str) -> Value {
    if let Ok(i) = s.parse::<i64>() {
        return Value::Int(i);
    }
    if let Ok(f) = s.parse::<f64>() {
        if affinity == "INTEGER" && f.fract() == 0.0 && f.is_finite() {
            return Value::Int(f as i64);
        }
        return Value::Float(f);
    }
    Value::Text(s.to_string())
}

/// Registry mapping table names to their ordered column metadata list.
///
/// Insertion order of tables is preserved (a `BTreeMap` would reorder them);
/// the index registry is stored for completeness and never selects a scan path.
#[derive(Debug, Default, Clone)]
pub struct Catalog {
    tables: Vec<(String, Vec<ColumnMeta>)>,
    // table → list of (index_name, column, is_partial). Recorded, never used.
    indexes: BTreeMap<String, Vec<(String, String, bool, bool)>>,
    // table → the original CREATE TABLE text, as the `sql` column of the
    // sqlite_master row for that table. Set when a CREATE TABLE is applied.
    ddl_sql: BTreeMap<String, String>,
    // table → its B-tree root page, as the `rootpage` column of the
    // sqlite_master row. Set by the meta layer after it allocates the root.
    rootpage: BTreeMap<String, i64>,
}

/// One synthesized `sqlite_master` row: the schema-object kind, its name, the
/// name of the table it belongs to, its root page, and the stored DDL text.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaRow {
    /// `"table"` or `"index"`.
    pub object_type: String,
    /// The object name (table name, or index name).
    pub name: String,
    /// The table the object belongs to (equals `name` for a table row).
    pub tbl_name: String,
    /// The object's root page, or 0 when unknown.
    pub rootpage: i64,
    /// The stored CREATE DDL text, or NULL when none was retained.
    pub sql: Value,
}

impl Catalog {
    /// An empty catalog.
    pub fn new() -> Self {
        Catalog::default()
    }

    fn table_index(&self, table: &str) -> Option<usize> {
        self.tables.iter().position(|(n, _)| n == table)
    }

    /// Register (or replace) a table with the given column list.
    pub fn register(&mut self, table: impl Into<String>, columns: Vec<ColumnMeta>) {
        let table = table.into();
        match self.table_index(&table) {
            Some(i) => self.tables[i] = (table, columns),
            None => self.tables.push((table, columns)),
        }
    }

    /// The column names for `table` in declaration order.
    pub fn column_names(&self, table: &str) -> Result<Vec<String>> {
        Ok(self.get(table)?.iter().map(|c| c.name.clone()).collect())
    }

    /// The `ColumnMeta` list for `table`. Errors when the table is absent.
    pub fn get(&self, table: &str) -> Result<&[ColumnMeta]> {
        self.table_index(table)
            .map(|i| self.tables[i].1.as_slice())
            .ok_or_else(|| EngineError::parse(format!("catalog: unknown table {table:?}")))
    }

    /// The PRIMARY KEY column names for `table` in declaration order. Empty when
    /// the table has no primary key.
    pub fn primary_key_columns(&self, table: &str) -> Result<Vec<String>> {
        Ok(self
            .get(table)?
            .iter()
            .filter(|c| c.primary_key)
            .map(|c| c.name.clone())
            .collect())
    }

    /// The PRAGMA `table_info(table)` rows: `(cid, name, type, notnull,
    /// dflt_value, pk)` per column. `dflt_value` is the declared default
    /// rendered to its stored cell, or `Value::Null` when none was declared.
    #[allow(clippy::type_complexity)]
    pub fn table_info(
        &self,
        table: &str,
    ) -> Result<Vec<(i64, String, String, i64, Value, i64)>> {
        let cols = self.get(table)?;
        Ok(cols
            .iter()
            .enumerate()
            .map(|(cid, c)| {
                (
                    cid as i64,
                    c.name.clone(),
                    c.affinity.clone(),
                    i64::from(!c.nullable),
                    c.default.clone().unwrap_or(Value::Null),
                    i64::from(c.primary_key),
                )
            })
            .collect())
    }

    /// Record a table's B-tree root page so the `sqlite_master.rootpage` column
    /// reports it. Called by the meta layer once a root is allocated or replayed.
    pub fn set_rootpage(&mut self, table: &str, root_page: i64) {
        self.rootpage.insert(table.to_string(), root_page);
    }

    /// The table names in declaration order, excluding the engine-internal
    /// `sqlite_`-prefixed objects (none exist here, but the filter keeps parity
    /// with stdlib's reserved-name convention).
    pub fn table_names(&self) -> Vec<String> {
        self.tables
            .iter()
            .map(|(n, _)| n.clone())
            .filter(|n| !n.starts_with("sqlite_"))
            .collect()
    }

    /// The synthesized `sqlite_master` rows: one `type='table'` row per table
    /// (in declaration order) followed by one `type='index'` row per tracked
    /// index. Matches stdlib sqlite3's `(type, name, tbl_name, rootpage, sql)`
    /// shape so a schema-introspection query reads identically across drivers.
    pub fn sqlite_master_rows(&self) -> Vec<SchemaRow> {
        let mut rows: Vec<SchemaRow> = Vec::new();
        for (table, _) in &self.tables {
            if table.starts_with("sqlite_") {
                continue;
            }
            rows.push(SchemaRow {
                object_type: "table".to_string(),
                name: table.clone(),
                tbl_name: table.clone(),
                rootpage: self.rootpage.get(table).copied().unwrap_or(0),
                sql: match self.ddl_sql.get(table) {
                    Some(s) => Value::Text(s.clone()),
                    None => Value::Null,
                },
            });
        }
        for (table, idx_list) in &self.indexes {
            for (index_name, column, _is_partial, is_unique) in idx_list {
                let unique_kw = if *is_unique { "UNIQUE " } else { "" };
                rows.push(SchemaRow {
                    object_type: "index".to_string(),
                    name: index_name.clone(),
                    tbl_name: table.clone(),
                    rootpage: 0,
                    sql: Value::Text(format!(
                        "CREATE {unique_kw}INDEX {index_name} ON {table} ({column})"
                    )),
                });
            }
        }
        rows
    }

    /// Update the catalog from a parsed statement.
    ///
    /// Handles the DDL kinds (`create_table`, `alter_add_column`,
    /// `alter_drop_column`, `alter_rename_table`, `drop_table`, `create_index`).
    /// PRAGMA and non-DDL statements are silently accepted.
    pub fn apply_ddl(&mut self, stmt: &Stmt) -> Result<()> {
        let Stmt::Ddl(DdlStmt { kind, raw }) = stmt else {
            return Ok(());
        };
        match kind.as_str() {
            "create_table" => self.apply_create_table(raw),
            "alter_add_column" => self.apply_alter_add_column(raw),
            "alter_drop_column" => self.apply_alter_drop_column(raw),
            "alter_rename_table" => self.apply_alter_rename(raw),
            "drop_table" => self.apply_drop_table(raw),
            "create_index" => self.apply_create_index(raw),
            _ => Ok(()), // pragma and others: accepted, no-op
        }
    }

    fn apply_create_table(&mut self, raw: &str) -> Result<()> {
        let tokens = tokenize(raw)?;
        let mut w = Walker::new(&tokens);

        w.expect("CREATE")?;
        w.expect("TABLE")?;
        if w.peek_upper() == "IF" {
            w.advance();
            w.expect("NOT")?;
            w.expect("EXISTS")?;
        }
        let table_name = w.advance();
        w.expect("(")?;

        let mut columns: Vec<ColumnMeta> = Vec::new();
        while !matches!(w.peek_upper().as_str(), ")" | "") {
            let head = w.peek_upper();
            // Table-level constraint clause (PRIMARY KEY (...), UNIQUE (...), ...).
            if matches!(
                head.as_str(),
                "PRIMARY" | "UNIQUE" | "CHECK" | "FOREIGN" | "CONSTRAINT"
            ) {
                let is_primary = head == "PRIMARY";
                let mut pk_cols: Vec<String> = Vec::new();
                // Skip up to the opening paren.
                while !w.at_end() && w.peek_raw() != "(" {
                    w.advance();
                }
                if !w.at_end() && w.peek_raw() == "(" {
                    w.advance(); // "("
                    let mut depth = 1;
                    while !w.at_end() && depth > 0 {
                        let v = w.peek_raw();
                        match v.as_str() {
                            "(" => {
                                depth += 1;
                                w.advance();
                            }
                            ")" => {
                                depth -= 1;
                                w.advance();
                                if depth == 0 {
                                    break;
                                }
                            }
                            "," => {
                                w.advance();
                            }
                            _ => {
                                if is_primary && depth == 1 {
                                    pk_cols.push(v);
                                }
                                w.advance();
                            }
                        }
                    }
                }
                if is_primary {
                    let pk_set: Vec<String> =
                        pk_cols.iter().map(|n| n.to_ascii_lowercase()).collect();
                    for cm in &mut columns {
                        if pk_set.contains(&cm.name.to_ascii_lowercase()) {
                            cm.primary_key = true;
                            cm.nullable = false; // PRIMARY KEY implies NOT NULL
                        }
                    }
                }
                if w.peek_upper() == "," {
                    w.advance();
                }
                continue;
            }

            // Column definition: name [type] [constraints...].
            let col_name = w.advance();
            let mut affinity = "TEXT".to_string();
            if w.peek_is_type_token() {
                affinity = affinity_for(&w.advance());
            }

            let mut nullable = true;
            let mut primary_key = false;
            let mut default: Option<Value> = None;
            while !matches!(w.peek_upper().as_str(), "," | ")" | "") {
                match w.peek_upper().as_str() {
                    "PRIMARY" => {
                        w.advance(); // PRIMARY
                        w.advance(); // KEY
                        primary_key = true;
                    }
                    "NOT" => {
                        w.advance(); // NOT
                        if w.peek_upper() == "NULL" {
                            w.advance();
                            nullable = false;
                        }
                    }
                    "AUTOINCREMENT" => {
                        w.advance();
                    }
                    "UNIQUE" => {
                        w.advance();
                    }
                    "DEFAULT" => {
                        w.advance();
                        default = Some(parse_default(&mut w));
                    }
                    "REFERENCES" => {
                        w.advance(); // REFERENCES
                        w.advance(); // referenced table
                        if w.peek_raw() == "(" {
                            w.advance(); // (
                            w.advance(); // col
                            w.advance(); // )
                        }
                    }
                    _ => break,
                }
            }

            let mut col = ColumnMeta::new(col_name, affinity);
            col.nullable = nullable;
            col.primary_key = primary_key;
            // A declared default is stored in the column's affinity, matching the
            // storage class sqlite3 would persist for that column type.
            col.default = default.map(|d| coerce_affinity(d, &col.affinity));
            if columns
                .iter()
                .any(|c| c.name.eq_ignore_ascii_case(&col.name))
            {
                return Err(EngineError::parse(format!(
                    "duplicate column name: {}",
                    col.name
                )));
            }
            columns.push(col);

            if w.peek_upper() == "," {
                w.advance();
            }
        }

        self.ddl_sql.insert(table_name.clone(), raw.trim().to_string());
        self.register(table_name, columns);
        Ok(())
    }

    fn apply_alter_add_column(&mut self, raw: &str) -> Result<()> {
        let tokens = tokenize(raw)?;
        let mut w = Walker::new(&tokens);
        w.expect("ALTER")?;
        w.expect("TABLE")?;
        let table_name = w.advance();
        w.expect("ADD")?;
        if w.peek_upper() == "COLUMN" {
            w.advance();
        }
        let col_name = w.advance();
        let mut affinity = "TEXT".to_string();
        if w.peek_is_type_token() {
            affinity = affinity_for(&w.advance());
        }
        let i = self.table_index(&table_name).ok_or_else(|| {
            EngineError::parse(format!(
                "catalog: unknown table {table_name:?} in ALTER TABLE"
            ))
        })?;
        if self.tables[i]
            .1
            .iter()
            .any(|c| c.name.eq_ignore_ascii_case(&col_name))
        {
            return Err(EngineError::parse(format!(
                "duplicate column name: {col_name}"
            )));
        }
        self.tables[i].1.push(ColumnMeta::new(col_name, affinity));
        Ok(())
    }

    fn apply_alter_drop_column(&mut self, raw: &str) -> Result<()> {
        let tokens = tokenize(raw)?;
        let mut w = Walker::new(&tokens);
        w.expect("ALTER")?;
        w.expect("TABLE")?;
        let table_name = w.advance();
        w.expect("DROP")?;
        if w.peek_upper() == "COLUMN" {
            w.advance();
        }
        let col_name = w.advance();
        if let Some(i) = self.table_index(&table_name) {
            self.tables[i].1.retain(|c| c.name != col_name);
        }
        Ok(())
    }

    fn apply_alter_rename(&mut self, raw: &str) -> Result<()> {
        let tokens = tokenize(raw)?;
        let mut w = Walker::new(&tokens);
        w.expect("ALTER")?;
        w.expect("TABLE")?;
        let old_name = w.advance();
        w.expect("RENAME")?;
        w.expect("TO")?;
        let new_name = w.advance();
        let i = self.table_index(&old_name).ok_or_else(|| {
            EngineError::parse(format!(
                "catalog: unknown table {old_name:?} in ALTER TABLE RENAME"
            ))
        })?;
        self.tables[i].0 = new_name.clone();
        if let Some(idx) = self.indexes.remove(&old_name) {
            self.indexes.insert(new_name.clone(), idx);
        }
        if let Some(page) = self.rootpage.remove(&old_name) {
            self.rootpage.insert(new_name.clone(), page);
        }
        // sqlite3 rewrites the table name inside the stored DDL on RENAME; mirror
        // that so the renamed table's `sql` column names the new table.
        if let Some(sql) = self.ddl_sql.remove(&old_name) {
            let rewritten = rewrite_ddl_table_name(&sql, &old_name, &new_name);
            self.ddl_sql.insert(new_name, rewritten);
        }
        Ok(())
    }

    fn apply_drop_table(&mut self, raw: &str) -> Result<()> {
        let tokens = tokenize(raw)?;
        let mut w = Walker::new(&tokens);
        w.expect("DROP")?;
        w.expect("TABLE")?;
        let mut if_exists = false;
        if w.peek_upper() == "IF" {
            w.advance();
            w.expect("EXISTS")?;
            if_exists = true;
        }
        let name = w.advance();
        match self.table_index(&name) {
            Some(i) => {
                self.tables.remove(i);
                self.indexes.remove(&name);
                self.ddl_sql.remove(&name);
                self.rootpage.remove(&name);
                Ok(())
            }
            None if if_exists => Ok(()),
            None => Err(EngineError::parse(format!(
                "catalog: unknown table {name:?} in DROP TABLE"
            ))),
        }
    }

    /// The single-column indexes declared for `table`, as `(column, is_partial)` pairs.
    ///
    /// Each recorded `CREATE INDEX` contributes one entry — its leading column and
    /// a flag indicating whether the index carries a `WHERE` predicate (partial).
    /// When the same column appears under more than one index, only one entry is
    /// returned; the non-partial flag takes precedence (a non-partial index subsumes
    /// the partial one for equality probes). Returns an empty `Vec` for an unknown
    /// table or a table with no recorded indexes.
    ///
    /// Callers that build a hash secondary index over these columns should skip
    /// partial-flagged entries unless they have special knowledge that the partial
    /// predicate is satisfied implicitly by their probe (e.g. `embedding_pending=1`
    /// for a `WHERE embedding_pending=1` partial index, where an equality probe
    /// `col = 1` is a subset of the partial's domain).
    pub fn indexed_columns(&self, table: &str) -> Vec<(String, bool)> {
        let Some(entries) = self.indexes.get(table) else {
            return Vec::new();
        };
        // Dedup by column: if a column appears both partial and non-partial, the
        // non-partial flag wins (the non-partial index is a superset of the partial).
        let mut seen: std::collections::BTreeMap<String, bool> = std::collections::BTreeMap::new();
        for (_index_name, col, is_partial, _is_unique) in entries {
            seen.entry(col.clone())
                .and_modify(|existing| {
                    // Non-partial wins over partial.
                    if !is_partial {
                        *existing = false;
                    }
                })
                .or_insert(*is_partial);
        }
        seen.into_iter().collect()
    }

    fn apply_create_index(&mut self, raw: &str) -> Result<()> {
        let tokens = tokenize(raw)?;
        let mut w = Walker::new(&tokens);
        w.advance(); // CREATE
        // CREATE UNIQUE INDEX carries the UNIQUE keyword before INDEX; consume it
        // so the index name is the actual name, not the keyword.
        let is_unique = w.peek_upper() == "UNIQUE";
        if is_unique {
            w.advance(); // UNIQUE
        }
        w.advance(); // INDEX
        if w.peek_upper() == "IF" {
            w.advance(); // IF
            w.advance(); // NOT
            w.advance(); // EXISTS
        }
        let index_name = w.advance();
        w.advance(); // ON
        let table_name = w.advance();
        w.advance(); // (
        let col_name = w.advance();
        w.advance(); // )
        let is_partial = w.peek_upper() == "WHERE";
        self.indexes
            .entry(table_name)
            .or_default()
            .push((index_name, col_name, is_partial, is_unique));
        Ok(())
    }

    /// `PRAGMA index_list(table)` rows, sqlite3-shaped:
    /// `(seq, name, unique, origin, partial)`, newest index first. Only
    /// explicitly created indexes are listed (this engine materialises no
    /// `sqlite_autoindex_*` rows for UNIQUE column constraints). An unknown
    /// table yields an empty list, matching sqlite3.
    pub fn index_list(&self, table: &str) -> Vec<(i64, String, bool, String, bool)> {
        let Some(entries) = self.indexes.get(table) else {
            return Vec::new();
        };
        entries
            .iter()
            .rev()
            .enumerate()
            .map(|(seq, (name, _col, is_partial, is_unique))| {
                (
                    seq as i64,
                    name.clone(),
                    *is_unique,
                    "c".to_string(),
                    *is_partial,
                )
            })
            .collect()
    }
}

/// Parse a `DEFAULT <literal>` value into a stored cell.
///
/// A hex blob literal `x'<hex>'` arrives as two tokens (an `x` ident followed by
/// a quoted string) when the DDL was round-tripped through the space-joined raw
/// form, so the `x` prefix is recognized and the following quoted run is decoded
/// to bytes. A single-quoted string strips its quotes; `NULL` becomes
/// `Value::Null`; an unquoted numeric literal becomes Int or Float.
fn parse_default(w: &mut Walker) -> Value {
    // A blob literal `x'<hex>'` re-tokenizes as a single Blob token; decode its
    // hex body. (When the raw text inserted a space — `x ''` — it instead
    // arrives as an `x` ident followed by a quoted string; both forms decode to
    // the same blob.)
    if w.peek_is_blob() {
        let blob_tok = w.advance();
        let inner = blob_tok
            .strip_prefix('x')
            .or_else(|| blob_tok.strip_prefix('X'))
            .unwrap_or(&blob_tok)
            .trim_matches('\'');
        return match decode_hex(inner) {
            Some(bytes) => Value::Blob(bytes),
            None => Value::Blob(Vec::new()),
        };
    }
    let first = w.advance();
    if first.eq_ignore_ascii_case("x") && w.peek_is_string() {
        let hex_str = w.advance();
        let inner = hex_str.trim_matches('\'');
        return match decode_hex(inner) {
            Some(bytes) => Value::Blob(bytes),
            None => Value::Blob(Vec::new()),
        };
    }
    if first.starts_with('\'') && first.ends_with('\'') && first.len() >= 2 {
        return Value::Text(first[1..first.len() - 1].to_string());
    }
    if first.eq_ignore_ascii_case("NULL") {
        return Value::Null;
    }
    if let Ok(i) = first.parse::<i64>() {
        return Value::Int(i);
    }
    if let Ok(f) = first.parse::<f64>() {
        return Value::Float(f);
    }
    Value::Text(first)
}

/// Rewrite the table name in a stored CREATE TABLE DDL after a RENAME, replacing
/// the name immediately after `CREATE TABLE` (honoring an `IF NOT EXISTS`
/// clause) with `new_name`. sqlite3 keeps the rest of the DDL verbatim.
fn rewrite_ddl_table_name(sql: &str, old_name: &str, new_name: &str) -> String {
    let upper = sql.to_ascii_uppercase();
    let Some(kw_pos) = upper.find("CREATE TABLE") else {
        return sql.to_string();
    };
    let after = kw_pos + "CREATE TABLE".len();
    let head = &sql[..after];
    let tail = &sql[after..];
    // Skip leading whitespace / an optional IF NOT EXISTS before the name.
    let trimmed = tail.trim_start();
    let lead_ws = &tail[..tail.len() - trimmed.len()];
    let rest_upper = trimmed.to_ascii_uppercase();
    let (ine, body) = if let Some(stripped) = rest_upper.strip_prefix("IF NOT EXISTS") {
        let consumed = trimmed.len() - stripped.len();
        (&trimmed[..consumed], &trimmed[consumed..])
    } else {
        ("", trimmed)
    };
    let body_trimmed = body.trim_start();
    let body_ws = &body[..body.len() - body_trimmed.len()];
    // The table name at the head of the body may be bare (`t`) or bracket-quoted
    // (`[t]`). Identify the whole name token's byte span and its unwrapped value,
    // then replace the entire token — a byte-prefix compare against the unwrapped
    // name would miss a `[t]` whose first byte is `[`, not `t`.
    if let Some((unwrapped, token_len)) = leading_name_token(body_trimmed) {
        if unwrapped.eq_ignore_ascii_case(old_name) {
            let suffix = &body_trimmed[token_len..];
            return format!("{head}{lead_ws}{ine}{body_ws}{new_name}{suffix}");
        }
    }
    sql.to_string()
}

/// The table-name token at the start of `s`, returned as its unwrapped value and
/// the byte length of the whole token in `s` (including any surrounding brackets).
/// Handles a bare identifier and a bracket-quoted `[name]`; returns `None` when no
/// identifier-shaped token starts `s`.
fn leading_name_token(s: &str) -> Option<(String, usize)> {
    let mut chars = s.char_indices();
    match chars.next() {
        // Bracket-quoted `[name]`: the token is everything through the closing `]`.
        Some((_, '[')) => {
            let close_rel = s[1..].find(']')?;
            let close = 1 + close_rel; // byte index of `]` in `s`
            if close <= 1 {
                return None; // empty `[]` is not a name
            }
            let name = s[1..close].to_string();
            Some((name, close + 1))
        }
        // Bare identifier: [A-Za-z_][A-Za-z0-9_]*.
        Some((_, c)) if c == '_' || c.is_ascii_alphabetic() => {
            let mut end = s.len();
            for (idx, ch) in s.char_indices() {
                if idx == 0 {
                    continue;
                }
                if !(ch == '_' || ch.is_ascii_alphanumeric()) {
                    end = idx;
                    break;
                }
            }
            Some((s[..end].to_string(), end))
        }
        _ => None,
    }
}

/// Decode a hex string to bytes; an empty string yields an empty blob.
fn decode_hex(s: &str) -> Option<Vec<u8>> {
    if s.is_empty() {
        return Some(Vec::new());
    }
    if s.len() % 2 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for pair in bytes.chunks(2) {
        let hi = (pair[0] as char).to_digit(16)?;
        let lo = (pair[1] as char).to_digit(16)?;
        out.push((hi * 16 + lo) as u8);
    }
    Some(out)
}

/// A forward token cursor over a tokenized DDL string, mirroring the reference
/// catalog's manual token walk.
struct Walker<'t> {
    tokens: &'t [Token],
    pos: usize,
}

impl<'t> Walker<'t> {
    fn new(tokens: &'t [Token]) -> Self {
        Walker { tokens, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len() || matches!(self.tokens.get(self.pos), Some(Token::Eof))
    }

    /// The raw (case-preserving) value of the token at the cursor.
    fn peek_raw(&self) -> String {
        match self.tokens.get(self.pos) {
            Some(t) => token_text(t),
            None => String::new(),
        }
    }

    /// The upper-cased value of the token at the cursor.
    fn peek_upper(&self) -> String {
        self.peek_raw().to_ascii_uppercase()
    }

    /// True when the cursor sits on a string-literal token (`'...'`).
    fn peek_is_string(&self) -> bool {
        matches!(self.tokens.get(self.pos), Some(Token::Str(_)))
    }

    /// True when the cursor sits on a blob-literal token (`x'..'`).
    fn peek_is_blob(&self) -> bool {
        matches!(self.tokens.get(self.pos), Some(Token::Blob(_)))
    }

    /// True when the cursor sits on a token that can serve as a column type
    /// (a keyword or identifier that is not a constraint keyword or delimiter).
    fn peek_is_type_token(&self) -> bool {
        match self.tokens.get(self.pos) {
            Some(Token::Keyword(_)) | Some(Token::Ident(_)) => !matches!(
                self.peek_upper().as_str(),
                "," | ")"
                    | "PRIMARY"
                    | "NOT"
                    | "UNIQUE"
                    | "DEFAULT"
                    | "REFERENCES"
                    | "CHECK"
                    | "CONSTRAINT"
                    | "AUTOINCREMENT"
            ),
            _ => false,
        }
    }

    /// Return the current token's raw value and advance.
    fn advance(&mut self) -> String {
        let v = self.peek_raw();
        self.pos += 1;
        v
    }

    /// Advance past a token whose upper-cased value must equal `value`.
    fn expect(&mut self, value: &str) -> Result<()> {
        let got = self.advance().to_ascii_uppercase();
        if got != value.to_ascii_uppercase() {
            return Err(EngineError::parse(format!(
                "DDL: expected {value:?}, got {got:?}"
            )));
        }
        Ok(())
    }
}

/// The raw source text of a token, matching the parser's `token_value` rendering
/// (string/blob keep their surrounding quotes; punctuation is its character).
fn token_text(tok: &Token) -> String {
    match tok {
        Token::Keyword(s) | Token::Ident(s) | Token::Op(s) | Token::Str(s) | Token::Blob(s) => {
            s.clone()
        }
        Token::Integer(v) => v.to_string(),
        Token::Real(f) => f.to_string(),
        Token::Placeholder => "?".to_string(),
        Token::NamedParam(name) => format!(":{name}"),
        Token::Punct(c) => c.to_string(),
        Token::Eof => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_ddl_str(cat: &mut Catalog, sql: &str) {
        let stmt = crate::parser::parse(sql).expect("parse failed");
        cat.apply_ddl(&stmt).expect("apply_ddl failed");
    }

    #[test]
    fn indexed_columns_non_partial_index() {
        let mut cat = Catalog::new();
        apply_ddl_str(&mut cat, "CREATE TABLE t (id TEXT, tier TEXT, n INTEGER)");
        apply_ddl_str(&mut cat, "CREATE INDEX idx_t_tier ON t(tier)");

        let cols = cat.indexed_columns("t");
        assert!(
            cols.iter().any(|(c, partial)| c == "tier" && !partial),
            "expected (\"tier\", false) in indexed_columns; got {cols:?}"
        );
    }

    #[test]
    fn indexed_columns_partial_index() {
        let mut cat = Catalog::new();
        apply_ddl_str(&mut cat, "CREATE TABLE t (id TEXT, embedding_pending INTEGER)");
        apply_ddl_str(
            &mut cat,
            "CREATE INDEX idx_t_pending ON t(embedding_pending) WHERE embedding_pending=1",
        );

        let cols = cat.indexed_columns("t");
        assert!(
            cols.iter().any(|(c, partial)| c == "embedding_pending" && *partial),
            "expected (\"embedding_pending\", true) for partial index; got {cols:?}"
        );
    }

    #[test]
    fn indexed_columns_multi_column_leading_only() {
        let mut cat = Catalog::new();
        apply_ddl_str(
            &mut cat,
            "CREATE TABLE budget_ledger (date TEXT, kind TEXT, amount INTEGER)",
        );
        apply_ddl_str(
            &mut cat,
            "CREATE INDEX idx_budget_date_kind ON budget_ledger(date, kind)",
        );

        let cols = cat.indexed_columns("budget_ledger");
        // Leading column "date" should appear as non-partial.
        assert!(
            cols.iter().any(|(c, partial)| c == "date" && !partial),
            "expected leading column \"date\" as non-partial; got {cols:?}"
        );
        // The trailing column "kind" should NOT appear (only leading column captured).
        assert!(
            !cols.iter().any(|(c, _)| c == "kind"),
            "trailing column \"kind\" must not appear in indexed_columns; got {cols:?}"
        );
    }

    #[test]
    fn indexed_columns_unknown_table_returns_empty() {
        let cat = Catalog::new();
        let cols = cat.indexed_columns("no_such_table");
        assert!(cols.is_empty(), "expected empty Vec for unknown table; got {cols:?}");
    }

    #[test]
    fn indexed_columns_dedup_prefers_non_partial() {
        // Same column indexed twice: once full, once partial. Non-partial wins.
        let mut cat = Catalog::new();
        apply_ddl_str(&mut cat, "CREATE TABLE t (id TEXT, x INTEGER)");
        apply_ddl_str(&mut cat, "CREATE INDEX idx_t_x_partial ON t(x) WHERE x=1");
        apply_ddl_str(&mut cat, "CREATE INDEX idx_t_x_full ON t(x)");

        let cols = cat.indexed_columns("t");
        let entry = cols.iter().find(|(c, _)| c == "x");
        assert!(entry.is_some(), "expected \"x\" in indexed_columns");
        assert!(!entry.unwrap().1, "non-partial index must win dedup; got is_partial=true");
        assert_eq!(cols.iter().filter(|(c, _)| c == "x").count(), 1, "dedup: only one entry for x");
    }
}

//! The SELECT read pipeline over the record store.
//!
//! Executes a [`SelectPlan`] against a [`lillibrain::Store`] data tree, returning
//! result rows and their output column names. The pipeline mirrors the
//! reference's Volcano operators as Rust passes over the materialized scan:
//! sequential scan → WHERE filter (three-valued) → projection → ORDER BY
//! (NULLs-first, stable, DESC-aware, datetime-normalized) → LIMIT / OFFSET. The
//! scalar aggregates (COUNT(*) with a decode-free fast path, SUM / MIN / MAX) and
//! the GROUP BY + HAVING COUNT(*) paths are handled before the row pipeline.
//!
//! `rowid` resolves to the `vec_label` column when the table has one (the
//! AUTOINCREMENT high-water mark, equal to sqlite3's rowid for an INTEGER PRIMARY
//! KEY table), else to the raw B-tree key. The `dbstat` virtual table is answered
//! by a constant size estimate without a cursor.

use std::collections::{BTreeMap, HashMap};

use lillibrain::{Store, Value};

use crate::ast::{DeleteStmt, Expr, InsertStmt, UpdateStmt};
use crate::catalog::{coerce_affinity, Catalog, ColumnMeta};
use crate::error::{EngineError, Result};
use crate::eval::{
    eval_predicate, eval_scalar, order_key, sort_value, sql_compare, DirectedKey, Row, SortKey,
};
use crate::meta::MetaTable;
use crate::plan::{
    too_few_bindings_error, too_many_bindings_error, ScanType, SelectPlan,
};
use crate::rowcodec::{decode_row, encode_row};

/// The pseudo-column name for the row identifier in a query.
const ROWID: &str = "rowid";
/// The column carrying the AUTOINCREMENT high-water mark (the `records` table).
const ROWID_SOURCE: &str = "vec_label";
/// The reserved row key threading the raw B-tree key for rowid fallback.
const ROWKEY: &str = "_rowkey";

/// One result column's value, keyed by name, in output order.
pub type OutRow = Vec<(String, Value)>;

/// The outcome of executing a SELECT: the result rows and their column names.
#[derive(Debug, Clone, PartialEq)]
pub struct ResultSet {
    /// The result rows; each is an ordered list of `(column, value)` pairs.
    pub rows: Vec<OutRow>,
    /// The output column names in result order.
    pub columns: Vec<String>,
}

/// A scan counter snapshot, used to assert the COUNT(*) / point-lookup fast paths
/// avoid a full row materialization in tests.
#[derive(Debug, Clone, Copy)]
pub struct ScanStats {
    /// True when the executor decoded full row payloads during the query.
    pub decoded_rows: bool,
}

/// Execute a planned SELECT against `store`, resolving the table's root page from
/// `root_map` and its columns from `catalog`.
pub fn execute_select(
    plan: &SelectPlan,
    store: &Store,
    catalog: &Catalog,
    root_map: &std::collections::BTreeMap<String, u32>,
    id_index: Option<&mut IdIndex>,
    col_index: Option<&mut ColIndex>,
    ordered_index: Option<&mut OrderedColIndex>,
) -> Result<ResultSet> {
    let (rs, _) = execute_select_instrumented(
        plan, store, catalog, root_map, id_index, col_index, ordered_index,
    )?;
    Ok(rs)
}

/// Execute a SELECT and also report whether full row payloads were decoded, so a
/// test can confirm the COUNT(*) fast path counts without a row decode.
pub fn execute_select_instrumented(
    plan: &SelectPlan,
    store: &Store,
    catalog: &Catalog,
    root_map: &std::collections::BTreeMap<String, u32>,
    mut id_index: Option<&mut IdIndex>,
    col_index: Option<&mut ColIndex>,
    mut ordered_index: Option<&mut OrderedColIndex>,
) -> Result<(ResultSet, ScanStats)> {
    if plan.scan_type == ScanType::Constant {
        return Ok((execute_constant(plan, store)?, ScanStats { decoded_rows: false }));
    }
    if plan.scan_type == ScanType::SqliteMaster {
        return Ok((execute_sqlite_master(plan, catalog)?, ScanStats { decoded_rows: true }));
    }

    let table = &plan.table;
    let root = *root_map
        .get(table)
        .ok_or_else(|| EngineError::parse(format!("execute: unknown table {table:?}")))?;
    let col_names = catalog.column_names(table)?;

    // Bind placeholders in textual order: select-list exprs, then WHERE, then
    // LIMIT, then OFFSET. A count mismatch fails loud like the reference.
    let mut binder = Binder::new(plan.params.clone(), plan.named_params.clone());
    let bound_select_exprs: Vec<(Expr, String)> = plan
        .select_exprs
        .iter()
        .map(|se| Ok((binder.bind(&se.expr)?, se.output_name.clone())))
        .collect::<Result<_>>()?;
    let bound_where = match &plan.where_clause {
        Some(w) => Some(binder.bind(w)?),
        None => None,
    };
    let effective_limit = binder.resolve_limit(plan.limit, plan.limit_is_param)?;
    let effective_offset = binder.resolve_limit(plan.offset, plan.offset_is_param)?;
    binder.finish()?;

    // Aliased / COALESCE projection path. When the WHERE reduces to an indexed
    // `id IN (...)` shape, resolve the candidate rows from the ColIndex (a union
    // of O(1) probes) so the wide batch-get projects only the matched rows instead
    // of decoding the whole table. The full predicate is re-applied to each
    // candidate, so the result is byte-identical to the scan path.
    if !bound_select_exprs.is_empty() {
        let indexed_rows = match (&bound_where, col_index) {
            (Some(where_expr), Some(cidx)) => {
                indexed_candidate_rows(store, root, &col_names, where_expr, cidx)?
            }
            _ => None,
        };
        let rs = execute_select_exprs(
            store, root, &col_names, &bound_select_exprs, &bound_where, effective_limit,
            effective_offset, indexed_rows, plan,
        )?;
        return Ok((rs, ScanStats { decoded_rows: true }));
    }

    // GROUP BY path.
    if plan.group_by_col.is_some() {
        let rs = execute_group_by(plan, store, root, &col_names, &bound_where, effective_limit, effective_offset)?;
        return Ok((rs, ScanStats { decoded_rows: true }));
    }

    // COUNT(*) fast path: a bare COUNT(*) with no WHERE counts keys without
    // decoding any row payload; a filtered COUNT(*) decodes only to evaluate the
    // predicate.
    if plan.aggregate.as_deref() == Some("count")
        && plan.group_by_col.is_none()
        && plan.order_by.is_none()
    {
        let label = plan.count_alias.clone().unwrap_or_else(|| "COUNT(*)".to_string());
        if bound_where.is_none() {
            let count = count_rows_no_decode(store, root)?;
            return Ok((
                ResultSet {
                    rows: vec![vec![(label.clone(), Value::Int(count))]],
                    columns: vec![label],
                },
                ScanStats { decoded_rows: false },
            ));
        }
        let where_expr = bound_where.as_ref().unwrap();
        // Derive the minimal column mask for the predicate; decode only those
        // columns per row — the embedding BLOB is never decoded for a narrow
        // WHERE. Does NOT call full_scan(), so full_scan_count stays at zero.
        let pred_cols = predicate_columns(Some(where_expr));
        let pred_mask = mask_from_names(&col_names, &pred_cols);
        let mut count = 0i64;
        store
            .tree(root)
            .scan_cells_with(|key, payload| {
                let mut vals = crate::rowcodec::decode_row_columns(&payload, &pred_mask)?;
                while vals.len() < col_names.len() {
                    vals.push(Value::Null);
                }
                let mut pairs: Vec<(String, Value)> = col_names
                    .iter()
                    .cloned()
                    .zip(vals.into_iter())
                    .collect();
                pairs.push((ROWKEY.to_string(), Value::Int(key)));
                let row = Row::from_pairs(pairs);
                if eval_predicate(where_expr, &row) == Some(true) {
                    count += 1;
                }
                Ok(())
            })
            .map_err(|e: EngineError| e)?;
        return Ok((
            ResultSet {
                rows: vec![vec![(label.clone(), Value::Int(count))]],
                columns: vec![label],
            },
            ScanStats { decoded_rows: true },
        ));
    }

    // MIN/MAX index-seek fast path (sqlite's min/max optimization). Costs
    // O(skipped + 1) row decodes instead of a full table decode; the NULL
    // bucket is skipped. Feeds the same `aggregate_scalar` the scan path uses
    // and preserves its tie-break order, so the result is byte-identical to
    // the scan path.
    if let Some(agg) = plan.aggregate.as_deref() {
        if (agg == "min" || agg == "max") && plan.group_by_col.is_none() {
            if let (Some(col), Some(oidx)) =
                (plan.aggregate_column.as_deref(), ordered_index.as_deref_mut())
            {
                if col == oidx.column() {
                    let tree = store.tree(root);
                    oidx.ensure_built(&tree, &col_names)?;
                    let null_tag = order_key(&Value::Null);
                    let lower = ordered_walk_lower_bound(bound_where.as_ref(), col);
                    let mut matched: Vec<Row> = Vec::new();
                    'minmax: for (tag, keys) in oidx.walk_buckets(agg == "max", lower.as_ref()) {
                        if *tag == null_tag {
                            continue;
                        }
                        for &key in keys.iter() {
                            if let Some(payload) = tree.get(key).map_err(store_err)? {
                                let mut row = decode_to_row(&payload, &col_names)?;
                                row.set(ROWKEY.to_string(), Value::Int(key));
                                let pass = bound_where
                                    .as_ref()
                                    .map(|w| eval_predicate(w, &row) == Some(true))
                                    .unwrap_or(true);
                                if pass {
                                    matched.push(row);
                                    break 'minmax;
                                }
                            }
                        }
                    }
                    let rs = aggregate_scalar(
                        agg,
                        plan.aggregate_column.as_deref(),
                        &matched,
                        effective_limit,
                        plan.count_alias.as_deref(),
                    );
                    return Ok((rs, ScanStats { decoded_rows: true }));
                }
            }
        }
    }

    // Point-lookup fast path: WHERE id = '<literal>' with LIMIT >= 1. The records
    // table keys rows by vec_label, so a lookup by `id` has no B-tree key descent
    // of its own. When an id-index is supplied the engine resolves the row by an
    // O(1) map probe plus one O(tree-height) `tree.get(row_key)` — the same access
    // path stdlib sqlite3 takes through `idx_records_id`. Without the index it
    // falls back to a single equality scan that stops at the first match. The
    // index is built once (lazily) and maintained by the write paths, so a recall
    // that issues dozens of per-candidate id lookups stays O(rows + lookups·logN)
    // instead of O(rows · lookups).
    if let (Some(where_expr), Some(lim)) = (&bound_where, effective_limit) {
        if lim >= 1 {
            if let Some(id_val) = extract_id_eq(where_expr) {
                let tree = store.tree(root);
                let matched: Vec<Row> = if let Some(idx) = id_index.as_mut() {
                    idx.ensure_built(&tree, &col_names)?;
                    match idx.lookup(&id_val) {
                        Some(row_key) => match tree.get(row_key).map_err(store_err)? {
                            Some(payload) => {
                                let mut row = decode_to_row(&payload, &col_names)?;
                                // Thread the raw B-tree key for rowid resolution,
                                // matching scan_rows so projection is identical.
                                row.set(ROWKEY.to_string(), Value::Int(row_key));
                                if eval_predicate(where_expr, &row) == Some(true) {
                                    vec![row]
                                } else {
                                    Vec::new()
                                }
                            }
                            // The index pointed at a row that is gone (a consistency
                            // lapse): fall back to the safe scan rather than miss it.
                            None => first_id_match_by_scan(
                                store, root, &col_names, &id_val, where_expr,
                            )?,
                        },
                        None => Vec::new(), // id not present: no row
                    }
                } else {
                    first_id_match_by_scan(store, root, &col_names, &id_val, where_expr)?
                };
                let out = project_rows(matched, &plan.columns, &col_names);
                let cols = output_columns(&plan.columns, &col_names);
                return Ok((ResultSet { rows: out, columns: cols }, ScanStats { decoded_rows: true }));
            }
        }
    }

    // `id IN (text-literals)` fast path: when the WHERE is a pure `id IN (...)`
    // over text literals and the id-index is available, resolve each id to its
    // row-key by an O(1) id-index probe instead of building the ColIndex with a
    // full-table scan.  Dedup the keys, decode each row, and re-apply the full
    // bound predicate — the index only narrows which rows are decoded, so the
    // result is byte-identical to the scan path.  A key that the tree no longer
    // holds (a consistency lapse) falls back to the per-id safe scan, mirroring
    // the single-id point-lookup fallback, so a stale index entry never drops a
    // real row.
    if let (Some(where_expr), Some(idx)) = (&bound_where, id_index.as_mut()) {
        if let Some(id_vals) = extract_id_in(where_expr) {
            let tree = store.tree(root);
            idx.ensure_built(&tree, &col_names)?;
            let mut seen_keys: std::collections::HashSet<i64> = std::collections::HashSet::new();
            let mut candidate_keys: Vec<(String, i64)> = Vec::new();
            for id_val in &id_vals {
                if let Some(row_key) = idx.lookup(id_val) {
                    if seen_keys.insert(row_key) {
                        candidate_keys.push((id_val.clone(), row_key));
                    }
                }
                // An id absent from the index yields no candidate (genuine miss).
            }
            let mut rows: Vec<Row> = Vec::with_capacity(candidate_keys.len());
            for (id_val, key) in candidate_keys {
                match tree.get(key).map_err(store_err)? {
                    Some(payload) => {
                        let mut row = decode_to_row(&payload, &col_names)?;
                        row.set(ROWKEY.to_string(), Value::Int(key));
                        if eval_predicate(where_expr, &row) == Some(true) {
                            rows.push(row);
                        }
                    }
                    // The index pointed at a row that is gone (a consistency lapse):
                    // fall back to the safe per-id scan rather than dropping the row.
                    None => {
                        let fallback = first_id_match_by_scan(
                            store, root, &col_names, &id_val, where_expr,
                        )?;
                        rows.extend(fallback);
                    }
                }
            }
            let rs = finish_row_pipeline(rows, plan, &col_names, effective_limit, effective_offset);
            return Ok((rs, ScanStats { decoded_rows: true }));
        }
    }

    // Indexed `col IN (...)` fast path (the recall edge-adjacency lookup): when a
    // ColIndex covers the columns the WHERE filters with `IN`, resolve the
    // candidate row-keys from the index (a union of O(1) probes) and decode only
    // those rows, instead of scanning the whole table. The full predicate is then
    // re-applied to each candidate, so the result is identical to the scan path
    // (the index only narrows which rows are decoded). Disjoint from the aggregate
    // / scalar paths above; reached only for a plain row SELECT.
    if let (Some(where_expr), Some(cidx)) = (&bound_where, col_index) {
        if let Some(in_cols) = indexed_in_columns(where_expr, cidx) {
            let tree = store.tree(root);
            cidx.ensure_built(&tree, &col_names)?;
            // Collect candidate keys: the union over every (indexed-column, value)
            // probe. A row may be referenced by several values (src and dst) — dedup.
            let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
            let mut candidate_keys: Vec<i64> = Vec::new();
            for (col, value) in &in_cols {
                for &k in cidx.probe(col, value) {
                    if seen.insert(k) {
                        candidate_keys.push(k);
                    }
                }
            }
            // Fetch all candidates in ONE ordered leaf-chain walk (sorted keys,
            // O(pages spanned) instead of one O(height) descent per key — a
            // hub-heavy adjacency fetch resolves thousands of candidates).
            // Sorting also makes the pre-ORDER row order equal the scan path's
            // ascending key order. Then apply the FULL predicate (e.g. the
            // `AND edge_type IN (...)` tail and the exact OR membership) so
            // the result equals the scan path.
            candidate_keys.sort_unstable();
            let fetched = tree.get_many(&candidate_keys).map_err(store_err)?;
            let mut rows: Vec<Row> = Vec::with_capacity(fetched.len());
            for (key, payload) in fetched {
                let mut row = decode_to_row(&payload, &col_names)?;
                row.set(ROWKEY.to_string(), Value::Int(key));
                if eval_predicate(where_expr, &row) == Some(true) {
                    rows.push(row);
                }
            }
            let rs = finish_row_pipeline(rows, plan, &col_names, effective_limit, effective_offset);
            return Ok((rs, ScanStats { decoded_rows: true }));
        }
    }

    // Ordered top-K fast path: `ORDER BY <indexed-col> DESC LIMIT k` narrows
    // the candidate set via the BTreeMap index, then delegates to the EXISTING
    // `order_rows` + truncate pipeline for byte-identical final order.
    //
    // The index only narrows which rows are decoded; it never emits BTreeMap
    // iteration order as the result.  The whole-bucket gather rule ensures the
    // candidate superset is always complete for `order_rows` + truncate to
    // produce the correct result, even when LIMIT cuts inside a tie group.
    if let Some(oidx) = ordered_index.as_deref_mut() {
        if let Some(spec) = extract_ordered_topk(plan, oidx, effective_limit, effective_offset) {
            let tree = store.tree(root);
            oidx.ensure_built(&tree, &col_names)?;
            let limit_usize = spec.limit_hint;
            let candidate_keys =
                oidx.candidates_desc(limit_usize, spec.ge_bound.as_deref());
            let mut rows: Vec<Row> = Vec::with_capacity(candidate_keys.len());
            for key in candidate_keys {
                if let Some(payload) = tree.get(key).map_err(store_err)? {
                    let mut row = decode_to_row(&payload, &col_names)?;
                    row.set(ROWKEY.to_string(), Value::Int(key));
                    // Re-apply the FULL bound predicate (including any equality
                    // conjuncts such as kind=? or severity=? that the index did
                    // not filter).  This is what produces byte-identical results.
                    if bound_where
                        .as_ref()
                        .map(|w| eval_predicate(w, &row) == Some(true))
                        .unwrap_or(true)
                    {
                        rows.push(row);
                    }
                }
            }
            let rs = finish_row_pipeline(rows, plan, &col_names, effective_limit, effective_offset);
            return Ok((rs, ScanStats { decoded_rows: true }));
        }
    }

    // General ordered LIMIT-k walk (predicate-aware): used when the strict
    // shape above declined because the WHERE carries residual conjuncts on
    // other columns. Early-stopping on the PASSING count (never the candidate
    // count) is what makes an arbitrary residual predicate sound — a passing
    // row can never hide below the stop point, because the walk only stops
    // after k passing rows are in hand and the current tie group is complete.
    // Only a plain (non-`datetime()`) single-key ORDER BY on the indexed
    // column qualifies; the walk start is seeded from any `col >= bound`
    // conjunct so an ascending bounded probe never decodes below its bound.
    if let Some(oidx) = ordered_index {
        if let (Some(k1), Some(lim)) = (plan.order_by.as_ref(), effective_limit) {
            if k1.col == oidx.column()
                && !k1.datetime
                && plan.order_by2.is_none()
                && plan.aggregate.is_none()
                && lim > 0
            {
                let target = lim as usize + effective_offset.unwrap_or(0).max(0) as usize;
                let tree = store.tree(root);
                oidx.ensure_built(&tree, &col_names)?;
                let lower = ordered_walk_lower_bound(bound_where.as_ref(), &k1.col);
                let mut rows: Vec<Row> = Vec::new();
                for (_tag, keys) in oidx.walk_buckets(k1.desc, lower.as_ref()) {
                    for &key in keys.iter() {
                        if let Some(payload) = tree.get(key).map_err(store_err)? {
                            let mut row = decode_to_row(&payload, &col_names)?;
                            row.set(ROWKEY.to_string(), Value::Int(key));
                            let pass = bound_where
                                .as_ref()
                                .map(|w| eval_predicate(w, &row) == Some(true))
                                .unwrap_or(true);
                            if pass {
                                rows.push(row);
                            }
                        }
                    }
                    if rows.len() >= target {
                        break; // bucket boundary: the tie group is complete
                    }
                }
                let rs =
                    finish_row_pipeline(rows, plan, &col_names, effective_limit, effective_offset);
                return Ok((rs, ScanStats { decoded_rows: true }));
            }
        }
    }

    // A plain-row `LIMIT 0` returns zero rows with the full column list — used as
    // a column-shape probe (e.g. counting a table's columns before an insert). The
    // result is empty regardless of table contents, so short-circuit BEFORE the
    // scan: reading, decoding, then discarding every row for a guaranteed-empty
    // result is pure waste and, on a large table, a full-scan tax per probe. This
    // matches the reference, which serves `LIMIT 0` from the schema without
    // touching a row. Only the plain-row path is short-circuited here — a scalar
    // aggregate still collapses to its single-row result below, unaffected.
    if effective_limit == Some(0) && plan.aggregate.is_none() {
        return Ok((
            ResultSet { rows: Vec::new(), columns: output_columns(&plan.columns, &col_names) },
            ScanStats { decoded_rows: false },
        ));
    }

    // Standard scan → filter → (aggregate | order → project → limit).
    //
    // Column-selective decode is safe only when the query projects a specific
    // named subset (not SELECT *) and is not a scalar aggregate: a MAX/MIN/SUM
    // on a named column needs that column decoded regardless of the WHERE. The
    // filtered-COUNT fast path above already handles COUNT(*) before this
    // point, so aggregate != None here means MIN/MAX/SUM (always full decode).
    let use_selective = !plan.columns.is_empty()
        && plan.columns != ["*"]
        && plan.aggregate.is_none();
    let mut rows: Vec<Row> = if use_selective {
        let mut needed: std::collections::HashSet<String> = predicate_columns(bound_where.as_ref());
        // Add projected columns.
        for c in &plan.columns {
            if c != "*" && c != ROWID {
                needed.insert(c.clone());
            }
        }
        // Add ORDER BY columns so the sort key is present on each row.
        if let Some(k) = &plan.order_by {
            if k.col != ROWID {
                needed.insert(k.col.clone());
            }
        }
        if let Some(k) = &plan.order_by2 {
            if k.col != ROWID {
                needed.insert(k.col.clone());
            }
        }
        let mask = mask_from_names(&col_names, &needed);
        let mut out: Vec<Row> = Vec::new();
        store
            .tree(root)
            .scan_cells_with(|key, payload| {
                let mut vals = crate::rowcodec::decode_row_columns(&payload, &mask)?;
                while vals.len() < col_names.len() {
                    vals.push(Value::Null);
                }
                let mut pairs: Vec<(String, Value)> = col_names
                    .iter()
                    .cloned()
                    .zip(vals.into_iter())
                    .collect();
                pairs.push((ROWKEY.to_string(), Value::Int(key)));
                out.push(Row::from_pairs(pairs));
                Ok(())
            })
            .map_err(|e: EngineError| e)?;
        if let Some(where_expr) = &bound_where {
            out.retain(|r| eval_predicate(where_expr, r) == Some(true));
        }
        out
    } else {
        let mut r = scan_rows(store, root, &col_names)?;
        if let Some(where_expr) = &bound_where {
            r.retain(|r| eval_predicate(where_expr, r) == Some(true));
        }
        r
    };

    // Scalar aggregates collapse to a single row (ORDER BY is irrelevant).
    if let Some(agg) = plan.aggregate.as_deref() {
        let rs = aggregate_scalar(agg, plan.aggregate_column.as_deref(), &rows, effective_limit, plan.count_alias.as_deref());
        return Ok((rs, ScanStats { decoded_rows: true }));
    }

    // ORDER BY before projection and LIMIT.
    if plan.order_by.is_some() {
        order_rows(&mut rows, plan, &col_names);
    }

    // rowid resolution for the projection.
    if plan.columns.iter().any(|c| c == ROWID) {
        for r in &mut rows {
            resolve_rowid(r);
        }
    }

    // OFFSET then LIMIT.
    if let Some(off) = effective_offset {
        if off > 0 {
            let off = off as usize;
            rows = if off >= rows.len() { Vec::new() } else { rows.split_off(off) };
        }
    }
    if let Some(lim) = effective_limit {
        if lim > 0 && (lim as usize) < rows.len() {
            rows.truncate(lim as usize);
        }
    }

    let out = project_rows(rows, &plan.columns, &col_names);
    let cols = output_columns(&plan.columns, &col_names);
    Ok((ResultSet { rows: out, columns: cols }, ScanStats { decoded_rows: true }))
}

/// The shared tail of the plain-row SELECT pipeline: ORDER BY → (LIMIT 0) →
/// rowid resolution → OFFSET → LIMIT → projection. Byte-identical to the standard
/// scan path's tail, so the indexed `IN` fast path produces the same ResultSet.
fn finish_row_pipeline(
    mut rows: Vec<Row>,
    plan: &SelectPlan,
    col_names: &[String],
    effective_limit: Option<i64>,
    effective_offset: Option<i64>,
) -> ResultSet {
    if plan.order_by.is_some() {
        order_rows(&mut rows, plan, col_names);
    }
    if effective_limit == Some(0) {
        return ResultSet { rows: Vec::new(), columns: output_columns(&plan.columns, col_names) };
    }
    if plan.columns.iter().any(|c| c == ROWID) {
        for r in &mut rows {
            resolve_rowid(r);
        }
    }
    if let Some(off) = effective_offset {
        if off > 0 {
            let off = off as usize;
            rows = if off >= rows.len() { Vec::new() } else { rows.split_off(off) };
        }
    }
    if let Some(lim) = effective_limit {
        if lim > 0 && (lim as usize) < rows.len() {
            rows.truncate(lim as usize);
        }
    }
    let out = project_rows(rows, &plan.columns, col_names);
    let cols = output_columns(&plan.columns, col_names);
    ResultSet { rows: out, columns: cols }
}

/// Spec produced by the ordered-index detector when the query shape is narrowable.
struct OrderedTopkSpec {
    /// The minimum candidate count to gather (= the effective LIMIT).  The
    /// gather always completes whole buckets, so the actual count may exceed this.
    limit_hint: usize,
    /// When the WHERE carries a `ts >= <literal>` lower bound, its text value.
    /// `candidates_desc` uses it as the BTreeMap range lower bound.
    ge_bound: Option<String>,
}

/// Detect whether the plan targets an ordered top-K shape the `OrderedColIndex`
/// can narrow: `ORDER BY <indexed-col> DESC LIMIT k`, with an optional
/// `<indexed-col> >= <literal>` lower bound in the WHERE.
///
/// Returns `Some(spec)` only when the gathered candidate superset is PROVABLY
/// complete — i.e. every row the scan path would return is guaranteed to be
/// among the candidates `candidates_desc` collects, in the right order.  That
/// holds only under a strict shape:
///   - The plan has `ORDER BY <col> DESC` where `<col>` is the index column.
///   - The `order_by2` secondary key is absent (a second ORDER BY key over a
///     non-gathered column would need the whole table; fall back to scan).
///   - `LIMIT` is present and positive.
///   - `OFFSET` is absent or zero.  A positive OFFSET would skip past an
///     early-stopped superset, leaving fewer rows than the scan path.
///   - The WHERE (if any) is EXACTLY a lower bound on the ordered column —
///     `<col> >= <literal>` / `<col> > <literal>`, optionally AND-chained with
///     more lower bounds on the SAME ordered column.  Every gathered candidate
///     satisfies such a bound, so re-applying it drops nothing the gather
///     missed.  ANY conjunct on a different column (an equality on a
///     non-ordered column, an OFFSET, an upper bound, an OR, an IN, an IS NULL)
///     makes the early-stopped gather an INCOMPLETE candidate set — a matching
///     row could sit below the top-`limit` ordered candidates and never be
///     gathered.  The detector returns `None` (→ scan) the moment it sees one.
///
/// Returns `None` for any unrecognized shape, falling back to the full scan.
/// Correctness never depends on the index recognising a predicate it cannot
/// soundly narrow.
fn extract_ordered_topk(
    plan: &SelectPlan,
    oidx: &OrderedColIndex,
    effective_limit: Option<i64>,
    effective_offset: Option<i64>,
) -> Option<OrderedTopkSpec> {
    // Must have ORDER BY on the indexed column in DESC direction.
    let order = plan.order_by.as_ref()?;
    if !order.desc || order.col != oidx.column() {
        return None;
    }
    // A second ORDER BY key over a non-gathered column would require the full
    // table; fall back to scan for safety.
    if plan.order_by2.is_some() {
        return None;
    }
    // A positive OFFSET skips past the under-gathered superset (the gather only
    // collects ~LIMIT candidates), so the result would be short by the offset.
    // Fall back to scan unless the offset is absent or zero.
    if effective_offset.unwrap_or(0) > 0 {
        return None;
    }
    // Must have a positive LIMIT.  Use the already-bound effective limit so
    // that `LIMIT ?` (a parameter placeholder resolved by the binder) is
    // recognised correctly.  `plan.limit` holds the literal from the parsed
    // plan and is None when the LIMIT was supplied as a `?` parameter.
    let lim = effective_limit.or(plan.limit)?;
    if lim <= 0 {
        return None;
    }
    let limit_hint = lim as usize;

    // The WHERE (if any) must be EXACTLY a lower bound on the ordered column.
    // Any conjunct on another column, any disjunction, or any non-bound
    // predicate makes the early-stopped gather an incomplete candidate set —
    // reject (→ scan).
    let ge_bound = match &plan.where_clause {
        None => None,
        Some(w) => strict_ordered_lower_bound(w, oidx.column())?,
    };

    Some(OrderedTopkSpec { limit_hint, ge_bound })
}

/// Classify a WHERE expression that the ordered fast path can soundly narrow.
///
/// The fast path stops gathering once ~LIMIT candidates are collected from the
/// top of the ordered column, so it is correct ONLY when every gathered
/// candidate is guaranteed to survive the WHERE filter — otherwise a row that
/// passes the filter could sit below the gathered window and silently vanish.
/// The single class of predicate that satisfies this is a lower bound on the
/// ordered column itself: `<col> >= literal` / `<col> > literal`.  Every
/// gathered candidate satisfies the bound by construction, so re-applying it
/// drops nothing.
///
/// Returns:
///   - `Some(None)` — the WHERE is entirely composed of lower bounds on the
///     ordered column but none narrowed the range (only AND-chained bounds that
///     reduce to a single `>=`); no `BTreeMap::range` lower bound is needed.
///   - `Some(Some(text))` — a `>= text` (or `> text`) lower bound to feed
///     `candidates_desc`.
///   - `None` — the WHERE contains ANY conjunct that is not a lower bound on the
///     ordered column (a foreign-column predicate, an OR, an upper bound, an IN,
///     an IS NULL, an equality on the ordered column, …).  The caller falls back
///     to the full scan, which is always byte-identical.
fn strict_ordered_lower_bound(expr: &Expr, ordered_col: &str) -> Option<Option<String>> {
    match expr {
        // `col >= literal` or `col > literal` on the ordered column ONLY.
        Expr::BinOp { op, left, right } if op == ">=" || op == ">" => {
            if let (Expr::Column(col), Expr::Literal(Value::Text(s))) =
                (left.as_ref(), right.as_ref())
            {
                if col == ordered_col {
                    return Some(Some(s.clone()));
                }
            }
            // A bound on any OTHER column (or against a non-text literal) is a
            // residual filter the early-stopped gather cannot enforce — reject.
            None
        }
        // AND: every conjunct must itself be a sound lower bound on the ordered
        // column.  Merge the bounds, preferring the tighter (greater) one so the
        // range walk starts as high as soundly possible.
        Expr::BinOp { op, left, right } if op == "AND" => {
            let lb = strict_ordered_lower_bound(left, ordered_col)?;
            let rb = strict_ordered_lower_bound(right, ordered_col)?;
            let merged = match (lb, rb) {
                (Some(a), Some(b)) => Some(if a >= b { a } else { b }),
                (Some(a), None) | (None, Some(a)) => Some(a),
                (None, None) => None,
            };
            Some(merged)
        }
        // Any other expression (OR, equality, IN, IS NULL, upper bound, a bound
        // on a foreign column, …) is a residual predicate the early-stopped
        // gather cannot soundly enforce — reject so the caller scans.
        _ => None,
    }
}

/// Detect the indexed `WHERE col IN (...)` shape the recall edge-adjacency lookup
/// uses, returning the list of `(indexed-column, candidate-value)` pairs to probe
/// when EVERY `IN (...)` disjunct in the predicate is over a column the `ColIndex`
/// covers and every IN item bound to a concrete literal. Returns `None` (fall back
/// to the scan) for any other shape, so correctness never depends on the index
/// recognizing a predicate it cannot fully resolve.
///
/// Supported shapes (after binding turns `?` into literals):
///   `col IN (v1, v2, ...)` where `col` is indexed
///   `srcCol IN (...) OR dstCol IN (...)` (both indexed)
///   `(<above>) AND <anything>` — the AND tail is re-checked by the full predicate
/// The returned pairs are the union to probe; the caller re-applies the full
/// predicate to the decoded candidates, so an over-broad candidate set is still
/// filtered to the exact result.
fn indexed_in_columns(expr: &Expr, idx: &ColIndex) -> Option<Vec<(String, Value)>> {
    let mut out: Vec<(String, Value)> = Vec::new();
    if collect_indexed_in(expr, idx, &mut out) && !out.is_empty() {
        Some(out)
    } else {
        None
    }
}

/// Walk the WHERE tree collecting indexed `col IN (literals)` probes. Returns true
/// only when the WHERE is fully reducible to indexed IN-probes at its top
/// disjunction: an `OR` requires BOTH sides reducible (else the index would miss
/// rows the other branch matches); an `AND` requires AT LEAST ONE side reducible
/// (the candidate superset from one indexed conjunct is then filtered by the full
/// predicate). A bare indexed `col IN (lits)` reduces directly.
fn collect_indexed_in(expr: &Expr, idx: &ColIndex, out: &mut Vec<(String, Value)>) -> bool {
    match expr {
        Expr::InList { operand, items, .. } => {
            if let Expr::Column(col) = operand.as_ref() {
                if idx.indexes(col) {
                    let mut vals: Vec<(String, Value)> = Vec::with_capacity(items.len());
                    for it in items {
                        match it {
                            Expr::Literal(v) => vals.push((col.clone(), v.clone())),
                            // A non-literal IN item (unbound) is not index-resolvable.
                            _ => return false,
                        }
                    }
                    out.extend(vals);
                    return true;
                }
            }
            false
        }
        // `col = <literal>` over an indexed column reduces to a single-value probe;
        // a `col = 1` filter on an indexed low-cardinality column (e.g. the pending
        // flag) resolves by index instead of a whole-table scan. The full predicate
        // is re-applied to the candidates, so an over-broad probe stays correct.
        Expr::BinOp { op, left, right } if op == "=" || op == "==" => {
            if let (Expr::Column(col), Expr::Literal(v)) = (left.as_ref(), right.as_ref()) {
                if idx.indexes(col) {
                    out.push((col.clone(), v.clone()));
                    return true;
                }
            }
            if let (Expr::Literal(v), Expr::Column(col)) = (left.as_ref(), right.as_ref()) {
                if idx.indexes(col) {
                    out.push((col.clone(), v.clone()));
                    return true;
                }
            }
            false
        }
        Expr::BinOp { op, left, right } if op == "OR" => {
            // OR: the index must cover BOTH branches, or it would miss matches from
            // the unindexed branch. Collect from both.
            let mut l: Vec<(String, Value)> = Vec::new();
            let mut r: Vec<(String, Value)> = Vec::new();
            if collect_indexed_in(left, idx, &mut l) && collect_indexed_in(right, idx, &mut r) {
                out.extend(l);
                out.extend(r);
                true
            } else {
                false
            }
        }
        Expr::BinOp { op, left, right } if op == "AND" => {
            // AND: one indexed conjunct yields a candidate SUPERSET; the full
            // predicate (re-applied by the caller) narrows it. Prefer the left
            // branch, then the right.
            let mut l: Vec<(String, Value)> = Vec::new();
            if collect_indexed_in(left, idx, &mut l) {
                out.extend(l);
                return true;
            }
            let mut r: Vec<(String, Value)> = Vec::new();
            if collect_indexed_in(right, idx, &mut r) {
                out.extend(r);
                return true;
            }
            false
        }
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Scan helpers
// ---------------------------------------------------------------------------

/// Collect the column names an expression reads into `out`. Used to compute the
/// minimal set of columns a filtering scan must materialize, so a wide row whose
/// large BLOB columns are neither filtered nor projected is read without copying
/// those bodies.
fn collect_expr_columns(expr: &Expr, out: &mut std::collections::HashSet<String>) {
    match expr {
        Expr::Column(c) | Expr::Excluded(c) => {
            out.insert(c.clone());
        }
        Expr::BinOp { left, right, .. } => {
            collect_expr_columns(left, out);
            collect_expr_columns(right, out);
        }
        Expr::UnaryOp { operand, .. } | Expr::IsNull { operand, .. } => {
            collect_expr_columns(operand, out);
        }
        Expr::InList { operand, items, .. } => {
            collect_expr_columns(operand, out);
            for it in items {
                collect_expr_columns(it, out);
            }
        }
        Expr::Coalesce(args) | Expr::FuncCall { args, .. } => {
            for a in args {
                collect_expr_columns(a, out);
            }
        }
        Expr::Literal(_) | Expr::Placeholder | Expr::NamedParam(_) => {}
    }
}

/// Build a per-column mask (indexed by catalog column position) from a set of
/// column names, always including the `rowid` source column so rowid resolution
/// and an ORDER BY rowid key see its value. A column absent from the catalog name
/// list contributes nothing; an unwanted column is decode-skipped.
fn mask_from_names(col_names: &[String], names: &std::collections::HashSet<String>) -> Vec<bool> {
    col_names
        .iter()
        .map(|c| names.contains(c) || c == ROWID_SOURCE)
        .collect()
}

/// The set of columns the WHERE predicate reads.
fn predicate_columns(where_expr: Option<&Expr>) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    if let Some(w) = where_expr {
        collect_expr_columns(w, &mut names);
    }
    names
}

/// The set of columns the projection reads.
fn projection_columns(projected: &[(Expr, String)]) -> std::collections::HashSet<String> {
    let mut names = std::collections::HashSet::new();
    for (e, _) in projected {
        collect_expr_columns(e, &mut names);
    }
    names
}

/// Scan the data tree applying the WHERE predicate during the scan, decoding each
/// row in two passes so a non-matching row never decodes the projection-only
/// columns (e.g. the large embedding BLOB):
///
///  1. Decode only the columns the predicate reads; evaluate the predicate.
///  2. For a surviving row, decode the projection-only columns from the same
///     payload and merge them in.
///
/// When `early_limit` is `Some(n)` the scan stops after `n` surviving rows — safe
/// only because the aliased-projection caller applies LIMIT without an ORDER BY,
/// so the first `n` matching rows in scan order are exactly what it would keep
/// after a full scan + filter + truncate. Every decoded value is byte-identical to
/// a full decode; the result equals the full-scan path.
fn scan_rows_two_phase(
    store: &Store,
    root: u32,
    col_names: &[String],
    pred_cols: &std::collections::HashSet<String>,
    proj_cols: &std::collections::HashSet<String>,
    where_expr: Option<&Expr>,
    early_limit: Option<usize>,
) -> Result<Vec<Row>> {
    let pred_mask = mask_from_names(col_names, pred_cols);
    // Second-pass mask: projection columns NOT already decoded in the first pass.
    let mut proj_only = proj_cols.clone();
    for c in pred_cols {
        proj_only.remove(c);
    }
    let proj_mask = mask_from_names(col_names, &proj_only);
    let need_phase2 = proj_mask.iter().any(|&b| b);

    // Stream cells without pre-materialising the whole table and without
    // incrementing the full-scan counter; decode only the predicate columns per
    // cell, then decode projection-only columns for survivors.
    //
    // Early-exit sentinel: when the `early_limit` row count is reached we stop
    // the scan by returning a recognisable `EngineError`. The `.or_else` below
    // swallows the sentinel and treats it as a clean stop.
    let mut rows: Vec<Row> = Vec::new();
    const EARLY_LIMIT_SENTINEL: &str = "__scan_early_limit__";

    let scan_result: Result<()> = store
        .tree(root)
        .scan_cells_with(|key, payload| {
            let mut vals = crate::rowcodec::decode_row_columns(&payload, &pred_mask)?;
            while vals.len() < col_names.len() {
                vals.push(Value::Null);
            }
            let mut pairs: Vec<(String, Value)> = col_names
                .iter()
                .cloned()
                .zip(vals)
                .collect();
            pairs.push((ROWKEY.to_string(), Value::Int(key)));
            let mut row = Row::from_pairs(pairs);
            if let Some(w) = where_expr {
                if eval_predicate(w, &row) != Some(true) {
                    return Ok(());
                }
            }
            // Second pass: a survivor — decode the projection-only columns from the
            // same payload and fill them in (the predicate columns are already present).
            if need_phase2 {
                let proj_vals = crate::rowcodec::decode_row_columns(&payload, &proj_mask)?;
                for (i, name) in col_names.iter().enumerate() {
                    if proj_mask.get(i).copied().unwrap_or(false) {
                        if let Some(v) = proj_vals.get(i) {
                            row.set(name.clone(), v.clone());
                        }
                    }
                }
            }
            rows.push(row);
            if let Some(n) = early_limit {
                if rows.len() >= n {
                    return Err(EngineError::parse(EARLY_LIMIT_SENTINEL));
                }
            }
            Ok(())
        });

    match scan_result {
        Ok(()) => {}
        Err(ref e) if e.to_string() == EARLY_LIMIT_SENTINEL => {}
        Err(e) => return Err(e),
    }
    Ok(rows)
}

/// Scan the data tree into decoded rows keyed by column name, padding a short
/// stored record (one written before an ALTER TABLE ADD COLUMN) with NULL and
/// threading the raw B-tree key for rowid fallback.
fn scan_rows(store: &Store, root: u32, col_names: &[String]) -> Result<Vec<Row>> {
    let raw = store
        .tree(root)
        .full_scan()
        .map_err(store_err)?;
    let mut rows = Vec::with_capacity(raw.len());
    for (key, payload) in raw {
        let mut vals = decode_row(&payload)?;
        while vals.len() < col_names.len() {
            vals.push(Value::Null);
        }
        let mut pairs: Vec<(String, Value)> = col_names
            .iter()
            .cloned()
            .zip(vals.into_iter())
            .collect();
        pairs.push((ROWKEY.to_string(), Value::Int(key)));
        rows.push(Row::from_pairs(pairs));
    }
    Ok(rows)
}

/// Count all rows without decoding any row payload.
///
/// Walks the leaf sibling chain reading only the page header's `num_cells`
/// field — no cell payload is read, no overflow chain is followed, and the
/// full-scan counter is not incremented. The count equals the number of rows
/// a full scan would return for every tree shape including overflow cells.
fn count_rows_no_decode(store: &Store, root: u32) -> Result<i64> {
    let n = store
        .tree(root)
        .count_cells()
        .map_err(store_err)?;
    Ok(n as i64)
}

// ---------------------------------------------------------------------------
// Aggregates
// ---------------------------------------------------------------------------

fn aggregate_scalar(
    agg: &str,
    agg_col: Option<&str>,
    rows: &[Row],
    effective_limit: Option<i64>,
    count_alias: Option<&str>,
) -> ResultSet {
    let label = match agg {
        "count" => count_alias.unwrap_or("COUNT(*)").to_string(),
        "sum" => format!("SUM({})", agg_col.unwrap_or("")),
        "min" => format!("MIN({})", agg_col.unwrap_or("")),
        "max" => format!("MAX({})", agg_col.unwrap_or("")),
        _ => agg.to_string(),
    };
    if effective_limit == Some(0) {
        return ResultSet { rows: Vec::new(), columns: vec![label] };
    }
    let value = match agg {
        "count" => Value::Int(rows.len() as i64),
        "sum" => sum_aggregate(agg_col.unwrap_or(""), rows),
        "min" | "max" => {
            let col = agg_col.unwrap_or("");
            let want_max = agg == "max";
            let mut best: Value = Value::Null;
            let mut best_key: Option<(u8, crate::eval::SortKey)> = None;
            for r in rows {
                let v = r.get_or_null(col);
                if matches!(v, Value::Null) {
                    continue; // NULLs are ignored by MIN/MAX
                }
                let k = order_key(&v);
                let take = match &best_key {
                    None => true,
                    Some(bk) => {
                        if want_max {
                            k > *bk
                        } else {
                            k < *bk
                        }
                    }
                };
                if take {
                    best_key = Some(k);
                    best = v;
                }
            }
            best
        }
        _ => Value::Null,
    };
    ResultSet { rows: vec![vec![(label.clone(), value)]], columns: vec![label] }
}

/// SUM over a column, matching sqlite3's `sumStep` semantics:
///
/// - NULLs are skipped; an empty / all-NULL column yields `NULL`.
/// - The total stays an integer while every addend is an integer (or
///   integer-valued numeric text), and promotes to REAL the moment any REAL
///   (or real-valued numeric text) is summed — sqlite3 returns REAL if any input
///   is REAL.
/// - The REAL accumulation uses Kahan-Babuška-Neumaier compensated summation, the
///   same algorithm sqlite3 uses, so a SUM over a REAL column is bit-identical to
///   stdlib sqlite3 (a naive running sum diverges in the low bits by summation
///   order).
fn sum_aggregate(col: &str, rows: &[Row]) -> Value {
    let mut int_total: i64 = 0;
    // Neumaier compensated sum: running total `sum` plus a correction `c`.
    let mut sum: f64 = 0.0;
    let mut c: f64 = 0.0;
    let mut is_float = false;
    let mut saw_any = false;

    // Add one f64 addend into the compensated running sum.
    fn neumaier_add(sum: &mut f64, c: &mut f64, v: f64) {
        let t = *sum + v;
        if sum.abs() >= v.abs() {
            *c += (*sum - t) + v;
        } else {
            *c += (v - t) + *sum;
        }
        *sum = t;
    }

    for r in rows {
        // Resolve the addend's numeric value and whether it is REAL. Numeric text
        // contributes its parsed value (integer string keeps the total integral, a
        // real string promotes it); a non-numeric text / blob contributes 0.
        let addend: Option<(f64, bool)> = match r.get(col) {
            None | Some(Value::Null) => None,
            Some(Value::Int(i)) => Some((*i as f64, false)),
            Some(Value::Float(f)) => Some((*f, true)),
            Some(Value::Text(s)) => {
                if let Ok(i) = s.parse::<i64>() {
                    Some((i as f64, false))
                } else if let Ok(f) = s.parse::<f64>() {
                    Some((f, true))
                } else {
                    Some((0.0, false)) // non-numeric text sums as 0
                }
            }
            Some(Value::Blob(_)) => Some((0.0, false)), // a blob sums as 0
        };
        let Some((v, addend_is_float)) = addend else {
            continue;
        };
        saw_any = true;
        if addend_is_float && !is_float {
            // Promote: seed the compensated float sum with the integer running
            // total accumulated so far.
            is_float = true;
            sum = int_total as f64;
            c = 0.0;
        }
        if is_float {
            neumaier_add(&mut sum, &mut c, v);
        } else {
            int_total += v as i64;
        }
    }

    if !saw_any {
        Value::Null
    } else if is_float {
        Value::Float(sum + c)
    } else {
        Value::Int(int_total)
    }
}

// ---------------------------------------------------------------------------
// ORDER BY
// ---------------------------------------------------------------------------

fn order_rows(rows: &mut [Row], plan: &SelectPlan, col_names: &[String]) {
    let resolve = |c: &str| -> String {
        if c != ROWID {
            return c.to_string();
        }
        if col_names.iter().any(|n| n == ROWID_SOURCE) {
            ROWID_SOURCE.to_string()
        } else {
            ROWKEY.to_string()
        }
    };
    let k1 = plan.order_by.as_ref().unwrap();
    let col1 = resolve(&k1.col);
    let (desc1, dt1) = (k1.desc, k1.datetime);

    if let Some(k2) = &plan.order_by2 {
        let col2 = resolve(&k2.col);
        let (desc2, dt2) = (k2.desc, k2.datetime);
        rows.sort_by(|a, b| {
            let ka = composite_key(a, &col1, desc1, dt1, &col2, desc2, dt2);
            let kb = composite_key(b, &col1, desc1, dt1, &col2, desc2, dt2);
            ka.cmp(&kb)
        });
    } else {
        rows.sort_by(|a, b| {
            let ka = sort_value(&a.get_or_null(&col1), desc1, dt1);
            let kb = sort_value(&b.get_or_null(&col1), desc1, dt1);
            ka.cmp(&kb)
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn composite_key(
    r: &Row,
    col1: &str,
    desc1: bool,
    dt1: bool,
    col2: &str,
    desc2: bool,
    dt2: bool,
) -> (DirectedKey, DirectedKey) {
    (
        sort_value(&r.get_or_null(col1), desc1, dt1),
        sort_value(&r.get_or_null(col2), desc2, dt2),
    )
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// Project rows to the requested columns. `SELECT *` passes the schema columns
/// through (stripping the reserved rowkey / rowid bookkeeping); a named-column
/// list selects those columns (resolving rowid first when present).
fn project_rows(rows: Vec<Row>, columns: &[String], col_names: &[String]) -> Vec<OutRow> {
    let select_star = columns == ["*"] || columns.is_empty();
    let mut out = Vec::with_capacity(rows.len());
    for mut row in rows {
        if select_star {
            let mut pairs: Vec<(String, Value)> = Vec::with_capacity(col_names.len());
            for name in col_names {
                pairs.push((name.clone(), row.get_or_null(name)));
            }
            out.push(pairs);
        } else {
            if columns.iter().any(|c| c == ROWID) {
                resolve_rowid(&mut row);
            }
            let pairs: Vec<(String, Value)> = columns
                .iter()
                .map(|c| (c.clone(), row.get_or_null(c)))
                .collect();
            out.push(pairs);
        }
    }
    out
}

/// The output column names for a projection.
fn output_columns(columns: &[String], col_names: &[String]) -> Vec<String> {
    if columns == ["*"] || columns.is_empty() {
        col_names.to_vec()
    } else {
        columns.to_vec()
    }
}

/// Resolve `rowid` for a row: prefer the vec_label column, fall back to the raw
/// B-tree key.
fn resolve_rowid(row: &mut Row) {
    let v = if row.contains(ROWID_SOURCE) {
        row.get_or_null(ROWID_SOURCE)
    } else {
        row.get_or_null(ROWKEY)
    };
    row.set(ROWID, v);
}

/// The point-lookup fallback when no id-index is available (or it pointed at a
/// gone row): scan the data tree and return the first row whose `id` matches and
/// satisfies the full predicate, threading the raw B-tree key for rowid
/// resolution exactly as `scan_rows` does. At most one row is returned (a unique
/// id yields at most one match), so the scan stops at the first hit.
fn first_id_match_by_scan(
    store: &Store,
    root: u32,
    col_names: &[String],
    id_val: &str,
    where_expr: &Expr,
) -> Result<Vec<Row>> {
    for row in scan_rows(store, root, col_names)? {
        if row.get("id") == Some(&Value::Text(id_val.to_string()))
            && eval_predicate(where_expr, &row) == Some(true)
        {
            return Ok(vec![row]);
        }
    }
    Ok(Vec::new())
}

/// Extract the literal id values from a pure `id IN ('<text>', ...)` expression
/// at the top of a WHERE tree, for the id-index IN fast path.  Returns `Some`
/// with the text values when the operand is `Expr::Column("id")` and every item
/// in the list is `Expr::Literal(Value::Text(_))`; returns `None` for any other
/// shape (mixed literals, non-id operand, empty list, non-InList), so the caller
/// falls back to the ColIndex or scan path without any loss of correctness.
///
/// The full bound predicate is re-applied to every decoded candidate by the
/// caller, so an over-broad probe (e.g. an AND-tail beyond the id IN) is still
/// correctly filtered.
fn extract_id_in(expr: &Expr) -> Option<Vec<String>> {
    if let Expr::InList { operand, items, .. } = expr {
        if let Expr::Column(col) = operand.as_ref() {
            if col == "id" && !items.is_empty() {
                let mut ids: Vec<String> = Vec::with_capacity(items.len());
                for it in items {
                    match it {
                        Expr::Literal(Value::Text(s)) => ids.push(s.clone()),
                        // A non-text or non-literal item is not resolvable through
                        // the id-index (ids are always TEXT); fall back to ColIndex/scan.
                        _ => return None,
                    }
                }
                return Some(ids);
            }
        }
    }
    None
}

/// Extract the literal id from a `id = '<literal>'` equality at the top of a
/// WHERE tree, for the point-lookup fast path. Returns `None` for any other shape.
fn extract_id_eq(expr: &Expr) -> Option<String> {
    if let Expr::BinOp { op, left, right } = expr {
        if op == "=" || op == "==" {
            if let (Expr::Column(name), Expr::Literal(Value::Text(s))) = (left.as_ref(), right.as_ref()) {
                if name == "id" {
                    return Some(s.clone());
                }
            }
            if let (Expr::Literal(Value::Text(s)), Expr::Column(name)) = (left.as_ref(), right.as_ref()) {
                if name == "id" {
                    return Some(s.clone());
                }
            }
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Aliased / COALESCE projection
// ---------------------------------------------------------------------------

/// Resolve the candidate rows for an indexed `id IN (...)`-shaped WHERE: when the
/// ColIndex covers every `IN` column the predicate filters on, probe the index
/// (a union of O(1) probes) and decode only those rows, threading the raw B-tree
/// key for rowid resolution exactly as `scan_rows` does. The caller re-applies the
/// FULL predicate to the returned rows, so the candidate set only narrows which
/// rows are decoded — never the result. Returns `None` when the predicate is not
/// fully index-resolvable, so the caller falls back to a scan and correctness
/// never depends on the index recognizing a shape it cannot resolve.
fn indexed_candidate_rows(
    store: &Store,
    root: u32,
    col_names: &[String],
    where_expr: &Expr,
    cidx: &mut ColIndex,
) -> Result<Option<Vec<Row>>> {
    let in_cols = match indexed_in_columns(where_expr, cidx) {
        Some(c) => c,
        None => return Ok(None),
    };
    let tree = store.tree(root);
    cidx.ensure_built(&tree, col_names)?;
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut candidate_keys: Vec<i64> = Vec::new();
    for (col, value) in &in_cols {
        for &k in cidx.probe(col, value) {
            if seen.insert(k) {
                candidate_keys.push(k);
            }
        }
    }
    // One ordered leaf-chain walk for the whole candidate set (see the
    // plain-row indexed-IN path): O(pages spanned) instead of one O(height)
    // descent per key, and ascending key order matches the scan path's row
    // order.
    candidate_keys.sort_unstable();
    let fetched = tree.get_many(&candidate_keys).map_err(store_err)?;
    let mut rows: Vec<Row> = Vec::with_capacity(fetched.len());
    for (key, payload) in fetched {
        let mut row = decode_to_row(&payload, col_names)?;
        row.set(ROWKEY.to_string(), Value::Int(key));
        rows.push(row);
    }
    Ok(Some(rows))
}

fn execute_select_exprs(
    store: &Store,
    root: u32,
    col_names: &[String],
    bound_select_exprs: &[(Expr, String)],
    bound_where: &Option<Expr>,
    effective_limit: Option<i64>,
    effective_offset: Option<i64>,
    indexed_rows: Option<Vec<Row>>,
    plan: &SelectPlan,
) -> Result<ResultSet> {
    let aliases: Vec<String> = bound_select_exprs.iter().map(|(_, a)| a.clone()).collect();
    let has_order_by = plan.order_by.is_some();
    // The index resolved a candidate set (an `id IN (...)` batch-get); decode only
    // those rows and re-apply the full predicate. Otherwise scan the table — but
    // decode only the columns the WHERE and the projection actually read, in two
    // phases so a non-matching wide row never decodes the projection-only blobs,
    // and (when there is no ORDER BY) stop after enough survivors to satisfy the
    // LIMIT. Both paths re-apply the full predicate, so the surviving rows are
    // identical to a full scan + filter.
    let mut rows: Vec<Row> = match indexed_rows {
        Some(mut r) => {
            if let Some(w) = bound_where {
                r.retain(|row| eval_predicate(w, row) == Some(true));
            }
            r
        }
        None => {
            let mut pred_cols = predicate_columns(bound_where.as_ref());
            let proj_cols = projection_columns(bound_select_exprs);
            // With an ORDER BY the projected rows must be sorted before LIMIT, so the
            // sort keys must be decoded onto each row (they may name a column outside
            // the WHERE and the projection). Fold them into the first-pass decode set.
            if has_order_by {
                for key in [plan.order_by.as_ref(), plan.order_by2.as_ref()].into_iter().flatten() {
                    if key.col != ROWID {
                        pred_cols.insert(key.col.clone());
                    }
                }
            }
            // Without an ORDER BY this path applies LIMIT/OFFSET in scan order, so the
            // first OFFSET+LIMIT survivors in scan order are exactly what a full scan +
            // filter + truncate would keep — stopping the scan there is correct and is
            // the fast path. With an ORDER BY the kept set depends on the sort, so the
            // full survivor set must be materialized before sorting; the early stop is
            // disabled and the sort runs on the already-WHERE-narrowed survivors.
            let early_limit = if has_order_by {
                None
            } else {
                effective_limit.and_then(|lim| {
                    if lim <= 0 {
                        None
                    } else {
                        let off = effective_offset.filter(|o| *o > 0).unwrap_or(0);
                        Some((lim + off) as usize)
                    }
                })
            };
            scan_rows_two_phase(
                store, root, col_names, &pred_cols, &proj_cols,
                bound_where.as_ref(), early_limit,
            )?
        }
    };
    // rowid may be referenced by a projected expression; resolve it on each row.
    for r in &mut rows {
        resolve_rowid(r);
    }

    // ORDER BY sorts the survivor rows before projection (and before OFFSET/LIMIT),
    // reusing the plain-row sort so DESC / NULLs-first / datetime normalization match.
    if has_order_by {
        order_rows(&mut rows, plan, col_names);
    }

    if effective_limit == Some(0) {
        return Ok(ResultSet { rows: Vec::new(), columns: aliases });
    }

    let mut out: Vec<OutRow> = rows
        .iter()
        .map(|r| {
            bound_select_exprs
                .iter()
                .map(|(expr, alias)| (alias.clone(), eval_scalar(expr, r)))
                .collect()
        })
        .collect();

    if let Some(off) = effective_offset {
        if off > 0 {
            let off = off as usize;
            out = if off >= out.len() { Vec::new() } else { out.split_off(off) };
        }
    }
    if let Some(lim) = effective_limit {
        if lim > 0 && (lim as usize) < out.len() {
            out.truncate(lim as usize);
        }
    }
    Ok(ResultSet { rows: out, columns: aliases })
}

// ---------------------------------------------------------------------------
// GROUP BY
// ---------------------------------------------------------------------------

fn execute_group_by(
    plan: &SelectPlan,
    store: &Store,
    root: u32,
    col_names: &[String],
    bound_where: &Option<Expr>,
    effective_limit: Option<i64>,
    effective_offset: Option<i64>,
) -> Result<ResultSet> {
    let group_col = plan.group_by_col.as_ref().unwrap();
    let mut rows: Vec<Row> = scan_rows(store, root, col_names)?;
    if let Some(w) = bound_where {
        rows.retain(|r| eval_predicate(w, r) == Some(true));
    }

    // Output projection: the grouped select list, or the plain columns.
    let projection: Vec<(String, String, String)> = if !plan.group_select.is_empty() {
        plan.group_select
            .iter()
            .map(|g| (g.kind.clone(), g.output_name.clone(), g.source_name.clone()))
            .collect()
    } else {
        plan.columns.iter().map(|c| ("col".to_string(), c.clone(), c.clone())).collect()
    };
    let out_names: Vec<String> = projection.iter().map(|(_, o, _)| o.clone()).collect();

    // Bucket by the storage-class key of the group column, keeping the first row
    // and a running count. The first-appearance order is recorded for the stable
    // tiebreak; the buckets are then sorted by key (NULL < number < text < blob).
    let mut buckets: Vec<((u8, crate::eval::SortKey), Row, i64)> = Vec::new();
    let mut index: HashMap<String, usize> = HashMap::new();
    for row in rows {
        let key = order_key(&row.get_or_null(group_col));
        let key_repr = format!("{:?}", key);
        match index.get(&key_repr) {
            Some(&i) => buckets[i].2 += 1,
            None => {
                index.insert(key_repr, buckets.len());
                buckets.push((key, row, 1));
            }
        }
    }
    buckets.sort_by(|a, b| a.0.cmp(&b.0));

    // Build one output row per group.
    let mut grouped: Vec<(OutRow, i64)> = Vec::with_capacity(buckets.len());
    for (_key, first_row, n) in &buckets {
        let mut out_row: OutRow = Vec::with_capacity(projection.len());
        for (kind, out_name, src_name) in &projection {
            if kind == "count" {
                out_row.push((out_name.clone(), Value::Int(*n)));
            } else {
                out_row.push((out_name.clone(), first_row.get_or_null(src_name)));
            }
        }
        grouped.push((out_row, *n));
    }

    // HAVING COUNT(*) <op> <int>.
    if let (Some(op), Some(rhs)) = (&plan.having_op, plan.having_value) {
        grouped.retain(|(_, n)| compare_int(op, *n, rhs));
    }

    let mut out_rows: Vec<OutRow> = grouped.into_iter().map(|(r, _)| r).collect();

    // ORDER BY on the grouped output rows (a key may name the COUNT alias).
    if let Some(k1) = &plan.order_by {
        let col1 = k1.col.clone();
        let (desc1, dt1) = (k1.desc, k1.datetime);
        if let Some(k2) = &plan.order_by2 {
            let col2 = k2.col.clone();
            let (desc2, dt2) = (k2.desc, k2.datetime);
            out_rows.sort_by(|a, b| {
                let ka = (sort_value(&out_get(a, &col1), desc1, dt1), sort_value(&out_get(a, &col2), desc2, dt2));
                let kb = (sort_value(&out_get(b, &col1), desc1, dt1), sort_value(&out_get(b, &col2), desc2, dt2));
                ka.cmp(&kb)
            });
        } else {
            out_rows.sort_by(|a, b| {
                sort_value(&out_get(a, &col1), desc1, dt1).cmp(&sort_value(&out_get(b, &col1), desc1, dt1))
            });
        }
    }

    if effective_limit == Some(0) {
        return Ok(ResultSet { rows: Vec::new(), columns: out_names });
    }
    if let Some(off) = effective_offset {
        if off > 0 {
            let off = off as usize;
            out_rows = if off >= out_rows.len() { Vec::new() } else { out_rows.split_off(off) };
        }
    }
    if let Some(lim) = effective_limit {
        if lim > 0 && (lim as usize) < out_rows.len() {
            out_rows.truncate(lim as usize);
        }
    }
    Ok(ResultSet { rows: out_rows, columns: out_names })
}

/// Read a named value from an output row (the grouped rows are keyed by alias).
fn out_get(row: &OutRow, name: &str) -> Value {
    row.iter().find(|(n, _)| n == name).map(|(_, v)| v.clone()).unwrap_or(Value::Null)
}

/// Integer comparison for HAVING COUNT(*).
fn compare_int(op: &str, lhs: i64, rhs: i64) -> bool {
    match op {
        "=" | "==" => lhs == rhs,
        "!=" | "<>" => lhs != rhs,
        "<" => lhs < rhs,
        "<=" => lhs <= rhs,
        ">" => lhs > rhs,
        ">=" => lhs >= rhs,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// dbstat constant plan
// ---------------------------------------------------------------------------

/// The `sqlite_master` column order, matching stdlib sqlite3.
const SCHEMA_COLS: &[&str] = &["type", "name", "tbl_name", "rootpage", "sql"];

/// Answer a `sqlite_master` / `sqlite_schema` query by synthesizing the schema
/// rows from the catalog and running them through the same WHERE / aggregate /
/// ORDER BY / projection pipeline a real-table scan uses. The `(type, name,
/// tbl_name, rootpage, sql)` shape matches stdlib so introspection reads
/// identically across drivers.
fn execute_sqlite_master(plan: &SelectPlan, catalog: &Catalog) -> Result<ResultSet> {
    let col_names: Vec<String> = SCHEMA_COLS.iter().map(|s| s.to_string()).collect();

    // Build one Row per synthesized schema object, in catalog order.
    let mut rows: Vec<Row> = catalog
        .sqlite_master_rows()
        .into_iter()
        .map(|r| {
            Row::from_pairs(vec![
                ("type".to_string(), Value::Text(r.object_type)),
                ("name".to_string(), Value::Text(r.name)),
                ("tbl_name".to_string(), Value::Text(r.tbl_name)),
                ("rootpage".to_string(), Value::Int(r.rootpage)),
                ("sql".to_string(), r.sql),
            ])
        })
        .collect();

    // Bind placeholders (WHERE, then LIMIT, then OFFSET) in textual order. The
    // schema table is read with plain columns / COUNT(*) only, so no aliased
    // select-list binding is needed; an aliased projection over the schema table
    // is rejected by the unbound-placeholder check below if it carries binds.
    let mut binder = Binder::new(plan.params.clone(), plan.named_params.clone());
    let bound_where = match &plan.where_clause {
        Some(w) => Some(binder.bind(w)?),
        None => None,
    };
    let effective_limit = binder.resolve_limit(plan.limit, plan.limit_is_param)?;
    let effective_offset = binder.resolve_limit(plan.offset, plan.offset_is_param)?;
    binder.finish()?;

    if let Some(where_expr) = &bound_where {
        rows.retain(|r| eval_predicate(where_expr, r) == Some(true));
    }

    // Scalar aggregate (e.g. COUNT(*) over the schema rows).
    if let Some(agg) = plan.aggregate.as_deref() {
        return Ok(aggregate_scalar(
            agg,
            plan.aggregate_column.as_deref(),
            &rows,
            effective_limit,
            plan.count_alias.as_deref(),
        ));
    }

    Ok(finish_row_pipeline(rows, plan, &col_names, effective_limit, effective_offset))
}

fn execute_constant(plan: &SelectPlan, store: &Store) -> Result<ResultSet> {
    // sqlite3's page size; the dbstat size estimate is page_count * PAGE_SIZE.
    const PAGE_SIZE: i64 = 4096;
    let page_count = store.db_size().map(i64::from).unwrap_or(0);
    let estimate = page_count * PAGE_SIZE;
    let col = plan.aggregate_column.clone().unwrap_or_else(|| "pgsize".to_string());
    Ok(ResultSet {
        rows: vec![vec![(col.clone(), Value::Int(estimate))]],
        columns: vec![col],
    })
}

// ---------------------------------------------------------------------------
// Placeholder binding
// ---------------------------------------------------------------------------

/// A textual-order binder for `?` and `:name` placeholders. The positional
/// queue is consumed left-to-right; named binds resolve from the map. Mixing the
/// two styles in one statement is rejected, matching the reference.
struct Binder {
    params: std::collections::VecDeque<Value>,
    named: Option<HashMap<String, Value>>,
    seen_positional: bool,
    seen_named: bool,
}

impl Binder {
    fn new(params: Vec<Value>, named: Option<HashMap<String, Value>>) -> Self {
        Binder {
            params: params.into_iter().collect(),
            named,
            seen_positional: false,
            seen_named: false,
        }
    }

    /// Bind an expression tree, replacing each placeholder with its literal.
    fn bind(&mut self, expr: &Expr) -> Result<Expr> {
        match expr {
            Expr::Placeholder => {
                self.seen_positional = true;
                self.reject_mixed()?;
                let v = self.params.pop_front().ok_or_else(too_few_bindings_error)?;
                Ok(Expr::Literal(v))
            }
            Expr::NamedParam(name) => {
                self.seen_named = true;
                self.reject_mixed()?;
                let named = self.named.as_ref().ok_or_else(|| {
                    EngineError::ProgrammingError(format!(
                        "a :name placeholder requires a dict param (none supplied for {name:?})"
                    ))
                })?;
                let v = named.get(name).cloned().ok_or_else(|| {
                    EngineError::ProgrammingError(format!(
                        "no value supplied for named parameter {name:?}"
                    ))
                })?;
                Ok(Expr::Literal(v))
            }
            Expr::Literal(v) => Ok(Expr::Literal(v.clone())),
            Expr::Column(n) => Ok(Expr::Column(n.clone())),
            Expr::Excluded(n) => Ok(Expr::Excluded(n.clone())),
            Expr::BinOp { op, left, right } => Ok(Expr::BinOp {
                op: op.clone(),
                left: Box::new(self.bind(left)?),
                right: Box::new(self.bind(right)?),
            }),
            Expr::UnaryOp { op, operand } => Ok(Expr::UnaryOp {
                op: op.clone(),
                operand: Box::new(self.bind(operand)?),
            }),
            Expr::IsNull { operand, negated } => Ok(Expr::IsNull {
                operand: Box::new(self.bind(operand)?),
                negated: *negated,
            }),
            Expr::InList { operand, items, .. } => {
                let bound_items: Vec<Expr> =
                    items.iter().map(|i| self.bind(i)).collect::<Result<_>>()?;
                // Attach an O(1) membership set when every bound item is a
                // TEXT literal (see `Expr::InList::text_set`); small lists
                // stay linear — the set only pays off when the list is large
                // enough that a per-row scan of it dominates.
                let text_set = if bound_items.len() >= 16
                    && bound_items
                        .iter()
                        .all(|i| matches!(i, Expr::Literal(Value::Text(_))))
                {
                    Some(std::sync::Arc::new(
                        bound_items
                            .iter()
                            .filter_map(|i| match i {
                                Expr::Literal(Value::Text(s)) => Some(s.clone()),
                                _ => None,
                            })
                            .collect::<std::collections::HashSet<String>>(),
                    ))
                } else {
                    None
                };
                Ok(Expr::InList {
                    operand: Box::new(self.bind(operand)?),
                    items: bound_items,
                    text_set,
                })
            }
            Expr::Coalesce(args) => Ok(Expr::Coalesce(
                args.iter().map(|a| self.bind(a)).collect::<Result<_>>()?,
            )),
            Expr::FuncCall { name, args } => Ok(Expr::FuncCall {
                name: name.clone(),
                args: args.iter().map(|a| self.bind(a)).collect::<Result<_>>()?,
            }),
        }
    }

    /// Resolve a LIMIT / OFFSET value: a literal stays, a `?` consumes the next
    /// positional param (a NULL value clears the clause, matching the reference).
    fn resolve_limit(&mut self, literal: Option<i64>, is_param: bool) -> Result<Option<i64>> {
        if !is_param {
            return Ok(literal);
        }
        self.seen_positional = true;
        self.reject_mixed()?;
        match self.params.pop_front() {
            Some(Value::Null) | None => Ok(None),
            Some(Value::Int(i)) => Ok(Some(i)),
            Some(Value::Float(f)) => Ok(Some(f as i64)),
            Some(Value::Text(s)) => Ok(s.parse::<i64>().ok()),
            Some(_) => Ok(None),
        }
    }

    fn reject_mixed(&self) -> Result<()> {
        if self.seen_positional && self.seen_named {
            return Err(EngineError::ProgrammingError(
                "cannot mix named (:name) and positional (?) parameters in one statement"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Fail loud on surplus positional params left unconsumed.
    fn finish(&self) -> Result<()> {
        if !self.params.is_empty() {
            return Err(too_many_bindings_error(self.params.len()));
        }
        Ok(())
    }
}

// ===========================================================================
// Write executor: INSERT / ON CONFLICT / UPDATE / DELETE / executemany.
// ===========================================================================

/// The result of a write statement: the autoincrement label exposed as the
/// cursor's `lastrowid` (when one was injected) and the affected row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WriteOutcome {
    /// The injected `vec_label` (the AUTOINCREMENT high-water mark) for an
    /// INSERT into a table with one, exposed as `lastrowid`; `None` otherwise.
    pub lastrowid: Option<i64>,
    /// The number of rows the statement inserted, updated, or deleted.
    pub rowcount: i64,
}

/// Whether a per-statement write opens its own store transaction or runs inside
/// an already-open batch transaction.
///
/// The reference executor wraps every single INSERT/UPDATE/DELETE in its own
/// `begin_write()/commit()`. Under `executemany` (or an explicit outer BEGIN)
/// that inner per-statement transaction is suppressed so the whole batch commits
/// as one transaction and never double-opens the pager — mirroring the
/// reference's `_suppress_pager_txn` no-op swap. `Suppressed` is that flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TxnScope {
    /// Open and commit a store transaction around this single statement.
    Own,
    /// Run inside the caller's already-open transaction; do not begin/commit.
    Suppressed,
}

impl TxnScope {
    fn owns(self) -> bool {
        matches!(self, TxnScope::Own)
    }
}

/// A per-table `id` → B-tree row-key index for the `WHERE id = '<literal>'`
/// point-lookup fast path (SELECT, UPDATE, DELETE).
///
/// The records table keys its B-tree on `vec_label` (the INTEGER PRIMARY KEY),
/// not on `id` (a separate TEXT column), so a lookup by `id` has no B-tree key
/// descent available — without this index every `WHERE id = ?` resolves by a
/// whole-table scan, and a recall that issues dozens of per-candidate `id`
/// lookups degrades to O(rows × lookups). The index turns each lookup into an
/// O(1) map probe plus one O(tree-height) `tree.get(row_key)`, mirroring how
/// stdlib sqlite3 serves the same query through `idx_records_id`.
///
/// Built and owned by the caller (the connection shim), maintained by the
/// executor on insert / update / delete, dropped on rollback / commit-reconcile.
/// Persisted to the on-disk sidecar alongside the [`ColIndex`] under the same
/// write-generation fingerprint, so a warm boot loads the id→key map instead of
/// paying the first-lookup `ensure_built` scan; the lazy rebuild is always the
/// fallback (correctness never depends on the sidecar). The `built` flag
/// distinguishes an empty-but-built index (an empty table, do not re-scan) from
/// an absent one (populate on first use).
/// The postings map lives behind an `Arc` so a `Clone` of the index is an
/// atomic refcount bump, never a deep copy of the postings — a decoded sidecar
/// snapshot can be shared across every read-only connection in the process,
/// and only a connection that actually MUTATES its index (the writer's
/// maintenance, or a lazy build where no sidecar was loadable) pays a
/// copy-on-write materialization via `Arc::make_mut`.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct IdIndex {
    map: std::sync::Arc<HashMap<String, i64>>,
    built: bool,
}

impl IdIndex {
    /// An empty, unbuilt index. The first [`ensure_built`](Self::ensure_built)
    /// populates it with one whole-table scan; later lookups are O(1).
    pub fn new() -> Self {
        IdIndex::default()
    }

    /// Populate the index with one whole-table scan the first time it is needed
    /// for a table; later calls are no-ops. Maps every existing row's `id` text
    /// to its B-tree key. Rows with a NULL / non-text `id` are skipped (they are
    /// not reachable by an `id = '<text>'` equality). After this returns,
    /// `lookup` / `insert` are O(1).
    pub fn ensure_built(&mut self, tree: &lillibrain::Tree, col_names: &[String]) -> Result<()> {
        if self.built {
            return Ok(());
        }
        // Column-selective build: decode ONLY the `id` cell per row — never
        // the wide payload (the embedding BLOB dominates a full decode and
        // has no business in an id→key index build). Keeps a lazy rebuild on
        // a fresh connection a cheap streaming pass instead of a whole-table
        // full decode on whatever read path tripped it.
        let mut fresh: HashMap<String, i64> = HashMap::new();
        let Some(id_pos) = col_names.iter().position(|c| c == "id") else {
            // No id column: an empty built index (no id lookup can match).
            self.map = std::sync::Arc::new(fresh);
            self.built = true;
            return Ok(());
        };
        let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
        needed.insert("id".to_string());
        let mask = mask_from_names(col_names, &needed);
        tree.scan_cells_with(|key, payload| {
            let vals = crate::rowcodec::decode_row_columns(&payload, &mask)?;
            if let Some(Value::Text(id)) = vals.get(id_pos) {
                fresh.insert(id.clone(), key);
            }
            Ok(())
        })
        .map_err(|e: EngineError| e)?;
        self.map = std::sync::Arc::new(fresh);
        self.built = true;
        Ok(())
    }

    /// True once the one-time population scan has run for this table.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Force this index to own a private copy of its postings (copy-on-write
    /// materialization), so later per-row maintenance never pays the deep copy
    /// on the write path. Called once at open for a read-write connection whose
    /// indexes were loaded from the shared sidecar snapshot.
    pub fn unshare(&mut self) {
        std::sync::Arc::make_mut(&mut self.map);
    }

    /// O(1) lookup of the B-tree key for an `id` text. Only meaningful after
    /// [`ensure_built`](Self::ensure_built).
    pub fn lookup(&self, id: &str) -> Option<i64> {
        self.map.get(id).copied()
    }

    /// Record an `id` → row-key entry for a freshly inserted / reconciled row.
    /// A no-op when the index has not been built, so the next `ensure_built`
    /// rebuilds from the tree (which already includes the new row) rather than
    /// priming a partial map.
    fn insert(&mut self, id: String, key: i64) {
        if self.built {
            std::sync::Arc::make_mut(&mut self.map).insert(id, key);
        }
    }

    /// Drop an `id` from the index when its row is deleted or its `id` rewritten.
    fn remove(&mut self, id: &str) {
        std::sync::Arc::make_mut(&mut self.map).remove(id);
    }

    /// Invalidate the index so the next lookup re-populates it from the tree.
    /// Used when a write path that does not maintain the map per-row (a non-id
    /// UPDATE / a multi-row DELETE) may have changed the indexed rows.
    fn invalidate(&mut self) {
        self.map = std::sync::Arc::new(HashMap::new());
        self.built = false;
    }
}

/// A per-table, non-unique `column-value → [B-tree row-key]` index for the
/// `WHERE col IN (...)` scan path (the recall edge-adjacency lookup).
///
/// The recall pipeline resolves contradiction/adjacency edges with
/// `SELECT ... FROM edges WHERE (src IN (...) OR dst IN (...))`. The edges table
/// has no B-tree key descent on `src` / `dst` (it keys on a synthetic row-key),
/// so without an index every such lookup scans the whole edges table — O(edges)
/// per recall. This index maps each indexed column's value to the row-keys that
/// carry it, turning the lookup into a union of O(1) probes plus one
/// `tree.get(row_key)` per candidate, mirroring how stdlib sqlite3 serves the
/// same query through `idx_edges_src` / `idx_edges_dst`.
///
/// A value maps to MANY row-keys (many edges share a `src`), so the map values
/// are vectors. Built and owned by the caller (the connection shim). It is
/// built lazily on first use OR loaded from the on-disk sidecar when the
/// stamped write-generation matches the table's committed generation (see
/// [`crate::conn::Connection::persist_col_indexes`] and the load gate in
/// `from_store`); maintained incrementally across committed writes
/// ([`Self::insert_row`] / [`Self::remove_row`]); cleared on rollback or a
/// tainted commit; re-emitted to the sidecar by the maintenance pass. The
/// sidecar is a boot accelerator — correctness never depends on it, because the
/// lazy [`Self::ensure_built`] rebuild is always available as the fallback.
/// The postings map lives behind an `Arc` (same copy-on-write discipline as
/// [`IdIndex`]): a `Clone` is a refcount bump, so a decoded sidecar snapshot is
/// shared across every read-only connection; only a mutating holder pays the
/// deep-copy materialization via `Arc::make_mut`.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct ColIndex {
    /// The indexed columns (e.g. `["src", "dst"]` for the edges table).
    columns: Vec<String>,
    /// `(column, value-tag) → row-keys`. The value tag is the same injective
    /// per-storage-class encoding the conflict key uses, so an int and a numeric
    /// string never collide.
    map: std::sync::Arc<HashMap<(String, String), Vec<i64>>>,
    built: bool,
}

impl ColIndex {
    /// An empty index over `columns`. Unbuilt until [`ensure_built`].
    pub fn new(columns: Vec<String>) -> Self {
        ColIndex { columns, map: std::sync::Arc::new(HashMap::new()), built: false }
    }

    /// True once the population scan has run.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// True when the index carries no postings (an empty table, or unbuilt).
    /// Exposed for tests that assert a persisted index was actually populated.
    pub fn map_is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Consume the index, returning its raw `(column, value-tag) → row-keys`
    /// postings map. Exposed for tests that need to normalize (sort) the
    /// postings before a cross-copy comparison — `HashMap` iteration order is
    /// not deterministic, so a byte comparison of two independently-persisted
    /// envelopes must sort both sides first.
    pub fn into_postings(self) -> HashMap<(String, String), Vec<i64>> {
        std::sync::Arc::try_unwrap(self.map).unwrap_or_else(|shared| (*shared).clone())
    }

    /// Force this index to own a private copy of its postings (copy-on-write
    /// materialization), so later per-row maintenance never pays the deep copy
    /// on the write path. Called once at open for a read-write connection whose
    /// indexes were loaded from the shared sidecar snapshot.
    pub fn unshare(&mut self) {
        std::sync::Arc::make_mut(&mut self.map);
    }

    /// Populate the index with one whole-table scan the first time it is needed.
    /// Maps every row's value in each indexed column to that row's B-tree key.
    /// NULL values are skipped (a `col IN (...)` never matches NULL). After this
    /// returns, [`probe`](Self::probe) is O(matches) per value.
    pub fn ensure_built(&mut self, tree: &lillibrain::Tree, col_names: &[String]) -> Result<()> {
        if self.built {
            return Ok(());
        }
        // Column-selective build: decode ONLY the indexed columns per row —
        // never the wide payload (the embedding BLOB dominates a full decode
        // and is never an indexed column). Keeps a lazy rebuild on a fresh
        // connection a cheap streaming pass instead of a whole-table full
        // decode on whatever read path tripped it.
        let mut fresh: HashMap<(String, String), Vec<i64>> = HashMap::new();
        let needed: std::collections::HashSet<String> = self.columns.iter().cloned().collect();
        let mask = mask_from_names(col_names, &needed);
        let positions: Vec<(usize, &String)> = self
            .columns
            .iter()
            .filter_map(|col| col_names.iter().position(|c| c == col).map(|p| (p, col)))
            .collect();
        tree.scan_cells_with(|key, payload| {
            let vals = crate::rowcodec::decode_row_columns(&payload, &mask)?;
            for &(pos, col) in &positions {
                let v = vals.get(pos).cloned().unwrap_or(Value::Null);
                if let Some(tag) = scalar_index_tag(&v) {
                    fresh.entry((col.clone(), tag)).or_default().push(key);
                }
            }
            Ok(())
        })
        .map_err(|e: EngineError| e)?;
        self.map = std::sync::Arc::new(fresh);
        self.built = true;
        Ok(())
    }

    /// The indexed column set this index covers.
    pub fn columns(&self) -> &[String] {
        &self.columns
    }

    /// True when `col` is one of the indexed columns.
    pub fn indexes(&self, col: &str) -> bool {
        self.columns.iter().any(|c| c == col)
    }

    /// True once the population scan has run for this table — the discriminator
    /// the write paths use to decide between an incremental entry mutation (built)
    /// and a no-op that defers to the next [`ensure_built`] rebuild (unbuilt).
    pub fn built(&self) -> bool {
        self.built
    }

    /// The row-keys carrying `value` in `col`. Empty when none (or `col` is not
    /// indexed / the value is NULL). Only meaningful after [`ensure_built`].
    fn probe(&self, col: &str, value: &Value) -> &[i64] {
        match scalar_index_tag(value) {
            Some(tag) => self
                .map
                .get(&(col.to_string(), tag))
                .map(|v| v.as_slice())
                .unwrap_or(&[]),
            None => &[],
        }
    }

    /// Record a freshly written row's indexed-column values, mapping each indexed
    /// column's value to `key`. A no-op when the index has not been built, so the
    /// next [`ensure_built`] rebuilds from the tree (which already includes the new
    /// row) rather than priming a partial map. NULL values are skipped (a
    /// `col IN (...)` never matches NULL), mirroring [`ensure_built`]. The supplied
    /// `row` need only carry the indexed columns; absent columns read as NULL.
    fn insert_row(&mut self, row: &Row, key: i64) {
        if !self.built {
            return;
        }
        let map = std::sync::Arc::make_mut(&mut self.map);
        for col in &self.columns {
            let v = row.get_or_null(col);
            if let Some(tag) = scalar_index_tag(&v) {
                let bucket = map.entry((col.clone(), tag)).or_default();
                if !bucket.contains(&key) {
                    bucket.push(key);
                }
            }
        }
    }

    /// Drop a removed row's `key` from every indexed column's posting list, using
    /// the row's prior values to find the buckets it was filed under. A no-op when
    /// unbuilt. An emptied bucket is removed so a later probe sees no stale empty
    /// posting list. The caller passes the row as it stood BEFORE the delete /
    /// before a value-changing rewrite, so the entries removed are exactly the ones
    /// the prior `insert_row` (or `ensure_built`) created.
    fn remove_row(&mut self, row: &Row, key: i64) {
        if !self.built {
            return;
        }
        let map = std::sync::Arc::make_mut(&mut self.map);
        for col in &self.columns {
            let v = row.get_or_null(col);
            if let Some(tag) = scalar_index_tag(&v) {
                let map_key = (col.clone(), tag);
                if let Some(bucket) = map.get_mut(&map_key) {
                    bucket.retain(|&k| k != key);
                    if bucket.is_empty() {
                        map.remove(&map_key);
                    }
                }
            }
        }
    }

    /// Invalidate the index so the next probe re-populates it from the tree.
    fn invalidate(&mut self) {
        self.map = std::sync::Arc::new(HashMap::new());
        self.built = false;
    }
}

/// Tag a scalar value with its storage class, injectively across types, for an
/// index map key. Returns `None` for NULL (which never matches an equality /
/// IN membership test).
fn scalar_index_tag(v: &Value) -> Option<String> {
    match v {
        Value::Null => None,
        Value::Int(i) => Some(format!("i{i}")),
        Value::Float(f) => Some(format!("f{}", f.to_bits())),
        Value::Text(s) => Some(format!("t{s}")),
        Value::Blob(b) => Some(format!("b{}", hex_lower(b))),
    }
}

/// A per-table `(conflict-key tuple) → B-tree row-key` index for the
/// `ON CONFLICT (...) DO UPDATE/NOTHING` / `OR IGNORE` / `OR REPLACE` path.
///
/// Without it, every conflicting INSERT compares the attempted row against every
/// existing row via one whole-table scan, so a bulk UPSERT degrades to
/// O(rows^2). The index turns that comparison into an O(1) map lookup keyed by
/// the serialized conflict columns, keeping a bulk UPSERT linear.
///
/// Built and owned by the caller (the connection shim), maintained by the
/// executor on insert / update / replace / delete, and never persisted. The
/// `built` flag records whether the one-time population scan has run for the
/// table — an empty-but-built index is distinct from an absent one, so an empty
/// table is not re-scanned on every insert.
#[derive(Debug, Default, Clone)]
pub struct ConflictIndex {
    map: HashMap<String, i64>,
    built: bool,
}

impl ConflictIndex {
    /// Populate the index with one whole-table scan the first time it is needed
    /// for a table; later calls are no-ops. Maps every existing row's serialized
    /// conflict-key tuple to its B-tree key. NULL-bearing tuples are skipped (they
    /// cannot conflict). After this returns, `lookup` / `insert` are O(1).
    fn ensure_built(
        &mut self,
        tree: &lillibrain::Tree,
        conflict_keys: &[String],
        col_names: &[String],
    ) -> Result<()> {
        if self.built {
            return Ok(());
        }
        // Column-selective build: decode ONLY the conflict-key columns per
        // row (same discipline as the id/col index builds) so a lazy rebuild
        // never pays a whole-table full decode of wide payload columns.
        let needed: std::collections::HashSet<String> = conflict_keys.iter().cloned().collect();
        let mask = mask_from_names(col_names, &needed);
        let positions: Vec<(usize, &String)> = conflict_keys
            .iter()
            .filter_map(|ck| col_names.iter().position(|c| c == ck).map(|p| (p, ck)))
            .collect();
        let map = &mut self.map;
        tree.scan_cells_with(|key, payload| {
            let vals = crate::rowcodec::decode_row_columns(&payload, &mask)?;
            let mut src: HashMap<String, Value> = HashMap::with_capacity(conflict_keys.len());
            for &(pos, ck) in &positions {
                src.insert(ck.clone(), vals.get(pos).cloned().unwrap_or(Value::Null));
            }
            if let Some(ckey) = conflict_key_string(conflict_keys, &src) {
                map.insert(ckey, key);
            }
            Ok(())
        })
        .map_err(|e: EngineError| e)?;
        self.built = true;
        Ok(())
    }

    /// O(1) lookup of the B-tree key for a serialized conflict-key tuple. Only
    /// valid after [`ensure_built`](Self::ensure_built).
    fn lookup(&self, ckey: &str) -> Option<i64> {
        self.map.get(ckey).copied()
    }

    /// Record the conflict-key → row-key entry for a freshly inserted row.
    fn insert(&mut self, ckey: String, key: i64) {
        self.map.insert(ckey, key);
    }

    /// Drop a conflict-key entry when a DO UPDATE / OR REPLACE rewrites a
    /// conflict-key column, so the old key value no longer maps to the rewritten
    /// row (a later insert of the old value inserts fresh, not falsely conflicts).
    /// Paired with [`insert`](Self::insert) of the new key at the same row-key.
    fn remove(&mut self, ckey: &str) {
        self.map.remove(ckey);
    }

    /// Invalidate the index so the next conflict probe re-populates it. Used when
    /// a write path that does not maintain the conflict map (a non-key UPDATE that
    /// could touch conflict columns, or a DELETE without per-row keys) may have
    /// changed the indexed rows.
    fn invalidate(&mut self) {
        self.map.clear();
        self.built = false;
    }
}

/// The ordered-index bucket tag: the `(storage-class rank, within-class key)`
/// collation pair from [`order_key`], so BTreeMap iteration order equals the
/// engine's ORDER BY order for a plain (non-`datetime()`) single-column key
/// across every storage class — NULL < number < text < blob.
type OrderedTag = (u8, SortKey);

/// A per-table ordered secondary index backed by a
/// `BTreeMap<OrderedTag, Vec<i64>>`.
///
/// Maps each value's collation tag ([`order_key`] output, total across every
/// storage class including NULL) to the B-tree row-keys that carry it.  Within
/// each bucket row-keys are stored in their B-tree scan order (ascending),
/// which is the same input order the full-scan path feeds into `order_rows`.
/// The stable sort in `order_rows` therefore produces byte-identical output
/// whether the row set came from the index or from a full table scan — the
/// tie order matches the scan path.
///
/// The index ONLY narrows the candidate row set; `order_rows` + truncate always
/// produces the final order.  BTreeMap iteration order is never emitted as the
/// result order.
///
/// Built lazily on first use (one column-selective streaming scan that never
/// decodes unrelated columns such as the embedding BLOB), maintained
/// incrementally on insert/update/delete (no-op when unbuilt), cleared on
/// rollback and tainted commit.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct OrderedColIndex {
    /// The single column this index covers (e.g. `"ts"` for events).
    column: String,
    /// `collation tag → row-keys (ascending scan order within bucket)`.
    map: BTreeMap<OrderedTag, Vec<i64>>,
    built: bool,
}

impl OrderedColIndex {
    /// An empty, unbuilt index over `column`.
    pub fn new(column: impl Into<String>) -> Self {
        OrderedColIndex { column: column.into(), map: BTreeMap::new(), built: false }
    }

    /// The column this index covers.
    pub fn column(&self) -> &str {
        &self.column
    }

    /// True once the one-time population scan has run.
    pub fn is_built(&self) -> bool {
        self.built
    }

    /// Populate the index with one column-selective streaming scan the first
    /// time it is needed.
    ///
    /// Files EVERY row (including NULL-valued rows, under the NULL-class tag)
    /// by its [`order_key`] collation tag, so an ordered walk covers the whole
    /// table across every storage class — the same total order `order_rows`
    /// sorts by.  Only the indexed column is decoded per row (never the large
    /// embedding BLOB), so the one-time build cost is a cheap streaming pass.
    /// Row-keys accumulate in ascending scan order within each bucket,
    /// matching the full-scan traversal order so `order_rows` produces
    /// byte-identical tie order.
    pub fn ensure_built(
        &mut self,
        tree: &lillibrain::Tree,
        col_names: &[String],
    ) -> Result<()> {
        if self.built {
            return Ok(());
        }
        self.map.clear();
        let Some(col_pos) = col_names.iter().position(|c| c == &self.column) else {
            // A column absent from the catalog can never be read by a query;
            // an empty built index keeps every walk a no-op.
            self.built = true;
            return Ok(());
        };
        let mut needed: std::collections::HashSet<String> = std::collections::HashSet::new();
        needed.insert(self.column.clone());
        let mask = mask_from_names(col_names, &needed);
        let map = &mut self.map;
        tree.scan_cells_with(|key, payload| {
            let vals = crate::rowcodec::decode_row_columns(&payload, &mask)?;
            let v = vals.get(col_pos).cloned().unwrap_or(Value::Null);
            map.entry(order_key(&v)).or_default().push(key);
            Ok(())
        })
        .map_err(|e: EngineError| e)?;
        self.built = true;
        Ok(())
    }

    /// Return a candidate row-key superset for `ORDER BY <col> DESC LIMIT limit`.
    ///
    /// Walks the BTreeMap in descending tag order from the top (or from the
    /// inclusive lower bound `ge_bound` when a `col >= ?` conjunct narrows the
    /// range).  Gathers whole buckets until at least `limit` keys have been
    /// collected AND the current bucket is fully consumed — never stops
    /// mid-bucket, so a tie group is always whole when `order_rows` truncates.
    ///
    /// The returned vector is a SUPERSET of the true top-k.  The caller must
    /// decode each key, re-apply the full bound predicate, then run `order_rows`
    /// + truncate to produce the final byte-identical result.
    pub fn candidates_desc(
        &self,
        limit: usize,
        ge_bound: Option<&str>,
    ) -> Vec<i64> {
        let mut out: Vec<i64> = Vec::new();
        let iter: Box<dyn Iterator<Item = (&OrderedTag, &Vec<i64>)>> = match ge_bound {
            Some(lo) => Box::new(self.map.range(order_key(&Value::Text(lo.to_string()))..)),
            None => Box::new(self.map.iter()),
        };
        // Collect into a vec so we can reverse (BTreeMap has no DoubleEndedIterator
        // for range in stable Rust without the nightly `rev` on range iterators;
        // for the full-map case `.iter().rev()` works — handle both paths).
        let buckets: Vec<(&OrderedTag, &Vec<i64>)> = iter.collect();
        for (_, keys) in buckets.iter().rev() {
            // Add the whole bucket before checking the limit so a tie group is
            // never split: if the first key in this bucket would reach the limit
            // we still add every key in the bucket (the caller's `order_rows` +
            // truncate handles the exact cut inside the group correctly).
            out.extend_from_slice(keys);
            if out.len() >= limit {
                // Bucket is fully consumed — safe to stop.
                break;
            }
        }
        out
    }

    /// Record a freshly written row's ordered-column value → row-key mapping.
    ///
    /// A no-op when the index has not been built, so the next `ensure_built`
    /// rebuilds from the tree (which already includes the new row) rather than
    /// priming a partial map.  Every storage class is filed (including NULL,
    /// under the NULL-class tag), mirroring `ensure_built`.
    ///
    /// The key is inserted in ascending-key position within its bucket, matching
    /// the full-scan B-tree traversal order `ensure_built` files keys in.  A
    /// DO UPDATE / OR REPLACE re-file (`remove_row` then `insert_row` with the
    /// SAME key) therefore restores the key to its sorted slot rather than the
    /// bucket tail, so the within-tie order stays byte-identical to the scan path
    /// regardless of write path.  The sorted `partition_point` probe also makes
    /// the duplicate-key guard an O(log n) presence check.
    pub fn insert_row(&mut self, row: &Row, key: i64) {
        if !self.built {
            return;
        }
        let tag = order_key(&row.get_or_null(&self.column));
        let bucket = self.map.entry(tag).or_default();
        let pos = bucket.partition_point(|&k| k < key);
        if bucket.get(pos) != Some(&key) {
            bucket.insert(pos, key);
        }
    }

    /// Drop a removed row's key from the index bucket it was filed under.
    ///
    /// A no-op when the index has not been built.  An emptied bucket is removed
    /// so a later ordered walk never returns stale empty buckets.  The
    /// caller passes the row as it stood BEFORE the delete / value-changing
    /// rewrite so the correct bucket entry is removed.
    pub fn remove_row(&mut self, row: &Row, key: i64) {
        if !self.built {
            return;
        }
        let tag = order_key(&row.get_or_null(&self.column));
        if let Some(bucket) = self.map.get_mut(&tag) {
            bucket.retain(|&k| k != key);
            if bucket.is_empty() {
                self.map.remove(&tag);
            }
        }
    }

    /// The buckets in walk order (`desc` reverses), each `(collation-tag,
    /// row-keys-in-scan-order)`, optionally range-restricted to tags at or
    /// above `lower` — the caller's extracted lower bound in [`order_key`]
    /// space.  For an ascending walk the bound is the start; for a descending
    /// walk it is the stop boundary.  The caller re-applies the full predicate
    /// to every candidate row, so the bound only narrows which buckets are
    /// visited, never what a visited row must satisfy.
    fn walk_buckets(&self, desc: bool, lower: Option<&OrderedTag>) -> Vec<(&OrderedTag, &Vec<i64>)> {
        let iter: Box<dyn Iterator<Item = (&OrderedTag, &Vec<i64>)>> = match lower {
            Some(lo) => Box::new(self.map.range(lo.clone()..)),
            None => Box::new(self.map.iter()),
        };
        let mut buckets: Vec<(&OrderedTag, &Vec<i64>)> = iter.collect();
        if desc {
            buckets.reverse();
        }
        buckets
    }

    /// Invalidate the index so the next probe re-populates it from the tree.
    pub fn invalidate(&mut self) {
        self.map.clear();
        self.built = false;
    }
}

/// Extract a walk-start lower bound for an ordered-index walk from
/// `col >= <literal>` / `col > <literal>` conjuncts in the WHERE, mapped into
/// [`order_key`] space with the same numeric-text coercion `sql_compare`
/// applies: a numeric or numeric-string bound seeds at its numeric position
/// (below every text bucket, so no numerically-coercible row is skipped) and a
/// non-numeric text bound seeds at its text position (numeric rows can never
/// satisfy `number >= text`, so skipping the numeric buckets is sound).
/// Returns the TIGHTEST (highest) such bound.
///
/// This is an optimization only — every walked candidate still passes the full
/// predicate — so an unrecognized conjunct is simply ignored, never rejected.
fn ordered_walk_lower_bound(where_expr: Option<&Expr>, col: &str) -> Option<OrderedTag> {
    fn bound_tag(v: &Value) -> Option<OrderedTag> {
        match v {
            Value::Int(_) | Value::Float(_) => Some(order_key(v)),
            Value::Text(s) => {
                if let Ok(i) = s.parse::<i64>() {
                    Some(order_key(&Value::Int(i)))
                } else if let Ok(f) = s.parse::<f64>() {
                    Some(order_key(&Value::Float(f)))
                } else {
                    Some(order_key(v))
                }
            }
            _ => None,
        }
    }
    fn collect(e: &Expr, col: &str, best: &mut Option<OrderedTag>) {
        match e {
            Expr::BinOp { op, left, right } if op == "AND" => {
                collect(left, col, best);
                collect(right, col, best);
            }
            Expr::BinOp { op, left, right } if op == ">=" || op == ">" => {
                if let (Expr::Column(c), Expr::Literal(v)) = (left.as_ref(), right.as_ref()) {
                    if c == col {
                        if let Some(tag) = bound_tag(v) {
                            if best.as_ref().map(|b| tag > *b).unwrap_or(true) {
                                *best = Some(tag);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    let mut best: Option<OrderedTag> = None;
    if let Some(w) = where_expr {
        collect(w, col, &mut best);
    }
    best
}

/// A NUL byte never appears inside a serialized scalar below, so it is a safe
/// field separator and a safe variant tag prefix — the joined key is injective
/// across the value types (an int `1` and the text `"1"` produce distinct keys).
const CKEY_SEP: char = '\u{0}';

/// Serialize a list of conflict-key values into a single injective map key.
///
/// Returns `None` when any value is NULL: SQL uniqueness treats `NULL` as
/// distinct from every value including another `NULL`, so a row with a NULL in
/// any conflict column never conflicts and must not be indexed (it always
/// inserts fresh). Each scalar is tagged by storage class so values of different
/// types never collide.
fn conflict_key_string(keys: &[String], source: &HashMap<String, Value>) -> Option<String> {
    let mut parts: Vec<String> = Vec::with_capacity(keys.len());
    for k in keys {
        match source.get(k).cloned().unwrap_or(Value::Null) {
            Value::Null => return None,
            Value::Int(i) => parts.push(format!("i{i}")),
            Value::Float(f) => parts.push(format!("f{}", f.to_bits())),
            Value::Text(s) => parts.push(format!("t{s}")),
            Value::Blob(b) => parts.push(format!("b{}", hex_lower(&b))),
        }
    }
    Some(parts.join(&CKEY_SEP.to_string()))
}

/// Serialize a row's conflict-key tuple into the same injective map key the
/// [`ConflictIndex`] uses. Returns `None` when any conflict-key value is NULL (a
/// NULL-bearing tuple is never indexed). Used to re-key the conflict index when a
/// DO UPDATE / OR REPLACE rewrites a conflict-key column.
fn row_conflict_key(keys: &[String], row: &Row) -> Option<String> {
    let mut src: HashMap<String, Value> = HashMap::with_capacity(keys.len());
    for k in keys {
        src.insert(k.clone(), row.get_or_null(k));
    }
    conflict_key_string(keys, &src)
}

/// Lowercase hex of a byte slice, for the blob conflict-key tag.
fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// An RAII guard around an owned store write transaction.
///
/// `begin` opens a transaction for an owned scope (a no-op for a suppressed scope,
/// which runs inside the caller's already-open batch). The work between `begin`
/// and `commit` may fail at any `?`; if it does, the guard's `Drop` rolls the
/// owned transaction back, so a mid-statement error never leaves the pager with a
/// leaked open transaction (which would make the next write fail with "a
/// transaction is already open"). A suppressed scope is never rolled back by the
/// guard — aborting the outer batch is the caller's decision, not a single
/// statement's. `commit` defuses the guard and commits; after a successful commit
/// the `Drop` does nothing.
struct TxnGuard<'a> {
    store: &'a Store,
    scope: TxnScope,
    /// Set once `commit` runs (or the scope is suppressed), so `Drop` is inert.
    done: bool,
}

impl<'a> TxnGuard<'a> {
    /// Open an owned write transaction (a no-op for a suppressed scope) and return
    /// a guard that rolls it back on `Drop` unless `commit` is invoked first.
    fn begin(store: &'a Store, scope: TxnScope) -> Result<TxnGuard<'a>> {
        if scope.owns() {
            store.begin_write().map_err(store_err)?;
        }
        // A suppressed scope owns no transaction, so the guard is inert from the
        // start — it must never touch the caller's outer batch.
        let done = !scope.owns();
        Ok(TxnGuard { store, scope, done })
    }

    /// Commit the owned transaction (a no-op for a suppressed scope) and defuse the
    /// guard so its `Drop` does not also roll back.
    fn commit(mut self) -> Result<()> {
        if self.scope.owns() {
            self.store.commit().map_err(store_err)?;
        }
        self.done = true;
        Ok(())
    }
}

impl Drop for TxnGuard<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        // An owned transaction is still open because an early `?` skipped `commit`.
        // Roll it back so no open transaction leaks. `Store::rollback` is safe even
        // when no transaction is open; any rollback error is intentionally ignored
        // here (Drop cannot return it) — the next write's `begin_write` would
        // surface a still-open transaction loudly.
        let _ = self.store.rollback();
    }
}

/// Adapt a storage-layer error into the engine error taxonomy, preserving the
/// corruption class so the boundary can surface it as a dedicated exception.
fn store_err(e: lillibrain::StoreError) -> EngineError {
    EngineError::from(e)
}

/// Decode a stored record into a row map, padding a short record (written before
/// an ALTER TABLE ADD COLUMN) with trailing NULLs.
fn decode_to_row(payload: &[u8], col_names: &[String]) -> Result<Row> {
    let mut vals = decode_row(payload)?;
    while vals.len() < col_names.len() {
        vals.push(Value::Null);
    }
    let pairs: Vec<(String, Value)> = col_names.iter().cloned().zip(vals).collect();
    Ok(Row::from_pairs(pairs))
}

/// True when `table`'s primary key is a single AUTOINCREMENT-style `vec_label`
/// integer column — the records-shaped table whose INSERT injects the next HWM.
fn has_autoincrement_vec_label(cols: &[ColumnMeta]) -> bool {
    cols.iter()
        .any(|c| c.name.eq_ignore_ascii_case("vec_label") && c.primary_key && c.affinity == "INTEGER")
}

/// Build the catalog-ordered insert payload from a name→value map: unsupplied
/// columns take their DEFAULT (when declared) else NULL, each value is coerced to
/// its column affinity, and NOT NULL is enforced before any write so a constraint
/// failure never leaves a transaction half-open.
fn build_payload(
    cols: &[ColumnMeta],
    value_map: &HashMap<String, Value>,
    table: &str,
) -> Result<Vec<Value>> {
    let mut out: Vec<Value> = Vec::with_capacity(cols.len());
    for cm in cols {
        let supplied = value_map.get(&cm.name).cloned();
        let mut val = match supplied {
            Some(Value::Null) | None => cm.default.clone().unwrap_or(Value::Null),
            Some(v) => v,
        };
        val = coerce_affinity(val, &cm.affinity);
        out.push(val);
    }
    enforce_not_null(cols, &out, table)?;
    Ok(out)
}

/// Reject a catalog-ordered value list that leaves a NOT NULL column NULL, with
/// the sqlite3-shaped error. Shared by the fresh-insert payload build and the
/// DO UPDATE rewrite path so both enforce NOT NULL identically.
fn enforce_not_null(cols: &[ColumnMeta], vals: &[Value], table: &str) -> Result<()> {
    for (cm, val) in cols.iter().zip(vals.iter()) {
        if !cm.nullable && matches!(val, Value::Null) {
            return Err(EngineError::parse(format!(
                "NOT NULL constraint failed: {table}.{}",
                cm.name
            )));
        }
    }
    Ok(())
}

/// Resolve the conflict-key columns for an INSERT: an explicit `ON CONFLICT
/// (keys)` clause wins; otherwise OR IGNORE / OR REPLACE fall back to the
/// catalog primary key, then to the first column when no PK is declared.
fn resolve_conflict_keys(stmt: &InsertStmt, cols: &[ColumnMeta]) -> Vec<String> {
    if !stmt.conflict_keys.is_empty() {
        return stmt.conflict_keys.clone();
    }
    if stmt.or_ignore || stmt.or_replace {
        let pk: Vec<String> = cols
            .iter()
            .filter(|c| c.primary_key)
            .map(|c| c.name.clone())
            .collect();
        if !pk.is_empty() {
            return pk;
        }
        if let Some(first) = cols.first() {
            return vec![first.name.clone()];
        }
    }
    Vec::new()
}

/// Execute an INSERT (with `?`/`:name` binds, vec_label AUTOINCREMENT injection,
/// OR IGNORE / OR REPLACE, and ON CONFLICT DO UPDATE/NOTHING) against `store`.
///
/// The next B-tree key comes from `Tree::max_key()` (O(tree-height)) on the
/// non-conflict path, so a bulk insert loop never full-scans per row — the
/// quadratic-insert regression (a per-row whole-table sweep) cannot return. A
/// conflict clause must compare the attempted row against every existing row on
/// the conflict columns (there is no engine index on arbitrary columns), so that
/// one path does a single full scan and derives the next key from the same pass.
///
/// `id_index`, when supplied, is updated with the new row's `id`→row-key entry so
/// a later `WHERE id = '<lit>'` UPDATE/DELETE point-lookup stays O(log N).
#[allow(clippy::too_many_arguments)]
pub fn execute_insert(
    stmt: &InsertStmt,
    store: &Store,
    meta: &MetaTable,
    catalog: &Catalog,
    root_map: &std::collections::BTreeMap<String, u32>,
    params: Vec<Value>,
    named: Option<HashMap<String, Value>>,
    scope: TxnScope,
    id_index: Option<&mut IdIndex>,
    mut conflict_index: Option<&mut ConflictIndex>,
    mut col_index: Option<&mut ColIndex>,
    mut ordered_index: Option<&mut OrderedColIndex>,
) -> Result<WriteOutcome> {
    let table = &stmt.table;
    let root = *root_map
        .get(table)
        .ok_or_else(|| EngineError::parse(format!("insert: unknown table {table:?}")))?;
    let cols = catalog.get(table)?.to_vec();
    let col_names: Vec<String> = cols.iter().map(|c| c.name.clone()).collect();
    // Whether `table` carries a col-index at all (a live cache handle was
    // supplied — the caller only supplies one for a table with declared
    // indexed columns, see `catalog_col_index_columns`). Gates the
    // write-generation bump: a table with no col-index has no on-disk sidecar
    // entry to ever invalidate, so bumping its generation would be pure
    // overhead (and would spuriously fire the one-time meta-tree disk-seed
    // scan on a table that never needs the fence).
    let has_col_index = col_index.is_some();

    // Bind the VALUES expressions in textual order; INSERT carries no LIMIT.
    let mut binder = Binder::new(params, named);
    let mut bound_values: Vec<Value> = Vec::with_capacity(stmt.values.len());
    for v in &stmt.values {
        bound_values.push(eval_scalar(&binder.bind(v)?, &Row::new()));
    }
    binder.finish()?;

    // vec_label AUTOINCREMENT injection: when the table is records-shaped and the
    // caller did not name vec_label, the next high-water mark is prepended as the
    // vec_label column value and exposed as lastrowid (the reference injects it in
    // the connection layer before the executor runs).
    let mut insert_cols: Vec<String> = stmt.columns.clone();
    let names_vec_label = insert_cols
        .iter()
        .any(|c| c.eq_ignore_ascii_case("vec_label"));
    let mut lastrowid: Option<i64> = None;
    if !names_vec_label && has_autoincrement_vec_label(&cols) {
        // The label bump persists a meta row; route it inside the caller's batch
        // transaction when one is open (suppressed) so the pager is not
        // double-opened and the bump commits atomically with the data row.
        let label = match scope {
            TxnScope::Own => meta.next_vec_label(store, table)?,
            TxnScope::Suppressed => meta.next_vec_label_in_txn(store, table)?,
        };
        insert_cols.insert(0, "vec_label".to_string());
        bound_values.insert(0, Value::Int(label));
        lastrowid = Some(label);
    }

    // Map supplied column → value; a short value list pads with NULL.
    let mut value_map: HashMap<String, Value> = HashMap::new();
    for (i, col) in insert_cols.iter().enumerate() {
        let v = bound_values.get(i).cloned().unwrap_or(Value::Null);
        value_map.insert(col.clone(), v);
    }

    let conflict_keys = resolve_conflict_keys(stmt, &cols);
    let needs_conflict_scan = (stmt.conflict_action.is_some()
        || stmt.or_ignore
        || stmt.or_replace)
        && !conflict_keys.is_empty();

    // Resolve the conflicting row, if any. Two paths:
    //
    //   Indexed (a ConflictIndex is supplied): the index is lazily populated with
    //   one whole-table scan on first use, then every later conflict probe is an
    //   O(1) map lookup keyed by the serialized conflict columns. The next B-tree
    //   key comes from an O(tree-height) `max_key()` descent — never a per-row
    //   scan — so a bulk UPSERT stays linear. This is the durable fix for the
    //   O(rows^2) conflict-scan shape.
    //
    //   Unindexed (no ConflictIndex — e.g. a one-off statement with no caller
    //   cache): the original single-scan probe is preserved as the correct
    //   fallback. A NULL in any conflict key never matches (SQL uniqueness treats
    //   NULL as distinct), so such a row always inserts fresh.
    let tree = store.tree(root);

    let attempted_ckey = conflict_key_string(&conflict_keys, &value_map);

    // Find the existing row key the attempted row conflicts with, plus its decoded
    // row when one is found (the DO UPDATE branch reads its current values).
    let conflict_hit: Option<(i64, Row)> = if !needs_conflict_scan {
        None
    } else if let Some(idx) = conflict_index.as_deref_mut() {
        // Indexed probe: lazily populate once, then O(1) lookup. A NULL-bearing
        // attempted tuple never conflicts (its serialized key is None).
        idx.ensure_built(&tree, &conflict_keys, &col_names)?;
        match attempted_ckey.as_deref().and_then(|ck| idx.lookup(ck)) {
            Some(key) => {
                let payload = tree.get(key).map_err(store_err)?;
                match payload {
                    Some(p) => Some((key, decode_to_row(&p, &col_names)?)),
                    // The index pointed at a row that is gone: a consistency lapse.
                    // Drop to the safe full-scan probe rather than miss the conflict.
                    None => scan_conflict(&tree, &conflict_keys, &value_map, &col_names)?,
                }
            }
            None => None,
        }
    } else {
        scan_conflict(&tree, &conflict_keys, &value_map, &col_names)?
    };

    // Next key for a fresh insert: O(tree-height) max-key descent.
    let next_key = tree.max_key().map_err(store_err)?.map_or(1, |k| k + 1);

    if needs_conflict_scan {
        if let Some((key, row)) = conflict_hit {
            // A match: DO UPDATE applies the SET assignments (excluded.col resolves
            // to the attempted-insert value); DO NOTHING / OR IGNORE skips; OR
            // REPLACE overwrites the existing row in place at its key.
            //
            // The `id` of the conflicting row at `key` is captured before the
            // rewrite so the id-index can be reconciled if a DO UPDATE /
            // OR REPLACE changes it (the production UPSERT updates only non-key
            // columns, so `id` is stable today, but the index invariant must hold
            // for any caller that does rewrite a key column).
            let old_id = match row.get("id") {
                Some(Value::Text(s)) => Some(s.clone()),
                _ => None,
            };
            // The conflict-key tuple of the row BEFORE the rewrite, so the conflict
            // index can be re-keyed if a DO UPDATE / OR REPLACE rewrites a
            // conflict-key column. The production UPSERT updates only non-key
            // columns (the key is stable), so this is latent today — but the index
            // invariant must hold for any caller that does rewrite a key column.
            let old_ckey = row_conflict_key(&conflict_keys, &row);
            let new_ckey: Option<String>;
            let new_id: Option<String>;
            if stmt.conflict_action.as_deref() == Some("update") {
                let mut updated = row.clone();
                for (set_col, set_expr) in &stmt.set_clauses {
                    let new_val = match set_expr {
                        Expr::Excluded(col) => {
                            value_map.get(col).cloned().unwrap_or(Value::Null)
                        }
                        other => eval_scalar(other, &row),
                    };
                    updated.set(set_col.clone(), new_val);
                }
                new_id = match updated.get("id") {
                    Some(Value::Text(s)) => Some(s.clone()),
                    _ => None,
                };
                new_ckey = row_conflict_key(&conflict_keys, &updated);
                // A DO UPDATE that moves the conflict key onto a value already
                // held by a DIFFERENT existing row is a UNIQUE violation — stdlib
                // sqlite3 rejects it rather than letting two rows share the key.
                // Probe before opening the write transaction so no rewrite lands
                // and nothing needs rolling back. NULL-bearing new keys are never
                // unique-constrained (they always insert fresh).
                if new_ckey != old_ckey {
                    if let Some(nk) = &new_ckey {
                        if let Some(owner) = existing_owner_of_ckey(
                            &tree,
                            conflict_index.as_deref(),
                            &conflict_keys,
                            &col_names,
                            nk,
                        )? {
                            if owner != key {
                                return Err(unique_constraint_error(table, &conflict_keys));
                            }
                        }
                    }
                }
                let payload_vals: Vec<Value> =
                    col_names.iter().map(|cn| updated.get_or_null(cn)).collect();
                let guard = TxnGuard::begin(store, scope)?;
                // Enforce NOT NULL on the rewritten row inside the open transaction,
                // matching stdlib sqlite3 which rejects a DO UPDATE that nulls a
                // NOT NULL column. The guard rolls the transaction back on this
                // early return so no open transaction leaks.
                enforce_not_null(&cols, &payload_vals, table)?;
                tree.insert(key, &encode_row(&payload_vals)).map_err(store_err)?;
                if has_col_index {
                    meta.bump_col_generation(store, table)?;
                }
                guard.commit()?;
                // The rewrite may have changed an indexed column (the records
                // pending flag flips on this exact DO UPDATE path). Re-file the
                // row's col-index entries: drop the pre-rewrite values, add the
                // rewritten ones, both at the same row-key.
                if let Some(cidx) = col_index.as_deref_mut() {
                    cidx.remove_row(&row, key);
                    cidx.insert_row(&updated, key);
                }
                if let Some(oidx) = ordered_index.as_deref_mut() {
                    oidx.remove_row(&row, key);
                    oidx.insert_row(&updated, key);
                }
            } else if stmt.or_replace {
                let payload_vals = build_payload(&cols, &value_map, table)?;
                new_id = match value_map.get("id") {
                    Some(Value::Text(s)) => Some(s.clone()),
                    _ => None,
                };
                let replacement = Row::from_pairs(
                    col_names.iter().cloned().zip(payload_vals.iter().cloned()).collect(),
                );
                new_ckey = row_conflict_key(&conflict_keys, &replacement);
                let guard = TxnGuard::begin(store, scope)?;
                tree.insert(key, &encode_row(&payload_vals)).map_err(store_err)?;
                if has_col_index {
                    meta.bump_col_generation(store, table)?;
                }
                guard.commit()?;
                // OR REPLACE overwrites every column in place: re-file the
                // col-index from the pre-replace row to the replacement row.
                if let Some(cidx) = col_index.as_deref_mut() {
                    cidx.remove_row(&row, key);
                    cidx.insert_row(&replacement, key);
                }
                if let Some(oidx) = ordered_index.as_deref_mut() {
                    oidx.remove_row(&row, key);
                    oidx.insert_row(&replacement, key);
                }
            } else {
                // DO NOTHING / OR IGNORE: the row is unchanged.
                new_id = old_id.clone();
                new_ckey = old_ckey.clone();
            }
            // Reconcile the id-index when the rewrite changed the row's id: drop
            // the stale `old_id → key` entry and record the new one at the same
            // row-key, so a later `WHERE id = '<new>'` point lookup still finds it.
            if let Some(idx) = id_index {
                if new_id != old_id {
                    if let Some(old) = &old_id {
                        idx.remove(old);
                    }
                    if let Some(new) = &new_id {
                        idx.insert(new.clone(), key);
                    }
                }
            }
            // Reconcile the conflict index when the rewrite changed the row's
            // conflict-key tuple: drop the stale old key (so a later insert of the
            // old value inserts fresh, not a false conflict) and file the new key at
            // the same row-key (so a later insert of the new value conflicts). Only
            // when the index is built — an unbuilt index rebuilds from the tree on
            // its next probe. A NULL-bearing tuple yields None and is never indexed.
            if let Some(cidx) = conflict_index.as_deref_mut() {
                if cidx.built && new_ckey != old_ckey {
                    if let Some(old) = &old_ckey {
                        cidx.remove(old);
                    }
                    if let Some(new) = &new_ckey {
                        cidx.insert(new.clone(), key);
                    }
                }
            }
            // The OR REPLACE / DO UPDATE may have changed the conflicting row in
            // place at `key`; the id-index and conflict-index already reconciled
            // above. DO NOTHING / OR IGNORE: skip silently. No row inserted.
            return Ok(WriteOutcome { lastrowid, rowcount: 0 });
        }
        // No conflict found — fall through to the normal insert.
    }

    let payload_vals = build_payload(&cols, &value_map, table)?;
    let guard = TxnGuard::begin(store, scope)?;
    tree.insert(next_key, &encode_row(&payload_vals)).map_err(store_err)?;
    if has_col_index {
        meta.bump_col_generation(store, table)?;
    }
    guard.commit()?;

    // Keep the id→row-key fast-path index current for this fresh row.
    if let Some(idx) = id_index {
        if let Some(Value::Text(id)) = value_map.get("id") {
            idx.insert(id.clone(), next_key);
        }
    }
    // Keep the conflict-key index current: a fresh non-NULL conflict tuple maps to
    // the row just written. NULL-bearing tuples are never indexed (they cannot
    // conflict). The index is left untouched on the unindexed fallback path. It
    // is safe to insert into a built index here; if the index was never built
    // (no conflict clause on this INSERT) the entry simply primes it for a later
    // ensure_built no-op only when `built` is already set — otherwise the next
    // conflict probe rebuilds from the tree, which includes this row.
    if let (Some(idx), Some(ck)) = (conflict_index, attempted_ckey) {
        if idx.built {
            idx.insert(ck, next_key);
        }
    }
    // Keep the col-index current for this fresh row: map each indexed column's
    // value to the new row-key. A no-op when the index is unbuilt (the next probe
    // rebuilds from the tree, which includes this row), so a write to a table whose
    // col-index was never built stays free.
    if let Some(cidx) = col_index {
        let inserted = Row::from_pairs(
            col_names.iter().cloned().zip(payload_vals.iter().cloned()).collect(),
        );
        cidx.insert_row(&inserted, next_key);
    }
    // Keep the ordered index current for the fresh row (same discipline as ColIndex).
    if let Some(oidx) = ordered_index {
        let inserted = Row::from_pairs(
            col_names.iter().cloned().zip(payload_vals.iter().cloned()).collect(),
        );
        oidx.insert_row(&inserted, next_key);
    }

    Ok(WriteOutcome { lastrowid, rowcount: 1 })
}

/// Resolve which existing row, if any, already owns a serialized conflict-key
/// tuple. Prefers an O(1) probe of a built [`ConflictIndex`]; falls back to one
/// whole-table scan when no built index is supplied. Returns the owning row's
/// B-tree key, or `None` when the key is free. Used to detect a UNIQUE collision
/// when a DO UPDATE rewrites a conflict-key column onto a value held by a third
/// row — the case stdlib sqlite3 rejects.
fn existing_owner_of_ckey(
    tree: &lillibrain::Tree,
    conflict_index: Option<&ConflictIndex>,
    conflict_keys: &[String],
    col_names: &[String],
    ckey: &str,
) -> Result<Option<i64>> {
    if let Some(idx) = conflict_index {
        if idx.built {
            return Ok(idx.lookup(ckey));
        }
    }
    for (k, payload) in tree.full_scan().map_err(store_err)? {
        let row = decode_to_row(&payload, col_names)?;
        if row_conflict_key(conflict_keys, &row).as_deref() == Some(ckey) {
            return Ok(Some(k));
        }
    }
    Ok(None)
}

/// The sqlite3-shaped UNIQUE-constraint message naming the table-qualified
/// conflict columns (`UNIQUE constraint failed: t.a, t.b`), matching stdlib
/// sqlite3 for a conflict-key collision.
fn unique_constraint_error(table: &str, conflict_keys: &[String]) -> EngineError {
    let cols = conflict_keys
        .iter()
        .map(|c| format!("{table}.{c}"))
        .collect::<Vec<_>>()
        .join(", ");
    EngineError::parse(format!("UNIQUE constraint failed: {cols}"))
}

/// The unindexed conflict probe: one whole-table scan comparing the attempted row
/// against every existing row on the conflict columns. The correct fallback when
/// no [`ConflictIndex`] is supplied; O(rows) per call.
fn scan_conflict(
    tree: &lillibrain::Tree,
    conflict_keys: &[String],
    value_map: &HashMap<String, Value>,
    col_names: &[String],
) -> Result<Option<(i64, Row)>> {
    for (key, payload) in tree.full_scan().map_err(store_err)? {
        let row = decode_to_row(&payload, col_names)?;
        let matched = conflict_keys.iter().all(|ck| {
            let lhs = row.get_or_null(ck);
            let rhs = value_map.get(ck).cloned().unwrap_or(Value::Null);
            sql_compare("=", lhs, rhs) == Some(true)
        });
        if matched {
            return Ok(Some((key, row)));
        }
    }
    Ok(None)
}

/// Execute an UPDATE: bind SET then WHERE in textual order, select matching rows
/// (full scan + three-valued WHERE, or the `id = '<lit>'` point-lookup fast path
/// when an id-index is available), apply the SET assignments, and report the
/// affected row count.
#[allow(clippy::too_many_arguments)]
pub fn execute_update(
    stmt: &UpdateStmt,
    store: &Store,
    meta: &MetaTable,
    catalog: &Catalog,
    root_map: &std::collections::BTreeMap<String, u32>,
    params: Vec<Value>,
    named: Option<HashMap<String, Value>>,
    scope: TxnScope,
    id_index: Option<&mut IdIndex>,
    conflict_index: Option<&mut ConflictIndex>,
    col_index: Option<&mut ColIndex>,
    ordered_index: Option<&mut OrderedColIndex>,
) -> Result<WriteOutcome> {
    let table = &stmt.table;
    let root = *root_map
        .get(table)
        .ok_or_else(|| EngineError::parse(format!("update: unknown table {table:?}")))?;
    let col_names = catalog.column_names(table)?;
    let tree = store.tree(root);
    // See `execute_insert`'s `has_col_index`: gates the write-generation bump to
    // tables that actually carry a col-index, so a table with none never pays
    // the one-time meta-tree disk-seed scan for a fence it will never need.
    let has_col_index = col_index.is_some();

    // SET ?s bind before WHERE ?s (SQL textual order). Each SET rhs is a literal
    // or placeholder evaluated against an empty row.
    let mut binder = Binder::new(params, named);
    let mut bound_set: Vec<(String, Value)> = Vec::with_capacity(stmt.set_clauses.len());
    for (set_col, set_expr) in &stmt.set_clauses {
        let bound = binder.bind(set_expr)?;
        bound_set.push((set_col.clone(), eval_scalar(&bound, &Row::new())));
    }
    let bound_where = match &stmt.where_clause {
        Some(w) => Some(binder.bind(w)?),
        None => None,
    };
    binder.finish()?;

    let mut to_update: Vec<(i64, Row)> = Vec::new();
    let mut used_fast_path = false;
    let mut id_index = id_index;

    if let (Some(w), Some(idx)) = (&bound_where, id_index.as_deref_mut()) {
        if let Some(id_val) = extract_id_eq(w) {
            used_fast_path = true;
            idx.ensure_built(&tree, &col_names)?;
            if let Some(row_key) = idx.lookup(&id_val) {
                if let Some(payload) = tree.get(row_key).map_err(store_err)? {
                    let row = decode_to_row(&payload, &col_names)?;
                    if eval_predicate(w, &row) == Some(true) {
                        to_update.push((row_key, row));
                    }
                }
            } else {
                // id absent from the built index → the row does not exist; no scan.
                return Ok(WriteOutcome { lastrowid: None, rowcount: 0 });
            }
        }
    }

    if !used_fast_path {
        for (key, payload) in tree.full_scan().map_err(store_err)? {
            let row = decode_to_row(&payload, &col_names)?;
            let pass = match &bound_where {
                None => true,
                Some(w) => eval_predicate(w, &row) == Some(true),
            };
            if pass {
                to_update.push((key, row));
            }
        }
    }

    if to_update.is_empty() {
        return Ok(WriteOutcome { lastrowid: None, rowcount: 0 });
    }

    let cols = catalog.get(table)?.to_vec();
    // Whether any SET assignment touches a col-indexed column. When none do, the
    // col-index needs no maintenance (the matched rows keep their indexed values),
    // so a non-indexed-column UPDATE leaves the index valid and triggers no rebuild.
    let touches_indexed_col = col_index
        .as_ref()
        .map(|cidx| bound_set.iter().any(|(c, _)| cidx.indexes(c)))
        .unwrap_or(false);
    // Re-file ops gathered during the write loop: (pre-update row, post-update row,
    // key). Applied after the commit so a mid-loop store error leaves the index
    // untouched (the loop's `?` aborts before any re-file lands).
    let mut col_refile: Vec<(Row, Row, i64)> = Vec::new();
    let guard = TxnGuard::begin(store, scope)?;
    for (key, row) in &to_update {
        let mut updated = row.clone();
        for (set_col, set_val) in &bound_set {
            // Coerce the assigned value to the target column's affinity so the
            // stored cell's storage class agrees with stdlib sqlite3.
            let aff = cols
                .iter()
                .find(|c| &c.name == set_col)
                .map(|c| c.affinity.as_str())
                .unwrap_or("TEXT");
            updated.set(set_col.clone(), coerce_affinity(set_val.clone(), aff));
        }
        let payload_vals: Vec<Value> =
            col_names.iter().map(|cn| updated.get_or_null(cn)).collect();
        tree.insert(*key, &encode_row(&payload_vals)).map_err(store_err)?;
        if touches_indexed_col {
            col_refile.push((row.clone(), updated, *key));
        }
    }
    // The write-generation is the readers' exactness fence for their adopted
    // col/id indexes and row counts. An UPDATE that touches no indexed column
    // changes none of those read models — same keys, same indexed values,
    // same count — so it must NOT bump: a bump here forced every reader to
    // drop and re-adopt per reinforcement write, and any adoption miss
    // degraded the recall seed fetch to a whole-table scan. Row DATA
    // freshness is the pager snapshot fence's job, not the generation's.
    if has_col_index && touches_indexed_col {
        meta.bump_col_generation(store, table)?;
    }
    guard.commit()?;

    // Re-file the col-index for every rewritten row whose indexed columns changed:
    // drop the pre-update values, add the post-update ones, at the same row-key.
    // A no-op when no SET touched an indexed column or the index is unbuilt.
    if let Some(cidx) = col_index {
        for (before, after, key) in &col_refile {
            cidx.remove_row(before, *key);
            cidx.insert_row(after, *key);
        }
    }
    // Re-file the ordered index for every updated row: invalidate is cheapest and
    // correct — an UPDATE may change the ordered column value, so drop the whole
    // index and let the next read rebuild from the tree (same discipline as the
    // conflict index, which also invalidates on any UPDATE).
    if let Some(oidx) = ordered_index {
        oidx.invalidate();
    }

    // An UPDATE may rewrite a conflict-key column, so any built conflict index is
    // now potentially stale. Invalidate it; the next conflict probe rebuilds from
    // the tree. (Cheap: the index is rebuilt at most once per bulk run, and the
    // edges UPSERT workload never interleaves UPDATEs.)
    if let Some(cidx) = conflict_index {
        cidx.invalidate();
    }

    // The id-index maps `id` → row-key. A non-fast-path UPDATE may have matched
    // arbitrary rows, and any UPDATE that assigns the `id` column moves a row's
    // key away from its old id. In either case a built id-index is now potentially
    // stale, so drop it; the next `id = ?` point lookup rebuilds it from the tree.
    // The fast-path branch that did not touch `id` leaves the index valid (the row
    // stayed at the same key under the same id), so it is not invalidated there.
    let touched_id = bound_set
        .iter()
        .any(|(c, _)| c.eq_ignore_ascii_case("id"));
    if let Some(idx) = id_index {
        if !used_fast_path || touched_id {
            idx.invalidate();
        }
    }

    Ok(WriteOutcome { lastrowid: None, rowcount: to_update.len() as i64 })
}

/// Execute a DELETE: select matching keys (full scan + three-valued WHERE, or all
/// keys when no WHERE), remove them, and report the affected row count. An
/// `id_index`, when supplied, has the deleted ids removed so it stays consistent.
#[allow(clippy::too_many_arguments)]
pub fn execute_delete(
    stmt: &DeleteStmt,
    store: &Store,
    meta: &MetaTable,
    catalog: &Catalog,
    root_map: &std::collections::BTreeMap<String, u32>,
    params: Vec<Value>,
    named: Option<HashMap<String, Value>>,
    scope: TxnScope,
    id_index: Option<&mut IdIndex>,
    conflict_index: Option<&mut ConflictIndex>,
    col_index: Option<&mut ColIndex>,
    ordered_index: Option<&mut OrderedColIndex>,
) -> Result<WriteOutcome> {
    let table = &stmt.table;
    let root = *root_map
        .get(table)
        .ok_or_else(|| EngineError::parse(format!("delete: unknown table {table:?}")))?;
    let col_names = catalog.column_names(table)?;
    let tree = store.tree(root);

    let mut binder = Binder::new(params, named);
    let bound_where = match &stmt.where_clause {
        Some(w) => Some(binder.bind(w)?),
        None => None,
    };
    binder.finish()?;

    // Whether the col-index (when supplied and built) needs per-row maintenance:
    // capture the deleted rows so their indexed-column entries can be dropped.
    let maintain_col = col_index
        .as_ref()
        .map(|cidx| cidx.built())
        .unwrap_or(false);
    // See `execute_insert`'s `has_col_index`: gates the write-generation bump to
    // tables that actually carry a col-index.
    let has_col_index = col_index.is_some();

    let mut to_delete: Vec<i64> = Vec::new();
    let mut deleted_ids: Vec<String> = Vec::new();
    let mut deleted_rows: Vec<(Row, i64)> = Vec::new();
    for (key, payload) in tree.full_scan().map_err(store_err)? {
        let row = decode_to_row(&payload, &col_names)?;
        let pass = match &bound_where {
            None => true,
            Some(w) => eval_predicate(w, &row) == Some(true),
        };
        if pass {
            to_delete.push(key);
            if let Value::Text(id) = row.get_or_null("id") {
                deleted_ids.push(id);
            }
            if maintain_col {
                deleted_rows.push((row, key));
            }
        }
    }

    if to_delete.is_empty() {
        return Ok(WriteOutcome { lastrowid: None, rowcount: 0 });
    }

    let guard = TxnGuard::begin(store, scope)?;
    for key in &to_delete {
        tree.delete(*key).map_err(store_err)?;
    }
    if has_col_index {
        meta.bump_col_generation(store, table)?;
    }
    guard.commit()?;

    if let Some(idx) = id_index {
        for id in deleted_ids {
            idx.remove(&id);
        }
    }
    // Drop the deleted rows' col-index entries at their freed keys, so a later
    // probe never returns a key pointing at a deleted (or pager-reused) row.
    if let Some(cidx) = col_index {
        for (row, key) in &deleted_rows {
            cidx.remove_row(row, *key);
        }
    }
    // Drop the deleted rows from the ordered index as well.
    if let Some(oidx) = ordered_index {
        for (row, key) in &deleted_rows {
            oidx.remove_row(row, *key);
        }
    }
    // A delete may remove a row a later INSERT would conflict with. The delete
    // path does not compute the deleted rows' conflict-key tuples, so invalidate
    // the conflict index — the next conflict probe rebuilds it from the tree
    // (which no longer contains the deleted rows).
    if let Some(cidx) = conflict_index {
        cidx.invalidate();
    }

    Ok(WriteOutcome { lastrowid: None, rowcount: to_delete.len() as i64 })
}

/// Execute a batch of INSERTs over one parameter row each, atomically.
///
/// When no transaction is open, the whole batch is wrapped in a single
/// `Store::begin_write` / `commit`; every per-statement INSERT runs under
/// `TxnScope::Suppressed` so it never opens a second pager transaction (the
/// reference's `_suppress_pager_txn` invariant). A mid-batch error rolls the
/// entire batch back, so a partial write never reaches the store.
///
/// vec_label injection still runs per row inside the batch, so each inserted row
/// receives its own monotonic high-water mark.
#[allow(clippy::too_many_arguments)]
pub fn execute_insert_many(
    stmt: &InsertStmt,
    store: &Store,
    meta: &MetaTable,
    catalog: &Catalog,
    root_map: &std::collections::BTreeMap<String, u32>,
    seq_of_params: Vec<Vec<Value>>,
    mut id_index: Option<&mut IdIndex>,
    mut conflict_index: Option<&mut ConflictIndex>,
    mut col_index: Option<&mut ColIndex>,
    mut ordered_index: Option<&mut OrderedColIndex>,
) -> Result<WriteOutcome> {
    if seq_of_params.is_empty() {
        return Ok(WriteOutcome { lastrowid: None, rowcount: 0 });
    }

    // Open one batch transaction; suppress every inner per-statement transaction
    // so the batch commits atomically and the pager is never double-opened.
    store.begin_write().map_err(store_err)?;
    let mut last: WriteOutcome = WriteOutcome::default();
    let mut total: i64 = 0;
    for params in seq_of_params {
        let idx = id_index.as_deref_mut();
        let cidx = conflict_index.as_deref_mut();
        let colx = col_index.as_deref_mut();
        let ordx = ordered_index.as_deref_mut();
        let outcome = match execute_insert(
            stmt,
            store,
            meta,
            catalog,
            root_map,
            params,
            None,
            TxnScope::Suppressed,
            idx,
            cidx,
            colx,
            ordx,
        ) {
            Ok(o) => o,
            Err(e) => {
                // Roll the whole batch back: a partial write must not survive.
                let _ = store.rollback();
                // The id-index gained `id → row-key` entries for the rolled-back
                // rows; invalidate it so a later point lookup rebuilds from the
                // committed tree rather than resolving an `id` to a reverted key.
                if let Some(idx) = id_index.as_deref_mut() {
                    idx.invalidate();
                }
                // The conflict index gained entries for the rolled-back rows;
                // invalidate it so a later probe rebuilds from the committed tree.
                if let Some(cidx) = conflict_index.as_deref_mut() {
                    cidx.invalidate();
                }
                // The col-index likewise gained incremental entries for the
                // rolled-back rows (mapping their indexed-column values at freed
                // keys); invalidate it so a later probe rebuilds from the committed
                // tree rather than returning a key the rollback reverted.
                if let Some(colx) = col_index.as_deref_mut() {
                    colx.invalidate();
                }
                // The ordered index gained incremental entries for rolled-back rows;
                // invalidate so the next read rebuilds from the committed tree.
                if let Some(oidx) = ordered_index.as_deref_mut() {
                    oidx.invalidate();
                }
                return Err(e);
            }
        };
        total += outcome.rowcount;
        last = outcome;
    }
    store.commit().map_err(store_err)?;

    Ok(WriteOutcome { lastrowid: last.lastrowid, rowcount: total })
}

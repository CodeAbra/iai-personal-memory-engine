//! Query planner: a parsed statement plus the catalog and bound parameters into
//! an immutable physical plan.
//!
//! Every real-table query emits a full scan (this engine has no secondary-index
//! scan path). The `dbstat` virtual table is intercepted before any catalog
//! lookup and answered by a constant plan — a size estimate with no cursor. The
//! planner validates that a real table exists (a missing table is the reference's
//! verbatim unknown-table error) and threads the WHERE / ORDER BY / projection /
//! aggregate / GROUP BY structure onto the plan for the executor.

use lillibrain::Value;

use crate::ast::{Expr, GroupSelectItem, OrderKey, SelectExpr, SelectStmt};
use crate::catalog::Catalog;
use crate::error::{EngineError, Result};

/// The scan strategy a plan resolves to.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ScanType {
    /// A full table scan over the data B-tree (the default strategy).
    #[default]
    Full,
    /// A constant answer with no cursor (the `dbstat` size estimate).
    Constant,
    /// The `sqlite_master` / `sqlite_schema` schema catalog, synthesized from the
    /// in-memory catalog and filtered/ordered/projected through the normal row
    /// pipeline.
    SqliteMaster,
}

/// An immutable physical plan for a SELECT statement.
///
/// Carries everything the executor needs: the scan strategy, the projection
/// (plain columns, aliased expressions, or a grouped projection), the WHERE
/// predicate, the ORDER BY keys, LIMIT / OFFSET, the scalar aggregate, and the
/// GROUP BY / HAVING structure. Positional `?` binds resolve from `params`;
/// named `:name` binds resolve from `named_params`.
#[derive(Debug, Clone, Default)]
pub struct SelectPlan {
    /// The source table name.
    pub table: String,
    /// The scan strategy.
    pub scan_type: ScanTypeField,
    /// The plain-column projection (`["*"]` for `SELECT *`), empty when an
    /// aliased / grouped projection is carried instead.
    pub columns: Vec<String>,
    /// The optional WHERE predicate.
    pub where_clause: Option<Expr>,
    /// A literal LIMIT value, or `None`.
    pub limit: Option<i64>,
    /// True when LIMIT is a `?` placeholder.
    pub limit_is_param: bool,
    /// A literal OFFSET value, or `None`.
    pub offset: Option<i64>,
    /// True when OFFSET is a `?` placeholder.
    pub offset_is_param: bool,
    /// The positional `?` bind values in textual order.
    pub params: Vec<Value>,
    /// The named `:name` bind values, or `None` for the positional path.
    pub named_params: Option<std::collections::HashMap<String, Value>>,
    /// The scalar aggregate kind (`count` / `sum` / `min` / `max`), or `None`.
    pub aggregate: Option<String>,
    /// The aggregate column for SUM/MIN/MAX, or `None`.
    pub aggregate_column: Option<String>,
    /// The output label for a sole `COUNT(*) AS alias`, or `None`.
    pub count_alias: Option<String>,
    /// The first ORDER BY key.
    pub order_by: Option<OrderKey>,
    /// The optional second ORDER BY key.
    pub order_by2: Option<OrderKey>,
    /// The aliased / COALESCE projection.
    pub select_exprs: Vec<SelectExpr>,
    /// The GROUP BY column, or `None`.
    pub group_by_col: Option<String>,
    /// The grouped projection items.
    pub group_select: Vec<GroupSelectItem>,
    /// The HAVING COUNT(*) operator, or `None`.
    pub having_op: Option<String>,
    /// The HAVING COUNT(*) value, or `None`.
    pub having_value: Option<i64>,
}

/// A serialization-free alias keeping a `Default` on the plan struct (the default
/// scan is a full scan).
pub type ScanTypeField = ScanType;

/// The virtual tables answered by a constant plan, matched case-insensitively.
const VIRTUAL_TABLES: &[&str] = &["dbstat"];

/// The schema-catalog virtual tables, matched case-insensitively. Both are the
/// same object in sqlite3 (`sqlite_schema` is the modern alias).
const SCHEMA_TABLES: &[&str] = &["sqlite_master", "sqlite_schema"];

/// Plan a SELECT statement against the catalog and bound parameters.
///
/// A `dbstat` query is intercepted before any catalog lookup and planned as a
/// constant. A real table is validated against the catalog (an unknown table is
/// an error); the remaining structure is threaded onto the plan unchanged. The
/// positional `?` count is not checked here — the executor binds in textual
/// order and fails loud on a count mismatch, mirroring the reference.
pub fn plan_select(
    stmt: &SelectStmt,
    catalog: &Catalog,
    params: Vec<Value>,
    named_params: Option<std::collections::HashMap<String, Value>>,
) -> Result<SelectPlan> {
    let table = stmt.table.clone();
    let is_virtual = VIRTUAL_TABLES
        .iter()
        .any(|v| v.eq_ignore_ascii_case(&table));
    let is_schema = SCHEMA_TABLES.iter().any(|v| v.eq_ignore_ascii_case(&table));

    let scan_type = if is_virtual {
        ScanType::Constant
    } else if is_schema {
        ScanType::SqliteMaster
    } else {
        // Validate the table exists (an unknown table is the reference's error).
        catalog.column_names(&table)?;
        ScanType::Full
    };

    Ok(SelectPlan {
        table,
        scan_type,
        columns: stmt.columns.clone(),
        where_clause: stmt.where_clause.clone(),
        limit: stmt.limit,
        limit_is_param: stmt.limit_is_param,
        offset: stmt.offset,
        offset_is_param: stmt.offset_is_param,
        params,
        named_params,
        aggregate: stmt.aggregate.clone(),
        aggregate_column: stmt.aggregate_column.clone(),
        count_alias: stmt.count_alias.clone(),
        order_by: stmt.order_by.clone(),
        order_by2: stmt.order_by2.clone(),
        select_exprs: stmt.select_exprs.clone(),
        group_by_col: stmt.group_by_col.clone(),
        group_select: stmt.group_select.clone(),
        having_op: stmt.having_op.clone(),
        having_value: stmt.having_value,
    })
}

/// The verbatim bind-count error the executor raises, matching the reference
/// engine's `sqlite3.ProgrammingError` text family.
pub fn too_few_bindings_error() -> EngineError {
    EngineError::ProgrammingError(
        "Incorrect number of bindings supplied: too few parameters for '?' placeholders"
            .to_string(),
    )
}

/// The verbatim surplus-bind error the executor raises.
pub fn too_many_bindings_error(surplus: usize) -> EngineError {
    EngineError::ProgrammingError(format!(
        "Incorrect number of bindings supplied: {surplus} surplus parameter(s)"
    ))
}

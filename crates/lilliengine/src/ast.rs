//! Statement and expression AST for the supported SQL subset.
//!
//! The node set mirrors the reference's shape so the later planner and executor
//! consume the same tree. Literal values reuse [`lillibrain::Value`] — no second
//! value enum is defined. Placeholders are carried as their own expression
//! variants (positional `?` and named `:name`) rather than as sentinel literals.

use lillibrain::Value;

/// An expression node.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// A constant value. `NULL` is carried as [`Value::Null`]; `TRUE`/`FALSE`
    /// are carried as `Value::Int(1)` / `Value::Int(0)` (reference parity).
    Literal(Value),
    /// A positional `?` bind placeholder, resolved positionally at bind time.
    Placeholder,
    /// A named `:name` bind placeholder, resolved from a dict param at bind
    /// time; carries the name without the leading colon.
    NamedParam(String),
    /// A reference to a column by name (a dotted `table.col` keeps the full
    /// dotted name, matching the reference).
    Column(String),
    /// A reference to `excluded.col` inside an `ON CONFLICT DO UPDATE SET`
    /// clause; resolves to the attempted-insert value.
    Excluded(String),
    /// A binary comparison or logical expression. `op` is one of the comparison
    /// operators, `AND`, `OR`, `LIKE`, or `NOT LIKE`.
    BinOp {
        /// The operator string.
        op: String,
        /// The left operand.
        left: Box<Expr>,
        /// The right operand.
        right: Box<Expr>,
    },
    /// A unary `NOT` expression.
    UnaryOp {
        /// The operator string (always `NOT`).
        op: String,
        /// The negated operand.
        operand: Box<Expr>,
    },
    /// An `IS [NOT] NULL` predicate.
    IsNull {
        /// The tested operand.
        operand: Box<Expr>,
        /// True for `IS NOT NULL`, false for `IS NULL`.
        negated: bool,
    },
    /// An `IN (...)` predicate over a fixed item list.
    InList {
        /// The tested operand.
        operand: Box<Expr>,
        /// The candidate items.
        items: Vec<Expr>,
        /// Bind-time membership set, present ONLY when every item is a TEXT
        /// literal: an O(1) probe for a TEXT operand with semantics identical
        /// to the linear item scan (byte equality, the TEXT=TEXT comparison).
        /// A large bound `IN` list re-evaluated per candidate row is otherwise
        /// an O(rows × items) scan of the item vector. `None` for any mixed /
        /// non-literal item list — those keep the linear evaluation.
        text_set: Option<std::sync::Arc<std::collections::HashSet<String>>>,
    },
    /// A two-argument `COALESCE(a, b)` — the first non-NULL argument.
    Coalesce(Vec<Expr>),
    /// A scalar function call. The supported subset uses `datetime(<expr>)`;
    /// `name` is lower-cased and `args` holds the call arguments in order.
    FuncCall {
        /// The lower-cased function name.
        name: String,
        /// The call arguments in source order.
        args: Vec<Expr>,
    },
}

/// One `ORDER BY` key: a column name, a DESC flag, and whether the key was
/// written as `datetime(<col>)` (so the executor sorts on the normalized
/// timestamp value).
#[derive(Debug, Clone, PartialEq)]
pub struct OrderKey {
    /// The column name to sort on.
    pub col: String,
    /// True for `DESC`, false for `ASC` (the default).
    pub desc: bool,
    /// True when the key was wrapped in `datetime(...)`.
    pub datetime: bool,
}

/// A single output column in an aliased / COALESCE projection.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectExpr {
    /// The projected expression.
    pub expr: Expr,
    /// The output column name (alias, or the source name when unaliased).
    pub output_name: String,
}

/// One item in a grouped (`GROUP BY` + `COUNT(*)`) projection. `kind` is
/// `"col"` for a plain grouped column (its first-row value, read from
/// `source_name`) or `"count"` for the `COUNT(*)` item (`source_name` empty).
#[derive(Debug, Clone, PartialEq)]
pub struct GroupSelectItem {
    /// `"col"` or `"count"`.
    pub kind: String,
    /// The output column name.
    pub output_name: String,
    /// The source column name (`""` for the count item).
    pub source_name: String,
}

/// A `SELECT` statement.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SelectStmt {
    /// The single source table.
    pub table: String,
    /// The plain-column projection (`["*"]` for `SELECT *`); empty when an
    /// aliased / grouped projection is carried instead.
    pub columns: Vec<String>,
    /// The optional `WHERE` predicate.
    pub where_clause: Option<Expr>,
    /// A literal `LIMIT` value, or `None`.
    pub limit: Option<i64>,
    /// True when `LIMIT` is a `?` placeholder.
    pub limit_is_param: bool,
    /// A literal `OFFSET` value, or `None`.
    pub offset: Option<i64>,
    /// True when `OFFSET` is a `?` placeholder.
    pub offset_is_param: bool,
    /// The aggregate kind: `"count"`, `"sum"`, `"min"`, `"max"`, or `None`.
    pub aggregate: Option<String>,
    /// The aggregate column for SUM/MIN/MAX; `None` for `COUNT(*)`.
    pub aggregate_column: Option<String>,
    /// The output label for a sole `COUNT(*) AS alias`, or `None`.
    pub count_alias: Option<String>,
    /// The first `ORDER BY` key, or `None`.
    pub order_by: Option<OrderKey>,
    /// The optional second `ORDER BY` key.
    pub order_by2: Option<OrderKey>,
    /// The aliased / COALESCE projection (mutually exclusive with `columns`).
    pub select_exprs: Vec<SelectExpr>,
    /// The `GROUP BY` column, or `None`.
    pub group_by_col: Option<String>,
    /// The output label for a grouped `COUNT(*)` item, or `None`.
    pub group_count_alias: Option<String>,
    /// The grouped projection items (populated only with a grouped COUNT).
    pub group_select: Vec<GroupSelectItem>,
    /// The `HAVING COUNT(*) <op>` operator, or `None`.
    pub having_op: Option<String>,
    /// The `HAVING COUNT(*) <op> <int>` value, or `None`.
    pub having_value: Option<i64>,
}

/// An `INSERT` statement.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InsertStmt {
    /// The target table.
    pub table: String,
    /// The explicit column list.
    pub columns: Vec<String>,
    /// The VALUES row expressions.
    pub values: Vec<Expr>,
    /// True for `INSERT OR IGNORE`.
    pub or_ignore: bool,
    /// True for `INSERT OR REPLACE`.
    pub or_replace: bool,
    /// The `ON CONFLICT (keys)` columns.
    pub conflict_keys: Vec<String>,
    /// The conflict action: `"update"`, `"nothing"`, or `None`.
    pub conflict_action: Option<String>,
    /// The `DO UPDATE SET col=expr` assignments.
    pub set_clauses: Vec<(String, Expr)>,
}

/// An `UPDATE` statement.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UpdateStmt {
    /// The target table.
    pub table: String,
    /// The `SET col=expr` assignments.
    pub set_clauses: Vec<(String, Expr)>,
    /// The optional `WHERE` predicate.
    pub where_clause: Option<Expr>,
}

/// A `DELETE` statement.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeleteStmt {
    /// The target table.
    pub table: String,
    /// The optional `WHERE` predicate.
    pub where_clause: Option<Expr>,
}

/// A DDL or PRAGMA statement. The frontend records the kind and the raw SQL
/// text; DDL is replayed by the catalog, not the row executor.
#[derive(Debug, Clone, PartialEq)]
pub struct DdlStmt {
    /// The DDL kind: `create_table`, `create_index`, `alter_add_column`,
    /// `alter_drop_column`, `alter_rename_table`, `drop_table`, or `pragma`.
    pub kind: String,
    /// The original SQL text, normalized to space-separated token values.
    pub raw: String,
}

/// A parsed statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// A `SELECT`.
    Select(SelectStmt),
    /// An `INSERT`.
    Insert(InsertStmt),
    /// An `UPDATE`.
    Update(UpdateStmt),
    /// A `DELETE`.
    Delete(DeleteStmt),
    /// A DDL / PRAGMA passthrough.
    Ddl(DdlStmt),
}

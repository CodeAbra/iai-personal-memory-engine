"""AST node dataclasses for the SQL frontend."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


@dataclass
class Literal:
    """Constant value expression."""

    value: Any


@dataclass
class NamedParam:
    """A :name bind placeholder; resolved from a dict param at bind time."""

    name: str


@dataclass
class ColumnRef:
    """Reference to a column by name."""

    name: str


@dataclass
class ExcludedRef:
    """Reference to excluded.col in an ON CONFLICT DO UPDATE SET clause."""

    col_name: str


@dataclass
class BinOp:
    """Binary comparison or logical expression."""

    op: str
    left: Expr
    right: Expr


@dataclass
class UnaryOp:
    """Unary NOT expression."""

    op: str
    operand: Expr


@dataclass
class IsNull:
    """IS NULL or IS NOT NULL predicate."""

    operand: Expr
    negated: bool = False


@dataclass
class InList:
    """IN (...) predicate with a variable-length item list."""

    operand: Expr
    items: list[Expr] = field(default_factory=list)


@dataclass
class Coalesce:
    """COALESCE(arg, fallback) — the value of the first non-NULL argument.

    Restricted to the two-argument form (a column or expression and a fallback
    value) that the supported query subset uses inside WHERE predicates.
    """

    args: list[Expr] = field(default_factory=list)


@dataclass
class FuncCall:
    """A scalar function call ``name(arg, ...)``.

    The supported subset uses a single scalar function — ``datetime(<expr>)`` —
    which normalises a timestamp value to the same second-resolution space-form
    string the underlying TEXT-comparison semantics expect. ``name`` is kept
    lower-case; ``args`` holds the call arguments in source order.
    """

    name: str
    args: list[Expr] = field(default_factory=list)


# Expr is the union of all expression node types.
Expr = (
    Literal | NamedParam | ColumnRef | ExcludedRef | BinOp | UnaryOp | IsNull
    | InList | Coalesce | FuncCall
)


# ---------------------------------------------------------------------------
# Statement nodes
# ---------------------------------------------------------------------------


@dataclass
class SelectStmt:
    """SELECT statement AST node.

    aggregate is "count" for COUNT(*), "sum"/"min"/"max" for the matching
    single-column aggregate, else None. aggregate_column holds the column name
    for SUM/MIN/MAX(col); None for COUNT(*). count_alias is the output label for
    a sole (ungrouped) ``COUNT(*) AS alias`` aggregate, or None when the scalar
    count has no alias (the default "COUNT(*)" label is used); it is the scalar
    counterpart to group_count_alias for the grouped select list.
    order_by_col holds the single ORDER BY column name (one column only), or
    None when no ORDER BY is present; order_by_desc is True for DESC, False for
    ASC (the default). order_by_col2/order_by_desc2 carry an optional second
    ORDER BY key (the ``ORDER BY a, b`` form); order_by_col2 is None when the
    ORDER BY has a single key. order_by_datetime/order_by_datetime2 are True when
    the matching key was written as ``datetime(<col>)`` — the executor then sorts
    by that column's second-resolution normalised value instead of its raw value,
    so the order matches the timestamp-comparison semantics; a plain ``ORDER BY
    <col>`` key leaves the flag False and sorts on the raw column value.
    select_exprs carries the projection when at least one select item is aliased
    or is a COALESCE expression: a list of (expression, output_name) pairs in
    source order, so positional access reproduces sqlite3's column order. When
    select_exprs is empty the plain-column ``columns`` fast path is used; the two
    are mutually exclusive carriers for the projection.
    group_by_col holds the GROUP BY column name, or None when no GROUP BY is
    present. group_count_alias is the output label for a ``COUNT(*) [AS alias]``
    item in the grouped select list (defaulting to "COUNT(*)" when present
    without an alias), or None when the grouped select list has no count item.
    group_select carries the grouped projection as ordered
    ``(kind, output_name, source_name)`` triples — kind "col" for a plain
    grouped column (its first-row-in-group value, read from source_name) and
    kind "count" for the COUNT(*) item (source_name "") — so positional access
    matches sqlite3's column order. It is populated only when the grouped select
    list contains a COUNT(*) item; an all-plain grouped list (``id, c GROUP BY
    c``) leaves it empty and projects ``columns`` instead. having_op/having_value
    carry an optional ``HAVING COUNT(*) <op> <int>`` group filter (op one of
    >=,>,<=,<,=,!=).
    """

    table: str
    columns: list[str]
    where: Expr | None = None
    limit: int | None = None
    limit_is_param: bool = False
    offset: int | None = None
    offset_is_param: bool = False
    aggregate: str | None = None
    aggregate_column: str | None = None
    count_alias: str | None = None
    order_by_col: str | None = None
    order_by_desc: bool = False
    order_by_col2: str | None = None
    order_by_desc2: bool = False
    order_by_datetime: bool = False
    order_by_datetime2: bool = False
    select_exprs: list[tuple[Expr, str]] = field(default_factory=list)
    group_by_col: str | None = None
    group_count_alias: str | None = None
    group_select: list[tuple[str, str, str]] = field(default_factory=list)
    having_op: str | None = None
    having_value: int | None = None


@dataclass
class InsertStmt:
    """INSERT statement AST node."""

    table: str
    columns: list[str]
    values: list[Expr] = field(default_factory=list)
    or_ignore: bool = False
    or_replace: bool = False
    conflict_keys: list[str] = field(default_factory=list)
    conflict_action: str | None = None  # "update" or "nothing"
    set_clauses: list[tuple[str, Expr]] = field(default_factory=list)


@dataclass
class UpdateStmt:
    """UPDATE statement AST node."""

    table: str
    set_clauses: list[tuple[str, Expr]] = field(default_factory=list)
    where: Expr | None = None


@dataclass
class DeleteStmt:
    """DELETE statement AST node."""

    table: str
    where: Expr | None = None


@dataclass
class DDLStmt:
    """DDL or PRAGMA statement AST node."""

    kind: str   # "create_table", "create_index", "alter_add_column", "pragma"
    raw: str    # the original SQL text; DDL is not executed by the executor


# Stmt is the union of all statement node types.
Stmt = SelectStmt | InsertStmt | UpdateStmt | DeleteStmt | DDLStmt

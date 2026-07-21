"""Query planner: AST statement → PhysicalPlan (full scan only).

Converts a parsed statement into an immutable PhysicalPlan. Every real-table
statement emits scan_type='full' (no secondary-index lookup). The dbstat
virtual size-estimate table is intercepted before any catalog lookup and
emits scan_type='constant' — the executor answers it without a cursor.
"""
from __future__ import annotations

from iai_mcp import errors
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.lillibrain.sql.ast_nodes import Expr, Stmt
    from iai_mcp.lillibrain.sql.catalog import Catalog


# ---------------------------------------------------------------------------
# Virtual table names answered by constant plans
# ---------------------------------------------------------------------------

_VIRTUAL_TABLES: frozenset[str] = frozenset({"dbstat"})


# ---------------------------------------------------------------------------
# Physical plan
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhysicalPlan:
    """Immutable physical execution plan produced by the planner.

    stmt_type: 'select', 'insert', 'update', 'delete', 'ddl'
    table: the target table name
    scan_type: 'full' for real-table scans, 'constant' for virtual tables
    columns: projected columns (['*'] for SELECT *, [] for aggregates/DML)
    where: the WHERE expression node, or None
    limit: integer LIMIT value, or None
    params: bound parameter values for '?' placeholders
    aggregate: 'count', 'sum', 'min', or 'max', or None
    aggregate_column: column name for SUM/MIN/MAX(col), or None
    count_alias: output label for a sole COUNT(*) AS alias, or None for the
        default "COUNT(*)" label (the scalar counterpart to group_count_alias)
    order_by_col: single ORDER BY column name, or None
    order_by_desc: True for ORDER BY ... DESC, False for ASC (the default)
    order_by_col2: second ORDER BY column name (the ``ORDER BY a, b`` form), or
        None when the ORDER BY has a single key
    order_by_desc2: True for the second key's DESC, False for ASC
    order_by_datetime: True when the first ORDER BY key was ``datetime(<col>)``,
        so the executor sorts on the column's second-resolution normalised value
    order_by_datetime2: the same flag for the second ORDER BY key
    named_params: name->value map supplied for ``:name`` placeholder binds, or
        None for the positional ``?`` path; the two are mutually exclusive (a
        statement mixing the styles is rejected at bind time, matching sqlite3)
    select_exprs: (expr, output_name) projection pairs when the SELECT list is
        aliased / has a COALESCE expression; empty for the plain-column path
    group_by_col: GROUP BY column name, or None when no GROUP BY is present
    group_count_alias: output label for a COUNT(*) item in a grouped select list
        (defaulting to "COUNT(*)"), or None when the grouped list has no count
    group_select: ordered (kind, name) grouped-projection pairs — kind "col" for
        a plain grouped column (its first-row-in-group value), "count" for the
        COUNT(*) item — empty for a non-grouped query
    having_op: comparison operator for a HAVING COUNT(*) group filter, or None
    having_value: integer RHS of the HAVING COUNT(*) comparison, or None
    insert_columns: column names for INSERT
    set_clauses: (column, expr) pairs for UPDATE / ON CONFLICT DO UPDATE
    conflict_keys: key columns in ON CONFLICT (...)
    conflict_action: 'update' or 'nothing', or None
    or_ignore: True for INSERT OR IGNORE
    """

    stmt_type: str
    table: str
    scan_type: str = "full"
    columns: tuple[str, ...] = field(default_factory=tuple)
    where: object = None                         # Expr | None at runtime
    limit: int | None = None
    limit_is_param: bool = False
    offset: int | None = None
    offset_is_param: bool = False
    params: tuple = field(default_factory=tuple)
    named_params: dict | None = None             # name->value map for :name binds
    aggregate: str | None = None
    aggregate_column: str | None = None
    count_alias: str | None = None
    order_by_col: str | None = None
    order_by_desc: bool = False
    order_by_col2: str | None = None
    order_by_desc2: bool = False
    order_by_datetime: bool = False
    order_by_datetime2: bool = False
    select_exprs: tuple = field(default_factory=tuple)   # tuple[(Expr, str)]
    group_by_col: str | None = None
    group_count_alias: str | None = None
    group_select: tuple = field(default_factory=tuple)   # tuple[(str, str)]
    having_op: str | None = None
    having_value: int | None = None
    insert_columns: tuple[str, ...] = field(default_factory=tuple)
    set_clauses: tuple = field(default_factory=tuple)   # tuple[(str, Expr)]
    conflict_keys: tuple[str, ...] = field(default_factory=tuple)
    conflict_action: str | None = None
    or_ignore: bool = False
    or_replace: bool = False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan_stmt(
    stmt: Stmt,
    catalog: Catalog,
    params: list,
    named_params: dict | None = None,
) -> PhysicalPlan:
    """Produce a PhysicalPlan from a parsed statement and the current catalog.

    For SELECT against a virtual table (dbstat), the dbstat interception fires
    before any catalog lookup so CatalogError is never raised for the virtual
    table name.

    named_params is the name->value map for ``:name`` placeholder binds; it is
    threaded onto the SELECT plan so the executor can resolve a NamedParam by
    name. The positional ``params`` list serves the ``?`` path; the two styles
    are mutually exclusive per statement (mixing is rejected at bind time).
    """
    from iai_mcp.lillibrain.sql.ast_nodes import (  # noqa: PLC0415
        DDLStmt,
        DeleteStmt,
        InsertStmt,
        SelectStmt,
        UpdateStmt,
    )

    if isinstance(stmt, SelectStmt):
        return _plan_select(stmt, catalog, params, named_params)
    if isinstance(stmt, InsertStmt):
        return _plan_insert(stmt, catalog, params, named_params)
    if isinstance(stmt, UpdateStmt):
        return _plan_update(stmt, catalog, params, named_params)
    if isinstance(stmt, DeleteStmt):
        return _plan_delete(stmt, catalog, params, named_params)
    if isinstance(stmt, DDLStmt):
        return _plan_ddl(stmt, params)

    raise ValueError(f"plan_stmt: unsupported statement type {type(stmt).__name__!r}")


# ---------------------------------------------------------------------------
# Per-statement planners
# ---------------------------------------------------------------------------


def _plan_select(
    stmt: SelectStmt,
    catalog: Catalog,
    params: list,
    named_params: dict | None = None,
) -> PhysicalPlan:
    table = stmt.table

    # Intercept virtual tables before any catalog lookup.
    if table.lower() in _VIRTUAL_TABLES:
        return PhysicalPlan(
            stmt_type="select",
            table=table,
            scan_type="constant",
            columns=tuple(stmt.columns),
            where=stmt.where,
            limit=stmt.limit,
            limit_is_param=stmt.limit_is_param,
            offset=stmt.offset,
            offset_is_param=stmt.offset_is_param,
            params=tuple(params),
            named_params=named_params,
            aggregate=stmt.aggregate,
            aggregate_column=stmt.aggregate_column,
            count_alias=stmt.count_alias,
            order_by_col=stmt.order_by_col,
            order_by_desc=stmt.order_by_desc,
            order_by_col2=stmt.order_by_col2,
            order_by_desc2=stmt.order_by_desc2,
            order_by_datetime=stmt.order_by_datetime,
            order_by_datetime2=stmt.order_by_datetime2,
            select_exprs=tuple(stmt.select_exprs),
            group_by_col=stmt.group_by_col,
            group_count_alias=stmt.group_count_alias,
            group_select=tuple(stmt.group_select),
            having_op=stmt.having_op,
            having_value=stmt.having_value,
        )

    # Validate the table exists in the catalog (raises CatalogError if absent).
    catalog.column_names(table)

    return PhysicalPlan(
        stmt_type="select",
        table=table,
        scan_type="full",
        columns=tuple(stmt.columns),
        where=stmt.where,
        limit=stmt.limit,
        limit_is_param=stmt.limit_is_param,
        offset=stmt.offset,
        offset_is_param=stmt.offset_is_param,
        params=tuple(params),
        named_params=named_params,
        aggregate=stmt.aggregate,
        aggregate_column=stmt.aggregate_column,
        count_alias=stmt.count_alias,
        order_by_col=stmt.order_by_col,
        order_by_desc=stmt.order_by_desc,
        order_by_col2=stmt.order_by_col2,
        order_by_desc2=stmt.order_by_desc2,
        order_by_datetime=stmt.order_by_datetime,
        order_by_datetime2=stmt.order_by_datetime2,
        select_exprs=tuple(stmt.select_exprs),
        group_by_col=stmt.group_by_col,
        group_count_alias=stmt.group_count_alias,
        group_select=tuple(stmt.group_select),
        having_op=stmt.having_op,
        having_value=stmt.having_value,
    )


def _plan_insert(
    stmt: InsertStmt,
    catalog: Catalog,
    params: list,
    named_params: dict | None = None,
) -> PhysicalPlan:
    # Validate table (raises CatalogError if absent).
    catalog.column_names(stmt.table)

    # Resolve VALUES expressions to concrete values in column order.
    # Each value expression is either:
    #   - Literal(value="?")  → consume the next positional param
    #   - Literal(value=<x>)  → use the literal value directly
    #   - NamedParam(name)    → resolve from named_params dict by name
    # This ensures INSERT … VALUES (42) or (:a, :b) stores the correct values.
    from iai_mcp.lillibrain.sql.ast_nodes import Literal, NamedParam  # noqa: PLC0415
    from iai_mcp.lillibrain.sql.types import SqlNull                    # noqa: PLC0415

    params_queue = list(params)
    resolved: list = []
    for val_expr in stmt.values:
        if isinstance(val_expr, NamedParam):
            if named_params is None or val_expr.name not in named_params:
                raise ValueError(
                    f"no value supplied for named parameter {val_expr.name!r} in INSERT"
                )
            v = named_params[val_expr.name]
            resolved.append(SqlNull if v is None else v)
        elif isinstance(val_expr, Literal):
            if val_expr.value == "?":
                # Placeholder — consume next positional param.
                # Fail loud on underflow: mirrors the ProgrammingError raised
                # for too-few bindings ("Incorrect number of bindings supplied").
                if not params_queue:
                    raise errors.ProgrammingError(
                        "Incorrect number of bindings supplied: too few parameters "
                        "for '?' placeholders in INSERT VALUES"
                    )
                val = params_queue.pop(0)
                resolved.append(SqlNull if val is None else val)
            else:
                resolved.append(val_expr.value)
        else:
            # Unsupported expression node in VALUES — fail loud rather than
            # silently storing NULL.
            raise ValueError(
                f"unsupported VALUES expression type {type(val_expr).__name__!r} in INSERT"
            )

    # Fail loud on surplus params — mirrors the ProgrammingError raised for
    # too-many bindings ("Incorrect number of bindings supplied").
    if params_queue:
        raise errors.ProgrammingError(
            "Incorrect number of bindings supplied: "
            f"{len(params_queue)} surplus parameter(s) for INSERT VALUES"
        )

    return PhysicalPlan(
        stmt_type="insert",
        table=stmt.table,
        scan_type="full",
        columns=tuple(stmt.columns),
        params=tuple(resolved),
        named_params=named_params,
        insert_columns=tuple(stmt.columns),
        set_clauses=tuple(stmt.set_clauses),
        conflict_keys=tuple(stmt.conflict_keys),
        conflict_action=stmt.conflict_action,
        or_ignore=stmt.or_ignore,
        or_replace=stmt.or_replace,
        where=None,
    )


def _plan_update(
    stmt: UpdateStmt,
    catalog: Catalog,
    params: list,
    named_params: dict | None = None,
) -> PhysicalPlan:
    catalog.column_names(stmt.table)

    return PhysicalPlan(
        stmt_type="update",
        table=stmt.table,
        scan_type="full",
        params=tuple(params),
        named_params=named_params,
        set_clauses=tuple(stmt.set_clauses),
        where=stmt.where,
    )


def _plan_delete(
    stmt: DeleteStmt,
    catalog: Catalog,
    params: list,
    named_params: dict | None = None,
) -> PhysicalPlan:
    catalog.column_names(stmt.table)

    return PhysicalPlan(
        stmt_type="delete",
        table=stmt.table,
        scan_type="full",
        params=tuple(params),
        named_params=named_params,
        where=stmt.where,
    )


def _plan_ddl(stmt: DDLStmt, params: list) -> PhysicalPlan:
    return PhysicalPlan(
        stmt_type="ddl",
        table="",
        scan_type="full",
        params=tuple(params),
    )

"""Volcano executor: PhysicalPlan → rows via BtreeCursor generator pipeline.

Implements a generator-based Volcano pipeline over the B+tree cursor:
  - seq_scan: full table scan yielding Row dicts keyed by column name
  - filter_op: WHERE filter with three-valued NULL logic
  - project_op: column projection
  - limit_op: LIMIT n stop-after
  - eval_expr: recursive AST expression evaluator — no eval() anywhere
  - execute: entry point returning (rows: list[Row], col_names: list[str])
  - run: convenience wrapper (parse → plan → execute)

NULL three-valued logic: any comparison involving a NULL operand propagates
SqlNull through sql_compare; filter_op rejects rows where the predicate result
is SqlNull or False. IS NULL / IS NOT NULL always return a real bool.
"""
from __future__ import annotations

import logging
import re as _re
import sqlite3
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from iai_mcp.lillibrain.constants import PAGE_SIZE
from iai_mcp.lillibrain.record import decode_record, decode_record_projected, encode_record
from iai_mcp.lillibrain.sql.types import (
    SqlNull,
    sql_and,
    sql_compare,
    sql_is_null,
    sql_not,
    sql_or,
)

if TYPE_CHECKING:
    from iai_mcp.lillibrain.btree.cursor import BtreeCursor
    from iai_mcp.lillibrain.pager import Pager
    from iai_mcp.lillibrain.sql.ast_nodes import Expr
    from iai_mcp.lillibrain.sql.catalog import Catalog
    from iai_mcp.lillibrain.sql.planner import PhysicalPlan

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Row type
# ---------------------------------------------------------------------------

Row = dict[str, Any]


# ---------------------------------------------------------------------------
# Reserved row-dict keys used by the engine for internal bookkeeping.
# These never appear in user-visible result rows.
# ---------------------------------------------------------------------------

_ROWID = "rowid"          # pseudo-column name in SQL queries
_ROWID_SOURCE = "vec_label"  # column carrying the AUTOINCREMENT HWM (records table)
_ROWKEY = "_rowkey"       # raw B+tree integer key threaded for rowid fallback


# ---------------------------------------------------------------------------
# Volcano operators
# ---------------------------------------------------------------------------


def seq_scan(
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
) -> Iterator[Row]:
    """Full table scan via the B+tree leaf sibling-link chain.

    Decodes each leaf payload into a Row dict keyed by column name.
    NULL values (decoded as SqlNull by the record codec) are left as SqlNull.
    """
    col_names = catalog.column_names(table)
    for btree_key, payload in cursor.full_scan():
        values, _ = decode_record(payload, 0)
        # Pad with SqlNull if the stored record has fewer columns than the
        # current schema (e.g. after ALTER TABLE ADD COLUMN).
        while len(values) < len(col_names):
            values.append(SqlNull)
        row = dict(zip(col_names, values))
        # Thread the raw B+tree key under a reserved name so _resolve_rowid
        # can fall back to it for tables that have no vec_label column.
        row["_rowkey"] = btree_key
        yield row


def seq_scan_projected(
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
    decode_indices: "frozenset[int]",
    all_col_names: list[str],
) -> Iterator[Row]:
    """Projected table scan — decodes only the columns in decode_indices.

    decode_indices is the union of the requested output columns and any
    additional columns needed for WHERE filtering.  The caller is responsible
    for resolving this set before calling.

    Yields rows containing all decoded columns (output + filter columns).
    The caller strips filter-only columns from the output after filtering.

    Traverses leaf pages directly so the page buffer is available as a
    memoryview.  When the pager's mapped-read path is active, BLOB cells in
    decode_indices are returned as zero-copy memoryview slices into the mapped
    page — no per-row bytes copy.  For non-mapped pagers the codec falls back
    to bytes copies (correct in both cases).

    all_col_names is the full catalog column list for the table, used to map
    column indices to names and to pad short stored records with SqlNull.
    """
    import struct as _struct  # noqa: PLC0415

    from iai_mcp.lillibrain.btree.overflow import read_overflow_chain  # noqa: PLC0415
    from iai_mcp.lillibrain.btree.page import (  # noqa: PLC0415
        _LEAF_PTR_ARRAY_START,
        get_next_leaf,
        read_leaf_header,
    )
    from iai_mcp.lillibrain.constants import OVERFLOW_THRESHOLD  # noqa: PLC0415
    from iai_mcp.lillibrain.record import decode_varint  # noqa: PLC0415

    n_catalog = len(all_col_names)
    pager = cursor._pager

    page_no = cursor._leftmost_leaf()
    while page_no != 0:
        page_buf = pager.read_page(page_no)
        page_mv = memoryview(page_buf)
        num_cells, _ccs, _fb, _frag = read_leaf_header(page_mv)

        for i in range(num_cells):
            (ptr,) = _struct.unpack_from(">H", page_mv, _LEAF_PTR_ARRAY_START + i * 2)
            _key, kn = decode_varint(page_mv, ptr)
            payload_len, vn = decode_varint(page_mv, ptr + kn)
            payload_start = ptr + kn + vn

            if payload_len > OVERFLOW_THRESHOLD:
                (first_ovf,) = _struct.unpack_from(">I", page_mv, payload_start)
                payload: bytes | memoryview = read_overflow_chain(pager, first_ovf, payload_len)
            else:
                # Pass the page memoryview slice directly — when the pager's
                # mapped-read path is ON, this is a zero-copy view into the mapped
                # page buffer.  When it is OFF the slice is still a memoryview of
                # the bytearray page copy, so decode_record_projected returns
                # memoryview slices of that bytearray — still avoids an extra copy.
                payload = page_mv[payload_start : payload_start + payload_len]

            by_index, _ = decode_record_projected(payload, 0, decode_indices)
            # Pad: columns beyond the stored record length default to SqlNull.
            row: Row = {}
            for idx in decode_indices:
                if idx < n_catalog:
                    row[all_col_names[idx]] = by_index.get(idx, SqlNull)
            # Thread the raw B+tree key so _resolve_rowid can fall back to it
            # for tables that have no vec_label column.
            row["_rowkey"] = _key
            yield row

        # Follow sibling link to the next leaf.
        page_no = get_next_leaf(pager, page_no)


def count_leaf_cells(cursor: BtreeCursor) -> int:
    """Count all rows by summing leaf-cell counts — no payload decode.

    Walks the leaf sibling-link chain and reads only each leaf's cell count
    from its header. No record payload is decoded, so a COUNT(*) over a wide
    table (large embedding BLOBs) touches only the leaf page headers instead of
    materialising every row. This keeps the scan's transient allocation flat
    (bounded by the page cache) regardless of row width.
    """
    from iai_mcp.lillibrain.btree.page import (  # noqa: PLC0415
        get_next_leaf,
        read_leaf_header,
    )

    pager = cursor._pager
    total = 0
    page_no = cursor._leftmost_leaf()
    while page_no != 0:
        page_buf = pager.read_page(page_no)
        num_cells, _ccs, _fb, _frag = read_leaf_header(memoryview(page_buf))
        total += num_cells
        page_no = get_next_leaf(pager, page_no)
    return total


def filter_op(
    source: Iterator[Row],
    predicate: Expr,
    params: list,
) -> Iterator[Row]:
    """Yield rows for which the predicate evaluates to a truthy non-NULL value."""
    for row in source:
        result = eval_expr(predicate, row, params)
        if result is not SqlNull and result:
            yield row


def project_op(
    source: Iterator[Row],
    columns: list[str],
) -> Iterator[Row]:
    """Project each row to the requested columns (or pass through for SELECT *).

    The reserved ``_rowkey`` field (B+tree key threaded for rowid resolution)
    is stripped from ``SELECT *`` output so it never surfaces in result rows.
    """
    for row in source:
        if columns == ["*"]:
            # Strip internal bookkeeping keys that must not appear in results.
            if _ROWKEY in row or _ROWID in row:
                row = {k: v for k, v in row.items() if k not in (_ROWKEY, _ROWID)}
            yield row
        else:
            yield {c: row[c] for c in columns}


def limit_op(
    source: Iterator[Row],
    n: int,
) -> Iterator[Row]:
    """Yield at most n rows from source."""
    for i, row in enumerate(source):
        if i >= n:
            return
        yield row


# Storage-class ranks for ordering: NULL < INTEGER/REAL < TEXT < BLOB.
# Cross-class comparison uses the rank alone, so values of different classes
# are never compared directly (avoids str-vs-int TypeError) while reproducing
# the sqlite3 collation order.
_CLASS_NULL = 0
_CLASS_NUMBER = 1
_CLASS_TEXT = 2
_CLASS_BLOB = 3


def _order_key(value: object) -> tuple[int, object]:
    """Map a row value to an ascending (storage_class_rank, value) sort key.

    NULL sorts before every non-NULL value; within a class, native value order
    applies. The placeholder for the NULL class never participates in a value
    comparison because the class rank already separates it.
    """
    if value is SqlNull or value is None:
        return (_CLASS_NULL, 0)
    if isinstance(value, bool):
        # bool is an int subclass; treat it as a number for ordering.
        return (_CLASS_NUMBER, int(value))
    if isinstance(value, (int, float)):
        return (_CLASS_NUMBER, value)
    if isinstance(value, str):
        return (_CLASS_TEXT, value)
    if isinstance(value, memoryview):
        return (_CLASS_BLOB, value.tobytes())
    if isinstance(value, (bytes, bytearray)):
        return (_CLASS_BLOB, bytes(value))
    # Any other type orders as a number-like fallback by its native value.
    return (_CLASS_NUMBER, value)


class _Descending:
    """Sort-key wrapper that inverts the natural order of the wrapped key.

    Wrapping the ascending key lets a single stable ``list.sort`` produce
    descending order while leaving equal-key rows in their original (insertion)
    order — a blanket ``reverse=True`` would instead flip the tie order and
    move NULLs to the wrong end.
    """

    __slots__ = ("key",)

    def __init__(self, key: tuple[int, object]) -> None:
        self.key = key

    def __lt__(self, other: "_Descending") -> bool:
        return other.key < self.key

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Descending) and other.key == self.key


def _sort_value(row: Row, col: str, as_datetime: bool) -> object:
    """Return a row's sort value for ``col``, normalised when datetime-wrapped.

    A plain key reads the raw column value. A ``datetime(<col>)`` key reads the
    column's second-resolution normalised value so the order matches the
    timestamp-comparison semantics; SqlNull stays SqlNull so NULL placement holds.
    """
    val = row.get(col, SqlNull)
    if as_datetime:
        return _datetime_scalar(val)
    return val


def _order_by(rows: list[Row], col: str, desc: bool, as_datetime: bool = False) -> list[Row]:
    """Return rows sorted by ``col`` reproducing sqlite3 single-column ORDER BY.

    ASC places NULLs first then ascending values; DESC places descending values
    then NULLs last. The sort is stable, so equal sort values keep their prior
    (insertion / rowid) order — matching sqlite3's tiebreak. ``rows`` is sorted
    on a copy; the caller passes a materialised list. ``as_datetime`` sorts by the
    column's normalised timestamp value (a ``datetime(<col>)`` key).
    """
    if desc:
        rows.sort(key=lambda r: _Descending(_order_key(_sort_value(r, col, as_datetime))))
    else:
        rows.sort(key=lambda r: _order_key(_sort_value(r, col, as_datetime)))
    return rows


def _order_by_keys(rows: list[Row], keys: list[tuple[str, bool, bool]]) -> list[Row]:
    """Return rows sorted by a sequence of ``(column, desc, as_datetime)`` keys.

    Composes a tuple sort key of per-element ``_order_key`` results, each wrapped
    in ``_Descending`` for its own direction, so a single stable ``list.sort``
    yields lexicographic order with the correct per-element NULL placement —
    never a blanket ``reverse=True`` (which flips NULL placement and the
    cross-key tiebreak). A key's ``as_datetime`` flag sorts that element by the
    column's normalised timestamp value. ``rows`` is sorted in place.
    """
    def composite(r: Row) -> tuple:
        parts: list[object] = []
        for col, desc, as_datetime in keys:
            k = _order_key(_sort_value(r, col, as_datetime))
            parts.append(_Descending(k) if desc else k)
        return tuple(parts)

    rows.sort(key=composite)
    return rows


# ---------------------------------------------------------------------------
# rowid pseudo-column
# ---------------------------------------------------------------------------


def _resolve_rowid(row: Row) -> Row:
    """Set ``row['rowid']`` in place.

    Prefers the vec_label column (the never-reused AUTOINCREMENT HWM that
    equals sqlite3's rowid for tables with an INTEGER PRIMARY KEY). Falls
    back to the raw B+tree key for tables that have no vec_label column —
    matching sqlite3's plain-rowid semantics (reusable after DELETE+INSERT).
    """
    if _ROWID_SOURCE in row:
        row[_ROWID] = row[_ROWID_SOURCE]
    else:
        row[_ROWID] = row.get(_ROWKEY, SqlNull)
    return row


# ---------------------------------------------------------------------------
# Expression evaluator (recursive, no eval())
# ---------------------------------------------------------------------------


def eval_expr(expr: Expr, row: Row, params: list) -> object:
    """Evaluate an expression AST node against a row, returning a Python value
    or SqlNull for NULL results.

    Consumes positional '?' parameters from params in left-to-right order.
    Caller must supply params as a mutable list; consumed items are popped
    from the front via a shared mutable context (passed as a list, not an
    index).

    Note: params is consumed destructively during evaluation. Wrap in a
    fresh list per call site; the execute() function handles this.
    """
    from iai_mcp.lillibrain.sql.ast_nodes import (  # noqa: PLC0415
        BinOp,
        Coalesce,
        ColumnRef,
        ExcludedRef,
        FuncCall,
        InList,
        IsNull,
        Literal,
        UnaryOp,
    )

    if isinstance(expr, Literal):
        return expr.value

    if isinstance(expr, ColumnRef):
        val = row.get(expr.name)
        # Treat Python None as SqlNull for any value coming from the row
        return SqlNull if val is None else val

    if isinstance(expr, ExcludedRef):
        # Resolved by execute() before calling eval_expr — should not appear
        # in normal WHERE evaluation. Return SqlNull as a safe default.
        return SqlNull

    if isinstance(expr, BinOp):
        op = expr.op.upper()
        if op == "AND":
            left_val = eval_expr(expr.left, row, params)
            right_val = eval_expr(expr.right, row, params)
            return sql_and(left_val, right_val)
        if op == "OR":
            left_val = eval_expr(expr.left, row, params)
            right_val = eval_expr(expr.right, row, params)
            return sql_or(left_val, right_val)
        if op == "LIKE":
            left_val = eval_expr(expr.left, row, params)
            right_val = eval_expr(expr.right, row, params)
            return _sql_like(left_val, right_val)
        if op == "NOT LIKE":
            # Produced by infix "col NOT LIKE pattern" in _parse_comparison.
            left_val = eval_expr(expr.left, row, params)
            right_val = eval_expr(expr.right, row, params)
            result = _sql_like(left_val, right_val)
            return sql_not(result)
        # Comparison operators
        left_val = eval_expr(expr.left, row, params)
        right_val = eval_expr(expr.right, row, params)
        # Coerce types for comparison: int vs str comparisons follow
        # sqlite3 type affinity (numeric < text).
        left_val, right_val = _coerce_for_compare(left_val, right_val)
        return sql_compare(op, left_val, right_val)

    if isinstance(expr, UnaryOp):
        operand_val = eval_expr(expr.operand, row, params)
        if expr.op.upper() == "NOT":
            return sql_not(operand_val)
        raise ValueError(f"eval_expr: unknown unary op {expr.op!r}")

    if isinstance(expr, IsNull):
        operand_val = eval_expr(expr.operand, row, params)
        is_null = sql_is_null(operand_val)
        return not is_null if expr.negated else is_null

    if isinstance(expr, InList):
        operand_val = eval_expr(expr.operand, row, params)
        if operand_val is SqlNull:
            return SqlNull
        found_null = False
        for item_expr in expr.items:
            item_val = eval_expr(item_expr, row, params)
            if item_val is SqlNull:
                # NULL in list — cannot determine membership (three-valued logic)
                found_null = True
                continue
            if sql_compare("=", operand_val, item_val) is True:
                return True
        # No match found: if any NULL was in the list, result is unknown (SqlNull)
        return SqlNull if found_null else False

    if isinstance(expr, Coalesce):
        # First non-NULL argument; SqlNull only if every argument is NULL.
        for arg_expr in expr.args:
            arg_val = eval_expr(arg_expr, row, params)
            if arg_val is not SqlNull:
                return arg_val
        return SqlNull

    if isinstance(expr, FuncCall):
        if expr.name == "datetime":
            arg_val = eval_expr(expr.args[0], row, params)
            return _datetime_scalar(arg_val)
        raise ValueError(f"eval_expr: unsupported scalar function {expr.name!r}")

    raise ValueError(f"eval_expr: unknown expression type {type(expr).__name__!r}")


# ---------------------------------------------------------------------------
# Expression evaluation helpers
# ---------------------------------------------------------------------------


def _datetime_scalar(value: object) -> object:
    """Normalise a timestamp value to sqlite3 ``datetime()`` output, or SqlNull.

    Reproduces the built-in scalar: parse a timestamp given with either an ISO
    'T' separator or a space separator (and an optional 'Z' / offset suffix),
    convert an offset-bearing value to UTC, truncate to whole seconds, drop the
    timezone, and return the second-resolution space-form string
    ``YYYY-MM-DD HH:MM:SS``. A bare date string yields midnight. A NULL argument
    propagates as SqlNull. An unparseable timestamp argument also yields SqlNull
    rather than raising, matching sqlite3's NULL handling for both NULL-the-value
    and NULL-on-garbage.
    """
    if value is SqlNull or value is None:
        return SqlNull
    import datetime as _datetime  # noqa: PLC0415

    if isinstance(value, _datetime.datetime):
        dt = value
    elif isinstance(value, _datetime.date):
        dt = _datetime.datetime(value.year, value.month, value.day)
    else:
        text = str(value).replace("Z", "+00:00").replace(" ", "T")
        try:
            dt = _datetime.datetime.fromisoformat(text)
        except ValueError:
            # sqlite3's datetime() returns NULL (it never errors) for any
            # unparseable timestamp argument; match that contract so a single
            # corrupted row degrades to NULL instead of aborting the query.
            return SqlNull
    # An offset-bearing value is converted to UTC (sqlite3 datetime() normalises
    # an explicit timezone to UTC); a naive value keeps its wall-clock components.
    # The tz and second-fraction are then dropped, matching sqlite3's output.
    if dt.tzinfo is not None:
        dt = dt.astimezone(_datetime.timezone.utc).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _collect_column_names(expr: "Expr") -> set[str]:
    """Return all column names referenced in an expression tree."""
    from iai_mcp.lillibrain.sql.ast_nodes import (  # noqa: PLC0415
        BinOp,
        Coalesce,
        ColumnRef,
        FuncCall,
        InList,
        IsNull,
        UnaryOp,
    )

    if isinstance(expr, ColumnRef):
        return {expr.name}
    if isinstance(expr, BinOp):
        return _collect_column_names(expr.left) | _collect_column_names(expr.right)
    if isinstance(expr, UnaryOp):
        return _collect_column_names(expr.operand)
    if isinstance(expr, IsNull):
        return _collect_column_names(expr.operand)
    if isinstance(expr, InList):
        names: set[str] = _collect_column_names(expr.operand)
        for item in expr.items:
            names |= _collect_column_names(item)
        return names
    if isinstance(expr, Coalesce):
        coalesce_names: set[str] = set()
        for arg in expr.args:
            coalesce_names |= _collect_column_names(arg)
        return coalesce_names
    if isinstance(expr, FuncCall):
        func_names: set[str] = set()
        for arg in expr.args:
            func_names |= _collect_column_names(arg)
        return func_names
    return set()


def _coerce_for_compare(a: object, b: object) -> tuple[object, object]:
    """Coerce two values for comparison following sqlite3 type affinity.

    When one side is int and the other is str, attempt numeric coercion of
    the str. If the str is not numeric, keep both as-is (str vs int will
    compare by Python rules; sqlite3 puts numerics before text).
    """
    if isinstance(a, int) and isinstance(b, str):
        try:
            return a, int(b)
        except (ValueError, TypeError):
            pass
    if isinstance(a, str) and isinstance(b, int):
        try:
            return int(a), b
        except (ValueError, TypeError):
            pass
    return a, b


def _sql_like(value: object, pattern: object) -> object:
    """ASCII case-insensitive LIKE: '%' = any sequence, '_' = one character.

    Converts LIKE wildcards to regex without relying on re.escape for '%' and '_',
    because Python 3.7+ does not escape those characters (they are not regex
    metacharacters). Instead, escape only the characters that are regex metacharacters
    in the range that users might type in LIKE patterns, then substitute wildcards.
    """
    if value is SqlNull or pattern is SqlNull:
        return SqlNull
    if not isinstance(pattern, str):
        return False
    # sqlite3 coerces numeric values to their text representation before LIKE matching.
    if not isinstance(value, str):
        value = str(value)
    # Build a regex by escaping real metacharacters in the pattern, then substituting
    # LIKE wildcards. Escape char-by-char to avoid touching % and _.
    _REGEX_META = frozenset(r"\.^$*+?{}[]|()")
    parts: list[str] = ["^"]
    for ch in pattern:
        if ch == "%":
            parts.append(".*")
        elif ch == "_":
            parts.append(".")
        elif ch in _REGEX_META:
            parts.append(_re.escape(ch))
        else:
            parts.append(ch)
    parts.append("$")
    regex = "".join(parts)
    return bool(_re.match(regex, value, _re.IGNORECASE))


# ---------------------------------------------------------------------------
# Placeholder parameter binding
# ---------------------------------------------------------------------------


def _bind_params(
    expr: Expr,
    params_queue: list,
    named: "dict | None" = None,
    seen: "dict | None" = None,
) -> Expr:
    """Substitute placeholders with their bound values.

    The parser emits Literal(value="?") for positional '?' placeholders and
    NamedParam(name) for ':name' placeholders. This function walks the
    expression tree in evaluation order and replaces each placeholder with its
    bound value: a '?' pops the next positional param from params_queue; a
    ':name' resolves from the ``named`` map by name.

    Mixing the two styles in one statement is rejected (sqlite3 parity): the
    ``seen`` dict tracks which styles appeared, and encountering the second
    style raises. A NamedParam with no ``named`` map (e.g. a tuple supplied for
    a named query) raises; a name absent from the map raises (no silent None).

    Returns a new expression tree with every placeholder resolved.
    """
    from iai_mcp.lillibrain.sql.ast_nodes import (  # noqa: PLC0415
        BinOp,
        Coalesce,
        ExcludedRef,
        FuncCall,
        InList,
        IsNull,
        Literal,
        NamedParam,
        UnaryOp,
    )

    if seen is None:
        seen = {"positional": False, "named": False}

    if isinstance(expr, NamedParam):
        if named is None:
            raise sqlite3.ProgrammingError(
                "bind_params: a :name placeholder requires a dict param "
                f"(none supplied for {expr.name!r})"
            )
        if expr.name not in named:
            raise sqlite3.ProgrammingError(
                f"bind_params: no value supplied for named parameter {expr.name!r}"
            )
        seen["named"] = True
        if seen["positional"]:
            raise sqlite3.ProgrammingError(
                "bind_params: cannot mix named (:name) and positional (?) "
                "parameters in one statement"
            )
        val = named[expr.name]
        return Literal(SqlNull if val is None else val)

    if isinstance(expr, Literal):
        if expr.value == "?":
            seen["positional"] = True
            if seen["named"]:
                raise sqlite3.ProgrammingError(
                    "bind_params: cannot mix named (:name) and positional (?) "
                    "parameters in one statement"
                )
            # This is a '?' placeholder — pop the next bound param
            if not params_queue:
                raise sqlite3.ProgrammingError(
                    "bind_params: too few bound parameters for '?' placeholders"
                )
            val = params_queue.pop(0)
            # Normalise Python None to SqlNull
            return Literal(SqlNull if val is None else val)
        return expr

    if isinstance(expr, BinOp):
        return BinOp(
            expr.op,
            _bind_params(expr.left, params_queue, named, seen),
            _bind_params(expr.right, params_queue, named, seen),
        )

    if isinstance(expr, UnaryOp):
        return UnaryOp(expr.op, _bind_params(expr.operand, params_queue, named, seen))

    if isinstance(expr, IsNull):
        return IsNull(_bind_params(expr.operand, params_queue, named, seen), expr.negated)

    if isinstance(expr, InList):
        return InList(
            _bind_params(expr.operand, params_queue, named, seen),
            [_bind_params(item, params_queue, named, seen) for item in expr.items],
        )

    if isinstance(expr, Coalesce):
        return Coalesce([_bind_params(arg, params_queue, named, seen) for arg in expr.args])

    if isinstance(expr, FuncCall):
        # Recurse into the call arguments so a '?' nested inside datetime(?) pops
        # the next positional param in evaluation order; without this the inner
        # placeholder survives into eval and the params queue desyncs.
        return FuncCall(
            expr.name,
            [_bind_params(arg, params_queue, named, seen) for arg in expr.args],
        )

    # ColumnRef, ExcludedRef — no placeholders inside
    return expr


def _bind_value_expr(expr: Expr, params_queue: list) -> Expr:
    """Bind a single value expression (INSERT VALUES item or SET rhs)."""
    from iai_mcp.lillibrain.sql.ast_nodes import Literal  # noqa: PLC0415

    if isinstance(expr, Literal) and expr.value == "?":
        if not params_queue:
            raise ValueError("bind_value_expr: too few bound parameters")
        val = params_queue.pop(0)
        return Literal(SqlNull if val is None else val)
    return expr


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def execute(
    plan: PhysicalPlan,
    pager: Pager,
    root_map: dict[str, int],
    id_index: "dict[str, int] | None" = None,
) -> tuple[list[Row], list[str]]:
    """Execute a PhysicalPlan and return (rows, col_names).

    The return shape is always a tuple so callers can reconstruct
    cursor.description without re-querying the catalog. COUNT(*) and
    SUM(col) each return a single-row result in this shape. LIMIT 0
    returns ([], populated col_names).
    """
    # ---------------------------------------------------------------------------
    # Constant / virtual-table plans (dbstat size estimate)
    # ---------------------------------------------------------------------------

    if plan.scan_type == "constant":
        return _execute_constant(plan, pager)

    # ---------------------------------------------------------------------------
    # DDL plans — DDL is applied to the catalog in run() before reaching here.
    # If a DDL plan arrives via direct execute() (not via run()), the catalog
    # was not updated — callers must use run() or call catalog.apply_ddl() first.
    # ---------------------------------------------------------------------------

    if plan.stmt_type == "ddl":
        return ([], [])

    # ---------------------------------------------------------------------------
    # Real table plans — validate root_map entry
    # ---------------------------------------------------------------------------

    table = plan.table
    if table not in root_map:
        raise ValueError(f"execute: unknown table {table!r}")

    root_page_no = root_map[table]

    from iai_mcp.lillibrain.btree.cursor import BtreeCursor  # noqa: PLC0415
    from iai_mcp.lillibrain.sql.catalog import Catalog  # noqa: PLC0415

    # Infer catalog from root_map context — executor receives catalog through
    # the run() helper. For direct execute() calls, catalog must be provided
    # via a closure or the plan itself. Here we require it via the _catalog
    # attribute that run() sets on the plan wrapper (see run()).
    # However, the public execute() signature does not carry catalog directly
    # per the plan's interface contract. We obtain it from the plan's internal
    # _catalog field when available, otherwise raise.
    catalog: Catalog = getattr(plan, "_catalog", None)  # type: ignore[assignment]
    if catalog is None:
        raise ValueError(
            "execute: no catalog attached to plan; use run() instead of execute() directly"
        )

    cursor = BtreeCursor(pager, root_page_no)

    if plan.stmt_type == "select":
        return _execute_select(plan, cursor, catalog, table, pager, id_index=id_index)
    if plan.stmt_type == "insert":
        return _execute_insert(plan, cursor, catalog, table, pager)
    if plan.stmt_type == "update":
        return _execute_update(plan, cursor, catalog, table, pager, id_index=id_index)
    if plan.stmt_type == "delete":
        return _execute_delete(plan, cursor, catalog, table, pager)

    raise ValueError(f"execute: unsupported stmt_type {plan.stmt_type!r}")


# ---------------------------------------------------------------------------
# Constant plan executor (dbstat virtual table)
# ---------------------------------------------------------------------------


def _execute_constant(
    plan: PhysicalPlan,
    pager: Pager,
) -> tuple[list[Row], list[str]]:
    """Answer a constant plan without touching a cursor.

    For the dbstat virtual table: return a single-row size estimate
    (total_pages * PAGE_SIZE). The caller (storage_maintain.py) reads
    row[0] as an int byte size and falls back to COUNT(*)*256 on None.
    """
    try:
        page_count = pager.read_db_header_field("db_size")
    except Exception:
        page_count = 0

    estimate = page_count * PAGE_SIZE

    # The aggregate column name is the argument to SUM() — 'pgsize' from
    # 'SELECT SUM(pgsize) FROM dbstat WHERE name = ?'
    col_name = plan.aggregate_column or "pgsize"
    row: Row = {col_name: estimate}
    logger.debug("executor.constant: dbstat estimate=%d bytes", estimate)
    return ([row], [col_name])


# ---------------------------------------------------------------------------
# Aliased / COALESCE projection executor
# ---------------------------------------------------------------------------


def _execute_select_exprs(
    plan: PhysicalPlan,
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
    col_names: list[str],
    bound_select_exprs: list[tuple[Expr, str]],
    bound_where: "Expr | None",
    effective_limit: int | None,
    effective_offset: int | None = None,
) -> tuple[list[Row], list[str]]:
    """Project a SELECT list that carries aliased / COALESCE expressions.

    Each output row is keyed by the alias (output name) in source order, so
    positional access reproduces sqlite3's column order. The source columns the
    expressions, WHERE, and ORDER BY reference are decoded on the projected scan;
    rows are filtered, NULL-aware stable-sorted (when ORDER BY is present), then
    each (expr, alias) is evaluated per row. ``?`` placeholders are already bound
    by the caller in textual order.
    """
    name_to_idx = {name: i for i, name in enumerate(col_names)}
    aliases = [alias for _, alias in bound_select_exprs]

    # Collect the catalog indices every projected expression needs.
    needed_names: set[str] = set()
    selects_rowid = False
    for expr, _ in bound_select_exprs:
        expr_names = _collect_column_names(expr)
        needed_names |= expr_names
        if _ROWID in expr_names:
            selects_rowid = True
    if bound_where is not None:
        needed_names |= _collect_column_names(bound_where)
    if plan.order_by_col is not None:
        needed_names.add(plan.order_by_col)
    if plan.order_by_col2 is not None:
        needed_names.add(plan.order_by_col2)

    decode_indices_set: set[int] = {
        name_to_idx[n] for n in needed_names if n in name_to_idx
    }
    # rowid aliases vec_label: decode the source column when rowid is projected
    # or ordered, so the alias value is available on the projected path.
    if (selects_rowid or plan.order_by_col == _ROWID) and _ROWID_SOURCE in name_to_idx:
        decode_indices_set.add(name_to_idx[_ROWID_SOURCE])
    decode_indices = frozenset(decode_indices_set)

    source: Iterator[Row] = seq_scan_projected(
        cursor, catalog, table, decode_indices, col_names
    )
    if bound_where is not None:
        source = filter_op(source, bound_where, [])

    # ORDER BY before projection so an unprojected sort column is still readable
    # and LIMIT keeps the top-N of the sorted set.
    if plan.order_by_col is not None:
        def _sort_col_exprs(c: str) -> str:
            if c != _ROWID:
                return c
            return _ROWID_SOURCE if _ROWID_SOURCE in name_to_idx else _ROWKEY

        if plan.order_by_col2 is not None:
            sort_keys_exprs = [
                (_sort_col_exprs(plan.order_by_col), plan.order_by_desc, plan.order_by_datetime),
                (_sort_col_exprs(plan.order_by_col2), plan.order_by_desc2, plan.order_by_datetime2),
            ]
            source = iter(_order_by_keys(list(source), sort_keys_exprs))
        else:
            source = iter(_order_by(
                list(source), _sort_col_exprs(plan.order_by_col),
                plan.order_by_desc, plan.order_by_datetime,
            ))

    # rowid is resolved per row before projection so eval_expr can read it.
    materialised = list(source)
    if selects_rowid or plan.order_by_col == _ROWID:
        for row in materialised:
            _resolve_rowid(row)

    out_rows: list[Row] = []
    for row in materialised:
        out_rows.append(
            {alias: eval_expr(expr, row, []) for expr, alias in bound_select_exprs}
        )

    if effective_limit == 0:
        return ([], aliases)
    # Apply OFFSET before LIMIT: skip the first N rows of the sorted set.
    if effective_offset is not None and effective_offset > 0:
        out_rows = out_rows[effective_offset:]
    if effective_limit is not None and effective_limit > 0:
        out_rows = out_rows[:effective_limit]

    logger.debug(
        "executor.select_exprs: table=%r rows=%d aliases=%r",
        table, len(out_rows), aliases,
    )
    return (out_rows, aliases)


def _execute_group_by(
    plan: PhysicalPlan,
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
    bound_where: "Expr | None",
    effective_limit: int | None,
    effective_offset: int | None = None,
) -> tuple[list[Row], list[str]]:
    """Execute a GROUP BY query: bucket WHERE-filtered rows, project, filter, sort.

    Rows are bucketed by the storage-class key of the group column (so a NULL
    group key is its own bucket, ordered before non-NULL values per sqlite3).
    Each bucket keeps the FIRST row it sees, reproducing sqlite3's choice of
    value for a bare (non-aggregated) select column, and a running count for the
    COUNT(*) item. HAVING filters the buckets by count; a multi-key ORDER BY
    sorts the output rows (the first key may be the COUNT alias); LIMIT keeps the
    top-N. The aggregate label is owned here, never routed through the per-row
    projector.
    """
    group_col = plan.group_by_col
    assert group_col is not None  # guarded by the caller

    # Full decode: the group key, any bare projected column, and the WHERE
    # columns must all be readable. Grouping runs on the already-filtered set
    # (O(#groups)); a full decode keeps first-row bare-column projection correct
    # without projected-scan index bookkeeping.
    source: Iterator[Row] = seq_scan(cursor, catalog, table)
    if bound_where is not None:
        source = filter_op(source, bound_where, [])

    # Determine the output projection: the grouped select list (with a COUNT
    # item) when present, else the plain ``columns`` (a bare-column grouped
    # select like ``id, c GROUP BY c``). Each entry is (kind, out_name, src_name).
    if plan.group_select:
        projection: list[tuple[str, str, str]] = list(plan.group_select)
    else:
        projection = [("col", c, c) for c in plan.columns]
    out_names = [out_name for _, out_name, _ in projection]

    # Bucket rows by the storage-class key of the group column, preserving the
    # first row of each group and a running count. Insertion order of the dict
    # records first-appearance order, matching sqlite3's stable tiebreak.
    buckets: dict[tuple[int, object], tuple[Row, int]] = {}
    for row in source:
        key = _order_key(row.get(group_col, SqlNull))
        if key in buckets:
            first_row, n = buckets[key]
            buckets[key] = (first_row, n + 1)
        else:
            buckets[key] = (row, 1)

    # Default group order: sqlite3 emits groups ordered by the group key in
    # storage-class order (NULL < number < text < blob), regardless of insertion
    # order. Sort the buckets by their key before projecting so the implicit
    # order matches and an explicit ORDER BY (applied stably below) breaks ties
    # on this base, as sqlite3 does.
    ordered_buckets = sorted(buckets.items(), key=lambda kv: kv[0])

    # Build one output row per group from its first row + count.
    grouped_pairs: list[tuple[Row, int]] = []
    for _key, (first_row, n) in ordered_buckets:
        out_row: Row = {}
        for kind, out_name, src_name in projection:
            if kind == "count":
                out_row[out_name] = n
            else:
                out_row[out_name] = first_row.get(src_name, SqlNull)
        grouped_pairs.append((out_row, n))

    # HAVING COUNT(*) <op> <int>: filter groups by their count.
    if plan.having_op is not None:
        op = plan.having_op
        rhs = plan.having_value
        grouped_pairs = [
            pair for pair in grouped_pairs
            if sql_compare(op, pair[1], rhs) is True
        ]

    grouped: list[Row] = [out_row for out_row, _ in grouped_pairs]

    # ORDER BY on the grouped output rows. A key may name the COUNT alias or a
    # projected column; both resolve against the output row keys. Compose
    # per-element direction (never blanket-reverse). The sort is stable, so an
    # explicit ORDER BY breaks ties on the implicit group-key order above.
    if plan.order_by_col is not None:
        # Grouped output rows are keyed by the projected/alias name; a
        # datetime(<col>) sort key is not part of the grouped consumer shapes, so
        # the normalisation flag is unset here.
        keys: list[tuple[str, bool, bool]] = [(plan.order_by_col, plan.order_by_desc, False)]
        if plan.order_by_col2 is not None:
            keys.append((plan.order_by_col2, plan.order_by_desc2, False))
        grouped = _order_by_keys(grouped, keys)

    # LIMIT / OFFSET: 0 limit -> empty (labels preserved); apply offset before
    # the top-N slice so OFFSET skips from the sorted set, matching sqlite3.
    if effective_limit == 0:
        return ([], out_names)
    if effective_offset is not None and effective_offset > 0:
        grouped = grouped[effective_offset:]
    if effective_limit is not None and effective_limit > 0:
        grouped = grouped[:effective_limit]

    logger.debug(
        "executor.group_by: table=%r group_col=%r groups=%d",
        table, group_col, len(grouped),
    )
    return (grouped, out_names)


# ---------------------------------------------------------------------------
# SELECT executor
# ---------------------------------------------------------------------------


def _execute_select(
    plan: PhysicalPlan,
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
    pager: Pager,
    id_index: "dict[str, int] | None" = None,
) -> tuple[list[Row], list[str]]:
    col_names = catalog.column_names(table)
    columns = list(plan.columns) if plan.columns else ["*"]

    # Bind '?' parameters in left-to-right textual order: SELECT-list expressions
    # (when the projection is aliased / has a COALESCE) bind before the WHERE, so
    # a placeholder inside a projected expression consumes its param before the
    # WHERE placeholders. The supported consumer queries put '?' only in LIMIT
    # (never inside a COALESCE), but the bind order is kept correct regardless.
    params_queue = list(plan.params)
    named = plan.named_params
    # Shared placeholder-style tracker so mixing :name with ? across the
    # SELECT-list / WHERE / LIMIT positions of one statement is rejected.
    seen = {"positional": False, "named": False}
    bound_select_exprs: list[tuple[Expr, str]] = []
    if plan.select_exprs:
        for expr, alias in plan.select_exprs:
            bound_select_exprs.append((_bind_params(expr, params_queue, named, seen), alias))
    bound_where = None
    if plan.where is not None:
        bound_where = _bind_params(plan.where, params_queue, named, seen)

    # Resolve LIMIT and OFFSET: literals are on plan.limit/plan.offset;
    # placeholder forms consume the next param(s) AFTER the WHERE binds
    # (LIMIT is textually last, OFFSET immediately follows it).
    effective_limit = plan.limit
    if plan.limit_is_param:
        seen["positional"] = True
        if seen["named"]:
            raise ValueError(
                "bind_params: cannot mix named (:name) and positional (?) "
                "parameters in one statement"
            )
        limit_val = params_queue.pop(0) if params_queue else None
        effective_limit = None if limit_val is None else int(limit_val)

    effective_offset: int | None = plan.offset
    if plan.offset_is_param:
        seen["positional"] = True
        if seen["named"]:
            raise ValueError(
                "bind_params: cannot mix named (:name) and positional (?) "
                "parameters in one statement"
            )
        offset_val = params_queue.pop(0) if params_queue else None
        effective_offset = None if offset_val is None else int(offset_val)

    # Aliased / COALESCE projection path: decode the source columns the select
    # expressions, WHERE, and ORDER BY reference; filter; sort; then evaluate
    # each (expr, alias) per row so the output is keyed by the alias in source
    # order (positional access matches sqlite3's column order). This path never
    # uses the id point-lookup fast path (which projects by plain column names).
    if bound_select_exprs:
        return _execute_select_exprs(
            plan=plan,
            cursor=cursor,
            catalog=catalog,
            table=table,
            col_names=col_names,
            bound_select_exprs=bound_select_exprs,
            bound_where=bound_where,
            effective_limit=effective_limit,
            effective_offset=effective_offset,
        )

    # GROUP BY path: bucket the WHERE-filtered rows by the group column, project
    # each grouped select item (a plain column takes its first-row-in-group
    # value; the COUNT(*) item is the group size), apply HAVING, sort, and LIMIT.
    # The grouping execute owns the aggregate label — it is never routed through
    # the per-row projector (which would collapse the rows incorrectly).
    if plan.group_by_col is not None:
        return _execute_group_by(
            plan=plan,
            cursor=cursor,
            catalog=catalog,
            table=table,
            bound_where=bound_where,
            effective_limit=effective_limit,
            effective_offset=effective_offset,
        )

    # Point-lookup fast path: WHERE id = 'literal' with LIMIT >= 1 and an index.
    # Navigates the B+tree in O(log N) instead of a full O(N) seq_scan.
    if (
        bound_where is not None
        and id_index is not None
        and effective_limit is not None
        and effective_limit >= 1
    ):
        id_val = _extract_id_eq(bound_where)
        if id_val is not None:
            if id_val not in id_index:
                # Row does not exist — return empty result immediately.
                out_col_names = col_names if columns == ["*"] else columns
                return ([], out_col_names)
            from iai_mcp.lillibrain.btree.page import read_all_leaf_cells  # noqa: PLC0415
            row_key = id_index[id_val]
            leaf_page_no, insert_idx = cursor.move_to(row_key)
            page_data = cursor._pager.read_page(leaf_page_no)
            cells = read_all_leaf_cells(memoryview(page_data))
            if insert_idx < len(cells) and cells[insert_idx][0] == row_key:
                payload = cells[insert_idx][1]
                vals, _ = decode_record(payload, 0)
                while len(vals) < len(col_names):
                    vals.append(SqlNull)
                row = dict(zip(col_names, vals))
                # Apply the WHERE predicate to confirm the cell is a true match
                # (guards against stale index entries).
                result = eval_expr(bound_where, row, [])
                if result is not SqlNull and result:
                    if columns != ["*"]:
                        if _ROWID in columns:
                            _resolve_rowid(row)
                        row = {c: row[c] for c in columns}
                    out_col_names = col_names if columns == ["*"] else columns
                    return ([row], out_col_names)
            # Index miss or predicate mismatch — fall through to seq_scan.

    # Decode-time column projection: when the query names specific columns
    # (not SELECT *) and is not an aggregate, resolve requested column names
    # to their catalog indices and route through seq_scan_projected so only
    # those cells are decoded.  The WHERE-referenced columns are unioned in
    # so filters can still evaluate correctly; those extra decoded columns are
    # stripped from the output rows before yielding.
    use_projected_scan = (
        columns != ["*"]
        and plan.aggregate is None
    )

    # COUNT(*) fast path: count rows without decoding any row payload.
    # A bare COUNT(*) (no WHERE) sums leaf-cell counts directly. A filtered
    # COUNT(*) decodes only the WHERE-referenced columns via the projected
    # scan, never the full record (e.g. the embedding BLOB). Either way the
    # wide embedding column is never materialised just to count rows.
    if plan.aggregate == "count" and plan.group_by_col is None and plan.order_by_col is None:
        label = plan.count_alias or "COUNT(*)"
        name_to_idx = {name: i for i, name in enumerate(col_names)}
        if bound_where is None:
            count = count_leaf_cells(cursor)
            logger.debug(
                "executor.select COUNT(*) cell-count: table=%r count=%d", table, count
            )
            return ([{label: count}], [label])
        where_col_indices = {
            name_to_idx[wc]
            for wc in _collect_column_names(bound_where)
            if wc in name_to_idx
        }
        decode_indices = frozenset(where_col_indices)
        source = seq_scan_projected(
            cursor, catalog, table, decode_indices, col_names
        )
        source = filter_op(source, bound_where, [])
        count = sum(1 for _ in source)
        logger.debug(
            "executor.select COUNT(*) projected: table=%r count=%d", table, count
        )
        return ([{label: count}], [label])

    if use_projected_scan:
        # Map requested output column names → catalog indices (skip unknowns).
        name_to_idx = {name: i for i, name in enumerate(col_names)}
        output_indices: set[int] = {
            name_to_idx[c] for c in columns if c in name_to_idx
        }
        # Union in any column referenced by the WHERE predicate so filters work.
        where_col_indices: set[int] = set()
        if bound_where is not None:
            for wc in _collect_column_names(bound_where):
                if wc in name_to_idx:
                    where_col_indices.add(name_to_idx[wc])
        # Union in the ORDER BY column(s) so the sort can read them even when
        # not in the SELECT projection. project_op strips them from the output
        # afterwards if they were not requested.
        order_col_indices: set[int] = set()
        if plan.order_by_col is not None and plan.order_by_col in name_to_idx:
            order_col_indices.add(name_to_idx[plan.order_by_col])
        if plan.order_by_col2 is not None and plan.order_by_col2 in name_to_idx:
            order_col_indices.add(name_to_idx[plan.order_by_col2])
        # rowid aliases vec_label: when rowid is selected or ordered, decode the
        # vec_label column so the alias value is available on the projected path.
        rowid_indices: set[int] = set()
        if (_ROWID in columns or plan.order_by_col == _ROWID) and _ROWID_SOURCE in name_to_idx:
            rowid_indices.add(name_to_idx[_ROWID_SOURCE])
        decode_indices = frozenset(
            output_indices | where_col_indices | order_col_indices | rowid_indices
        )
        source: Iterator[Row] = seq_scan_projected(
            cursor, catalog, table, decode_indices, col_names
        )
    else:
        source = seq_scan(cursor, catalog, table)

    # Apply WHERE filter.
    if bound_where is not None:
        source = filter_op(source, bound_where, [])

    # ORDER BY: materialise the already-filtered rows and sort them NULL-aware
    # and stable BEFORE projection and LIMIT, so an unprojected sort column is
    # still available and LIMIT keeps the top-N of the sorted set. Aggregates
    # collapse to one row, so the sort is irrelevant there and left unapplied.
    if plan.order_by_col is not None and plan.aggregate is None:
        # rowid: prefer vec_label (never-reused HWM) when the table has that
        # column; fall back to the raw B+tree key (_rowkey) otherwise, which
        # matches sqlite3's plain-rowid semantics for ordinary tables.
        def _sort_col_name(c: str) -> str:
            if c != _ROWID:
                return c
            return _ROWID_SOURCE if _ROWID_SOURCE in name_to_idx else _ROWKEY

        if plan.order_by_col2 is not None:
            # Two-key ORDER BY: use the multi-key helper so the second key is
            # applied as a tiebreaker with its own direction, not dropped.
            sort_keys = [
                (_sort_col_name(plan.order_by_col), plan.order_by_desc, plan.order_by_datetime),
                (_sort_col_name(plan.order_by_col2), plan.order_by_desc2, plan.order_by_datetime2),
            ]
            source = iter(_order_by_keys(list(source), sort_keys))
        else:
            source = iter(
                _order_by(
                    list(source), _sort_col_name(plan.order_by_col),
                    plan.order_by_desc, plan.order_by_datetime,
                )
            )

    # LIMIT 0 shortcut: return empty rows with populated col_names.
    # Aggregates with LIMIT 0 also return empty rows (matches sqlite3 behaviour).
    if effective_limit == 0:
        if plan.aggregate == "count":
            return ([], [plan.count_alias or "COUNT(*)"])
        if plan.aggregate == "sum":
            agg_col = plan.aggregate_column or ""
            return ([], [f"SUM({agg_col})"])
        if plan.aggregate == "min":
            agg_col = plan.aggregate_column or ""
            return ([], [f"MIN({agg_col})"])
        if plan.aggregate == "max":
            agg_col = plan.aggregate_column or ""
            return ([], [f"MAX({agg_col})"])
        out_col_names = col_names if columns == ["*"] else columns
        return ([], out_col_names)

    # Materialise the filtered rows for aggregates or standard projection.
    if plan.aggregate == "count":
        count = sum(1 for _ in source)
        label = plan.count_alias or "COUNT(*)"
        logger.debug("executor.select COUNT(*): table=%r count=%d", table, count)
        return ([{label: count}], [label])

    if plan.aggregate == "sum":
        agg_col = plan.aggregate_column or ""
        total = 0
        for row in source:
            val = row.get(agg_col, SqlNull)
            if val is not SqlNull and isinstance(val, int):
                total += val
        label = f"SUM({agg_col})"
        logger.debug("executor.select SUM(%s): table=%r sum=%d", agg_col, table, total)
        return ([{label: total}], [label])

    if plan.aggregate in ("min", "max"):
        agg_col = plan.aggregate_column or ""
        want_max = plan.aggregate == "max"
        best: object = SqlNull
        best_key: tuple[int, object] | None = None
        for row in source:
            val = row.get(agg_col, SqlNull)
            if val is SqlNull or val is None:
                continue  # NULLs are ignored by MIN/MAX (sqlite3 semantics)
            key = _order_key(val)
            if best_key is None or (key > best_key if want_max else key < best_key):
                best_key = key
                best = val
        # Empty set or all-NULL column -> NULL (best_key stayed None).
        label = f"{'MAX' if want_max else 'MIN'}({agg_col})"
        logger.debug(
            "executor.select %s: table=%r result=%r", label, table, best
        )
        return ([{label: best}], [label])

    # Standard projection.
    # For SELECT *: project_op passes rows through unchanged.
    # For named-column projected scan: strip any filter-only columns that were
    # decoded for predicate evaluation but are not in the requested output.
    # rowid is an explicit selectable only (never injected into SELECT *):
    # alias it to vec_label on each row before project_op reads row['rowid'].
    if _ROWID in columns:
        source = (_resolve_rowid(row) for row in source)
    source = project_op(source, columns)
    # OFFSET skips the first N rows AFTER ordering, BEFORE the LIMIT top-N slice.
    if effective_offset is not None and effective_offset > 0:
        rows_before_limit = list(source)
        rows_before_limit = rows_before_limit[effective_offset:]
        source = iter(rows_before_limit)
    if effective_limit is not None and effective_limit > 0:
        source = limit_op(source, effective_limit)

    rows = list(source)
    out_col_names = col_names if columns == ["*"] else columns
    logger.debug("executor.select: table=%r rows=%d", table, len(rows))
    return (rows, out_col_names)


# ---------------------------------------------------------------------------
# INSERT executor
# ---------------------------------------------------------------------------


def _execute_insert(
    plan: PhysicalPlan,
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
    pager: Pager,
) -> tuple[list[Row], list[str]]:
    from iai_mcp.lillibrain.sql.ast_nodes import ExcludedRef  # noqa: PLC0415

    col_names = catalog.column_names(table)
    insert_cols = list(plan.insert_columns)
    params_queue = list(plan.params)

    # plan.params is fully resolved by the planner (literals and placeholders
    # are both pre-evaluated there); params_queue is a concrete value per column.
    # We must reconstruct the original InsertStmt to get the values list,
    # but plan does not carry it — we use params directly in column order.
    #
    # The INSERT plan carries insert_columns; the params are in column order.
    # Build the value map from insert_cols + params.
    if len(params_queue) < len(insert_cols):
        # Pad short param lists (unlikely but safe)
        params_queue.extend([SqlNull] * (len(insert_cols) - len(params_queue)))

    value_map: dict[str, object] = {}
    for col, val in zip(insert_cols, params_queue):
        value_map[col] = SqlNull if val is None else val

    # Conflict resolution: check if a row with the same primary-key column
    # already exists. For OR IGNORE without an explicit ON CONFLICT clause,
    # fall back to the catalog's primary-key columns.
    conflict_keys = list(plan.conflict_keys)
    conflict_action = plan.conflict_action
    or_ignore = plan.or_ignore
    or_replace = plan.or_replace

    # Build the catalog-ordered payload (DEFAULT fill + NOT NULL enforcement)
    # once, so a replace-on-conflict overwrite honors the same constraints as a
    # fresh insert.
    def _build_payload() -> list:
        from iai_mcp.lillibrain.sql.catalog import _SENTINEL  # noqa: PLC0415

        col_meta_local = catalog.get(table)
        out = []
        for cm in col_meta_local:
            val = value_map.get(cm.name, SqlNull)
            if (val is SqlNull or val is None) and cm.default is not _SENTINEL:
                val = cm.default if cm.default is not None else SqlNull
            out.append(val)
        for cm, val in zip(col_meta_local, out):
            if not cm.nullable and (val is SqlNull or val is None):
                raise ValueError(
                    f"NOT NULL constraint failed: {table}.{cm.name}"
                )
        return out

    # For OR IGNORE / OR REPLACE without explicit conflict_keys, derive them
    # from the catalog.
    if (or_ignore or or_replace) and not conflict_keys:
        conflict_keys = [
            cm.name for cm in catalog.get(table) if cm.primary_key
        ]
        if not conflict_keys:
            # No primary key declared — treat first column as key (SQLite rowid analogy).
            all_cols = catalog.column_names(table)
            if all_cols:
                conflict_keys = [all_cols[0]]

    # The next B+tree key is one past the current maximum. The tree is strictly
    # key-ordered, so the maximum is the rightmost leaf's last cell — an
    # O(tree-height) descent, not an O(rows) scan. This runs on every insert, so
    # deriving it without materialising the whole table is what keeps bulk insert
    # linear instead of quadratic.
    needs_conflict_scan = bool(
        (conflict_action or or_ignore or or_replace) and conflict_keys
    )
    if needs_conflict_scan:
        # A conflict clause must compare the attempted row against every existing
        # row on the conflict columns (no engine index on arbitrary columns), so
        # the full scan is unavoidable here. Derive next_key from the same pass
        # rather than walking the tree twice.
        existing_pairs = list(cursor.full_scan())
        existing_keys = [k for k, _ in existing_pairs]
        next_key = (max(existing_keys) + 1) if existing_keys else 1
    else:
        max_existing = cursor.max_key()
        next_key = (max_existing + 1) if max_existing is not None else 1
        existing_pairs = []

    # Check for conflict on the primary / conflict key columns.
    if needs_conflict_scan:
        # Find an existing row that matches all conflict key values.
        for key, payload in existing_pairs:
            existing_vals, _ = decode_record(payload, 0)
            while len(existing_vals) < len(col_names):
                existing_vals.append(SqlNull)
            existing_row = dict(zip(col_names, existing_vals))
            # NULL != NULL for uniqueness purposes (sqlite3 semantics):
            # two rows with NULL in a conflict key column are never considered
            # a conflict, so multiple NULL PKs are allowed.
            match = all(
                sql_compare("=", existing_row.get(ck, SqlNull), value_map.get(ck, SqlNull)) is True
                for ck in conflict_keys
            )
            if match:
                if conflict_action == "update":
                    # ON CONFLICT DO UPDATE: apply set_clauses
                    # ExcludedRef(col) resolves to the attempted-insert value.
                    updated_row = dict(existing_row)
                    for set_col, set_expr in plan.set_clauses:
                        if isinstance(set_expr, ExcludedRef):
                            updated_row[set_col] = value_map.get(set_expr.col_name, SqlNull)
                        else:
                            updated_row[set_col] = eval_expr(set_expr, existing_row, [])
                    payload_vals = [updated_row.get(cn, SqlNull) for cn in col_names]
                    pager.begin_write()
                    cursor.insert(key, encode_record(payload_vals))
                    pager.commit()
                elif or_replace:
                    # REPLACE-on-conflict: overwrite the existing row in place
                    # with the attempted-insert values (delete-then-insert
                    # semantics collapse to an in-place rewrite at the same key).
                    # Build the payload (which enforces NOT NULL and may raise)
                    # before opening the write transaction, so a constraint
                    # failure never leaves a transaction half-open.
                    replace_payload = _build_payload()
                    pager.begin_write()
                    cursor.insert(key, encode_record(replace_payload))
                    pager.commit()
                # "nothing" or or_ignore: skip silently
                return ([], [])

        # No conflict found — proceed to normal insert below.

    # Build payload in catalog column order; unsupplied columns get their
    # DEFAULT if one is declared, otherwise SqlNull. NOT NULL is enforced
    # before writing (shared with the replace-on-conflict overwrite path).
    payload_vals = _build_payload()

    pager.begin_write()
    cursor.insert(next_key, encode_record(payload_vals))
    pager.commit()
    logger.debug("executor.insert: table=%r key=%d", table, next_key)
    return ([], [])


# ---------------------------------------------------------------------------
# UPDATE executor
# ---------------------------------------------------------------------------


def _extract_id_eq(where: "Expr") -> "str | None":
    """Return the string literal if WHERE is exactly ``id = 'literal'``, else None.

    Detects the pattern produced by merge_insert provenance updates:
      BinOp(op='=', left=ColumnRef(name='id'), right=Literal(value=str))
    Returns the string value so the caller can use a cached row-key index
    instead of a full table scan.
    """
    from iai_mcp.lillibrain.sql.ast_nodes import BinOp, ColumnRef, Literal  # noqa: PLC0415

    if (
        isinstance(where, BinOp)
        and where.op == "="
        and isinstance(where.left, ColumnRef)
        and where.left.name == "id"
        and isinstance(where.right, Literal)
        and isinstance(where.right.value, str)
    ):
        return where.right.value
    return None


def _execute_update(
    plan: PhysicalPlan,
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
    pager: Pager,
    id_index: "dict[str, int] | None" = None,
) -> tuple[list[Row], list[str]]:
    col_names = catalog.column_names(table)
    params_queue = list(plan.params)
    named = plan.named_params
    seen = {"positional": False, "named": False}

    # Params in SQL textual order: SET clause ?s appear before WHERE ?s.
    # Bind SET expressions first, collecting the bound values; then bind WHERE.
    bound_set_clauses: list[tuple[str, object]] = []
    for set_col, set_expr in plan.set_clauses:
        bound_expr = _bind_params(set_expr, params_queue, named, seen)
        # Evaluate against an empty row to get the literal value (SET rhs
        # is always a literal or placeholder, never a column reference).
        bound_val = eval_expr(bound_expr, {}, [])
        bound_set_clauses.append((set_col, bound_val))

    # Remaining params are for the WHERE clause.
    bound_where = None
    if plan.where is not None:
        bound_where = _bind_params(plan.where, params_queue, named, seen)

    # Point-lookup fast path: when WHERE is ``id = 'literal'`` and an in-memory
    # id→row_key index is available, skip the O(N) full scan and fetch the
    # target row in O(log N) via cursor.move_to(key).
    # move_to() returns (leaf_page_no, insert_idx); we read that leaf cell
    # directly instead of calling cursor.next() which restarts from page 1.
    to_update: list[tuple[int, Row]] = []
    used_fast_path = False
    if bound_where is not None and id_index is not None:
        id_val = _extract_id_eq(bound_where)
        if id_val is not None and id_val in id_index:
            from iai_mcp.lillibrain.btree.page import read_all_leaf_cells  # noqa: PLC0415
            row_key = id_index[id_val]
            leaf_page_no, insert_idx = cursor.move_to(row_key)
            page_data = cursor._pager.read_page(leaf_page_no)
            cells = read_all_leaf_cells(memoryview(page_data))
            if insert_idx < len(cells) and cells[insert_idx][0] == row_key:
                payload = cells[insert_idx][1]
                vals, _ = decode_record(payload, 0)
                while len(vals) < len(col_names):
                    vals.append(SqlNull)
                row = dict(zip(col_names, vals))
                result = eval_expr(bound_where, row, [])
                if result is not SqlNull and result:
                    to_update.append((row_key, row))
            used_fast_path = True
        elif id_val is not None and id_val not in id_index:
            # id not in index → row does not exist; skip the full scan.
            return ([], [])

    if not to_update and not used_fast_path:
        # Fall back to full scan for WHERE clauses the index cannot serve.
        for key, payload in cursor.full_scan():
            vals, _ = decode_record(payload, 0)
            while len(vals) < len(col_names):
                vals.append(SqlNull)
            row = dict(zip(col_names, vals))
            if bound_where is None:
                to_update.append((key, row))
            else:
                result = eval_expr(bound_where, row, [])
                if result is not SqlNull and result:
                    to_update.append((key, row))

    if not to_update:
        return ([], [])

    pager.begin_write()
    for key, row in to_update:
        updated = dict(row)
        for set_col, set_val in bound_set_clauses:
            updated[set_col] = set_val
        payload_vals = [updated.get(cn, SqlNull) for cn in col_names]
        cursor.insert(key, encode_record(payload_vals))
    pager.commit()
    logger.debug("executor.update: table=%r rows_updated=%d", table, len(to_update))
    return ([], [])


# ---------------------------------------------------------------------------
# DELETE executor
# ---------------------------------------------------------------------------


def _execute_delete(
    plan: PhysicalPlan,
    cursor: BtreeCursor,
    catalog: Catalog,
    table: str,
    pager: Pager,
) -> tuple[list[Row], list[str]]:
    col_names = catalog.column_names(table)
    params_queue = list(plan.params)
    named = plan.named_params
    seen = {"positional": False, "named": False}

    # Bind WHERE parameters.
    bound_where = None
    if plan.where is not None:
        bound_where = _bind_params(plan.where, params_queue, named, seen)

    # Collect keys to delete.
    to_delete: list[int] = []
    for key, payload in cursor.full_scan():
        vals, _ = decode_record(payload, 0)
        while len(vals) < len(col_names):
            vals.append(SqlNull)
        row = dict(zip(col_names, vals))
        if bound_where is None:
            to_delete.append(key)
        else:
            result = eval_expr(bound_where, row, [])
            if result is not SqlNull and result:
                to_delete.append(key)

    if not to_delete:
        return ([], [])

    pager.begin_write()
    for key in to_delete:
        cursor.delete(key)
    pager.commit()
    logger.debug("executor.delete: table=%r rows_deleted=%d", table, len(to_delete))
    return ([], [])


# ---------------------------------------------------------------------------
# Public convenience entry point
# ---------------------------------------------------------------------------


def run(
    sql: str,
    params: list,
    pager: Pager,
    catalog: Catalog,
    root_map: dict[str, int],
    id_index: "dict[str, int] | None" = None,
    named_params: "dict | None" = None,
) -> tuple[list[Row], list[str]]:
    """Parse → plan → execute a SQL string and return (rows, col_names).

    This is the single entry point for callers — parse_sql, plan_stmt, and
    execute are composed here. Catalog is threaded into the plan via a
    transient _catalog attribute so execute() can access it without altering
    the frozen PhysicalPlan dataclass.

    DDL statements (CREATE TABLE, ALTER TABLE, CREATE INDEX, PRAGMA) are applied
    to the catalog here before any planner or executor step, so subsequent DML
    against the same catalog sees the updated schema.

    id_index: optional per-table ``{id_str → row_key}`` map supplied by the
    connection layer. When present and the plan is an UPDATE with a simple
    ``id = 'literal'`` WHERE clause, the executor uses a O(log N) B+tree
    point lookup instead of an O(N) full scan.

    named_params: optional ``{name → value}`` map for ``:name`` placeholder
    binds. When supplied, the positional ``params`` list is empty and the
    executor resolves each NamedParam by name; mixing the two styles raises.
    """
    from iai_mcp.lillibrain.sql.ast_nodes import DDLStmt   # noqa: PLC0415
    from iai_mcp.lillibrain.sql.parser import parse_sql    # noqa: PLC0415
    from iai_mcp.lillibrain.sql.planner import plan_stmt   # noqa: PLC0415

    stmt = parse_sql(sql)

    # DDL: apply to catalog immediately and return — no plan/execute needed.
    if isinstance(stmt, DDLStmt):
        catalog.apply_ddl(stmt)
        return ([], [])

    plan = plan_stmt(stmt, catalog, params, named_params)

    # Attach catalog to the plan via object __dict__ (frozen dataclasses allow
    # setting attributes that are not declared fields — they land in __dict__).
    object.__setattr__(plan, "_catalog", catalog)

    return execute(plan, pager, root_map, id_index=id_index)

"""SQL three-valued NULL type and comparison operators.

Implements the SQL NULL sentinel and the five three-valued logic operators
required for correct WHERE-clause evaluation. All functions are pure (no I/O,
no module-level side effects beyond constructing the singleton).

NULL semantics follow the SQLite specification: https://www.sqlite.org/nulls.html
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# SqlNull singleton
# ---------------------------------------------------------------------------


class _SqlNullType:
    """Singleton sentinel for SQL NULL in three-valued logic."""

    _instance: _SqlNullType | None = None

    def __new__(cls) -> _SqlNullType:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self) -> bool:  # noqa: ANN101
        # NULL is never truthy in a WHERE predicate — treat as False.
        return False

    def __repr__(self) -> str:
        return "SqlNull"


SqlNull = _SqlNullType()


# ---------------------------------------------------------------------------
# sql_compare — three-valued equality and ordering
# ---------------------------------------------------------------------------


def sql_compare(op: str, a: object, b: object) -> bool | _SqlNullType:
    """Three-valued comparison: any NULL operand propagates to SqlNull (not False).

    Supports operators: =, !=, <, <=, >, >=.
    Raises ValueError for any other operator string.
    """
    if a is SqlNull or b is SqlNull:
        return SqlNull
    if op == "=":
        return a == b
    if op == "!=":
        return a != b
    if op == "<":
        return a < b  # type: ignore[operator]
    if op == "<=":
        return a <= b  # type: ignore[operator]
    if op == ">":
        return a > b  # type: ignore[operator]
    if op == ">=":
        return a >= b  # type: ignore[operator]
    raise ValueError(f"sql_compare: unknown operator {op!r}")


# ---------------------------------------------------------------------------
# sql_is_null — IS NULL predicate; always returns a real bool, never SqlNull
# ---------------------------------------------------------------------------


def sql_is_null(a: object) -> bool:
    """IS NULL — returns True/False (never SqlNull), even when operand is SqlNull."""
    return a is SqlNull


# ---------------------------------------------------------------------------
# sql_and / sql_or / sql_not — three-valued boolean operators
# ---------------------------------------------------------------------------


def sql_and(a: object, b: object) -> bool | _SqlNullType:
    """Three-valued AND: NULL AND FALSE → FALSE; NULL AND TRUE → NULL.

    Uses bool() rather than identity (is False) so integer 0 (the FALSE
    literal emitted by the parser) short-circuits correctly.
    """
    if (a is not SqlNull and not bool(a)) or (b is not SqlNull and not bool(b)):
        return False
    if a is SqlNull or b is SqlNull:
        return SqlNull
    return bool(a) and bool(b)


def sql_or(a: object, b: object) -> bool | _SqlNullType:
    """Three-valued OR: NULL OR TRUE → TRUE; NULL OR FALSE → NULL.

    Uses bool() rather than identity (is True) so integer 1 (the TRUE
    literal emitted by the parser) short-circuits correctly.
    """
    if (a is not SqlNull and bool(a)) or (b is not SqlNull and bool(b)):
        return True
    if a is SqlNull or b is SqlNull:
        return SqlNull
    return bool(a) or bool(b)


def sql_not(a: object) -> bool | _SqlNullType:
    """Three-valued NOT: NOT NULL → NULL."""
    if a is SqlNull:
        return SqlNull
    return not bool(a)

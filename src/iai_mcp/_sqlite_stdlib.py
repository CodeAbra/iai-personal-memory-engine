"""Lazy stdlib-SQLite access: the only module allowed to import sqlite3.

The native engine is the default storage driver; the stdlib driver serves
legacy-format stores and dev-only oracle checks. Importing this module does
NOT import sqlite3 — the import happens on first `connect()` / attribute
access, so a runtime that never opens a legacy store never loads sqlite3.

Every connection returned by `connect()` is wrapped so sqlite3 exceptions
surface as the `iai_mcp.errors` hierarchy — callers match ONE hierarchy
regardless of driver. Corruption (`sqlite3.DatabaseError` proper) maps to
the bare `DatabaseError`, never a subclass, so `except OperationalError`
cannot swallow disk damage.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

from iai_mcp import errors


def _mod() -> Any:
    return importlib.import_module("sqlite3")


def __getattr__(name: str) -> Any:
    if name in ("Row", "Connection", "Cursor", "PARSE_DECLTYPES", "PARSE_COLNAMES"):
        return getattr(_mod(), name)
    raise AttributeError(name)


def sqlite_version() -> str:
    return _mod().sqlite_version


def is_stdlib_connection(obj: object) -> bool:
    if getattr(obj, "_iai_stdlib_conn", None) is not None:
        return True
    sqlite3 = sys.modules.get("sqlite3")
    return sqlite3 is not None and isinstance(obj, sqlite3.Connection)


def translate_error(exc: BaseException) -> BaseException | None:
    """The `iai_mcp.errors` equivalent for a stdlib sqlite3 exception, else None.

    Most-derived classes first; `sqlite3.DatabaseError` proper stays the bare
    `DatabaseError` so operational handlers never swallow corruption.
    """
    sqlite3 = _mod()
    mapping = (
        (sqlite3.OperationalError, errors.OperationalError),
        (sqlite3.IntegrityError, errors.IntegrityError),
        (sqlite3.ProgrammingError, errors.ProgrammingError),
        (sqlite3.DataError, errors.DataError),
        (sqlite3.InterfaceError, errors.InterfaceError),
        (sqlite3.DatabaseError, errors.DatabaseError),
        (sqlite3.Error, errors.Error),
    )
    for src, dst in mapping:
        if isinstance(exc, src):
            return dst(*exc.args)
    return None


def _call(bound: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return bound(*args, **kwargs)
    except Exception as exc:
        translated = translate_error(exc)
        if translated is not None:
            raise translated from exc
        raise


class _TranslatingCursor:
    __slots__ = ("_c",)

    def __init__(self, cursor: Any) -> None:
        object.__setattr__(self, "_c", cursor)

    def execute(self, *args: Any, **kwargs: Any) -> "_TranslatingCursor":
        _call(self._c.execute, *args, **kwargs)
        return self

    def executemany(self, *args: Any, **kwargs: Any) -> "_TranslatingCursor":
        _call(self._c.executemany, *args, **kwargs)
        return self

    def fetchone(self) -> Any:
        return _call(self._c.fetchone)

    def fetchmany(self, *args: Any, **kwargs: Any) -> Any:
        return _call(self._c.fetchmany, *args, **kwargs)

    def fetchall(self) -> Any:
        return _call(self._c.fetchall)

    def __iter__(self) -> Any:
        while True:
            row = _call(self._c.fetchone)
            if row is None:
                return
            yield row

    def close(self) -> None:
        self._c.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_c"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_c"), name, value)


class _TranslatingConnection:
    __slots__ = ("_iai_stdlib_conn",)

    def __init__(self, conn: Any) -> None:
        object.__setattr__(self, "_iai_stdlib_conn", conn)

    def execute(self, *args: Any, **kwargs: Any) -> _TranslatingCursor:
        return _TranslatingCursor(
            _call(self._iai_stdlib_conn.execute, *args, **kwargs)
        )

    def executemany(self, *args: Any, **kwargs: Any) -> _TranslatingCursor:
        return _TranslatingCursor(
            _call(self._iai_stdlib_conn.executemany, *args, **kwargs)
        )

    def executescript(self, *args: Any, **kwargs: Any) -> _TranslatingCursor:
        return _TranslatingCursor(
            _call(self._iai_stdlib_conn.executescript, *args, **kwargs)
        )

    def cursor(self) -> _TranslatingCursor:
        return _TranslatingCursor(_call(self._iai_stdlib_conn.cursor))

    def commit(self) -> None:
        _call(self._iai_stdlib_conn.commit)

    def rollback(self) -> None:
        _call(self._iai_stdlib_conn.rollback)

    def close(self) -> None:
        _call(self._iai_stdlib_conn.close)

    def backup(self, target: Any, **kwargs: Any) -> None:
        inner = getattr(target, "_iai_stdlib_conn", target)
        _call(self._iai_stdlib_conn.backup, inner, **kwargs)

    def __enter__(self) -> "_TranslatingConnection":
        self._iai_stdlib_conn.__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> Any:
        return _call(self._iai_stdlib_conn.__exit__, *exc_info)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_iai_stdlib_conn"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_iai_stdlib_conn"), name, value)


def connect(*args: Any, **kwargs: Any) -> _TranslatingConnection:
    return _TranslatingConnection(_call(_mod().connect, *args, **kwargs))

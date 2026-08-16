"""Regression for the raw-writer fence: a bare (opt-in-less) writer open on the
lilli driver bypasses the read-only pool's generation fence, so it must fail
loud instead of silently handing back an unfenced writer.

Covers:
- Default `get_lilli_raw_conn` / `open_store_conn` (no allow_writer) raise.
- `allow_writer=True` returns a usable writer whose committed write is visible
  to a fresh read-only snapshot (the engine autocommits every execute).
- Every read_only=True path is unchanged (no opt-in required, no raise).
- Zero writer-path call sites remain under src/iai_mcp/ (with a positive
  control proving the scanner actually flags a planted violation).
"""
from __future__ import annotations

import ast
import contextlib
import os
import tempfile
from pathlib import Path

import pytest

from iai_mcp import errors
from iai_mcp.hippo._db import _open_storage_connection
from iai_mcp.hippo._raw_open import open_store_conn


SRC = Path(__file__).resolve().parent.parent / "src" / "iai_mcp"


@pytest.fixture
def lilli_driver(monkeypatch):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


def _path() -> str:
    return os.path.join(tempfile.mkdtemp(), "t.lilli")


# ---------------------------------------------------------------------------
# Default (opt-in-less) writer open raises — registry hit
# ---------------------------------------------------------------------------


def test_get_lilli_raw_conn_default_raises_registry_hit(lilli_driver):
    """A live writer connection is registered; a bare get_lilli_raw_conn call
    (no allow_writer) must raise instead of silently borrowing the writer."""
    from iai_mcp.lillibrain.connection import get_lilli_raw_conn

    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")

    with pytest.raises(errors.DatabaseError, match="generation fence"):
        get_lilli_raw_conn(path)


def test_open_store_conn_default_raises_registry_hit(lilli_driver):
    """open_store_conn forwards the same fence: a bare call raises."""
    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")

    with pytest.raises(errors.DatabaseError, match="generation fence"):
        open_store_conn(path)


# ---------------------------------------------------------------------------
# allow_writer=True opts in: writer executes+commits, visible to a fresh
# read-only snapshot (engine autocommits every execute).
# ---------------------------------------------------------------------------


def test_allow_writer_opt_in_write_visible_to_fresh_ro_snapshot(lilli_driver):
    from iai_mcp_native import engine as _engine

    from iai_mcp.lillibrain.connection import get_lilli_raw_conn

    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY, v TEXT)")

    raw = get_lilli_raw_conn(path, allow_writer=True)
    assert raw is not None
    raw.execute("INSERT INTO t (id, v) VALUES ('a', 'first')")
    raw.commit()

    ro = _engine.Connection.open_read_only(path, 0)
    try:
        rows = ro.execute("SELECT id, v FROM t WHERE id = 'a'").fetchall()
        assert len(rows) == 1
        assert rows[0]["v"] == "first"
    finally:
        # Read-only engine connections cannot flush on close (no write path);
        # the read data is already materialized above, so a close error is
        # not a lost-read.
        with contextlib.suppress(Exception):
            ro.close()
    raw.close()


def test_open_store_conn_allow_writer_returns_usable_writer(lilli_driver):
    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY, v TEXT)")

    writer = open_store_conn(path, allow_writer=True)
    assert writer is not None
    writer.execute("INSERT INTO t (id, v) VALUES ('b', 'second')")
    writer.commit()
    rows = writer.execute("SELECT id, v FROM t WHERE id = 'b'").fetchall()
    assert len(rows) == 1
    assert rows[0]["v"] == "second"
    writer.close()


# ---------------------------------------------------------------------------
# read_only=True paths are unchanged — no opt-in required, no raise.
# ---------------------------------------------------------------------------


def test_read_only_path_unchanged_no_opt_in_required(lilli_driver):
    from iai_mcp.lillibrain.connection import get_lilli_raw_conn

    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO t (id) VALUES ('a')")

    raw = get_lilli_raw_conn(path, read_only=True)
    assert raw is not None
    rows = raw.execute("SELECT id FROM t").fetchall()
    assert len(rows) == 1
    raw.close()


def test_open_store_conn_read_only_unchanged(lilli_driver):
    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO t (id) VALUES ('a')")

    reader = open_store_conn(path, read_only=True)
    assert reader is not None
    rows = reader.execute("SELECT id FROM t").fetchall()
    assert len(rows) == 1
    reader.close()


# ---------------------------------------------------------------------------
# Absent-path default open keeps its prior contract (RuntimeError from
# open_store_conn / None from get_lilli_raw_conn) — the opt-in raise must sit
# AFTER the not-exists return, not before.
# ---------------------------------------------------------------------------


def test_absent_path_default_open_still_raises_runtimeerror(tmp_path: Path, lilli_driver):
    absent = tmp_path / "does_not_exist" / "brain.sqlite3"
    with pytest.raises(RuntimeError, match="lilli driver active but no engine connection"):
        open_store_conn(absent)


def test_absent_path_get_lilli_raw_conn_returns_none(tmp_path: Path, lilli_driver):
    from iai_mcp.lillibrain.connection import get_lilli_raw_conn

    absent = str(tmp_path / "does_not_exist" / "brain.sqlite3")
    assert get_lilli_raw_conn(absent) is None


# ---------------------------------------------------------------------------
# Registry-miss writer branch also fences: a closed store's file still raises
# on a bare default writer open (the second WRITER-OPEN site).
# ---------------------------------------------------------------------------


def test_registry_miss_default_writer_open_raises(lilli_driver):
    from iai_mcp.lillibrain.connection import get_lilli_raw_conn

    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    conn.close()
    from iai_mcp.lillibrain.connection import deregister_lilli_conn

    deregister_lilli_conn(path, conn)

    with pytest.raises(errors.DatabaseError, match="generation fence"):
        get_lilli_raw_conn(path)


def test_registry_miss_allow_writer_still_works(lilli_driver):
    from iai_mcp.lillibrain.connection import deregister_lilli_conn, get_lilli_raw_conn

    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY, v TEXT)")
    conn.close()
    deregister_lilli_conn(path, conn)

    raw = get_lilli_raw_conn(path, allow_writer=True)
    assert raw is not None
    raw.execute("INSERT INTO t (id, v) VALUES ('c', 'third')")
    raw.commit()
    rows = raw.execute("SELECT id, v FROM t WHERE id = 'c'").fetchall()
    assert len(rows) == 1
    assert rows[0]["v"] == "third"
    raw.close()


# ---------------------------------------------------------------------------
# Zero-src-writer-site guard: a pure-function scanner over source text, with
# a mandatory positive control proving it actually flags a planted violation
# before trusting its zero-hits pass over the live tree.
# ---------------------------------------------------------------------------


def _find_writer_call_sites(source: str, filename: str = "<source>") -> list[str]:
    """Return a description string for every writer-INTENT call site.

    Flags a call to `get_lilli_raw_conn(` or `open_store_conn(` when either:
    - it carries no `read_only=True` keyword and no `allow_writer=` keyword
      (a bare/default call — defaults to the writer branch), or
    - it carries an `allow_writer=` keyword whose value is anything other than
      the literal `False` (`True`, a Name, or any expression is writer intent;
      only literal `allow_writer=False` passes).

    A file that will not parse is itself returned as an offender — an
    unparseable production file cannot be certified writer-free.

    Call-shaped only (an ast.Call node) — prose mentions (docstrings,
    `__all__` entries) never parse as a Call and are never flagged.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:
        # A production file that will not parse cannot be certified writer-free;
        # surface it as an offender rather than skipping it silently.
        return [f"{filename}: UNPARSEABLE ({exc})"]

    targets = {"get_lilli_raw_conn", "open_store_conn"}
    hits: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
            name = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in targets:
                has_read_only_true = False
                has_allow_writer = False
                writer_intent = False
                for kw in node.keywords:
                    if kw.arg == "read_only" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        has_read_only_true = True
                    if kw.arg == "allow_writer":
                        has_allow_writer = True
                        literal_false = isinstance(kw.value, ast.Constant) and kw.value.value is False
                        # Only a literal allow_writer=False is safe; True, a Name,
                        # or any other expression is writer intent (and if it
                        # resolves truthy at runtime it also escapes the fence raise).
                        if not literal_false:
                            writer_intent = True
                if writer_intent:
                    hits.append(f"{filename}:{node.lineno}: {name}(allow_writer=<not False>)")
                elif not has_read_only_true and not has_allow_writer:
                    hits.append(f"{filename}:{node.lineno}: bare/default {name}(...)")
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def test_writer_call_scanner_flags_planted_violations():
    """Positive control: the scanner must flag a planted bare writer call, a
    planted `allow_writer=True` call, AND a planted non-literal
    `allow_writer=<expr>` call (which would escape a literal-only rule), or the
    zero-hits pass below is vacuous."""
    source = (
        "open_store_conn(x, read_only=False)\n"
        "get_lilli_raw_conn(x, allow_writer=True)\n"
        "get_lilli_raw_conn(x, allow_writer=flag)\n"
    )
    hits = _find_writer_call_sites(source, filename="<synthetic>")
    assert len(hits) == 3, f"expected 3 planted violations flagged, got {hits}"


def test_writer_call_scanner_does_not_flag_read_only_or_literal_false():
    source = (
        "open_store_conn(x, read_only=True)\n"
        "get_lilli_raw_conn(x, read_only=True)\n"
        "get_lilli_raw_conn(x, allow_writer=False)\n"
    )
    hits = _find_writer_call_sites(source, filename="<synthetic>")
    assert hits == [], f"read_only=True / allow_writer=False calls must never be flagged, got {hits}"


# Line-keyed exemptions: the two function definitions and the single
# forwarder line inside open_store_conn. NEVER a file-level skip — a real
# future production writer would live in exactly these two files.
_EXEMPT = {
    ("src/iai_mcp/lillibrain/connection.py", "def get_lilli_raw_conn"),
    ("src/iai_mcp/hippo/_raw_open.py", "def open_store_conn"),
    ("src/iai_mcp/hippo/_raw_open.py", "get_lilli_raw_conn(bare, read_only=read_only, allow_writer=allow_writer)"),
}


def test_zero_writer_call_sites_under_src():
    """Runtime scan (line-keyed exemptions only) of the live src/iai_mcp tree:
    zero writer-INTENT call sites outside the two primitive definitions and
    the single forwarder line."""
    files = sorted(SRC.rglob("*.py"))
    # A guard that scans nothing is neutered: a wrong-layout or installed-package
    # run resolves SRC off-tree and passes green over zero files. Require the tree
    # to exist and a sane floor of files (well below the current count) to be walked.
    assert SRC.is_dir() and len(files) > 100, (
        f"scan scope empty or wrong: {SRC} ({len(files)} files)"
    )
    offenders: list[str] = []
    for f in files:
        text = f.read_text()
        rel = str(f.relative_to(SRC.parent.parent))
        for hit in _find_writer_call_sites(text, filename=rel):
            parts = hit.split(":")
            if len(parts) < 2 or not parts[1].strip().isdigit():
                # No line number (e.g. an UNPARSEABLE file): never exemptable.
                offenders.append(hit)
                continue
            lineno = int(parts[1])
            line_text = text.splitlines()[lineno - 1].strip()
            exempt = any(
                rel == ex_file and ex_marker in line_text
                for ex_file, ex_marker in _EXEMPT
            )
            if not exempt:
                offenders.append(hit)
    assert not offenders, (
        f"production writer-path call site(s) found under src/iai_mcp/: {offenders} "
        "— use the proxy writer (store.db._conn) or pass allow_writer=True with "
        "an explicit review, never a silent default"
    )

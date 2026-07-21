"""Guards for the sqlite3-free runtime contract.

The runtime import graph must not load stdlib sqlite3; only the lazy
legacy-driver module may reach it, and only when a legacy-format store is
actually opened. Statically, `import sqlite3` is banned from the runtime
tree — the legacy module goes through importlib so this stays greppable.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src" / "iai_mcp"

_IMPORT_RE = re.compile(r"^\s*(import sqlite3\b|from sqlite3\b)", re.MULTILINE)


def test_no_static_sqlite3_import_in_runtime_tree() -> None:
    offenders = []
    for py in _SRC_ROOT.rglob("*.py"):
        if py.name == "_sqlite_stdlib.py":
            continue
        if _IMPORT_RE.search(py.read_text(encoding="utf-8")):
            offenders.append(str(py.relative_to(_SRC_ROOT)))
    assert not offenders, f"static sqlite3 imports crept back in: {offenders}"


def test_runtime_import_graph_does_not_load_sqlite3() -> None:
    code = (
        "import sys\n"
        "import iai_mcp\n"
        "import iai_mcp.hippo\n"
        "import iai_mcp.store\n"
        "import iai_mcp.retrieve\n"
        "import iai_mcp.doctor\n"
        "import iai_mcp.lillibrain.connection\n"
        "import iai_mcp.hippo._ro_pool\n"
        "import iai_mcp.runtime_graph_cache_ro_export\n"
        "assert 'sqlite3' not in sys.modules, 'sqlite3 leaked at import time'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr


def test_legacy_module_translates_to_own_hierarchy() -> None:
    from iai_mcp import _sqlite_stdlib, errors

    conn = _sqlite_stdlib.connect(":memory:")
    try:
        with pytest.raises(errors.OperationalError):
            conn.execute("SELEKT broken")
        conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
        conn.execute("INSERT INTO t (id) VALUES ('a')")
        with pytest.raises(errors.IntegrityError):
            conn.execute("INSERT INTO t (id) VALUES ('a')")
        with pytest.raises(errors.ProgrammingError):
            conn.execute("SELECT id FROM t WHERE id = ?", ("a", "b"))
    finally:
        conn.close()


def test_corruption_not_swallowed_by_operational_handler() -> None:
    from iai_mcp import errors

    with pytest.raises(errors.DatabaseError):
        try:
            raise errors.DatabaseError("disk damage")
        except errors.OperationalError:  # pragma: no cover - must not match
            pytest.fail("OperationalError handler swallowed bare DatabaseError")


def test_native_engine_raises_own_hierarchy(tmp_path: Path) -> None:
    from iai_mcp import errors

    engine = pytest.importorskip("iai_mcp_native.engine")

    assert issubclass(engine.SnapshotFenceError, errors.OperationalError)

    conn = engine.Connection.open(str(tmp_path / "t.db"), 4)
    try:
        conn.execute("CREATE TABLE t (a INTEGER)")
        with pytest.raises(errors.ProgrammingError):
            conn.execute("INSERT INTO t (a) VALUES (?)", (1, 2))
        with pytest.raises(errors.OperationalError):
            conn.execute("SELEKT 1")
    finally:
        conn.close()

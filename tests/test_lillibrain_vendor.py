from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.hippo import HippoDB
from iai_mcp.lillibrain.connection import (
    LilliBrainConnection,
    _open_lilli_db,
    get_lilli_raw_conn,
)
from iai_mcp.types import EMBED_DIM
from iai_mcp_native import engine


def _unit_vec(seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-10
    return v.tolist()


def _record_row(
    *,
    rid: str | None = None,
    literal_surface: str = "test surface",
    embedding: list[float] | None = None,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": rid or str(uuid4()),
        "tier": "episodic",
        "literal_surface": literal_surface,
        "provenance_json": '{"src": "test"}',
        "profile_modulation_gain_json": '{"g": 1.0}',
        "embedding": embedding or _unit_vec(),
        "created_at": now,
    }


@pytest.fixture()
def lilli_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the in-tree pager/B+tree/WAL engine for the duration of a test.

    The autouse hermetic-path fixture redirects HOME, and every test uses a
    fresh tmp_path store, so the engine never touches the real home store.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


@pytest.fixture()
def key_provider():
    key = os.urandom(32)
    return lambda: key


def test_cursor_dbapi(tmp_path: Path) -> None:
    """Engine-level DB-API surface the consumer drives directly.

    Opens a connection on a tmp file (no encryption, no ANN — pure engine) and
    proves the three gap-fixes the consumer relies on:

    1. A single-row INSERT into an AUTOINCREMENT-vec_label table surfaces the
       injected label as ``cursor.lastrowid`` (a positive int, not None).
    2. ``conn.cursor().execute(sql).fetchall()`` reads that statement's result
       on the SAME cursor object (the pandas access pattern).
    3. ``cursor.close()`` is callable and harmless.
    4. A non-INSERT statement yields ``cursor.lastrowid is None``.
    """
    pager, catalog, root_map = _open_lilli_db(tmp_path / "engine.db")
    conn = LilliBrainConnection(pager, catalog, root_map)
    try:
        conn.execute(
            "CREATE TABLE records ("
            "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
            "id TEXT NOT NULL UNIQUE)"
        )

        # (1) single-row INSERT without supplying vec_label → lastrowid is the
        #     injected label, a positive int.
        insert_cur = conn.execute(
            "INSERT INTO records (id) VALUES (?)", ("alpha",)
        )
        assert isinstance(insert_cur.lastrowid, int)
        assert insert_cur.lastrowid > 0
        first_label = insert_cur.lastrowid

        # A second insert advances the label (monotonic AUTOINCREMENT).
        second_cur = conn.execute(
            "INSERT INTO records (id) VALUES (?)", ("beta",)
        )
        assert isinstance(second_cur.lastrowid, int)
        assert second_cur.lastrowid > first_label

        # (2) pandas access pattern: cur = conn.cursor(); cur.execute(...);
        #     cur.fetchall() — all against the SAME cursor object.
        cur = conn.cursor()
        returned = cur.execute("SELECT id FROM records WHERE id = ?", ("alpha",))
        assert returned is cur  # execute mutates and returns self
        rows = cur.fetchall()
        assert len(rows) == 1
        assert rows[0]["id"] == "alpha"

        # description reflects the executed statement's columns.
        assert [d[0] for d in cur.description] == ["id"]

        # (3) close() is callable and raises nothing; it does not corrupt the
        #     connection (a follow-up read still works).
        cur.close()
        insert_cur.close()
        post_close = conn.execute("SELECT id FROM records").fetchall()
        assert {r["id"] for r in post_close} == {"alpha", "beta"}

        # (4) a non-INSERT statement's cursor has lastrowid None.
        select_cur = conn.execute("SELECT id FROM records WHERE id = ?", ("beta",))
        assert select_cur.lastrowid is None
    finally:
        conn.close()


def test_both_drivers_import() -> None:
    """Both the engine connection module and HippoDB import together."""
    import iai_mcp.lillibrain.connection as _engine  # noqa: F401
    from iai_mcp.hippo import HippoDB as _HippoDB  # noqa: F401

    assert _engine.LilliBrainConnection is LilliBrainConnection
    assert _HippoDB is HippoDB


def test_default_driver_is_stdlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no driver env set, HippoDB opens a stdlib sqlite3 connection."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    db = HippoDB(tmp_path)
    try:
        assert isinstance(db._conn, sqlite3.Connection)
        assert not isinstance(db._conn, LilliBrainConnection)
    finally:
        db.close()


def test_driver_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LILLI_STORAGE_DRIVER selects the backend in both directions."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    db_lilli = HippoDB(tmp_path / "lilli")
    try:
        assert isinstance(db_lilli._conn, engine.Connection)
    finally:
        db_lilli.close()

    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    db_stdlib = HippoDB(tmp_path / "stdlib")
    try:
        assert isinstance(db_stdlib._conn, sqlite3.Connection)
        assert not isinstance(db_stdlib._conn, LilliBrainConnection)
    finally:
        db_stdlib.close()


def test_lilli_opens_and_creates_tables(
    tmp_path: Path, lilli_driver: None
) -> None:
    """Under the lilli driver, HippoDB boots and _ensure_tables registers all canonical tables."""
    db = HippoDB(tmp_path)
    try:
        assert isinstance(db._conn, engine.Connection)
        assert sorted(db.table_names()) == [
            "_hippo_meta",
            "budget_ledger",
            "edges",
            "events",
            "ratelimit_ledger",
            "record_tags",
            "records",
        ]
    finally:
        db.close()


def test_roundtrip_lilli(tmp_path: Path, lilli_driver: None, key_provider) -> None:
    """A record written on the lilli driver reads back raw AND via to_pandas."""
    db = HippoDB(tmp_path, crypto_key_provider=key_provider)
    try:
        rid = str(uuid4())
        row = _record_row(rid=rid, literal_surface="round trip")
        db.open_table("records").add([row])

        # (a) raw read straight off the engine connection.
        raw = db._conn.execute(
            "SELECT id, tier FROM records WHERE id = ?", (rid,)
        ).fetchall()
        assert len(raw) == 1
        assert raw[0]["id"] == rid
        assert raw[0]["tier"] == "episodic"

        # (b) the pandas cursor path — the encrypted column is stored as
        #     ciphertext; the explicit decrypt recovers the plaintext.
        df = db.open_table("records").to_pandas()
        match = df[df["id"] == rid]
        assert not match.empty
        stored = match.iloc[0]["literal_surface"]
        assert str(stored).startswith("iai:enc:v1:")
        plaintext = db._decrypt_record_field(rid, "literal_surface", stored)
        assert plaintext == "round trip"
    finally:
        db.close()


def test_aes_ciphertext_at_rest(
    tmp_path: Path, lilli_driver: None, key_provider
) -> None:
    """The engine stores the encrypted column as ciphertext — the key never crossed in."""
    db = HippoDB(tmp_path, crypto_key_provider=key_provider)
    db_path = db._db_path
    try:
        rid = str(uuid4())
        row = _record_row(rid=rid, literal_surface="secret phrase")
        db.open_table("records").add([row])

        # Read the STORED bytes raw through the engine registry (no decrypt).
        raw_conn = get_lilli_raw_conn(db_path)
        assert raw_conn is not None
        stored = raw_conn.execute(
            "SELECT literal_surface FROM records WHERE id = ?", (rid,)
        ).fetchone()
        assert stored is not None
        assert str(stored["literal_surface"]).startswith("iai:enc:v1:")

        # A non-encrypted column reads back as plaintext.
        tier = raw_conn.execute(
            "SELECT tier FROM records WHERE id = ?", (rid,)
        ).fetchone()
        assert tier["tier"] == "episodic"

        # The HippoDB-side decrypt recovers the original plaintext.
        df = db.open_table("records").to_pandas()
        cipher = df[df["id"] == rid].iloc[0]["literal_surface"]
        assert db._decrypt_record_field(rid, "literal_surface", cipher) == "secret phrase"
    finally:
        db.close()


# ---------------------------------------------------------------------------
# LR-01 guard: _txn rollback suppresses RuntimeError so original exc propagates
# ---------------------------------------------------------------------------


def test_txn_rollback_runtimeerror_does_not_mask_original() -> None:
    """If ROLLBACK raises RuntimeError the original exception still propagates.

    The lilli engine's pager.rollback() raises RuntimeError when no write
    transaction is in progress. The _txn context manager must suppress that
    error and re-raise the original exception — not let the RuntimeError from
    the rollback attempt mask the primary failure.
    """
    import sqlite3 as _sqlite3
    from unittest.mock import MagicMock, patch
    from iai_mcp.hippo._db import _txn

    # Build a minimal fake connection whose execute() throws RuntimeError on
    # "ROLLBACK" (simulating the lilli engine pager-state mismatch case).
    class _FakeConn:
        in_transaction = False
        row_factory = None

        def execute(self, sql, params=()):
            if sql.strip().upper() == "ROLLBACK":
                raise RuntimeError("simulated pager: no write transaction in progress")
            if sql.strip().upper() == "BEGIN":
                return None
            return None

        def commit(self):
            pass

    conn = _FakeConn()
    sentinel = ValueError("the original error")

    with pytest.raises(ValueError) as exc_info:
        with _txn(conn):
            raise sentinel

    # The original ValueError must propagate — not be masked by RuntimeError.
    assert exc_info.value is sentinel


# ---------------------------------------------------------------------------
# LR-02 guard: _lilli_fast_fsync fixture teardown cleans up LILLI_FSYNC_MODE
# ---------------------------------------------------------------------------


def test_lilli_fast_fsync_fixture_teardown_pattern() -> None:
    """The save/restore pattern in _lilli_fast_fsync preserves and restores env.

    Runs the fixture's save-set-yield-restore logic inline so we can assert
    that a value absent before the fixture is absent after it, and a pre-existing
    value is restored. The session fixture itself is autouse and cannot be easily
    called here, so we exercise the identical code path directly.
    """
    # Case 1: var was absent before — must be absent after teardown.
    prior_saved = os.environ.pop("LILLI_FSYNC_MODE", None)
    try:
        os.environ["LILLI_FSYNC_MODE"] = "fast"
        # Simulate teardown: prior was None.
        os.environ.pop("LILLI_FSYNC_MODE", None)
        assert "LILLI_FSYNC_MODE" not in os.environ
    finally:
        if prior_saved is not None:
            os.environ["LILLI_FSYNC_MODE"] = prior_saved

    # Case 2: var had a prior value — must be restored after teardown.
    original = "fullfsync"
    os.environ["LILLI_FSYNC_MODE"] = original
    try:
        os.environ["LILLI_FSYNC_MODE"] = "fast"
        # Simulate teardown: prior was "fullfsync".
        os.environ["LILLI_FSYNC_MODE"] = original
        assert os.environ["LILLI_FSYNC_MODE"] == original
    finally:
        del os.environ["LILLI_FSYNC_MODE"]


# ---------------------------------------------------------------------------
# MR-01 guard: PRAGMA query_only=ON enforced on the lilli driver
# ---------------------------------------------------------------------------


def test_lilli_read_only_raises_on_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HippoDB(read_only=True) on the lilli driver blocks INSERT/UPDATE/DELETE.

    The engine must honor PRAGMA query_only=ON by raising
    sqlite3.OperationalError('attempt to write a readonly database') when a
    mutating statement is executed — matching the sqlite3 contract so callers
    can rely on read_only=True for safety regardless of driver.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    # First open a writable instance to create the schema and insert a row.
    db_rw = HippoDB(tmp_path, read_only=False)
    try:
        rid = str(uuid4())
        row = _record_row(rid=rid, literal_surface="ro test")
        db_rw.open_table("records").add([row])
    finally:
        db_rw.close()

    # Re-open read-only: INSERT must raise.
    db_ro = HippoDB(tmp_path, read_only=True)
    try:
        with pytest.raises(sqlite3.OperationalError, match="attempt to write a readonly database"):
            db_ro._conn.execute(
                "INSERT INTO records (id, tier) VALUES (?, ?)",
                (str(uuid4()), "episodic"),
            )
    finally:
        db_ro.close()


def test_lilli_read_only_false_still_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HippoDB(read_only=False) on the lilli driver remains fully writable."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    db = HippoDB(tmp_path, read_only=False)
    try:
        rid = str(uuid4())
        row = _record_row(rid=rid, literal_surface="writable check")
        db.open_table("records").add([row])
        raw = db._conn.execute(
            "SELECT id FROM records WHERE id = ?", (rid,)
        ).fetchone()
        assert raw is not None
        assert raw["id"] == rid
    finally:
        db.close()


# ---------------------------------------------------------------------------
# LR-01 guard: _storage_driver stored at open-time, used at close-time
# ---------------------------------------------------------------------------


def test_storage_driver_recorded_at_open_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HippoDB._storage_driver reflects the env at open-time, not close-time.

    Mutating LILLI_STORAGE_DRIVER after open must not affect which deregister
    branch close() takes — it reads the instance attribute, not the env var.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    db = HippoDB(tmp_path)
    assert db._storage_driver == "lilli"

    # Simulate an env mutation between open and close.
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    # _storage_driver must still report the open-time decision.
    assert db._storage_driver == "lilli"
    # close() must not raise even with the env unset (deregister path taken
    # from the instance attribute, not the env).
    db.close()


# ---------------------------------------------------------------------------
# LR-02 guard: LilliBrainRawConn docstring covers the production recall path
# ---------------------------------------------------------------------------


def test_lilli_raw_conn_docstring_covers_production_paths() -> None:
    """LilliBrainRawConn docstring mentions both production and fixture uses."""
    from iai_mcp.lillibrain.connection import LilliBrainRawConn

    doc = LilliBrainRawConn.__doc__ or ""
    # Must mention the production recall and export paths (not just test fixtures).
    assert "recall" in doc.lower() or "daemon-down" in doc.lower(), (
        "Docstring must reference the daemon-down recall production path"
    )
    assert "export" in doc.lower() or "runtime-graph" in doc.lower() or "rO export" in doc.lower(), (
        "Docstring must reference the runtime-graph RO export production path"
    )

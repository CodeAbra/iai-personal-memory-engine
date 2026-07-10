"""Unit tests for tests/_store_raw.py::open_store_raw.

Covers three branches:
- Default (stdlib) driver: returns a stdlib sqlite3.Connection.
- Lilli driver: returns the engine raw DB-API connection when HippoDB is open.
- Fail-loud: lilli driver + absent path raises RuntimeError.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests._store_raw import open_store_raw


# ---------------------------------------------------------------------------
# Branch 1: default (stdlib) driver
# ---------------------------------------------------------------------------


def test_open_store_raw_default_returns_sqlite3(tmp_path: Path, monkeypatch) -> None:
    """With LILLI_STORAGE_DRIVER unset, open_store_raw returns a stdlib sqlite3.Connection."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    db_file = tmp_path / "test.sqlite3"
    # Create a real sqlite file so the connection is meaningful.
    seed = sqlite3.connect(str(db_file))
    seed.execute("CREATE TABLE t (x TEXT)")
    seed.execute("INSERT INTO t VALUES ('hello')")
    seed.commit()
    seed.close()

    conn = open_store_raw(db_file)
    try:
        assert isinstance(conn, sqlite3.Connection), (
            f"Expected sqlite3.Connection, got {type(conn)}"
        )
        row = conn.execute("SELECT x FROM t").fetchone()
        assert row is not None
        assert row[0] == "hello"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Branch 2: lilli driver — returns engine raw conn reading ciphertext
# ---------------------------------------------------------------------------


def test_open_store_raw_lilli_returns_engine_conn(tmp_path: Path, monkeypatch) -> None:
    """With LILLI_STORAGE_DRIVER=lilli and a HippoDB open, returns an engine raw conn."""
    import os

    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    # Set in the process env too so HippoDB picks it up during construction.
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    # Import HippoDB only here — keep the stdlib test free of engine imports.
    from iai_mcp.hippo import HippoDB
    from iai_mcp.types import EMBED_DIM
    import numpy as np
    from datetime import datetime, timezone
    from uuid import uuid4

    key = os.urandom(32)
    db = HippoDB(tmp_path, crypto_key_provider=lambda: key)
    try:
        # Seed one record so there is a row to SELECT.
        rng = np.random.default_rng(42)
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-10
        row = {
            "id": str(uuid4()),
            "tier": "episodic",
            "literal_surface": "engine raw conn test",
            "provenance_json": '{"src": "test"}',
            "profile_modulation_gain_json": '{"g": 1.0}',
            "embedding": vec.tolist(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        db.open_table("records").add([row])

        brain_path = tmp_path / "hippo" / "brain.sqlite3"
        raw_conn = open_store_raw(brain_path)
        try:
            # Must NOT be a stdlib sqlite3.Connection.
            assert not isinstance(raw_conn, sqlite3.Connection), (
                "Expected engine raw conn, got sqlite3.Connection"
            )
            # SELECT must work and return the ciphertext wire prefix on the
            # encrypted column (literal_surface is always encrypted by HippoDB).
            result = raw_conn.execute(
                "SELECT literal_surface FROM records WHERE id = ?", (row["id"],)
            ).fetchone()
            assert result is not None, "SELECT returned no row"
            val = result[0]
            assert isinstance(val, (str, bytes)), f"Unexpected type: {type(val)}"
            val_str = val.decode() if isinstance(val, bytes) else val
            assert val_str.startswith("iai:enc:v1:"), (
                f"Expected ciphertext prefix, got: {val_str!r}"
            )
        finally:
            raw_conn.close()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Branch 3: lilli driver + absent path → RuntimeError
# ---------------------------------------------------------------------------


def test_open_store_raw_lilli_missing_path_raises(tmp_path: Path, monkeypatch) -> None:
    """With LILLI_STORAGE_DRIVER=lilli and an absent path, open_store_raw raises RuntimeError."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    absent = tmp_path / "does_not_exist" / "brain.sqlite3"
    with pytest.raises(RuntimeError, match="lilli driver active but no engine connection"):
        open_store_raw(absent)

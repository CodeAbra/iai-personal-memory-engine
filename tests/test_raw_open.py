"""Unit tests for src/iai_mcp/hippo/_raw_open.py::open_store_conn.

Covers four branches:
- Default driver (env unset): returns None (the sentinel — caller keeps its
  existing sqlite3.connect verbatim).
- Lilli driver + encrypted HippoDB open: returns engine raw conn; SELECT on
  an encrypted column returns the ciphertext wire prefix (AES-at-rest proof).
- Lilli driver + file URI form: strips file:...?mode=ro to bare path before
  reaching get_lilli_raw_conn, returns the engine conn.
- Lilli driver + absent path: raises RuntimeError (fail loud, never
  silently mis-route to a broken connection).
"""
from __future__ import annotations

import os
import sqlite3

from iai_mcp import errors
from pathlib import Path

import pytest

from iai_mcp.hippo._raw_open import _bare_path, open_store_conn


# ---------------------------------------------------------------------------
# Sanity: _bare_path URI strip
# ---------------------------------------------------------------------------


def test_bare_path_strips_file_uri_with_mode(tmp_path: Path) -> None:
    """_bare_path removes the file: scheme and ?mode=ro query string."""
    result = _bare_path("file:/a/b.sqlite3?mode=ro")
    assert result == "/a/b.sqlite3"


def test_bare_path_passthrough_on_bare_path() -> None:
    """_bare_path returns the path unchanged when no file: prefix is present."""
    assert _bare_path("/a/b.sqlite3") == "/a/b.sqlite3"


def test_bare_path_strips_file_uri_no_query() -> None:
    """_bare_path handles file:/ URIs with no query string."""
    assert _bare_path("file:/a/b.sqlite3") == "/a/b.sqlite3"


# ---------------------------------------------------------------------------
# Branch 1: default driver — None sentinel
# ---------------------------------------------------------------------------


def test_default_returns_none_sentinel(tmp_path: Path, monkeypatch) -> None:
    """On an existing legacy sqlite file, open_store_conn returns None regardless
    of the driver env — on-disk format detection wins over env intent, which
    only governs a fresh/absent path (see _resolve_effective_driver)."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    path = tmp_path / "legacy.sqlite3"
    # A bare connect+close leaves a zero-byte file (sqlite3 lazily writes the
    # header); a CREATE TABLE forces the header page to land so the on-disk
    # sniff in _resolve_effective_driver has bytes to detect.
    legacy_conn = sqlite3.connect(str(path))
    legacy_conn.execute("CREATE TABLE t (id INTEGER)")
    legacy_conn.commit()
    legacy_conn.close()

    result = open_store_conn(path)
    assert result is None, f"Expected None sentinel, got {type(result)}"


def test_stdlib_explicit_returns_none_sentinel(tmp_path: Path, monkeypatch) -> None:
    """With LILLI_STORAGE_DRIVER=stdlib, open_store_conn returns None."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")

    result = open_store_conn(tmp_path / "any.sqlite3")
    assert result is None, f"Expected None sentinel on stdlib driver, got {type(result)}"


def test_unknown_driver_returns_none_sentinel(tmp_path: Path, monkeypatch) -> None:
    """Any driver value that is not 'lilli' returns the None sentinel."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "duckdb")

    result = open_store_conn(tmp_path / "any.sqlite3")
    assert result is None


# ---------------------------------------------------------------------------
# Helpers shared by the lilli branches
# ---------------------------------------------------------------------------


def _seed_encrypted_hippodb(tmp_path: Path, key: bytes):
    """Build a minimal HippoDB with one encrypted record; return the db-file path."""
    import numpy as np
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.hippo import HippoDB
    from iai_mcp.types import EMBED_DIM

    db = HippoDB(tmp_path, crypto_key_provider=lambda: key)
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-10
    row = {
        "id": str(uuid4()),
        "tier": "episodic",
        "literal_surface": "open_store_conn factory test record",
        "provenance_json": '{"src": "test"}',
        "profile_modulation_gain_json": '{"g": 1.0}',
        "embedding": vec.tolist(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    db.open_table("records").add([row])
    db_file = tmp_path / "hippo" / "brain.sqlite3"
    return db, db_file, row["id"]


# ---------------------------------------------------------------------------
# Branch 2: lilli driver — engine conn + AES ciphertext-at-rest proof
# ---------------------------------------------------------------------------


def test_lilli_returns_engine_conn_reading_ciphertext(
    tmp_path: Path, monkeypatch
) -> None:
    """With LILLI_STORAGE_DRIVER=lilli and an open HippoDB, returns an engine
    raw conn. SELECT literal_surface returns iai:enc:v1: ciphertext — proves
    the engine is crypto-blind and the key never crossed into the factory."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    _prior = os.environ.get("LILLI_STORAGE_DRIVER")
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    key = os.urandom(32)
    db, db_file, record_id = _seed_encrypted_hippodb(tmp_path, key)
    try:
        conn = open_store_conn(db_file, read_only=True)
        assert conn is not None, "Expected engine conn, got None"
        assert not isinstance(conn, sqlite3.Connection), (
            "Expected engine raw conn, got stdlib sqlite3.Connection"
        )
        try:
            result = conn.execute(
                "SELECT literal_surface FROM records WHERE id = ?",
                (record_id,),
            ).fetchone()
            assert result is not None, "SELECT returned no row"
            val = result[0]
            val_str = val.decode() if isinstance(val, bytes) else val
            # AES-at-rest proof: the stored bytes carry the ciphertext wire prefix.
            assert val_str.startswith("iai:enc:v1:"), (
                f"Expected ciphertext (iai:enc:v1:...), got: {val_str!r} — "
                "key may have leaked into the engine"
            )
            # Confirm the plaintext fragment is NOT visible (belt-and-suspenders).
            assert "open_store_conn factory test record" not in val_str, (
                "Plaintext visible through engine — AES boundary violated"
            )
        finally:
            conn.close()
    finally:
        db.close()
        if _prior is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = _prior


# ---------------------------------------------------------------------------
# Branch 3: lilli driver + file URI form — URI strip works
# ---------------------------------------------------------------------------


def test_uri_form_is_stripped_on_lilli(tmp_path: Path, monkeypatch) -> None:
    """Passing a file:...?mode=ro URI still resolves to the engine conn after
    _bare_path strips it — the URI form must NOT reach get_lilli_raw_conn."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    _prior = os.environ.get("LILLI_STORAGE_DRIVER")
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    key = os.urandom(32)
    db, db_file, record_id = _seed_encrypted_hippodb(tmp_path, key)
    try:
        uri_path = f"file:{db_file}?mode=ro"
        conn = open_store_conn(uri_path, read_only=True)
        assert conn is not None, (
            f"URI form was not stripped correctly — got None from open_store_conn({uri_path!r})"
        )
        assert not isinstance(conn, sqlite3.Connection)
        try:
            result = conn.execute(
                "SELECT literal_surface FROM records WHERE id = ?",
                (record_id,),
            ).fetchone()
            assert result is not None, "SELECT returned no row after URI strip"
            val = result[0]
            val_str = val.decode() if isinstance(val, bytes) else val
            assert val_str.startswith("iai:enc:v1:"), (
                f"Expected ciphertext after URI strip, got: {val_str!r}"
            )
        finally:
            conn.close()
    finally:
        db.close()
        if _prior is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = _prior


# ---------------------------------------------------------------------------
# Branch 4: lilli driver + absent path — RuntimeError (fail loud)
# ---------------------------------------------------------------------------


def test_lilli_absent_path_raises_runtimeerror(tmp_path: Path, monkeypatch) -> None:
    """With LILLI_STORAGE_DRIVER=lilli and a non-existent path, raises RuntimeError."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    _prior = os.environ.get("LILLI_STORAGE_DRIVER")
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    absent = tmp_path / "does_not_exist" / "brain.sqlite3"
    try:
        with pytest.raises(RuntimeError, match="lilli driver active but no engine connection"):
            open_store_conn(absent)
    finally:
        if _prior is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = _prior


# ---------------------------------------------------------------------------
# CR-01 guards: RO adapter isolation — writer not bricked by RO open
# ---------------------------------------------------------------------------


def test_ro_open_store_conn_does_not_brick_shared_writer(
    tmp_path: Path, monkeypatch
) -> None:
    """A RO open_store_conn on a live HippoDB must not disable the writer.

    This is the exact CR-01 repro: open_store_conn(read_only=True) returns an
    adapter backed by the shared writer connection. Executing PRAGMA query_only=ON
    on that adapter must NOT set _query_only on the shared LilliBrainConnection.
    A subsequent write through the shared connection must succeed.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    key = os.urandom(32)
    db, db_file, _record_id = _seed_encrypted_hippodb(tmp_path, key)
    try:
        # Open a RO adapter — this must not poison the shared writer conn.
        ro_conn = open_store_conn(db_file, read_only=True)
        assert ro_conn is not None
        try:
            ro_conn.execute("PRAGMA query_only=ON")
            _ = ro_conn.execute("SELECT id FROM records").fetchall()
        finally:
            ro_conn.close()

        # The writer's read-only state must remain off after the RO adapter
        # closes. _query_only is a white-box flag that exists only on the
        # reference Python connection; the active connection may be the native
        # engine (no such attribute) or the stdlib sqlite3 connection (no such
        # attribute). Assert it only when present — the authoritative,
        # driver-independent CR-01 guard is the write-succeeds check below.
        writer_query_only = getattr(db._conn, "_query_only", False)
        assert not writer_query_only, (
            "RO adapter set _query_only=True on the shared writer connection — "
            "the CR-01 bug is still present"
        )

        # A write through the shared connection must succeed (not raise).
        from uuid import uuid4
        import numpy as np
        from datetime import datetime, timezone
        from iai_mcp.types import EMBED_DIM

        rng = np.random.default_rng(7)
        vec = rng.standard_normal(EMBED_DIM).astype(np.float32)
        vec /= np.linalg.norm(vec) + 1e-10
        row = {
            "id": str(uuid4()),
            "tier": "episodic",
            "literal_surface": "post-ro-open write check",
            "provenance_json": '{"src": "test"}',
            "profile_modulation_gain_json": '{"g": 1.0}',
            "embedding": vec.tolist(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        # This must not raise OperationalError("attempt to write a readonly database").
        db.open_table("records").add([row])
    finally:
        db.close()


def test_ro_adapter_rejects_writes_through_itself(
    tmp_path: Path, monkeypatch
) -> None:
    """A RO adapter returned by open_store_conn(read_only=True) must reject writes.

    The adapter enforces the RO constraint at its own layer via _ro flag,
    so callers cannot accidentally mutate state through a read-only handle.
    """
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    key = os.urandom(32)
    db, db_file, _record_id = _seed_encrypted_hippodb(tmp_path, key)
    try:
        ro_conn = open_store_conn(db_file, read_only=True)
        assert ro_conn is not None
        try:
            from uuid import uuid4
            with pytest.raises(errors.OperationalError, match="attempt to write a readonly database"):
                ro_conn.execute(
                    "INSERT INTO records (id, tier) VALUES (?, ?)",
                    (str(uuid4()), "episodic"),
                )
        finally:
            ro_conn.close()
    finally:
        db.close()

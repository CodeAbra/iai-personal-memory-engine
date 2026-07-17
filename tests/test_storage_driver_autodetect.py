"""On-disk storage format governs driver selection, not the ambient env var.

Regression guard for the failure where a client process (e.g. the user-shell
``iai recall`` direct-store fallback) opens a native-engine store while
``LILLI_STORAGE_DRIVER`` is unset in its environment, defaulting to the stdlib
``sqlite3`` driver, which raises ``file is not a database`` and silently returns
zero results. The on-disk file format is authoritative: a store written by the
engine must reopen through the engine regardless of the ambient variable, and a
genuine stdlib store must keep opening stdlib even when the env asks for the
engine.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iai_mcp.hippo import _db
from iai_mcp.hippo._db import _resolve_effective_driver
from iai_mcp.lillibrain.constants import DB_MAGIC


def _write_magic(path: Path, magic: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(magic + b"\x00" * 64)


@pytest.fixture(autouse=True)
def _clear_driver_detect_cache() -> None:
    """Keep the module-level detection cache from leaking across tests."""
    _db._driver_detect_cache_clear()
    yield
    _db._driver_detect_cache_clear()


def test_resolve_driver_prefers_ondisk_lilli_over_unset_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "hippo" / "brain.sqlite3"
    _write_magic(db, DB_MAGIC)
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    assert _resolve_effective_driver(str(db)) == "lilli"


def test_resolve_driver_prefers_ondisk_stdlib_over_lilli_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "hippo" / "brain.sqlite3"
    _write_magic(db, b"SQLite format 3\x00")
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    assert _resolve_effective_driver(str(db)) == "stdlib"


def test_resolve_driver_honours_env_for_absent_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "hippo" / "brain.sqlite3"  # not created
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    assert _resolve_effective_driver(str(db)) == "lilli"
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    assert _resolve_effective_driver(str(db)) == "stdlib"


def test_resolve_driver_defers_to_env_on_unknown_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db = tmp_path / "hippo" / "brain.sqlite3"
    _write_magic(db, b"NOTADBFILE\x00")
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    assert _resolve_effective_driver(str(db)) == "lilli"


def test_resolve_driver_rejects_six_byte_sqlite_prefix_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A header matching only the first 6 bytes of "SQLite" (not the full
    16-byte magic) must NOT be detected as stdlib -- the comparison must be
    against the complete SQLite header magic, not a 6-byte prefix.
    """
    db = tmp_path / "hippo" / "brain.sqlite3"
    # First 6 bytes match b"SQLite", but bytes 6-15 diverge from the real
    # "SQLite format 3\x00" magic -- e.g. a corrupt or foreign file that
    # happens to start with the word "SQLite".
    _write_magic(db, b"SQLite\x00CORRUPTxx")
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    # Full-magic comparison must treat this as an unknown header and defer
    # to the env, not misdetect it as a genuine stdlib store.
    assert _resolve_effective_driver(str(db)) == "lilli"


@pytest.mark.skipif(
    os.environ.get("LILLI_STORAGE_DRIVER", "").lower() != "lilli",
    reason="engine-written store requires the lilli driver at write time",
)
def test_lilli_store_reopens_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lilli-written store reopens and reads back with the env unset.

    Mirrors the client-shell condition: the store was created by the daemon
    (engine env set), then a client process with no ``LILLI_STORAGE_DRIVER``
    opens it. Auto-detection must select the engine so the read succeeds rather
    than raising ``file is not a database``.
    """
    import numpy as np

    from iai_mcp.store import MemoryStore, flush_record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store_root = tmp_path / ".iai-mcp"
    store_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    surface = "the migrated store must reopen when the client shell has no driver env"
    rid = uuid.uuid4()

    # Write with the engine (LILLI_STORAGE_DRIVER=lilli is set in this session).
    writer = MemoryStore(store_root)
    try:
        vec = np.random.RandomState(seed=13).randn(EMBED_DIM).tolist()
        writer.insert(
            MemoryRecord(
                id=rid,
                tier="episodic",
                literal_surface=surface,
                aaak_index="",
                embedding=vec,
                community_id=None,
                centrality=0.0,
                detail_level=1,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{"session_id": "s", "role": "user"}],
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                tags=["role:user"],
                language="en",
            )
        )
        flush_record_buffer(writer)
    finally:
        writer.close()

    # The on-disk file is engine format.
    db_path = store_root / "hippo" / "brain.sqlite3"
    assert db_path.read_bytes()[: len(DB_MAGIC)] == DB_MAGIC

    # Now simulate the client shell: NO driver env. Detection must pick lilli.
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    assert _resolve_effective_driver(str(db_path)) == "lilli"

    reader = MemoryStore(store_root)
    try:
        assert reader.db._storage_driver == "lilli"
        got = reader.get(rid)
        assert got is not None
        assert got.literal_surface == surface
    finally:
        reader.close()


def test_resolve_driver_does_not_cache_absent_file_then_detects_from_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "absent file" branch must not be cached.

    First call honours the env because the file doesn't exist yet. Once the
    file is created with a real header, the next call must detect from the
    header instead of returning the stale env-honouring result — proving the
    absent-file path was never written to the cache.
    """
    db = tmp_path / "hippo" / "brain.sqlite3"  # not created yet
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    assert _resolve_effective_driver(str(db)) == "stdlib"

    _write_magic(db, DB_MAGIC)
    assert _resolve_effective_driver(str(db)) == "lilli"


def test_resolve_driver_caches_after_header_detection_and_skips_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once detected from a real header, subsequent calls must not reopen the
    file at all — that raw open()/close() cycle is what silently drops the
    live sqlite connection's POSIX locks (the WAL-deletion footgun)."""
    db = tmp_path / "hippo" / "brain.sqlite3"
    _write_magic(db, DB_MAGIC)
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    # First call reads the header and populates the cache.
    assert _resolve_effective_driver(str(db)) == "lilli"

    open_calls: list[str] = []
    real_open = open

    def _counting_open(file, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        if str(file) == str(db):
            open_calls.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _counting_open)

    for _ in range(3):
        assert _resolve_effective_driver(str(db)) == "lilli"

    assert open_calls == []

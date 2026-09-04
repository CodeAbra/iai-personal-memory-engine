"""Focused driver test: the HNSW-rebuild spawn child reads the store via
inherited LILLI_STORAGE_DRIVER.  Verifies that a spawn child launched from
this process inherits the env var and branches correctly inside _worker_entry,
producing a ("done", {"rebuilt_count": N}) result on both the lilli and the
stdlib driver.
"""
from __future__ import annotations

import multiprocessing
import os
import struct
import tempfile
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.hippo import HNSW_EF, HNSW_EF_CONSTRUCTION, HNSW_M, HippoDB
from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.hnsw_rebuild_worker import _worker_entry
from iai_mcp.types import EMBED_DIM


def _rand_embedding() -> bytes:
    vec = np.random.rand(EMBED_DIM).astype(np.float32)
    return struct.pack(f"<{EMBED_DIM}f", *vec.tolist())


def _seed_records(db: HippoDB, n: int) -> int:
    """Insert n active records with real embeddings; return active count."""
    rows = []
    for _ in range(n):
        rows.append({
            "id": str(uuid4()),
            "tier": "episodic",
            "literal_surface": "test record",
            "embedding": np.random.rand(EMBED_DIM).astype(np.float32).tolist(),
            "created_at": "2026-01-01T00:00:00+00:00",
        })
    tbl = db.open_table("records")
    tbl.add(rows)
    return n


def _run_worker(db_path: str, tmp_dir: str) -> tuple[str, object]:
    """Spawn the _worker_entry child, send params, wait for result."""
    ctx = multiprocessing.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=True)
    tmp_path = str(Path(tmp_dir) / "test_rebuild.hnsw.tmp")
    process = ctx.Process(target=_worker_entry, args=(child_conn,), daemon=True)
    process.start()
    child_conn.close()

    parent_conn.send(("params", {
        "db_path": db_path,
        "tmp_path": tmp_path,
        "embed_dim": EMBED_DIM,
        "hnsw_m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "ef": HNSW_EF,
        "capacity": 200,
    }))

    deadline = time.monotonic() + 60.0
    kind = payload = None
    try:
        while time.monotonic() < deadline:
            if parent_conn.poll(timeout=1.0):
                kind, payload = parent_conn.recv()
                break
    finally:
        parent_conn.close()
        process.join(timeout=5.0)
        if process.is_alive():
            process.terminate()

    return kind, payload


@pytest.mark.skipif(
    os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() != "lilli",
    reason="lilli-driver spawn child test — requires LILLI_STORAGE_DRIVER=lilli",
)
def test_hnsw_rebuild_worker_lilli_driver(tmp_path: pytest.TempPathFactory, monkeypatch):
    """The spawn child rebuilds the HNSW index from a lilli store via inherited env."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    db = HippoDB(tmp_path)
    n_seeded = _seed_records(db, 5)
    db_path = str(tmp_path / "hippo" / "brain.sqlite3")
    db.close()

    with tempfile.TemporaryDirectory() as tmp_dir:
        kind, payload = _run_worker(db_path, tmp_dir)

    assert kind == "done", f"worker returned error: {payload!r}"
    assert isinstance(payload, dict)
    assert "rebuilt_count" in payload
    assert payload["rebuilt_count"] == n_seeded, (
        f"expected {n_seeded} rebuilt records, got {payload['rebuilt_count']}"
    )


@pytest.mark.skipif(
    os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() != "lilli",
    reason="proves the worker detects a lilli-format store from on-disk magic "
    "even when LILLI_STORAGE_DRIVER is unset in the worker's own env — the "
    "split-brain regression: a nightly rebuild spawned without the env "
    "inherited must not pick stdlib and fail to open the lilli-format file",
)
def test_hnsw_rebuild_worker_lilli_store_with_env_unset(
    tmp_path: pytest.TempPathFactory, monkeypatch
):
    """Store written by the lilli engine; worker's LILLI_STORAGE_DRIVER unset.

    Regression for the split-brain where hnsw_rebuild_worker._worker_entry
    selected its driver from os.environ["LILLI_STORAGE_DRIVER"] instead of
    the on-disk file magic. A spawn child that does not inherit the env
    (e.g. launched from a context where it was never set) would default to
    "stdlib" and call sqlite3.connect(...ro) on the lilli-format file,
    raising "file is not a database" and failing the nightly rebuild.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    db = HippoDB(tmp_path)
    n_seeded = _seed_records(db, 5)
    db_path = str(tmp_path / "hippo" / "brain.sqlite3")
    db.close()

    # The regression condition: this process (which spawns the worker) has
    # no LILLI_STORAGE_DRIVER — the worker must still detect "lilli" from
    # the on-disk magic bytes of db_path, not default to stdlib.
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    assert "LILLI_STORAGE_DRIVER" not in os.environ

    with tempfile.TemporaryDirectory() as tmp_dir:
        kind, payload = _run_worker(db_path, tmp_dir)

    assert kind == "done", f"worker returned error: {payload!r}"
    assert isinstance(payload, dict)
    assert payload["rebuilt_count"] == n_seeded, (
        f"expected {n_seeded} rebuilt records, got {payload['rebuilt_count']}"
    )


def test_hnsw_rebuild_worker_stdlib_driver(tmp_path: pytest.TempPathFactory, monkeypatch):
    """The spawn child rebuilds the HNSW index from a stdlib store (regression guard)."""
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")

    db = HippoDB(tmp_path)
    n_seeded = _seed_records(db, 4)
    db_path = str(tmp_path / "hippo" / "brain.sqlite3")
    db.close()

    with tempfile.TemporaryDirectory() as tmp_dir:
        kind, payload = _run_worker(db_path, tmp_dir)

    assert kind == "done", f"worker returned error: {payload!r}"
    assert isinstance(payload, dict)
    assert payload["rebuilt_count"] == n_seeded, (
        f"expected {n_seeded} rebuilt records, got {payload['rebuilt_count']}"
    )


def test_hnsw_rebuild_worker_stdlib_store_with_env_lilli(
    tmp_path: pytest.TempPathFactory, monkeypatch
):
    """Reverse mismatch: store written by stdlib sqlite3; worker's env says lilli.

    The worker must detect "stdlib" from the on-disk SQLite magic bytes and
    open the plain `file:...?mode=ro` connection, not attempt to reopen the
    genuine sqlite3 file through the lilli engine because the env says so.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")

    db = HippoDB(tmp_path)
    n_seeded = _seed_records(db, 4)
    db_path = str(tmp_path / "hippo" / "brain.sqlite3")
    db.close()

    # The regression condition: worker's env says "lilli" but the file is a
    # genuine stdlib sqlite3 store — on-disk magic must win.
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    with tempfile.TemporaryDirectory() as tmp_dir:
        kind, payload = _run_worker(db_path, tmp_dir)

    assert kind == "done", f"worker returned error: {payload!r}"
    assert isinstance(payload, dict)
    assert payload["rebuilt_count"] == n_seeded, (
        f"expected {n_seeded} rebuilt records, got {payload['rebuilt_count']}"
    )

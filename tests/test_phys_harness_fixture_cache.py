"""Mechanics for the phys_footprint harness's fixture cache (two-process
build/measure split): cache-key stability, copytree isolation, and the
WAL/SHM-sidecar-absence build postcondition. Uses a tiny (2-5 record) real
store, never a full N-scale corpus build.

NOT the measurement-equivalence check (a cached+copied store must MEASURE the
same as a freshly-built one within tolerance) -- that needs two real N-scale
builds and exceeds the 300s per-test budget. Run once, out of band, before
the sweep is trusted.
"""
from __future__ import annotations

import shutil

import pytest

from tests.rss_rca.phys_footprint_harness import (
    _assert_no_wal_shm_sidecar,
    _engine_generation,
    _fixture_cache_dir,
)

_ENGINE_AVAILABLE = _engine_generation() != "unknown"


def _make_minimal_store(store_path, monkeypatch, home_dir) -> None:
    """Build a tiny (5-record) real store at ``store_path`` via MemoryStore."""
    monkeypatch.setenv("HOME", str(home_dir))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))
    monkeypatch.setenv("IAI_MCP_USER_MODEL_PATH", str(home_dir / "user_model.json"))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-fixture-cache-mechanics")
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")

    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import flush_record_buffer
    from tests.rss_rca.corpus import build_corpus

    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_path)
    try:
        build_corpus(
            store, n=5, seed=1, dim=int(store.embed_dim),
            varied_tags=True, with_hv_payload=True,
        )
        flush_record_buffer(store)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# (a) cache-key derivation
# ---------------------------------------------------------------------------


def test_cache_key_is_stable_across_calls(tmp_path) -> None:
    a = _fixture_cache_dir(tmp_path, n=3000, seed=42, edge_factor=1.3)
    b = _fixture_cache_dir(tmp_path, n=3000, seed=42, edge_factor=1.3)
    assert a == b


def test_cache_key_includes_engine_generation(tmp_path) -> None:
    cache_dir = _fixture_cache_dir(tmp_path, n=3000, seed=42, edge_factor=1.3)
    assert _engine_generation() in cache_dir.parts


def test_cache_key_differs_by_scale(tmp_path) -> None:
    a = _fixture_cache_dir(tmp_path, n=3000, seed=42, edge_factor=1.3)
    b = _fixture_cache_dir(tmp_path, n=8000, seed=42, edge_factor=1.3)
    assert a != b


def test_cache_key_differs_by_seed(tmp_path) -> None:
    a = _fixture_cache_dir(tmp_path, n=3000, seed=1, edge_factor=1.3)
    b = _fixture_cache_dir(tmp_path, n=3000, seed=2, edge_factor=1.3)
    assert a != b


def test_cache_key_differs_by_edge_factor(tmp_path) -> None:
    a = _fixture_cache_dir(tmp_path, n=3000, seed=42, edge_factor=1.3)
    b = _fixture_cache_dir(tmp_path, n=3000, seed=42, edge_factor=2.0)
    assert a != b


# ---------------------------------------------------------------------------
# (b) copytree into a fresh tmp root: openable + mutation-isolated
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _ENGINE_AVAILABLE, reason="native engine not built")
def test_copied_store_is_openable_and_isolated_from_source(tmp_path, monkeypatch) -> None:
    from iai_mcp.store import MemoryStore

    source = tmp_path / "source-store"
    _make_minimal_store(source, monkeypatch, tmp_path / "source-home")

    source_snapshot = {
        p.relative_to(source): p.read_bytes()
        for p in source.rglob("*") if p.is_file()
    }

    copy_dest = tmp_path / "trial-copy"
    shutil.copytree(source, copy_dest)

    monkeypatch.setenv("HOME", str(tmp_path / "trial-home"))
    monkeypatch.setenv("IAI_MCP_STORE", str(copy_dest))
    store = MemoryStore(path=copy_dest)
    try:
        # Openable: the copy is a real, usable store, not just files that
        # happen to exist on disk.
        assert store.embed_dim > 0

        from iai_mcp.store._buffers import flush_record_buffer
        from tests.rss_rca.corpus import build_corpus

        build_corpus(
            store, n=2, seed=99, dim=store.embed_dim,
            varied_tags=True, with_hv_payload=True,
        )
        flush_record_buffer(store)
    finally:
        store.close()

    # Mutation isolation: writing to the COPY must never appear in the
    # SOURCE -- otherwise a trial's mutation would bleed into the cache or a
    # sibling trial that also copies from it.
    source_resnapshot = {
        p.relative_to(source): p.read_bytes()
        for p in source.rglob("*") if p.is_file()
    }
    assert source_snapshot == source_resnapshot


# ---------------------------------------------------------------------------
# (c) WAL/SHM-sidecar-absence build postcondition
# ---------------------------------------------------------------------------


def test_wal_sidecar_with_pending_data_raises() -> None:
    # Larger than the 32-byte empty-header threshold -- represents real
    # uncheckpointed content, the case the postcondition must catch.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        store_dir = Path(d)
        (store_dir / "brain.sqlite3").write_bytes(b"fake-db")
        (store_dir / "brain.sqlite3-wal").write_bytes(b"x" * 100)
        with pytest.raises(RuntimeError, match="sidecar"):
            _assert_no_wal_shm_sidecar(store_dir)


def test_shm_sidecar_with_pending_data_raises() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        store_dir = Path(d)
        (store_dir / "brain.sqlite3").write_bytes(b"fake-db")
        (store_dir / "brain.sqlite3-shm").write_bytes(b"x" * 100)
        with pytest.raises(RuntimeError, match="sidecar"):
            _assert_no_wal_shm_sidecar(store_dir)


def test_no_sidecar_does_not_raise() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        store_dir = Path(d)
        (store_dir / "brain.sqlite3").write_bytes(b"fake-db")
        _assert_no_wal_shm_sidecar(store_dir)  # must not raise


def test_empty_header_only_wal_does_not_raise() -> None:
    # A checkpointed WAL persists as a header-only artifact (verified
    # empirically against this engine) -- that is the correct terminal
    # state after close(), not a violation.
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        store_dir = Path(d)
        (store_dir / "brain.sqlite3").write_bytes(b"fake-db")
        (store_dir / "brain.sqlite3-wal").write_bytes(b"x" * 32)
        _assert_no_wal_shm_sidecar(store_dir)  # must not raise

"""Regression: migrate-time stale runtime_graph_cache invalidation.

migrate_sqlite_to_lilli must delete a destination-root runtime_graph_cache.json
whose corpus fingerprint (payload_record_count) drifts too far from the
migrated store's ACTIVE record count -- fingerprint-checked, up front, never
lazy. A cache within drift tolerance is left byte-untouched. A corrupt/
unreadable cache is deleted defensively. A store with no cache at all still
recalls correctly via the ANN-only community-gate degrade (no false
negatives from the deletion).

Hermetic: tmp HOME + tmp store + a synthetic passphrase, single process, no
xdist. Both storage-driver env settings are exercised (the migrate target is
lilli by nature; the module itself must import and run cleanly under either
driver env setting since a caller could have LILLI_STORAGE_DRIVER set from a
prior step).
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from iai_mcp.migrate import MigrateReport, migrate_sqlite_to_lilli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(i: int, vec: list[float], *, created_at: datetime):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=f"alice recalls memory fragment number {i} about topology",
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
        provenance=[{"session_id": "sess", "role": "user"}],
        created_at=created_at,
        updated_at=created_at,
        tags=["role:user"],
        language="en",
    )


def _build_source_store(
    root: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
    n_records: int,
) -> tuple[str, list[float]]:
    """Build a synthetic stdlib source store; return (brain.sqlite3 path, record-0 vec)."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(root)
    rng = np.random.RandomState(7)
    base = datetime(2026, 5, 1, 9, 0, 0, tzinfo=timezone.utc)
    first_vec: list[float] | None = None
    for i in range(n_records):
        vec = rng.randn(384).tolist()
        if i == 0:
            first_vec = vec
        rec = _make_record(i, vec, created_at=base + timedelta(seconds=i))
        store.insert(rec)
    flush_record_buffer(store)

    src_db = str(root / "hippo" / "brain.sqlite3")
    store.db.close()
    assert first_vec is not None
    return src_db, first_vec


def _write_raw_cache(dst_root: Path, *, payload_record_count) -> None:
    """Write a minimal encrypted runtime_graph_cache.json at dst_root.

    Bypasses the full ``_cache_key`` gate (which requires an exact windowed
    edges/pending/embed_dim match) -- the invalidation helper under test only
    ever reads the compact ``payload_record_count`` field via
    ``_load_and_decrypt_cache``, exactly like this fixture does, so this is
    the correct minimal shape to exercise the fingerprint check in isolation.
    """
    from iai_mcp import runtime_graph_cache
    from iai_mcp.crypto import CryptoKey, encrypt_field

    data = {
        "cache_version": runtime_graph_cache.CACHE_VERSION,
        "key": [],
        "assignment": {"node_to_community": {}, "community_centroids": {}},
        "rich_club": [],
        "node_payload": {},
        "centrality": {"11111111-1111-1111-1111-111111111111": 0.5},
        "payload_record_count": payload_record_count,
        "max_degree": 0,
        "node_degrees": {},
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "generation": 0,
        "rebuild_timestamp": "",
    }
    key = CryptoKey(store_root=dst_root).get_or_create()
    plaintext = json.dumps(data)
    ciphertext = encrypt_field(plaintext, key, runtime_graph_cache._CACHE_AAD)
    cache_path = dst_root / runtime_graph_cache.CACHE_FILENAME
    cache_path.write_text(ciphertext, encoding="utf-8")


def _write_corrupt_cache(dst_root: Path) -> None:
    from iai_mcp import runtime_graph_cache

    cache_path = dst_root / runtime_graph_cache.CACHE_FILENAME
    # Not encrypted (no iai:enc:v1: prefix) -- is_encrypted() returns False,
    # so _load_and_decrypt_cache returns None: unreadable-by-construction.
    cache_path.write_text("not a valid cache payload", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _migrate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_records: int = 30,
    pre_cache_writer=None,
) -> tuple[Path, Path, MigrateReport, list[float]]:
    src_root = tmp_path / "src"
    dst_root = tmp_path / "dst"
    src_root.mkdir()

    src_db, first_vec = _build_source_store(
        src_root, monkeypatch=monkeypatch, n_records=n_records
    )

    if pre_cache_writer is not None:
        dst_root.mkdir(parents=True, exist_ok=True)
        pre_cache_writer(dst_root)

    report = migrate_sqlite_to_lilli(src_db, dst_root, batch=10)
    return dst_root, src_root, report, first_vec


# ---------------------------------------------------------------------------
# Group 1: mismatch branch -- deleted, reason=drift
# ---------------------------------------------------------------------------


def test_mismatch_cache_deleted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A cache whose fingerprint is wildly off vs. the migrated corpus is deleted."""
    from iai_mcp import runtime_graph_cache

    dst_root, _src_root, report, _first_vec = _migrate(
        tmp_path,
        monkeypatch,
        n_records=30,
        pre_cache_writer=lambda root: _write_raw_cache(root, payload_record_count=38631),
    )

    cache_path = dst_root / runtime_graph_cache.CACHE_FILENAME
    # The pre-warm step (which runs after invalidation) writes its OWN fresh
    # cache, so the file may exist again by the time migrate returns -- what
    # matters is that the invalidation step itself reported a drift-deletion,
    # not merely "file absent at the very end".
    assert report.cache_deleted is True, (
        f"expected the stale (38631 vs ~30) cache to be deleted; "
        f"got cache_deleted={report.cache_deleted}, reason={report.cache_invalidation_reason}"
    )
    assert report.cache_invalidation_reason == "drift"
    # Regardless of the later pre-warm, a cache now on disk (if any) must
    # reflect the migrated corpus, never the stale 38631 fingerprint.
    if cache_path.exists():
        results = _read_raw_payload_count(dst_root)
        assert results != 38631


def _read_raw_payload_count(dst_root: Path):
    from iai_mcp import runtime_graph_cache
    from iai_mcp.crypto import CryptoKey, decrypt_field

    cache_path = dst_root / runtime_graph_cache.CACHE_FILENAME
    raw = cache_path.read_text(encoding="utf-8")
    key = CryptoKey(store_root=dst_root).get_or_create()
    plaintext = decrypt_field(raw, key, runtime_graph_cache._CACHE_AAD)
    data = json.loads(plaintext)
    return data.get("payload_record_count")


# ---------------------------------------------------------------------------
# Group 2: match branch -- untouched, sha256 pre == post
# ---------------------------------------------------------------------------


def test_matching_cache_left_untouched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A cache whose fingerprint matches the migrated corpus is left byte-identical."""
    n_records = 30
    captured_hash: dict[str, str] = {}

    def _pre_cache_writer(root: Path) -> None:
        _write_raw_cache(root, payload_record_count=n_records)
        captured_hash["sha256"] = _sha256(root / "runtime_graph_cache.json")

    dst_root, _src_root, report, _first_vec = _migrate(
        tmp_path,
        monkeypatch,
        n_records=n_records,
        pre_cache_writer=_pre_cache_writer,
    )

    assert report.cache_deleted is False, (
        f"a within-tolerance cache must not be deleted; reason={report.cache_invalidation_reason}"
    )
    assert report.cache_invalidation_reason == "within_tolerance"

    # NOTE: the pre-warm step that runs AFTER invalidation overwrites the
    # cache file unconditionally with its own fresh build -- so this test
    # asserts on the INVALIDATION STEP'S reported outcome (verified above),
    # not on the final on-disk bytes, which the pre-warm legitimately changes
    # as an independent, later step. The invalidation step itself is proven
    # not to have touched the file by checking cache_deleted is False.


def test_matching_cache_untouched_when_prewarm_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """With no records to embed (degenerate corpus skip) the pre-warm bails out
    before writing, isolating the invalidation step's byte-untouched guarantee
    all the way to the final on-disk file.
    """
    from iai_mcp.migrate import _to_lilli as to_lilli_mod

    n_records = 30
    captured_hash: dict[str, str] = {}

    def _pre_cache_writer(root: Path) -> None:
        _write_raw_cache(root, payload_record_count=n_records)
        captured_hash["sha256"] = _sha256(root / "runtime_graph_cache.json")

    # Force the pre-warm step to no-op (simulate its degenerate-corpus skip)
    # so only the invalidation step under test can touch the cache file.
    monkeypatch.setattr(
        to_lilli_mod,
        "_prewarm_graph_cache",
        lambda dst_root, report, peak_rss, *, src_root=None: peak_rss,
    )

    dst_root, _src_root, report, _first_vec = _migrate(
        tmp_path,
        monkeypatch,
        n_records=n_records,
        pre_cache_writer=_pre_cache_writer,
    )

    assert report.cache_deleted is False
    assert report.cache_invalidation_reason == "within_tolerance"

    cache_path = dst_root / "runtime_graph_cache.json"
    assert cache_path.exists(), "matching cache must survive migrate untouched"
    assert _sha256(cache_path) == captured_hash["sha256"], (
        "byte-identical cache expected pre/post migrate for a within-tolerance fingerprint"
    )


# ---------------------------------------------------------------------------
# Group 3: corrupt branch -- deleted, migrate still succeeds
# ---------------------------------------------------------------------------


def test_corrupt_cache_deleted_migrate_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """An unreadable/corrupt cache is deleted defensively; migrate still succeeds."""
    dst_root, _src_root, report, _first_vec = _migrate(
        tmp_path,
        monkeypatch,
        n_records=25,
        pre_cache_writer=_write_corrupt_cache,
    )

    assert report.cache_deleted is True
    assert report.cache_invalidation_reason == "unreadable"
    # Migrate itself completed and copied rows -- corruption of the sidecar
    # cache file must never abort the row copy.
    assert report.rows_copied.get("records", 0) == 25


# ---------------------------------------------------------------------------
# Group 4: no-cache recall smoke -- ANN-only degrade, hits > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver_env", [None, "lilli"])
def test_no_cache_recall_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, driver_env
):
    """A migrated store with NO runtime_graph_cache.json still recalls with hits > 0.

    Proves deletion cannot cause false negatives: the community gate degrades
    gracefully to ANN-only when the cache is absent (semantic_recall.py's
    ANN-only fallback / runtime_graph_cache.py's cold_degrade path).
    """
    from iai_mcp import runtime_graph_cache
    from iai_mcp.store import MemoryStore

    dst_root, _src_root, _report, first_vec = _migrate(tmp_path, monkeypatch, n_records=20)

    cache_path = dst_root / runtime_graph_cache.CACHE_FILENAME
    # Force the no-cache condition regardless of whether pre-warm wrote one --
    # this test isolates SC-2 (recall correctness with an absent cache), not
    # the invalidation step itself.
    cache_path.unlink(missing_ok=True)
    assert not cache_path.exists()

    if driver_env is None:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver_env)
    monkeypatch.setenv("IAI_MCP_STORE", str(dst_root))

    store = MemoryStore(dst_root)
    try:
        # Query with the EXACT vector used for record 0 at insert time -- a
        # deterministic ANN self-match, independent of any embedder model, so
        # this test isolates the recall-path/cache-absence behavior (SC-2)
        # rather than embedding-quality.
        hits = store.query_similar(first_vec, k=5)
        assert len(hits) > 0, (
            "expected hits > 0 from the ANN-only degrade path with no "
            "runtime_graph_cache present; got zero hits"
        )
    finally:
        store.db.close()


# ---------------------------------------------------------------------------
# Group 5: boundary -- exact edge behavior matches _within_drift_tolerance
# ---------------------------------------------------------------------------


def test_boundary_within_tolerance_minus_one_kept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """cached_count exactly at the tolerance boundary is within tolerance (kept)."""
    from iai_mcp.retrieve import _drift_tolerance, _within_drift_tolerance

    n_records = 30
    tolerance = _drift_tolerance(n_records)
    # cached_count chosen so |records_count - cached_count| == tolerance (edge
    # is inclusive per _within_drift_tolerance's <=).
    boundary_cached_count = n_records + tolerance
    assert _within_drift_tolerance(boundary_cached_count, n_records) is True

    def _pre_cache_writer(root: Path) -> None:
        _write_raw_cache(root, payload_record_count=boundary_cached_count)

    _dst_root, _src_root, report, _first_vec = _migrate(
        tmp_path,
        monkeypatch,
        n_records=n_records,
        pre_cache_writer=_pre_cache_writer,
    )

    assert report.cache_deleted is False
    assert report.cache_invalidation_reason == "within_tolerance"


def test_boundary_tolerance_plus_one_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """cached_count one past the tolerance boundary is over tolerance (deleted)."""
    from iai_mcp.retrieve import _drift_tolerance, _within_drift_tolerance

    n_records = 30
    tolerance = _drift_tolerance(n_records)
    boundary_cached_count = n_records + tolerance + 1
    assert _within_drift_tolerance(boundary_cached_count, n_records) is False

    def _pre_cache_writer(root: Path) -> None:
        _write_raw_cache(root, payload_record_count=boundary_cached_count)

    _dst_root, _src_root, report, _first_vec = _migrate(
        tmp_path,
        monkeypatch,
        n_records=n_records,
        pre_cache_writer=_pre_cache_writer,
    )

    assert report.cache_deleted is True
    assert report.cache_invalidation_reason == "drift"

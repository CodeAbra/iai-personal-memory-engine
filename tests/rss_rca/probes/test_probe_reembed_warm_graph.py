"""Diagnose-only pytest probe: re-embedded (pending->ready) row vs warm graph.

Mirrors test_runtime_graph_drift_tolerance::test_small_drift_does_not_trigger_
betweenness_rebuild but makes the count change come from a re-embed (a pending
row flipping embedding_pending 1->0) instead of a fresh insert. Confirms whether
the warm-reuse path folds the re-embedded row into the community graph when the
+1 active-count delta is within drift tolerance.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4, UUID

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "store")
    s.root = tmp_path
    return s


def _make_rec(seed: int, store: MemoryStore) -> MemoryRecord:
    rng = np.random.default_rng(seed)
    vec = rng.random(store.embed_dim).astype(np.float32)
    vec = (vec / np.linalg.norm(vec)).tolist()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic",
        literal_surface=f"surface number {seed} carrying real text",
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=now, updated_at=now, tags=[f"tag{seed % 3}"],
        language="en",
    )


def _seed_connected(store: MemoryStore, n: int, seed_base: int = 0) -> list:
    recs = [_make_rec(seed_base + i, store) for i in range(n)]
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [rec.id for rec in recs]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(pairs, delta=1.0, edge_type="hebbian")
    return recs


def test_reembedded_row_excluded_from_warm_graph_within_drift(store, monkeypatch):
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_ABS", "3")
    monkeypatch.setenv("IAI_MCP_RGC_DRIFT_FRAC", "0.0")

    recs = _seed_connected(store, n=16, seed_base=200)
    ids = [r.id for r in recs]
    runtime_graph_cache.invalidate(store)

    graph0, assign0, _rc0 = retrieve.build_runtime_graph(store)
    assert graph0.node_count() == 16
    assert not retrieve._runtime_graph_rebuild_needed(store), (
        "freshly-built centrality-bearing cache must be warm"
    )
    count0 = store.active_records_count()

    # Insert a genuine PENDING row (excluded from active count, absent from graph).
    pid = str(uuid4())
    store.db.insert_pending_row(
        record_id=pid, tier="episodic",
        literal_surface="pending surface carrying real text",
        tags_json=json.dumps(["tag0"]), provenance_json=json.dumps([]),
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    assert store.active_records_count() == count0, "pending row must not count active"

    # Re-embed it (pending 1->0). The production reembed path
    # (reembed_pending_rows) flips embedding_pending and then invalidates the
    # corpus-count cache for both the active and pending keys. A raw UPDATE on
    # the connection bypasses every write chokepoint, so the probe must mirror
    # that invalidation explicitly — otherwise the cached active count stays
    # warm at its pre-reembed value and the +1 transition is invisible.
    with store.db._conn_lock:
        rng = np.random.default_rng(999)
        v = (rng.random(store.embed_dim).astype(np.float32))
        v = v / np.linalg.norm(v)
        blob = v.astype(np.float32).tobytes()
        store.db._conn.execute(
            "UPDATE records SET embedding = ?, embedding_pending = 0 WHERE id = ?",
            (blob, pid),
        )
        store.db._conn.commit()
    store._invalidate_corpus_count("active", "pending")
    # NOTE: deliberately NOT adding an edge — a pure re-embed must not change the
    # edges-count window, isolating the records-count drift path.

    count_after = store.active_records_count()
    from iai_mcp import runtime_graph_cache as _rgc
    print(
        f"CACHEKEY before-vs-after pure reembed: "
        f"{_rgc._cache_key(store)}"
    )
    assert count_after == count0 + 1, "re-embed must increment active count by 1"

    rebuild_needed = retrieve._runtime_graph_rebuild_needed(store)
    graph2, _assign2, _rc2 = retrieve.build_runtime_graph(store)

    # Diagnostic asserts — record the actual behavior.
    print(
        f"DIAG count0={count0} count_after={count_after} "
        f"drift_tol={retrieve._drift_tolerance(count0)} "
        f"rebuild_needed={rebuild_needed} "
        f"graph2_nodes={graph2.node_count()} "
        f"reembed_in_graph={graph2.has_node(UUID(pid))}"
    )
    # The bug claim: the +1 is within drift (<=3) -> no rebuild -> row excluded.
    assert not rebuild_needed, "EXPECTED within-drift warm reuse (no rebuild)"
    assert not graph2.has_node(UUID(pid)), (
        "LATENT BUG: re-embedded row should be excluded by warm reuse"
    )

from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


def test_w5_provenance_overflow_sustained_load(tmp_path, monkeypatch):
    from iai_mcp.provenance_queue import ProvenanceWriteQueue
    from iai_mcp.store import MemoryStore
    from tests.test_store import _make as _make_record

    store = MemoryStore(path=tmp_path / "store")
    r = _make_record()
    store.insert(r)

    monkeypatch.setenv("HOME", str(tmp_path))

    flushed: list = []
    release = threading.Event()
    real_batch = store.append_provenance_batch

    def slow_batch(pairs, records_cache=None):
        release.wait(timeout=15.0)
        flushed.extend(pairs)
        return real_batch(pairs, records_cache=records_cache)

    store.append_provenance_batch = slow_batch  # type: ignore[method-assign]

    q = ProvenanceWriteQueue(
        store, coalesce_ms=10, max_queue_size=2, max_batch_pairs=1,
    )
    q.start()
    try:
        for i in range(10):
            q.enqueue([(r.id, {
                "ts": f"t{i}", "cue": f"sustained-{i}", "session_id": "soak",
            })])
        time.sleep(0.15)
        overflow_dir = store.root / ".provenance-overflow"
        spilled = list(overflow_dir.glob("*.jsonl"))
        assert len(spilled) >= 1, (
            f"expected ≥1 spilled file under sustained overload; got {spilled}"
        )

        release.set()

        deadline = time.time() + 15.0
        while time.time() < deadline:
            q.flush(timeout=2.0)
            q._drain_overflow_dir()
            q.flush(timeout=2.0)
            if not list(overflow_dir.glob("*.jsonl")):
                break
            time.sleep(0.05)
    finally:
        q.stop()

    cues = [p[1]["cue"] for p in flushed]
    assert sorted(cues) == [f"sustained-{i}" for i in range(10)], (
        f"expected all 10 cues exactly once; got {sorted(cues)}"
    )
    overflow_dir = store.root / ".provenance-overflow"
    assert list(overflow_dir.glob("*.jsonl")) == []


def test_w5_capture_drain_partial_failure_preserves_evidence(tmp_path, monkeypatch):
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lance"))

    deferred = tmp_path / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True)
    fpath = deferred / "soak-mixed-1.jsonl"
    fpath.write_text(
        json.dumps({
            "version": 1,
            "deferred_at": "2026-04-30T00:00:00Z",
            "session_id": "soak-2",
            "cwd": "/tmp",
        }) + "\n"
        + json.dumps({
            "cue": "good a", "text": "first valid event with ample length here",
            "tier": "episodic", "role": "user",
        }) + "\n"
        + json.dumps({
            "cue": "poison", "text": "INSERT_FAIL_SENTINEL_SOAK middle event",
            "tier": "episodic", "role": "user",
        }) + "\n"
        + json.dumps({
            "cue": "good b", "text": "third valid event with sufficient text",
            "tier": "episodic", "role": "user",
        }) + "\n"
    )

    # The backlog drain routes new events through `_drain_write_pending` (the
    # two-phase pending-row write), not `MemoryStore.insert`. Inject the failure
    # there, content-specifically: only the poison sentinel turn fails, while the
    # two good turns still flow through the real drain path and insert. This keeps
    # the partial-failure intent — evidence is preserved by promoting the file to
    # `.failed-*` even though some events in it landed.
    import iai_mcp.capture as capture_mod

    real_drain_write_pending = capture_mod._drain_write_pending

    def drain_or_fail(store, *, text, **kwargs):
        if "INSERT_FAIL_SENTINEL_SOAK" in text:
            return {
                "status": "skipped",
                "record_id": None,
                "reason": "insert-failed: simulated failure at soak",
            }
        return real_drain_write_pending(store, text=text, **kwargs)

    monkeypatch.setattr(capture_mod, "_drain_write_pending", drain_or_fail)

    store = MemoryStore()
    counts = drain_deferred_captures(store)

    assert not fpath.exists()
    failed = list(deferred.glob("soak-mixed-1.failed-*.jsonl"))
    assert len(failed) == 1, (
        f"expected 1 .failed-* file; got {failed} "
        f"(deferred contents: {list(deferred.iterdir())})"
    )
    assert counts["events_inserted"] == 2, counts
    assert counts["events_skipped_insert_failed"] == 1, counts
    assert counts["files_drained"] == 0, counts
    assert counts["files_failed"] == 1, counts


def test_w5_graph_cache_encryption_no_plaintext_canary(tmp_path):
    from iai_mcp import runtime_graph_cache
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path

    rid = uuid4()
    canary = "PLAINTEXT_CANARY_SOAK_aaak"
    node_payload = {
        str(rid): {
            "embedding": [0.1] * 384,
            "surface": canary,
            "centrality": 0.3,
            "tier": "episodic",
            "pinned": False,
            "tags": [],
            "language": "en",
        }
    }
    assignment = CommunityAssignment(
        node_to_community={rid: rid},
        community_centroids={rid: [0.1] * 384},
        modularity=0.4,
        backend="leiden",
        top_communities=[rid],
        mid_regions={rid: [rid]},
    )
    rich_club = [rid]

    ok = runtime_graph_cache.save(
        store, assignment, rich_club,
        node_payload=node_payload, max_degree=2,
    )
    assert ok is True

    cache_path = tmp_path / "runtime_graph_cache.json"
    raw_bytes = cache_path.read_bytes()
    assert canary.encode("utf-8") not in raw_bytes, (
        "plaintext canary leaked into the on-disk sidecar"
    )
    assert raw_bytes.startswith(b"iai:enc:v1:")

    loaded = runtime_graph_cache.try_load(store)
    assert loaded is not None
    _, _, payload, _, _ = loaded
    assert payload[str(rid)]["surface"] == canary


def test_w5_recall_survives_malformed_anti_edge(tmp_path):
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import _find_anti_hits
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord

    store = MemoryStore(path=tmp_path / "lancedb")

    rid_hit = uuid4()
    rid_anti = uuid4()
    now = datetime.now(timezone.utc)
    for rid, surface in [(rid_hit, "primary"), (rid_anti, "anti")]:
        store.insert(MemoryRecord(
            id=rid, tier="episodic", literal_surface=surface,
            aaak_index="", embedding=[0.1] * EMBED_DIM,
            community_id=None, centrality=0.0, detail_level=2,
            pinned=False, stability=0.0, difficulty=0.0,
            last_reviewed=None, never_decay=False, never_merge=False,
            provenance=[], created_at=now, updated_at=now,
            tags=[], language="en",
        ))

    edges = store.db.open_table("edges")
    edges.add([
        {"src": str(rid_hit), "dst": str(rid_anti),
         "edge_type": "contradicts", "weight": 1.0,
         "updated_at": now},
        {"src": str(rid_hit), "dst": "not-a-uuid-soak",
         "edge_type": "contradicts", "weight": 1.0,
         "updated_at": now},
    ])

    hit = MemoryHit(
        record_id=rid_hit, score=0.9, reason="soak",
        literal_surface="primary", adjacent_suggestions=[],
    )

    caplog_records: list = []

    class _Capture(logging.Handler):
        def emit(self, record):
            caplog_records.append(record.getMessage())

    handler = _Capture(level=logging.WARNING)
    logging.getLogger("iai_mcp.pipeline").addHandler(handler)
    try:
        anti = _find_anti_hits(
            [hit], store, MemoryGraph(), k=3, records_cache=None,
        )
    finally:
        logging.getLogger("iai_mcp.pipeline").removeHandler(handler)

    assert len(anti) == 1
    assert anti[0].record_id == rid_anti
    assert any("anti_hits_skip_malformed_edge" in m for m in caplog_records), (
        f"expected log line; got {caplog_records}"
    )

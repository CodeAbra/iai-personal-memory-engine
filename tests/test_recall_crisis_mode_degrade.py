"""Hermetic test for the crisis_mode honest-degrade guard in
``core.dispatch(method="memory_recall", ...)``.

When the lifecycle_state's ``crisis_mode`` flag is True (the scheduler is
looping a deferred step and cannot advance), the daemon's warm recall
path would otherwise serve stale schema-dominated results. The guard
short-circuits to a bypass-safe degraded tier: raw ANN similarity plus
the exact-cosine authority, both of which read only consolidation-
independent structures. The response carries ``_degraded: True`` so the
wrapper falls back to bank-recall via its existing socket-unreachable
code path, but the hits themselves are real whenever matching data
exists in the store — the hippocampus invariant that consolidation
health can never zero recall.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _clear_crisis_cache():
    """Clear the module-level crisis-state TTL cache before each test so
    monkeypatches to lifecycle_state.load_state take effect immediately."""
    import iai_mcp.core as _core
    _core._CRISIS_STATE_CACHE.clear()
    yield
    _core._CRISIS_STATE_CACHE.clear()


def _make_record(
    text: str = "hello world",
    vec: list[float] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec if vec is not None else [0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def _make_store(tmp_path):
    """Open a hermetic on-disk MemoryStore under tmp_path."""
    from iai_mcp.store import MemoryStore

    return MemoryStore(path=tmp_path)


def _set_crisis(monkeypatch: pytest.MonkeyPatch, crisis_mode: bool):
    monkeypatch.setattr(
        "iai_mcp.lifecycle_state.load_state",
        lambda *_a, **_kw: {"crisis_mode": crisis_mode, "current_state": "SLEEP" if crisis_mode else "WAKE"},
    )


class _FakeEmbedder:
    """Deterministic fake embedder: always returns a vector near the seeded
    record's embedding, regardless of cue text — the point under test is the
    degraded branch's serving path, not real semantic encoding."""

    def __init__(self, vec: list[float]):
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


class _RaisingEmbedder:
    def embed(self, _text: str) -> list[float]:
        raise RuntimeError("synthetic embed failure")


def test_crisis_recall_serves_hits_when_data_matches(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """Under crisis_mode, a store with matching data must still return real
    hits — never the old unconditional empty list."""
    from iai_mcp.core import dispatch

    store = _make_store(tmp_path)
    target_vec = [0.9] + [0.05] * (EMBED_DIM - 1)
    seeded = _make_record(text="the target memory", vec=target_vec)
    store.insert(seeded)
    store.insert(_make_record(text="unrelated memory one", vec=[0.0] * EMBED_DIM))
    store.insert(_make_record(text="unrelated memory two", vec=[-0.2] * EMBED_DIM))

    _set_crisis(monkeypatch, True)
    monkeypatch.setattr(
        "iai_mcp.embed.embedder_for_store",
        lambda _store: _FakeEmbedder(target_vec),
    )

    resp = dispatch(store, "memory_recall", {"cue": "find the target memory"})

    assert resp["_degraded"] is True
    assert resp["_reason"] == "daemon_consolidation_stuck"
    assert len(resp["hits"]) > 0
    hit = resp["hits"][0]
    assert hit["literal_surface"] == "the target memory"
    assert set(["record_id", "score", "reason", "literal_surface"]).issubset(hit.keys())


def test_crisis_recall_empty_store_returns_empty_hits_degraded(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """An empty store organically yields hits=[] under crisis — not a
    special case, just the natural result of empty ANN/authority scans."""
    from iai_mcp.core import dispatch

    store = _make_store(tmp_path)

    _set_crisis(monkeypatch, True)
    monkeypatch.setattr(
        "iai_mcp.embed.embedder_for_store",
        lambda _store: _FakeEmbedder([0.1] * EMBED_DIM),
    )

    resp = dispatch(store, "memory_recall", {"cue": "anything"})

    assert resp["hits"] == []
    assert resp["_degraded"] is True
    assert resp["_reason"] == "daemon_consolidation_stuck"


def test_crisis_recall_internal_failure_falls_back_to_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """If the degraded tier itself fails internally (e.g. embed raises), the
    branch must degrade to the last-resort empty response, never raise."""
    from iai_mcp.core import dispatch

    store = _make_store(tmp_path)
    store.insert(_make_record(text="some memory"))

    _set_crisis(monkeypatch, True)
    monkeypatch.setattr(
        "iai_mcp.embed.embedder_for_store",
        lambda _store: _RaisingEmbedder(),
    )

    resp = dispatch(store, "memory_recall", {"cue": "anything"})

    assert resp["hits"] == []
    assert resp["_degraded"] is True
    assert resp["_reason"] == "daemon_consolidation_stuck"


def test_crisis_recall_authority_off_still_serves_ann_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """With the exact-authority tier disabled, ANN hits alone must still
    serve under crisis."""
    from iai_mcp.core import dispatch

    store = _make_store(tmp_path)
    target_vec = [0.9] + [0.05] * (EMBED_DIM - 1)
    store.insert(_make_record(text="target via ann only", vec=target_vec))

    _set_crisis(monkeypatch, True)
    monkeypatch.setattr(
        "iai_mcp.embed.embedder_for_store",
        lambda _store: _FakeEmbedder(target_vec),
    )
    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")

    resp = dispatch(store, "memory_recall", {"cue": "find it"})

    assert resp["_degraded"] is True
    assert len(resp["hits"]) > 0
    assert resp["hits"][0]["literal_surface"] == "target via ann only"


def test_crisis_recall_one_bad_hit_does_not_collapse_all(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """A single hit whose serialization raises must be skipped, not collapse
    the whole degraded response to the last-resort empty."""
    from iai_mcp.core import dispatch
    import iai_mcp.core._serializers as _serializers

    store = _make_store(tmp_path)
    target_vec = [0.9] + [0.05] * (EMBED_DIM - 1)
    store.insert(_make_record(text="good memory one", vec=target_vec))
    store.insert(_make_record(text="good memory two", vec=[0.85] + [0.06] * (EMBED_DIM - 1)))

    _set_crisis(monkeypatch, True)
    monkeypatch.setattr(
        "iai_mcp.embed.embedder_for_store",
        lambda _store: _FakeEmbedder(target_vec),
    )

    _real_hit_to_json = _serializers._hit_to_json
    _calls = {"n": 0}

    def _flaky_hit_to_json(hit):
        _calls["n"] += 1
        if _calls["n"] == 1:
            raise RuntimeError("synthetic serialization failure on first hit")
        return _real_hit_to_json(hit)

    monkeypatch.setattr(_serializers, "_hit_to_json", _flaky_hit_to_json)

    resp = dispatch(store, "memory_recall", {"cue": "find memories"})

    assert resp["_degraded"] is True
    assert resp["_reason"] == "daemon_consolidation_stuck"
    # The bad hit was skipped; the remaining hit still surfaced (not zeroed).
    assert len(resp["hits"]) >= 1


def test_crisis_last_resort_returns_fresh_hits_list(monkeypatch: pytest.MonkeyPatch):
    """Two last-resort responses must not alias the same ``hits`` list."""
    import iai_mcp.core as _core

    a = _core._crisis_degraded_last_resort()
    b = _core._crisis_degraded_last_resort()

    assert a["hits"] == [] and b["hits"] == []
    assert a["hits"] is not b["hits"]
    a["hits"].append("mutated")
    assert b["hits"] == []


def test_memory_recall_warm_path_when_crisis_mode_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """When crisis_mode=False, the guard does not interfere and the dispatch
    proceeds to the normal recall path. We assert on the absence of the
    degraded sentinel rather than the full response shape — that way the
    test stays focused on what changed."""
    from iai_mcp.core import dispatch

    _set_crisis(monkeypatch, False)

    store = _make_store(tmp_path)
    resp = dispatch(store, "memory_recall", {"cue": "Hello there"})

    # The non-degraded path returns a dict that does NOT carry the degrade
    # sentinel. The exact contents depend on the warm path (or fallback), so
    # we only check the absence of the crisis-mode short-circuit.
    assert resp.get("_degraded") is not True or resp.get("_reason") != "daemon_consolidation_stuck"


def test_memory_recall_crisis_state_read_is_cached_per_ttl_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """N rapid recalls within one TTL window must cause at most 1 filesystem
    read, not N reads. The crisis-state TTL cache (1 s) on the recall hot
    path must absorb all calls after the first population."""
    import iai_mcp.core as _core

    call_count = 0

    def _counting_load(*_a, **_kw):
        nonlocal call_count
        call_count += 1
        return {"crisis_mode": False, "current_state": "WAKE"}

    # Clear the module-level crisis cache before the test so we start fresh.
    _core._CRISIS_STATE_CACHE.clear()

    monkeypatch.setattr("iai_mcp.lifecycle_state.load_state", _counting_load)

    store = _make_store(tmp_path)
    # Fire 5 rapid recalls — all within the 1 s TTL window.
    for _ in range(5):
        _core.dispatch(store, "memory_recall", {"cue": "bench"})

    # Only the first call should have reached load_state.
    assert call_count == 1, (
        f"Expected 1 filesystem read within the TTL window, got {call_count}"
    )


def test_memory_recall_guards_against_load_state_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    """If daemon_state.load_state() itself raises, the guard must NOT crash
    recall — it falls through to the warm path (the constitutional
    always-available invariant: the guard cannot itself break recall)."""
    from iai_mcp.core import dispatch

    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic lifecycle_state read failure")

    monkeypatch.setattr("iai_mcp.lifecycle_state.load_state", _boom)

    store = _make_store(tmp_path)
    resp = dispatch(store, "memory_recall", {"cue": "Hello there"})

    # Did not short-circuit to degraded — fell through to warm path.
    assert resp.get("_reason") != "daemon_consolidation_stuck"

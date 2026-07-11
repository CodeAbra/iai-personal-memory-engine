from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import MemoryRecord


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


class _DetEmbedder:

    def __init__(self, dim: int = 384) -> None:
        self.DIM = dim
        self.DEFAULT_DIM = dim
        self.DEFAULT_MODEL_KEY = "test"

    def embed(self, text: str) -> list[float]:
        import hashlib
        import random as _random

        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = _random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        n = sum(x * x for x in v) ** 0.5
        return [x / n for x in v] if n > 0 else v


def _make_record(vec: list[float], text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
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
        created_at=now,
        updated_at=now,
        tags=["t"],
        language="en",
    )


def test_centrality_row_warn_when_no_events(monkeypatch) -> None:
    from iai_mcp import doctor as _doctor
    import iai_mcp.events as _events

    monkeypatch.setattr(_events, "query_events", lambda *a, **kw: [])
    import iai_mcp.store as _store_mod
    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    result = _doctor.check_u_recall_centrality_regression()
    assert result.status == "WARN"
    assert "no recall_timing" in result.detail.lower()


def test_centrality_row_warn_when_median_above_threshold(monkeypatch) -> None:
    from iai_mcp import doctor as _doctor
    import iai_mcp.events as _events

    synthetic = [
        {"data": {"centrality_ms": 45.0}},
        {"data": {"centrality_ms": 55.0}},
        {"data": {"centrality_ms": 50.0}},
    ]
    monkeypatch.setattr(_events, "query_events", lambda *a, **kw: synthetic)
    monkeypatch.setattr(_events, "write_event", lambda *a, **kw: None)
    import iai_mcp.store as _store_mod
    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    result = _doctor.check_u_recall_centrality_regression()
    assert result.status == "WARN"
    assert "50.0ms" in result.detail
    assert "30ms threshold" in result.detail


def test_centrality_row_pass_when_median_below_threshold(monkeypatch) -> None:
    from iai_mcp import doctor as _doctor
    import iai_mcp.events as _events

    synthetic = [
        {"data": {"centrality_ms": 10.0}},
        {"data": {"centrality_ms": 20.0}},
        {"data": {"centrality_ms": 15.0}},
    ]
    monkeypatch.setattr(_events, "query_events", lambda *a, **kw: synthetic)
    import iai_mcp.store as _store_mod
    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    result = _doctor.check_u_recall_centrality_regression()
    assert result.status == "PASS"
    assert "15.0ms" in result.detail


def test_centrality_row_emits_health_concern_when_warn(monkeypatch) -> None:
    from iai_mcp import doctor as _doctor
    import iai_mcp.events as _events

    synthetic = [
        {"data": {"centrality_ms": 60.0}},
        {"data": {"centrality_ms": 70.0}},
    ]
    monkeypatch.setattr(_events, "query_events", lambda *a, **kw: synthetic)

    write_event_calls: list[dict] = []

    def _capture(store, kind, data, **kwargs):
        write_event_calls.append({"kind": kind, "data": data})
        return uuid4()

    monkeypatch.setattr(_events, "write_event", _capture)
    import iai_mcp.store as _store_mod
    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    result = _doctor.check_u_recall_centrality_regression()
    assert result.status == "WARN"
    assert any(c["kind"] == "health_concern" for c in write_event_calls)
    health = [c for c in write_event_calls if c["kind"] == "health_concern"][0]
    assert "centrality_median_ms" in health["data"]
    assert health["data"]["centrality_median_ms"] == pytest.approx(65.0)


def test_recall_timing_event_emitted_with_payload_shape(
    tmp_path, monkeypatch
) -> None:
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.store import MemoryStore
    import iai_mcp.events as _events

    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")

    captured: list[dict] = []
    original_write_event = _events.write_event

    def _spy(store, kind, data, **kwargs):
        captured.append({"kind": kind, "data": data})
        return original_write_event(store, kind, data, **kwargs)

    monkeypatch.setattr(_events, "write_event", _spy)
    import iai_mcp.pipeline as _pipeline
    monkeypatch.setattr(_pipeline, "write_event", _spy)

    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    emb = _DetEmbedder(dim=store.embed_dim)
    for i in range(6):
        rec = _make_record(emb.embed(f"timing-fact-{i}"), f"timing fact {i}")
        store.insert(rec)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue="probe timing fact 3 from the seeded set",
        session_id="s",
        budget_tokens=1500,
    )

    timing = [c for c in captured if c["kind"] == "recall_timing"]
    assert timing, "expected at least one recall_timing event under 1.0 sample rate"
    payload = timing[0]["data"]
    assert "centrality_ms" in payload
    assert "sigma_ms" in payload
    assert "pool_collection_ms" in payload
    assert "n_nodes" in payload
    assert isinstance(payload["centrality_ms"], float)
    assert isinstance(payload["sigma_ms"], float)
    assert isinstance(payload["pool_collection_ms"], float)
    assert isinstance(payload["n_nodes"], int)


def test_recall_timing_write_event_failure_does_not_break_recall(
    tmp_path, monkeypatch
) -> None:
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.store import MemoryStore
    import iai_mcp.pipeline as _pipeline

    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")

    def _boom(store, kind, data, **kwargs):
        if kind == "recall_timing":
            raise RuntimeError("simulated event-store outage")
        return uuid4()

    monkeypatch.setattr(_pipeline, "write_event", _boom)

    store = MemoryStore(path=tmp_path / "lancedb")
    store.root = tmp_path
    emb = _DetEmbedder(dim=store.embed_dim)
    for i in range(4):
        rec = _make_record(emb.embed(f"outage-fact-{i}"), f"outage fact {i}")
        store.insert(rec)
    graph, assignment, rich_club = retrieve.build_runtime_graph(store)

    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rich_club,
        embedder=emb,
        cue="outage fact 2",
        session_id="s",
        budget_tokens=1500,
    )
    assert resp is not None
    assert isinstance(resp.hits, list)

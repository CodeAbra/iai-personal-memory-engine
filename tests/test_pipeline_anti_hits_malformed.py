from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.pipeline import _find_anti_hits
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord

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

@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "lancedb")

def _make_record(rid: UUID, surface: str = "topic") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        tags=[],
        language="en",
    )

def _add_edge_row(
    store: MemoryStore,
    *,
    src: str,
    dst: str,
    edge_type: str = "contradicts",
    weight: float = 1.0,
) -> None:
    tbl = store.db.open_table("edges")
    tbl.add([{
        "src": src,
        "dst": dst,
        "edge_type": edge_type,
        "weight": float(weight),
        "updated_at": datetime.now(timezone.utc),
    }])

def _make_hit(rid: UUID, surface: str = "primary topic") -> MemoryHit:
    return MemoryHit(
        record_id=rid,
        score=0.9,
        reason="test_hit",
        literal_surface=surface,
        adjacent_suggestions=[],
    )

def test_malformed_dst_does_not_crash_and_valid_anti_surfaces(store, caplog):
    rid_hit = uuid4()
    rid_anti = uuid4()
    store.insert(_make_record(rid_hit, "primary topic"))
    store.insert(_make_record(rid_anti, "anti topic"))

    _add_edge_row(store, src=str(rid_hit), dst=str(rid_anti),
                  edge_type="contradicts", weight=1.0)
    _add_edge_row(store, src=str(rid_hit), dst="not-a-uuid",
                  edge_type="contradicts", weight=1.0)

    from iai_mcp.graph import MemoryGraph
    graph = MemoryGraph()

    hit = _make_hit(rid_hit)

    with caplog.at_level(logging.WARNING, logger="iai_mcp.pipeline"):
        anti = _find_anti_hits([hit], store, graph, k=3, records_cache=None)

    assert len(anti) == 1, (
        f"expected 1 valid anti-hit; got {len(anti)} "
        f"(records: {[h.record_id for h in anti]})"
    )
    assert anti[0].record_id == rid_anti

    assert any(
        "anti_hits_skip_malformed_edge" in r.getMessage()
        for r in caplog.records
    ), f"expected log line; got {[r.getMessage() for r in caplog.records]}"

def test_malformed_src_filtered_at_upstream_step(store, caplog):
    rid_hit = uuid4()
    rid_anti = uuid4()
    store.insert(_make_record(rid_hit))
    store.insert(_make_record(rid_anti))

    _add_edge_row(store, src=str(rid_hit), dst=str(rid_anti),
                  edge_type="contradicts", weight=1.0)
    _add_edge_row(store, src="zzz-bad-src", dst=str(rid_hit),
                  edge_type="contradicts", weight=1.0)

    from iai_mcp.graph import MemoryGraph
    graph = MemoryGraph()
    hit = _make_hit(rid_hit)

    with caplog.at_level(logging.WARNING, logger="iai_mcp.pipeline"):
        anti = _find_anti_hits([hit], store, graph, k=3, records_cache=None)

    assert len(anti) == 1
    assert anti[0].record_id == rid_anti
    assert any(
        "anti_hits_skip_malformed_edge" in r.getMessage()
        for r in caplog.records
    )
    assert not any(
        "anti_hits_skip_malformed_lid" in r.getMessage()
        for r in caplog.records
    ), "upstream filter must remove bad rows before the inner UUID(lid) call"

def test_no_contradicts_edges_returns_empty_clean(store):
    rid_hit = uuid4()
    store.insert(_make_record(rid_hit))

    from iai_mcp.graph import MemoryGraph
    graph = MemoryGraph()
    hit = _make_hit(rid_hit)

    anti = _find_anti_hits([hit], store, graph, k=3, records_cache=None)
    assert anti == []

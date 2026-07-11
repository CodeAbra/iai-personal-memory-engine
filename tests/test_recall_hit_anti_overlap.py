"""A record must never be BOTH a hit and an anti-hit: with fewer than
k_hits+k_anti survivors the head/tail windows of the same candidate list
overlap, and a top memory would be served as a low-similarity anti-hit —
anti_hits carry the same contractual weight as hits."""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


@pytest.fixture
def small_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    for i in range(3):
        vec = [0.0] * EMBED_DIM
        vec[i] = 1.0
        vec[10 + i] = 0.4
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"small corpus record {i}",
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
            provenance=[{"session_id": "seed", "role": "user"}],
            created_at=now,
            updated_at=now,
            tags=["role:user"],
            language="en",
            s5_trust_score=0.5,
            profile_modulation_gain={},
        ))
    yield store
    store.close()


def test_small_corpus_hits_and_anti_hits_never_overlap(small_store):
    from iai_mcp.retrieve import recall
    from iai_mcp.types import EMBED_DIM

    cue = [0.0] * EMBED_DIM
    cue[0] = 1.0
    resp = recall(small_store, cue, "small corpus", "sess-overlap", k_hits=5, k_anti=3)

    hit_ids = {str(h.record_id) for h in resp.hits}
    anti_ids = {str(a.record_id) for a in resp.anti_hits}
    assert hit_ids, "expected at least one hit on a seeded corpus"
    assert not (hit_ids & anti_ids), (
        f"records served as BOTH hit and anti-hit: {hit_ids & anti_ids}"
    )

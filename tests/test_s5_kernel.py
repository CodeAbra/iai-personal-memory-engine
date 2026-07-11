from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

def _anchor(
    *,
    text: str = "User is Alice",
    vec: list[float] | None = None,
    s5_trust_score: float = 0.9,
    tier: str = "semantic",
    tags: list[str] | None = None,
    language: str = "en",
) -> MemoryRecord:
    if vec is None:
        vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=5,
        pinned=True,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=now,
        never_decay=True,
        never_merge=True,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or ["identity"]),
        language=language,
        s5_trust_score=s5_trust_score,
    )

class _FakeEmbedder:

    DIM = EMBED_DIM

    def embed(self, text: str) -> list[float]:
        return [1.0] + [0.0] * (EMBED_DIM - 1)

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]

@pytest.fixture(autouse=True)
def _patch_embedder(monkeypatch):
    from iai_mcp import embed as embed_mod

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)
    yield

def test_s5_constants():
    from iai_mcp import s5

    assert s5.IDENTITY_VIGILANCE_RHO == 0.99
    assert s5.S5_CONSENSUS_M == 3
    assert s5.S5_CONSENSUS_N == 5
    assert s5.COOLDOWN_HOURS == 48
    assert s5.TRUST_THRESHOLD_IDENTITY == 0.9

def test_s5_exports_propose_invariant_update():
    from iai_mcp import s5

    assert callable(getattr(s5, "propose_invariant_update", None))

def test_s5_exports_check_identity_anchor_on_write():
    from iai_mcp import s5

    assert callable(getattr(s5, "check_identity_anchor_on_write", None))

def test_propose_invariant_update_first_proposal_stages(tmp_path):
    from iai_mcp.s5 import propose_invariant_update
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)

    verdict, pid = propose_invariant_update(
        store, anchor.id, "new identity fact", session_id="s1"
    )
    assert verdict == "staged"
    assert isinstance(pid, UUID)

def test_propose_invariant_update_consensus_commits(tmp_path):
    from iai_mcp.s5 import propose_invariant_update
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)

    r1 = propose_invariant_update(store, anchor.id, "fact", "s1")
    r2 = propose_invariant_update(store, anchor.id, "fact", "s2")
    r3 = propose_invariant_update(store, anchor.id, "fact", "s3")
    assert r1[0] == "staged"
    assert r2[0] == "staged"
    assert r3[0] == "committed"

def test_propose_invariant_update_insufficient_consensus_rejected(tmp_path, monkeypatch):
    from iai_mcp import embed as embed_mod
    from iai_mcp.s5 import propose_invariant_update
    from iai_mcp.store import MemoryStore

    call_count = {"n": 0}

    class _AlternatingEmbedder:
        DIM = EMBED_DIM

        def embed(self, text):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return [1.0] + [0.0] * (EMBED_DIM - 1)
            vec = [0.0] * EMBED_DIM
            vec[call_count["n"] % EMBED_DIM] = 1.0
            return vec

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _AlternatingEmbedder)

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)

    verdicts = []
    for i in range(5):
        v, _ = propose_invariant_update(store, anchor.id, f"fact {i}", f"s{i}")
        verdicts.append(v)
    assert verdicts[-1] == "rejected"

def test_propose_invariant_update_cooldown(tmp_path):
    from iai_mcp.s5 import propose_invariant_update
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)

    propose_invariant_update(store, anchor.id, "fact", "s1")
    propose_invariant_update(store, anchor.id, "fact", "s2")
    verdict_commit, _ = propose_invariant_update(store, anchor.id, "fact", "s3")
    assert verdict_commit == "committed"

    verdict_next, pid = propose_invariant_update(
        store, anchor.id, "another fact", "s4"
    )
    assert verdict_next == "cooldown"
    assert pid is None

def test_propose_invariant_update_writes_event(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s5 import propose_invariant_update
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)

    propose_invariant_update(store, anchor.id, "fact", "s1")
    propose_invariant_update(store, anchor.id, "fact", "s2")
    propose_invariant_update(store, anchor.id, "fact", "s3")

    events = query_events(store, kind="s5_invariant_update")
    assert len(events) == 1
    ev = events[0]
    assert ev["data"]["anchor_id"] == str(anchor.id)
    assert "new_record_id" in ev["data"]
    assert "session_ids" in ev["data"]
    assert "agree_count" in ev["data"]

def test_propose_invariant_update_vigilance_099(tmp_path, monkeypatch):
    from iai_mcp import embed as embed_mod
    from iai_mcp.s5 import propose_invariant_update
    from iai_mcp.store import MemoryStore

    class _LowCosineEmbedder:
        DIM = EMBED_DIM
        _n = 0

        def embed(self, text):
            vec = [0.0] * EMBED_DIM
            vec[1] = 1.0
            return vec

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _LowCosineEmbedder)

    store = MemoryStore(path=tmp_path)
    anchor = _anchor()
    store.insert(anchor)

    verdicts = []
    for i in range(5):
        v, _ = propose_invariant_update(store, anchor.id, f"fact {i}", f"s{i}")
        verdicts.append(v)
    assert "committed" not in verdicts
    assert verdicts[-1] == "rejected"

def test_propose_invariant_update_unknown_anchor_rejected(tmp_path):
    from iai_mcp.s5 import propose_invariant_update
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    ghost = uuid4()
    verdict, pid = propose_invariant_update(store, ghost, "fact", "s")
    assert verdict == "rejected"
    assert pid is None

def test_check_identity_anchor_on_write_blocks_direct(tmp_path):
    from iai_mcp.s5 import check_identity_anchor_on_write
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    identity_rec = _anchor(s5_trust_score=0.95, tags=["identity"])
    ok, reason = check_identity_anchor_on_write(store, identity_rec, {})
    assert ok is False
    assert "identity-tier" in reason.lower() or "propose" in reason.lower()

def test_check_identity_anchor_on_write_allows_low_trust(tmp_path):
    from iai_mcp.s5 import check_identity_anchor_on_write
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _anchor(s5_trust_score=0.5)
    ok, reason = check_identity_anchor_on_write(store, rec, {})
    assert ok is True

def test_check_identity_anchor_on_write_allows_with_consensus_marker(tmp_path):
    from iai_mcp.s5 import check_identity_anchor_on_write
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _anchor(s5_trust_score=0.95, tags=["identity", "s5_consensus"])
    ok, reason = check_identity_anchor_on_write(store, rec, {})
    assert ok is True

def test_guarded_insert_blocks_direct_identity_write(tmp_path):
    from iai_mcp.store import MemoryStore
    from iai_mcp.write import guarded_insert

    store = MemoryStore(path=tmp_path)
    rec = _anchor(s5_trust_score=0.95)
    ok, reason = guarded_insert(store, rec, {})
    assert ok is False

def test_guarded_insert_allows_low_trust_write(tmp_path):
    from iai_mcp.store import MemoryStore
    from iai_mcp.write import guarded_insert

    store = MemoryStore(path=tmp_path)
    rec = _anchor(s5_trust_score=0.5)
    ok, reason = guarded_insert(store, rec, {})
    assert ok is True

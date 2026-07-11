from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


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


def _make_record(
    rid: UUID,
    surface: str = "topic",
    *,
    tier: str = "episodic",
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=surface,
        aaak_index="",
        embedding=list(embedding) if embedding is not None else [0.1] * EMBED_DIM,
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


def test_query_similar_accepts_tier_kwarg(store):
    rid_e = uuid4()
    rid_s = uuid4()
    store.insert(_make_record(rid_e, "episodic-cue", tier="episodic"))
    store.insert(_make_record(rid_s, "semantic-cue", tier="semantic"))

    embedding = [0.1] * EMBED_DIM
    out = store.query_similar(embedding, k=10, tier="episodic")
    ids = {r.id for r, _ in out}
    assert rid_e in ids, "episodic record should be returned by tier='episodic'"
    assert rid_s not in ids, "semantic record must be filtered out by tier='episodic'"

    with pytest.raises(ValueError):
        store.query_similar(embedding, k=10, tier="bogus")

    out_none = store.query_similar(embedding, k=10, tier=None)
    ids_none = {r.id for r, _ in out_none}
    assert rid_e in ids_none and rid_s in ids_none


def test_capture_turn_dedups_on_high_cos_match(store):
    text = "the user prefers Russian on the surface; English in storage"
    cue = "lang preference"

    r1 = capture_turn(
        store=store, text=text, cue=cue, tier="semantic",
        session_id="s1", role="user",
    )
    assert r1["status"] == "inserted", f"first capture should insert, got {r1}"

    r2 = capture_turn(
        store=store, text=text, cue=cue, tier="semantic",
        session_id="s1", role="user",
    )
    assert r2["status"] == "reinforced", f"second capture should reinforce, got {r2}"
    assert "cos=" in r2["reason"], f"reason should record cosine score, got {r2}"

    rows = list(store.iter_records())
    assert len([r for r in rows if r.tier == "semantic"]) == 1


def test_capture_turn_inserts_on_low_cos(store):
    r1 = capture_turn(
        store=store, text="apples are red", cue="apple",
        tier="episodic", session_id="s1", role="user",
    )
    r2 = capture_turn(
        store=store,
        text="quantum chromodynamics describes the strong force",
        cue="qcd", tier="episodic", session_id="s1", role="user",
    )
    assert r1["status"] == "inserted", f"first insert expected, got {r1}"
    assert r2["status"] == "inserted", f"second insert expected, got {r2}"

    rows = list(store.iter_records())
    assert len([r for r in rows if r.tier == "episodic"]) == 2


def test_reinforce_record_increments_edge_weight(store):
    rid = uuid4()
    store.insert(_make_record(rid, "anchor-record"))

    w1 = store.reinforce_record(rid)
    w2 = store.reinforce_record(rid)

    key = (str(rid), str(rid))
    assert key in w1, f"self-loop key missing from first call: {w1}"
    assert key in w2, f"self-loop key missing from second call: {w2}"
    assert w2[key] > w1[key], (
        f"weight must strictly increase across calls: w1={w1[key]} w2={w2[key]}"
    )

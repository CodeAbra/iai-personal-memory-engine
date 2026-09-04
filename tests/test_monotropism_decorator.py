"""Consumer 2 of monotropism_depth -- the monotropic-focus hit-reorder,
rekeyed off `community_id` resolved through the same boot-cached
community_names map the writer and the other consumers use, so the
`domain:`-tag reorder (which nothing ever emitted) becomes reachable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import core
from iai_mcp.core._serializers import _hit_to_json
from iai_mcp.response_decorator import _apply_monotropic_focus, apply_profile
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord
from tests._helpers import stub_embedder_for_store


@pytest.fixture(autouse=True)
def _restore_community_names():
    saved = dict(core._community_names_cache)
    yield
    core.set_community_names(saved)


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


def _hit(record_id: str, reason: str, community_id: str | None) -> dict:
    return {
        "record_id": record_id,
        "score": 0.5,
        "reason": reason,
        "literal_surface": "x",
        "adjacent_suggestions": [],
        "community_id": community_id,
    }


def test_promotes_exact_authority_hit_in_hot_topic():
    hot_id = uuid4()
    cold_id = uuid4()
    core.set_community_names({str(hot_id): "jazz", str(cold_id): "cooking"})
    state = {"monotropism_depth": {"jazz": 0.8}}
    resp = {
        "hits": [
            _hit("r-cold-1", "cosine 0.900", str(cold_id)),
            _hit("r-cold-2", "cosine 0.800", str(cold_id)),
            _hit("r-hot-authority", "exact-cosine", str(hot_id)),
        ],
        "anti_hits": [],
    }

    apply_profile(resp, state)

    ids = [h["record_id"] for h in resp["hits"]]
    assert ids[0] == "r-hot-authority", (
        f"exact-authority hit in the hot topic was not promoted to the front: {ids}"
    )
    assert set(ids[1:]) == {"r-cold-1", "r-cold-2"}


def test_noop_when_community_names_map_is_empty():
    core.set_community_names({})
    state = {"monotropism_depth": {"jazz": 0.9}}
    resp = {
        "hits": [
            _hit("r1", "cosine 0.9", str(uuid4())),
            _hit("r2", "cosine 0.8", str(uuid4())),
        ],
        "anti_hits": [],
    }

    apply_profile(resp, state)

    ids = [h["record_id"] for h in resp["hits"]]
    assert ids == ["r1", "r2"], f"empty map must never reorder hits: {ids}"


@pytest.mark.parametrize("depth", [0.0, 0.65, 0.7])
def test_noop_at_or_below_reachability_threshold(depth):
    hot_id = uuid4()
    core.set_community_names({str(hot_id): "jazz"})
    state = {"monotropism_depth": {"jazz": depth}}
    resp = {
        "hits": [
            _hit("r1", "cosine 0.9", str(uuid4())),
            _hit("r2", "exact-cosine", str(hot_id)),
        ],
        "anti_hits": [],
    }

    apply_profile(resp, state)

    ids = [h["record_id"] for h in resp["hits"]]
    assert ids == ["r1", "r2"], (
        f"depth <= 0.7 (auto cap is 0.65) must never reach the reorder: {ids}"
    )


def test_hit_without_community_id_sorts_non_hot_never_crashes():
    hot_id = uuid4()
    core.set_community_names({str(hot_id): "jazz"})
    state = {"monotropism_depth": {"jazz": 0.9}}
    resp = {
        "hits": [
            {"record_id": "r-no-cid", "score": 0.5, "reason": "cosine",
             "literal_surface": "x", "adjacent_suggestions": []},
            _hit("r-none-cid", "cosine", None),
            _hit("r-hot", "exact-cosine", str(hot_id)),
        ],
        "anti_hits": [],
    }

    apply_profile(resp, state)

    ids = [h["record_id"] for h in resp["hits"]]
    assert ids[0] == "r-hot"
    assert set(ids[1:]) == {"r-no-cid", "r-none-cid"}


def test_stable_sort_preserves_relative_order_within_bucket():
    hot_id = uuid4()
    core.set_community_names({str(hot_id): "jazz"})
    state = {"monotropism_depth": {"jazz": 0.9}}
    resp = {
        "hits": [
            _hit("r-hot-a", "cosine 0.6", str(hot_id)),
            _hit("r-cold-a", "cosine 0.9", str(uuid4())),
            _hit("r-hot-b", "exact-cosine", str(hot_id)),
            _hit("r-cold-b", "cosine 0.8", str(uuid4())),
        ],
        "anti_hits": [],
    }

    apply_profile(resp, state)

    ids = [h["record_id"] for h in resp["hits"]]
    assert ids == ["r-hot-a", "r-hot-b", "r-cold-a", "r-cold-b"], (
        f"stable sort must preserve original relative order within each bucket: {ids}"
    )


def test_community_names_resolved_once_per_decoration(monkeypatch):
    hot_id = uuid4()
    real_map = {str(hot_id): "jazz"}
    core.set_community_names(real_map)
    calls = []
    real_get = core.get_community_names

    def _counting_get():
        calls.append(1)
        return real_get()

    monkeypatch.setattr(core, "get_community_names", _counting_get)

    state = {"monotropism_depth": {"jazz": 0.9}}
    resp = {
        "hits": [
            _hit("r1", "cosine", str(uuid4())),
            _hit("r2", "cosine", str(uuid4())),
            _hit("r3", "exact-cosine", str(hot_id)),
            _hit("r4", "cosine", str(uuid4())),
            _hit("r5", "cosine", str(uuid4())),
        ],
        "anti_hits": [],
    }

    _apply_monotropic_focus(resp, state)

    assert len(calls) == 1, (
        f"get_community_names must resolve once before the sort, not per comparison: {len(calls)} calls"
    )


def test_hit_to_json_serializes_community_id():
    class _FakeHit:
        record_id = uuid4()
        score = 0.5
        reason = "cosine"
        literal_surface = "x"
        adjacent_suggestions = []
        valid_from = None
        valid_to = None
        session_id = None
        captured_at = None
        community_id = uuid4()

    out = _hit_to_json(_FakeHit())
    assert out["community_id"] == str(_FakeHit.community_id)


def test_hit_to_json_serializes_missing_community_id_as_none():
    class _FakeHitNoCommunity:
        record_id = uuid4()
        score = 0.5
        reason = "cosine"
        literal_surface = "x"
        adjacent_suggestions = []
        valid_from = None
        valid_to = None
        session_id = None
        captured_at = None

    out = _hit_to_json(_FakeHitNoCommunity())
    assert out["community_id"] is None


def test_hit_to_json_serializes_community_id_on_real_memoryhit():
    cid = uuid4()
    hit = MemoryHit(
        record_id=uuid4(),
        score=0.5,
        reason="cosine",
        literal_surface="x",
        adjacent_suggestions=[],
        community_id=cid,
    )
    out = _hit_to_json(hit)
    assert out["community_id"] == str(cid)


# ---------------------------------------------------------------------------
# Full-chain integration: real store -> dispatch -> serializer -> decorator.
#
# The dict-level tests above prove the decorator's own logic. This test
# proves the thing a scored-only partial fix would still pass: that the
# EXACT-AUTHORITY hit-construction site (core/__init__.py) actually carries
# community_id through to the response the decorator sees, not just the
# scored-hit site (pipeline.py).
# ---------------------------------------------------------------------------


class _StubEmbedder:
    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _blend_vec(base: list[float], other_seed: int, weight: float) -> list[float]:
    """A vector cosine-close to `base` but strictly less similar to it than
    `base` is to itself -- gives two authority hits distinct exact-cosine
    scores so their pre-decorator head order is deterministic."""
    other = np.array(_seeded_vec(other_seed))
    b = np.array(base)
    mixed = (1 - weight) * b + weight * other
    return (mixed / np.linalg.norm(mixed)).tolist()


def _make_record(rid, *, embedding, community_id=None, surface="x") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
        community_id=community_id,
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
        tags=["capture"],
        language="en",
    )


def test_authority_hit_carries_community_id_end_to_end_and_gets_promoted(tmp_path, monkeypatch):
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    cue_vec = _seeded_vec(1001)
    hot_id = uuid4()
    cold_id = uuid4()
    community_hot = uuid4()

    # cold's embedding IS the cue vector (cosine 1.0); hot's is a blend, so
    # its exact-cosine is strictly lower -- without the decorator, cold sorts
    # first in the authority head (never re-ranked by score).
    hot_rec = _make_record(hot_id, embedding=_blend_vec(cue_vec, 2002, 0.35), community_id=community_hot)
    cold_rec = _make_record(cold_id, embedding=cue_vec, community_id=None)

    store = MemoryStore(path=tmp_path / "store")
    store.insert(hot_rec)
    store.insert(cold_rec)
    for i in range(3):
        store.insert(_make_record(uuid4(), embedding=_seeded_vec(3000 + i), surface=f"filler {i}"))
    flush_record_buffer(store)

    _orig_query_similar = store.query_similar

    def _query_similar_missing_both(vec, *args, **kwargs):
        pairs = _orig_query_similar(vec, *args, **kwargs)
        return [(r, s) for r, s in pairs if getattr(r, "id", None) not in (hot_id, cold_id)]

    monkeypatch.setattr(store, "query_similar", _query_similar_missing_both)
    stub_embedder_for_store(monkeypatch, _StubEmbedder(cue_vec))

    # Pre-hydrate this store root so `dispatch`'s own (now community-name-
    # aware) hydration call is a no-op on the store's empty persisted map
    # and does not clobber the cache set immediately below.
    _core.ensure_profile_hydrated(store)
    core.set_community_names({str(community_hot): "jazz"})
    saved_profile = dict(_core._profile_state)
    _core._profile_state["monotropism_depth"] = {"jazz": 0.8}
    try:
        store._build_exact_index_sync()
        _pm._last_recall_latency_ms = 0.0
        resp = _core.dispatch(store, "memory_recall", {
            "cue": "irrelevant, embedder stubbed",
            "session_id": "monotropism-decorator-authority-test",
            "budget_tokens": 2000,
            "cue_embedding": cue_vec,
        })
    finally:
        _core._profile_state.clear()
        _core._profile_state.update(saved_profile)

    assert resp.get("exact_authority_used") is True, f"test setup: authority path did not fire: {resp}"
    hits = resp["hits"]
    authority_hits = [h for h in hits if h["reason"].startswith("exact-cosine")]
    authority_ids = {h["record_id"] for h in authority_hits}
    assert {str(hot_id), str(cold_id)}.issubset(authority_ids), (
        f"test setup: both records must surface via exact-authority: {hits}"
    )

    hot_hit = next(h for h in hits if h["record_id"] == str(hot_id))
    assert hot_hit["community_id"] == str(community_hot), (
        "the exact-authority construction site must carry community_id "
        "through to the serialized hit"
    )
    assert hits[0]["record_id"] == str(hot_id), (
        f"the lower-cosine authority hit in the hot topic must be promoted "
        f"to the front by the decorator, ahead of the higher-cosine cold "
        f"authority hit: {[h['record_id'] for h in hits]}"
    )

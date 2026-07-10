from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def test_theta_skip_constant():
    from iai_mcp.gate import THETA_SKIP

    assert THETA_SKIP == 0.2


def test_efer_empty_is_zero():
    from iai_mcp.gate import expected_free_energy_reduction

    assert expected_free_energy_reduction("") == 0.0


def test_efer_trivial_greeting_is_below_theta():
    from iai_mcp.gate import THETA_SKIP, expected_free_energy_reduction

    for cue in ("hi", "hello", "thanks", "ok", "yes", "no"):
        val = expected_free_energy_reduction(cue)
        assert val < THETA_SKIP, f"cue={cue!r} val={val}"


def test_efer_rich_is_above_theta():
    from iai_mcp.gate import THETA_SKIP, expected_free_energy_reduction

    rich = (
        "explain how CLS replay interacts with schema induction under "
        "monotropic attention"
    )
    val = expected_free_energy_reduction(rich)
    assert val > THETA_SKIP


def test_should_skip_retrieval_trivial():
    from iai_mcp.gate import should_skip_retrieval

    skip, reason = should_skip_retrieval("hi")
    assert skip is True
    assert reason


def test_should_skip_retrieval_informative():
    from iai_mcp.gate import should_skip_retrieval

    skip, _reason = should_skip_retrieval(
        "What did we discuss about auth last week?"
    )
    assert skip is False


def test_should_skip_very_short_cue():
    from iai_mcp.gate import should_skip_retrieval

    skip, _ = should_skip_retrieval("a")
    assert skip is True
    skip, _ = should_skip_retrieval("")
    assert skip is True


def test_pipeline_recall_skip_path_returns_minimal_response(tmp_path, monkeypatch):
    from iai_mcp import embed as embed_mod
    from iai_mcp.core import _seed_l0_identity, dispatch

    class _FakeEmbedder:
        DIM = EMBED_DIM
        DEFAULT_DIM = EMBED_DIM
        DEFAULT_MODEL_KEY = "fake"

        def __init__(self, *args, **kwargs):
            self.DIM = EMBED_DIM

        def embed(self, text: str) -> list[float]:
            return [1.0] + [0.0] * (EMBED_DIM - 1)

        def embed_batch(self, texts):
            return [self.embed(t) for t in texts]

    monkeypatch.setattr(embed_mod, "Embedder", _FakeEmbedder)

    store = MemoryStore(path=tmp_path)
    _seed_l0_identity(store)
    now = datetime.now(timezone.utc)
    for i in range(3):
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"extra fact {i}",
            aaak_index="",
            embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
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
        store.insert(rec)

    resp = dispatch(store, "memory_recall", {"cue": "hi", "session_id": "s-trivial"})
    assert "budget_used" in resp
    assert resp["budget_used"] < 200

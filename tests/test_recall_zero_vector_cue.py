"""The SLEEP/exception fallback dispatchers pad an absent cue_embedding with
zeros; recall must embed the cue TEXT itself — ranking by distance to the
zero vector returns cue-irrelevant memories, breaking 'the hippocampus
answers correctly on every path' exactly during consolidation."""
from __future__ import annotations

import pytest


@pytest.fixture
def seeded_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    texts = [
        "The quarterly budget review moved to Thursday afternoon.",
        "Hippocampal replay consolidates memories during deep sleep.",
        "The rust compiler caches incremental build artifacts on disk.",
    ]
    ids = {}
    for i, t in enumerate(texts):
        res = capture_turn(
            store, cue="", text=t, tier="episodic",
            session_id=f"seed-{i}", role="user",
        )
        assert res["status"] == "inserted", res
        ids[t] = res["record_id"]
    yield store, ids
    store.close()


def test_zero_vector_cue_is_reembedded_from_text(seeded_store):
    from iai_mcp.retrieve import recall
    from iai_mcp.types import EMBED_DIM

    store, ids = seeded_store
    target = "Hippocampal replay consolidates memories during deep sleep."

    resp = recall(
        store,
        [0.0] * EMBED_DIM,
        "how does sleep consolidate hippocampal memories?",
        "sess-zero-cue",
        k_hits=3,
    )
    assert resp.hits, "recall must return hits on a seeded corpus"
    top = resp.hits[0]
    assert str(top.record_id) == ids[target], (
        f"zero-vector cue must be re-embedded from text; top hit was "
        f"{top.literal_surface!r} (score {top.score:.3f})"
    )
    assert top.score > 0.3, "re-embedded cue must produce a real cosine"


def test_absent_cue_embedding_is_embedded_from_text(seeded_store):
    from iai_mcp.retrieve import recall

    store, ids = seeded_store
    resp = recall(
        store,
        [],
        "incremental rust build artifact caching",
        "sess-empty-cue",
        k_hits=3,
    )
    assert resp.hits
    assert str(resp.hits[0].record_id) == ids[
        "The rust compiler caches incremental build artifacts on disk."
    ]

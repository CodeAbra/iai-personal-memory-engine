"""Regression gate for the zero-vector-sink CLASS: any write or recall path
whose embedding defaults to zeros produces records that are semantically
invisible (cosine ~0 to everything) while occupying ANN slots. Two review
rounds found members of this class one at a time; these tests pin every
known sink shut."""
from __future__ import annotations

import pytest


@pytest.fixture
def seeded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    res = capture_turn(
        store, cue="",
        text="The deployment pipeline uses blue-green switching on Fridays.",
        tier="episodic", session_id="seed", role="user",
    )
    assert res["status"] == "inserted"
    yield store, res["record_id"]
    store.close()


def test_contradict_without_vector_stores_a_real_embedding(seeded):
    """memory_contradict's dispatcher pads an omitted vector with zeros; the
    corrector must be re-embedded or the CURRENT belief becomes semantically
    unfindable while the superseded one keeps surfacing."""
    from uuid import UUID

    from iai_mcp.retrieve import contradict
    from iai_mcp.types import EMBED_DIM

    store, old_id = seeded
    receipt = contradict(
        store,
        UUID(old_id),
        "The deployment pipeline switched to canary releases on Mondays.",
        [0.0] * EMBED_DIM,
    )
    new_rec = store.get(receipt.new_record_id)
    assert new_rec is not None
    assert any(v != 0.0 for v in new_rec.embedding), (
        "corrector stored with a zero embedding — semantically invisible"
    )
    # And it must actually surface on a matching cue.
    from iai_mcp.embed import embedder_for_store

    cue_vec = list(embedder_for_store(store).embed(
        "canary release deployment schedule"
    ))
    got = store.query_similar(cue_vec, k=3)
    assert any(str(r.id) == str(receipt.new_record_id) for r, _ in got), (
        "re-embedded corrector must be ANN-findable"
    )


def test_l0_identity_seed_has_real_embedding(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    from iai_mcp.core._identity import L0_ID, _seed_l0_identity
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    try:
        _seed_l0_identity(store)
        seed = store.get(L0_ID)
        assert seed is not None
        assert any(v != 0.0 for v in seed.embedding), (
            "identity anchor stored with a zero embedding — never surfaces "
            "on identity cues"
        )
    finally:
        store.close()


def test_reflection_embed_failure_skips_insert(tmp_path, monkeypatch):
    """A reflection whose embed failed must NOT be inserted (zero-embedded
    junk); the pipeline step skips and reports."""
    from iai_mcp import dmn_reflection as dmn
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    store = MemoryStore(path=tmp_path)
    try:
        import iai_mcp.embed as embed_mod

        def _boom(_store):
            raise RuntimeError("encoder down")

        monkeypatch.setattr(embed_mod, "embedder_for_store", _boom)
        rec = dmn.ReflectionAgent().synthesize(store, window_hours=24)
        assert rec.provenance[0].get("embed_failed") is True
    finally:
        store.close()

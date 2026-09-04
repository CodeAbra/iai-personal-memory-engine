"""Contract: `retrieve.recall` never serves a tier=procedural record's
literal_surface, in any cue mode -- not only under mode="verbatim".

`retrieve.recall` is a second, independent recall implementation from
`core._passes_mode_filter` (which excludes procedural unconditionally). It
backs several fallback dispatch sites and `_first_turn_recall_hook`, which
hardcodes mode="concept" and fires on every session's first cue-bearing
recall -- the most-reachable leak path of the two.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest import mock
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import retrieve
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

PROC_SURFACE = "PROCEDURAL SENTINEL SURFACE never visible as text"
CONTROL_SURFACE = "control episodic surface, must remain visible"


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


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, tier: str, surface: str, embedding: list[float], tags=None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=surface,
        aaak_index="",
        embedding=list(embedding),
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
        tags=tags or ["capture"],
        language="en",
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def _plant_procedural_and_control(store: MemoryStore, cue_vec: list[float]) -> tuple[UUID, UUID]:
    proc_id = uuid4()
    control_id = uuid4()
    store.insert(_make_rec(proc_id, "procedural", PROC_SURFACE, cue_vec, tags=["chunk", "source:cofire"]))
    store.insert(_make_rec(control_id, "episodic", CONTROL_SURFACE, cue_vec))
    flush_record_buffer(store)
    return proc_id, control_id


def test_retrieve_recall_excludes_procedural_in_concept_mode(store):
    cue_vec = _seeded_vec(42)
    proc_id, control_id = _plant_procedural_and_control(store, cue_vec)

    response = retrieve.recall(
        store=store,
        cue_embedding=cue_vec,
        cue_text="irrelevant, cue vector is planted directly",
        session_id="cr01-concept-mode",
        mode="concept",
        allow_cue_reembed=False,
    )

    served_ids = {h.record_id for h in response.hits} | {h.record_id for h in response.anti_hits}
    served_surfaces = {h.literal_surface for h in response.hits} | {
        h.literal_surface for h in response.anti_hits
    }

    assert proc_id not in served_ids, f"procedural record leaked into served hit_ids: {served_ids}"
    assert PROC_SURFACE not in served_surfaces, (
        f"procedural literal_surface leaked into served text: {served_surfaces}"
    )
    assert control_id in served_ids, "control episodic record must still be served (proves filter isn't blanket)"


def test_first_turn_recall_hook_excludes_procedural(store):
    """The most-reachable live path: `_first_turn_recall_hook` hardcodes
    mode="concept" and calls the real `retrieve.recall` on every session's
    first cue-bearing recall -- no degraded state needed to trigger it."""
    from iai_mcp import core as _core

    cue_vec = _seeded_vec(43)
    proc_id, control_id = _plant_procedural_and_control(store, cue_vec)

    response: dict = {}
    params = {
        "session_id": "cr01-first-turn",
        "cue": "irrelevant, cue vector is planted directly",
        "cue_embedding": cue_vec,
    }

    with mock.patch("iai_mcp.daemon_state.consume_first_turn", return_value=True), \
         mock.patch("iai_mcp.daemon_state.load_state", return_value={}), \
         mock.patch("iai_mcp.hippea_cascade.snapshot_warm_ids", return_value=[uuid4()]):
        _core._first_turn_recall_hook(response, params=params, store=store)

    assert "first_turn_recall" in response
    served = response["first_turn_recall"]["hits"]
    served_ids = {h["record_id"] for h in served}
    served_surfaces = {h["literal_surface"] for h in served}

    assert str(proc_id) not in served_ids, (
        f"procedural record leaked via _first_turn_recall_hook: {served_ids}"
    )
    assert PROC_SURFACE not in served_surfaces
    assert str(control_id) in served_ids, "control episodic record must still be served"

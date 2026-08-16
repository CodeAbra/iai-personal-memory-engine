"""Consumers 1 and 3 of monotropism_depth, rekeyed from `domain:` tags onto
topic names resolved through the boot-cached community_names map -- the
same resolution on write (the tuner) and read (these consumers), so they
never drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp import core
from iai_mcp.events import query_events
from iai_mcp.lilli.profile.knobs import profile_modulation_for_record
from iai_mcp.s4 import S4_MONOTROPIC_THETA, monotropic_proactive_check
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(autouse=True)
def _restore_community_names():
    saved = dict(core._community_names_cache)
    yield
    core.set_community_names(saved)


def _record(*, community_id=None, detail_level: int = 2, vec=None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="x",
        aaak_index="",
        embedding=vec if vec is not None else [0.1] * EMBED_DIM,
        community_id=community_id,
        centrality=0.0,
        detail_level=detail_level,
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


@dataclass
class _ViewWithoutCommunityId:
    """A lightweight record view lacking `community_id` entirely (the
    graph-sourced fast path) -- consumer 1 must degrade, never raise."""

    tags: list


# ---------------------------------------------------------------------------
# Consumer 1: profile_modulation_for_record
# ---------------------------------------------------------------------------


def test_consumer1_applies_gain_when_community_resolves_to_dict_key() -> None:
    cid = uuid4()
    core.set_community_names({str(cid): "music"})
    rec = _record(community_id=cid)
    state = {"monotropism_depth": {"music": 0.4}}

    gains = profile_modulation_for_record(rec, state)

    assert gains["monotropism_depth"] == pytest.approx(1.4)


def test_consumer1_no_gain_when_community_id_is_none() -> None:
    rec = _record(community_id=None)
    state = {"monotropism_depth": {"music": 0.4}}

    gains = profile_modulation_for_record(rec, state)

    assert "monotropism_depth" not in gains


def test_consumer1_no_gain_when_community_id_absent_from_map() -> None:
    core.set_community_names({})
    rec = _record(community_id=uuid4())
    state = {"monotropism_depth": {"music": 0.4}}

    gains = profile_modulation_for_record(rec, state)

    assert "monotropism_depth" not in gains


def test_consumer1_no_gain_when_resolved_name_absent_from_dict() -> None:
    cid = uuid4()
    core.set_community_names({str(cid): "film"})
    rec = _record(community_id=cid)
    state = {"monotropism_depth": {"music": 0.4}}

    gains = profile_modulation_for_record(rec, state)

    assert "monotropism_depth" not in gains


def test_consumer1_never_raises_on_view_without_community_id_attribute() -> None:
    cid = uuid4()
    core.set_community_names({str(cid): "music"})
    view = _ViewWithoutCommunityId(tags=[])
    state = {"monotropism_depth": {"music": 0.4}}

    gains = profile_modulation_for_record(view, state)

    assert "monotropism_depth" not in gains


# ---------------------------------------------------------------------------
# Consumer 3: s4.monotropic_proactive_check
# ---------------------------------------------------------------------------


def test_consumer3_gate_reachable_only_above_theta(tmp_path) -> None:
    cid = uuid4()
    core.set_community_names({str(cid): "music"})
    store = MemoryStore(path=tmp_path)
    rec = _record(community_id=cid, detail_level=5)
    store.insert(rec)

    below = monotropic_proactive_check(
        store, rec, {"monotropism_depth": {"music": S4_MONOTROPIC_THETA}}, session_id="t",
    )
    assert below == []


def test_consumer3_and_consumer1_resolve_the_same_name_for_the_same_record(tmp_path) -> None:
    """Write/read parity: the same community_id resolves to the same map
    key for both consumers, so neither silently drifts from the other."""
    cid = uuid4()
    core.set_community_names({str(cid): "music"})
    depth = 0.9  # above S4_MONOTROPIC_THETA (0.7), a manual profile_set value
    state = {"monotropism_depth": {"music": depth}}

    rec = _record(community_id=cid, detail_level=5)
    store = MemoryStore(path=tmp_path)
    store.insert(rec)

    gains = profile_modulation_for_record(rec, state)
    assert gains["monotropism_depth"] == pytest.approx(1.0 + depth)

    hints = monotropic_proactive_check(store, rec, state, session_id="t")
    # No sibling records share the resolved name, so no contradiction fires
    # -- but the gate itself (name resolves, depth clears theta) must not
    # short-circuit before the detail/pairwise checks.
    assert hints == []


def test_consumer3_same_domain_rebuilt_by_resolved_name_not_by_tag(tmp_path) -> None:
    music_cid = uuid4()
    film_cid = uuid4()
    core.set_community_names({str(music_cid): "music", str(film_cid): "film"})
    store = MemoryStore(path=tmp_path)

    music_sibling = _record(community_id=music_cid, detail_level=2)
    film_sibling = _record(community_id=film_cid, detail_level=2)
    store.insert(music_sibling)
    store.insert(film_sibling)

    new_rec = _record(community_id=music_cid, detail_level=5)
    store.insert(new_rec)

    state = {"monotropism_depth": {"music": 0.9}}
    # A near-duplicate embedding on the music sibling should be scored; the
    # film sibling (different resolved name) must never enter same_domain.
    music_sibling.embedding = list(new_rec.embedding)
    store.update(music_sibling)

    hints = monotropic_proactive_check(store, new_rec, state, session_id="t")
    source_ids = {sid for h in hints for sid in h["source_ids"]}
    assert str(film_sibling.id) not in source_ids


def test_consumer3_skip_event_carries_no_topic_name(tmp_path) -> None:
    cid = uuid4()
    core.set_community_names({str(cid): "music"})
    store = MemoryStore(path=tmp_path)
    for i in range(101):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        store.insert(_record(community_id=cid, detail_level=1, vec=vec))

    new_rec = _record(community_id=cid, detail_level=5)
    store.insert(new_rec)

    state = {"monotropism_depth": {"music": 0.9}}
    result = monotropic_proactive_check(store, new_rec, state, session_id="t")
    assert result == []

    events = query_events(store, kind="s4_monotropic_skip")
    assert events
    assert "domain" not in events[0]["data"]
    assert not events[0]["domain"]
    assert "music" not in str(events[0]["data"])

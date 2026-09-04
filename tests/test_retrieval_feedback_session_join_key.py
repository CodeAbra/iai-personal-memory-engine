"""Unattributed session ids never join across the retrieval-feedback pair.

memory_recall and memory_reinforce record DIFFERENT literals when the caller
supplies no session id ("unknown" and "-"). Both are unattributed, and the
nightly pairing helper refuses to join on either: an unattributable row is
dropped, never cross-paired with another session's recall.

The two producer literals are pinned here on purpose. bench/proc_corpus_census.py
partitions its counted population on "-", so moving recall onto that literal
would silently change what that census counts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp import core
from iai_mcp.events import (
    UNATTRIBUTED_SESSION_IDS,
    is_attributed_session,
    query_events,
)
from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune import _pair_retrieval_feedback
from iai_mcp.lilli.profile.retrieval_tuning import observe_retrieval_weight
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _row(ts: datetime, session_id: str, data: dict) -> dict:
    return {"ts": ts, "session_id": session_id, "data": data}


def test_unattributed_set_covers_both_historical_literals() -> None:
    assert "-" in UNATTRIBUTED_SESSION_IDS
    assert "unknown" in UNATTRIBUTED_SESSION_IDS
    assert is_attributed_session("a-real-session") is True
    for value in ("-", "unknown", "", None):
        assert is_attributed_session(value) is False


def test_pairing_drops_every_unattributed_literal() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for sid in sorted(UNATTRIBUTED_SESSION_IDS):
        used_rows = [_row(t0, sid, {"hit_ids": ["h1", "h2"]})]
        reinforced_rows = [_row(t0 + timedelta(seconds=5), sid, {"reinforced_ids": ["h1"]})]

        assert _pair_retrieval_feedback(reinforced_rows, used_rows) == [], (
            f"session_id={sid!r} must never join"
        )


def test_unattributed_rows_cannot_cross_pair_across_sessions() -> None:
    """Two unrelated callers that both omitted session_id land in the same
    bucket; the helper must abstain rather than pair them."""
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    used_rows = [
        _row(t0, "unknown", {"hit_ids": ["alice-hit"]}),
        _row(t0 + timedelta(seconds=1), "unknown", {"hit_ids": ["bob-hit"]}),
    ]
    reinforced_rows = [
        _row(t0 + timedelta(seconds=2), "unknown", {"reinforced_ids": ["bob-hit"]}),
    ]

    window_rows = _pair_retrieval_feedback(reinforced_rows, used_rows)

    assert window_rows == []
    observed, n, _signal = observe_retrieval_weight(window_rows)
    assert n == 0
    assert observed == 0.0


def test_a_supplied_session_id_still_joins() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    hit_ids = ["h-0", "h-1", "h-2", "h-3"]
    used_rows = [_row(t0, "sess-7", {"hit_ids": hit_ids})]
    reinforced_rows = [
        _row(t0 + timedelta(seconds=5), "sess-7", {"reinforced_ids": hit_ids[:2]}),
    ]

    window_rows = _pair_retrieval_feedback(reinforced_rows, used_rows)

    assert len(window_rows) == 1
    observed, n, _signal = observe_retrieval_weight(window_rows)
    assert n == 1
    assert observed == pytest.approx(0.5)


def _insert_probe_record(store: MemoryStore, now: datetime) -> tuple[list[float], str]:
    vec = [0.0] * EMBED_DIM
    vec[0] = 1.0
    rec_id = uuid4()
    store.insert(
        MemoryRecord(
            id=rec_id, tier="episodic", literal_surface="join key probe", aaak_index="",
            embedding=vec, community_id=None, centrality=0.0, detail_level=2, pinned=False,
            stability=0.0, difficulty=0.0, last_reviewed=None, never_decay=False,
            never_merge=False, provenance=[], created_at=now, updated_at=now, tags=[],
            language="en",
        )
    )
    return vec, str(rec_id)


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_default_session_ids_stay_per_producer_and_never_join(
    tmp_path, monkeypatch, driver
) -> None:
    from tests._helpers import stub_embedder_for_store

    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)
    vec, rec_id = _insert_probe_record(store, now)

    class _FixedEmbedder:
        def embed(self, text: str) -> list[float]:
            return list(vec)

    stub_embedder_for_store(monkeypatch, _FixedEmbedder())

    resp = core.dispatch(store, "memory_recall", {
        "cue": "join key probe", "budget_tokens": 2000, "cue_embedding": vec,
    })
    assert "error" not in resp
    assert resp["hits"], "expected a hit for the cosine=1.0 probe record"

    core.dispatch(store, "memory_reinforce", {"ids": [rec_id, str(uuid4())]})

    from iai_mcp.events import flush_event_buffer

    flush_event_buffer(store)

    used = query_events(store, kind="retrieval_used", limit=20)
    reinforced = query_events(store, kind="retrieval_reinforced", limit=20)
    assert used, "recall must emit retrieval_used"
    assert reinforced, "reinforce must emit retrieval_reinforced"

    used_sids = {row.get("session_id") for row in used}
    reinforced_sids = {row.get("session_id") for row in reinforced}

    # Producer literals are pinned: moving recall onto "-" would silently change
    # what bench/proc_corpus_census.py counts.
    assert used_sids == {"unknown"}, used_sids
    assert reinforced_sids == {"-"}, reinforced_sids
    assert not any(is_attributed_session(sid) for sid in used_sids | reinforced_sids)
    assert (used_sids | reinforced_sids) <= UNATTRIBUTED_SESSION_IDS

    assert _pair_retrieval_feedback(reinforced, used) == []

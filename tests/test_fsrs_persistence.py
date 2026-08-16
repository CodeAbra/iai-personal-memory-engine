from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _record(
    *,
    text: str = "fsrs-target",
    stability: float = 0.1,
    prov_seconds_ago: int = 30,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    prov_ts = (now - timedelta(seconds=prov_seconds_ago)).isoformat()
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=stability,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": prov_ts, "cue": "recall", "session_id": "s1"}],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def test_update_record_writes_back_fsrs_columns(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _record(stability=0.1)
    store.insert(rec)

    rec.stability = 0.55
    rec.difficulty = 0.42
    new_reviewed = datetime.now(timezone.utc)
    rec.last_reviewed = new_reviewed

    store.update_record(rec)

    fresh = store.get(rec.id)
    assert fresh is not None
    assert fresh.stability == pytest.approx(0.55, abs=1e-3)
    assert fresh.difficulty == pytest.approx(0.42, abs=1e-3)
    assert fresh.last_reviewed is not None


def test_update_record_rejects_unknown_id(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    phantom = _record(stability=0.9)

    before = store.db.open_table("records").count_rows()

    store.update_record(phantom)

    after = store.db.open_table("records").count_rows()
    assert after == before


def test_update_record_does_not_touch_untouched_columns(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _record(stability=0.1)
    rec.tags = ["important", "keep-me"]
    rec.provenance = [
        {"ts": "2026-04-16T00:00:00Z", "cue": "seed", "session_id": "s0"},
    ]
    store.insert(rec)

    rec.stability = 0.6
    rec.last_reviewed = datetime.now(timezone.utc)
    store.update_record(rec)

    fresh = store.get(rec.id)
    assert fresh is not None
    assert fresh.stability == pytest.approx(0.6, abs=1e-3)
    assert fresh.tags == ["important", "keep-me"]
    assert len(fresh.provenance) == 1
    assert fresh.provenance[0]["cue"] == "seed"

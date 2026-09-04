"""Round-trip proof for MemoryRecord.epistemic_status through store.insert()/get().

Guards against the dead-column class of bug: a schema column that reconciles
cleanly but is never wired into _to_row/_from_row, leaving the field
permanently unreadable regardless of what was written.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _record(*, epistemic_status: str | None = None, text: str = "the sky is blue") -> MemoryRecord:
    kwargs = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.01] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    if epistemic_status is not None:
        kwargs["epistemic_status"] = epistemic_status
    return MemoryRecord(**kwargs)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_epistemic_status_round_trips_explicit_value(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    rec = _record(epistemic_status="estimate")
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.epistemic_status == "estimate"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_epistemic_status_defaults_to_unknown(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    rec = _record()
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.epistemic_status == "unknown"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_literal_surface_byte_identical_alongside_epistemic_status(tmp_path, monkeypatch, driver):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)
    store = MemoryStore(path=tmp_path)

    text = "line1\x00line2 \U0001f9e0 漢字 العربية"
    rec = _record(epistemic_status="hypothesis", text=text)
    store.insert(rec)

    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == text
    assert got.epistemic_status == "hypothesis"

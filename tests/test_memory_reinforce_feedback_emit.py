"""memory_reinforce writes a durable, session-scoped retrieval_reinforced
event the nightly tuner reads back, joined against retrieval_used by
session_id.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from iai_mcp.core import dispatch
from iai_mcp.events import flush_event_buffer, query_events
from iai_mcp.store import MemoryStore
from tests.test_store import _make

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


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_reinforce_emits_one_queryable_event_with_session_id(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    recs = [_make(), _make()]
    for r in recs:
        store.insert(r)
    ids = [str(r.id) for r in recs]

    dispatch(store, "memory_reinforce", {"ids": ids, "session_id": "sess-42"})
    flush_event_buffer(store)

    events = query_events(store, kind="retrieval_reinforced")
    assert len(events) == 1
    data = events[0]["data"]
    assert data["session_id"] == "sess-42"
    assert data["reinforced_ids"] == ids
    datetime.fromisoformat(data["timestamp"])


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_reinforce_without_session_id_defaults_to_dash(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)
    recs = [_make(), _make()]
    for r in recs:
        store.insert(r)
    ids = [str(r.id) for r in recs]

    dispatch(store, "memory_reinforce", {"ids": ids})
    flush_event_buffer(store)

    events = query_events(store, kind="retrieval_reinforced")
    assert len(events) == 1
    assert events[0]["data"]["session_id"] == "-"


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_reinforce_malformed_id_raises_and_emits_nothing(tmp_path, monkeypatch, driver):
    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path)

    with pytest.raises(ValueError):
        dispatch(store, "memory_reinforce", {"ids": ["not-a-uuid"], "session_id": "sess-x"})

    flush_event_buffer(store)
    assert query_events(store, kind="retrieval_reinforced") == []

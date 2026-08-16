from __future__ import annotations

from iai_mcp.events import write_event
from iai_mcp.store import MemoryStore
from iai_mcp.trajectory import m4_profile_movement_live

def test_m4_zero_on_empty_store(tmp_path):
    store = MemoryStore(path=tmp_path)
    assert m4_profile_movement_live(store) == 0.0

def test_m4_reads_latest_moved_count(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(store, kind="profile_tuned", data={"moved_count": 2}, severity="info")
    write_event(store, kind="profile_tuned", data={"moved_count": 5}, severity="info")
    assert m4_profile_movement_live(store) == 5.0

def test_m4_skips_non_numeric_and_missing_moved_count(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(store, kind="profile_tuned", data={"moved_count": 3}, severity="info")
    write_event(store, kind="profile_tuned", data={"moved_count": True}, severity="info")
    write_event(store, kind="profile_tuned", data={"moved_count": "nope"}, severity="info")
    write_event(store, kind="profile_tuned", data={}, severity="info")
    assert m4_profile_movement_live(store) == 3.0

def test_m4_never_reads_profile_updated(tmp_path):
    store = MemoryStore(path=tmp_path)
    write_event(
        store, kind="profile_updated",
        data={"knob": "interest_boost", "old": 0.0, "new": 1.0},
        severity="info",
    )
    assert m4_profile_movement_live(store) == 0.0

"""write_continuity_cache's downgrade-guard must distinguish an EXPLICIT
caller-initiated focus=""/next_action="" clear on a still-open task from an
INCIDENTAL task-switch thin-park. isinstance("", str) is True, so an empty
string reaches update_task as a real value either way -- only the
explicit_clear signal (threaded update_task -> _persist_entry ->
write_continuity_cache(allow_downgrade=...)) tells the two apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp import session
from iai_mcp import working_tier as wt
from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "lancedb")
    yield s
    s.close()


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_explicit_clear_removes_retracted_focus_from_eager_file(driver, store, monkeypatch):
    """Drives the real memory_capture RPC dispatch, not a hand-flagged
    direct call -- this is the ONLY caller of update_task's explicit_clear
    kwarg, and its derivation (a literal empty focus/next_action, not the
    is-None unset gate) is what must be under test."""
    _select_driver(driver, monkeypatch)

    wt.open_task("investigate the recall regression", session_id="sess-x")
    wt.update_task(
        focus="the dispatch hot path",
        next_action="profile with the sampler",
        session_id="sess-x",
        store=store,
    )
    cache_path = session._continuity_cache_path(store)
    seeded = cache_path.read_text(encoding="utf-8")
    assert "focus: the dispatch hot path" in seeded
    assert "next action: profile with the sampler" in seeded

    result = dispatch(
        store, "memory_capture",
        {
            "text": "folding this sub-step, nothing queued yet",
            "cue": "c",
            "session_id": "sess-x",
            "role": "user",
            "focus": "",
            "next_action": "",
        },
    )
    assert result["status"] == "inserted", result

    cleared = cache_path.read_text(encoding="utf-8")
    assert "focus: the dispatch hot path" not in cleared
    assert "next action: profile with the sampler" not in cleared
    assert "focus: " not in cleared
    assert "next action: " not in cleared


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_incidental_thin_park_still_preserves_prior_block(driver, store, monkeypatch):
    _select_driver(driver, monkeypatch)

    wt.open_task("session A's original goal", session_id="sess-a")
    wt.update_task(
        focus="A's real focus",
        next_action="A's real next action",
        session_id="sess-a",
        store=store,
    )
    cache_path = session._continuity_cache_path(store)
    seeded = cache_path.read_text(encoding="utf-8")
    assert "focus: A's real focus" in seeded
    assert "next action: A's real next action" in seeded

    wt.open_task("session B's fresh goal", session_id="sess-b")
    wt.update_task(sub_goal="a step for B", session_id="sess-b", store=store)

    after_switch = cache_path.read_text(encoding="utf-8")
    assert "focus: A's real focus" in after_switch, (
        "an incidental task-switch thin-park must never clobber a "
        "substantive prior block"
    )
    assert "next action: A's real next action" in after_switch

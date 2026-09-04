"""The live-state block delivers unconditionally under every wake depth,
including minimal -- no depth gate suppresses the current focal goal or the
immediate next action.
"""

from __future__ import annotations

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.community import CommunityAssignment
from iai_mcp.session import _compose_session_start_payload, format_payload_as_markdown
from iai_mcp.store import MemoryStore


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


@pytest.mark.parametrize("wake_depth", ["minimal", "standard", "deep"])
def test_live_state_delivers_under_every_wake_depth(wake_depth, tmp_path):
    wt.open_task("wake-depth coverage goal")
    wt.update_task(next_action="deliver under every wake depth")

    store = MemoryStore(path=tmp_path)
    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        session_id="s1", profile_state={"wake_depth": wake_depth},
    )
    assert "wake-depth coverage goal" in payload.live_state
    assert "deliver under every wake depth" in payload.live_state

    rendered = format_payload_as_markdown(payload)
    assert "## Session continuity (always active)" in rendered
    assert "wake-depth coverage goal" in rendered


def test_minimal_wake_depth_does_not_drop_live_state_even_though_l0_l1_are_suppressed(tmp_path):
    wt.open_task("minimal-mode goal")
    wt.update_task(next_action="stay present on the minimal path")

    store = MemoryStore(path=tmp_path)
    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        session_id="s1", profile_state={"wake_depth": "minimal"},
    )
    assert payload.l0 == ""
    assert payload.l1 == ""
    assert "stay present on the minimal path" in payload.live_state

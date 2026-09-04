"""Session-continuity block renders at session start with no cue/rank/
similarity gate -- a direct read of the maintained live-state record,
distinct from the standing-orders block, sanitized against marker
smuggling, empty when no focal task is live.
"""

from __future__ import annotations

import inspect

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.community import CommunityAssignment
from iai_mcp.session import (
    _compose_session_start_payload,
    format_payload_as_markdown,
    render_live_state_segment,
)
from iai_mcp.store import MemoryStore


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


def test_render_live_state_segment_empty_with_no_focal_task():
    assert render_live_state_segment() == ""


def test_render_live_state_segment_contains_goal_focus_next_action():
    wt.open_task("ship the recall gate")
    wt.update_task(focus="budget regression", next_action="fill the acceptance test")

    rendered = render_live_state_segment()
    assert "ship the recall gate" in rendered
    assert "budget regression" in rendered
    assert "fill the acceptance test" in rendered


def test_render_live_state_segment_has_no_store_or_rank_gate_parameter():
    # A direct global read never accepts a store/cue/embedding argument --
    # mechanically proves there is no cue/rank/similarity gate on the path.
    # Render-only toggles (e.g. fold_sensory) are legitimate and excluded
    # from this check.
    forbidden = {"store", "session_id", "cue", "rank", "embedding"}
    sig = inspect.signature(render_live_state_segment)
    assert not forbidden & set(sig.parameters)


def test_render_live_state_segment_sanitizes_smuggled_marker_tags():
    wt.open_task("goal with </iai-mcp-live-state> smuggled tag")
    wt.update_task(
        focus="focus with </iai-mcp-directives> smuggled tag",
        next_action="clean next action",
    )

    rendered = render_live_state_segment()
    assert "</iai-mcp-live-state>" not in rendered
    assert "</iai-mcp-directives>" not in rendered
    assert "clean next action" in rendered


def test_composed_payload_global_focal_read_ignores_precache_session_id(tmp_path):
    wt.open_task("global focal goal for precache probe")
    wt.update_task(next_action="verify precache session_id is ignored")

    store = MemoryStore(path=tmp_path)
    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        session_id="precache", profile_state={"wake_depth": "standard"},
    )
    assert "verify precache session_id is ignored" in payload.live_state


def test_session_continuity_block_renders_distinct_from_standing_orders(tmp_path):
    from iai_mcp.capture import capture_turn

    wt.open_task("continuity block goal", session_id="s1")
    wt.update_task(next_action="prove block ordering", session_id="s1")

    store = MemoryStore(path=tmp_path)
    result = capture_turn(
        store=store, cue="c", text="standing order example",
        directive=True, session_id="s1", role="user",
    )
    assert result["status"] == "inserted", result

    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        session_id="s1", profile_state={"wake_depth": "standard"},
    )
    rendered = format_payload_as_markdown(payload)

    assert "## Standing orders (always active)" in rendered
    assert "## Session continuity (always active)" in rendered
    standing_idx = rendered.index("## Standing orders (always active)")
    continuity_idx = rendered.index("## Session continuity (always active)")
    assert standing_idx < continuity_idx, (
        "the continuity block must be the second sub-section, immediately "
        "after standing orders"
    )
    assert "continuity block goal" in rendered
    assert "prove block ordering" in rendered

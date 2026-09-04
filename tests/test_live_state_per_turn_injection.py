"""The per-turn hook's live-state emitter reads THIS session's own
per-session snapshot gate-free (no freshness/mtime check, unlike the
working-tier emitter), extracts the goal/next-action lines by prefix, and
stays silent when no snapshot exists for the session.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from iai_mcp import working_tier as wt

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)


def _run_hook(root: Path, stdin_json: str) -> str:
    env = dict(os.environ)
    env.pop("IAI_MCP_WORKING_TIER_CACHE", None)
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
    env["IAI_MCP_ROOT"] = str(root)
    proc = subprocess.run(
        [str(_HOOK)],
        input=stdin_json,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, f"hook must always exit 0: {proc.stderr}"
    return proc.stdout


def _write_snapshot(root: Path, session_id: str, *, goal: str, next_action: str) -> Path:
    entry = wt.WorkingSetEntry(goal=goal, session_id=session_id, next_action=next_action)
    text = wt._render_snapshot(entry)
    path = root / f".working-tier.{session_id}.cached.md"
    path.write_text(text, encoding="utf-8")
    return path


def test_per_turn_working_tier_emission_reads_default_snapshot_path(tmp_path):
    """A fresh snapshot at the default path surfaces via the working-tier
    block; emit_live_state is suppressed since it would duplicate the same
    goal/next-action lines (see the dedup regression test)."""
    _write_snapshot(
        tmp_path, "sess-live",
        goal="ship the live-state producer seam",
        next_action="wire memory_capture to update_task",
    )

    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-live"}')

    assert "<iai-mcp-working-tier>" in out
    assert "</iai-mcp-working-tier>" in out
    assert "goal: ship the live-state producer seam" in out
    assert "next action: wire memory_capture to update_task" in out
    assert "<iai-mcp-live-state>" not in out


def test_per_turn_live_state_ignores_freshness_gate(tmp_path):
    cache = _write_snapshot(
        tmp_path, "sess-old",
        goal="a task opened long ago", next_action="still the plan",
    )
    thirty_days_ago = time.time() - 30 * 24 * 3600
    os.utime(cache, (thirty_days_ago, thirty_days_ago))

    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-old"}')

    assert "<iai-mcp-live-state>" in out, (
        "emit_live_state must inject regardless of snapshot age, unlike "
        "emit_working_tier's freshness gate"
    )
    assert "a task opened long ago" in out
    assert "still the plan" in out


def _live_state_block(out: str) -> str:
    start = out.index("<iai-mcp-live-state>")
    end = out.index("</iai-mcp-live-state>")
    return out[start:end]


def test_per_turn_live_state_extraction_is_prefix_based_not_positional(tmp_path):
    # Backdated past FRESH_SEC so emit_working_tier stays silent and
    # emit_live_state fires alone -- otherwise the working-tier block would
    # carry the same lines and this would test extraction from the wrong
    # emitter's output (mirrors test_per_turn_live_state_ignores_freshness_gate).
    session_id = "sess-reorder"
    text = (
        "# Working tier — active task\n"
        "session: sess-reorder\n"
        "an extra unrelated section\n"
        "goal: prefix-extracted goal\n"
        "more filler\n"
        "next action: prefix-extracted next action\n"
    )
    cache = tmp_path / f".working-tier.{session_id}.cached.md"
    cache.write_text(text, encoding="utf-8")
    thirty_days_ago = time.time() - 30 * 24 * 3600
    os.utime(cache, (thirty_days_ago, thirty_days_ago))

    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-reorder"}')
    block = _live_state_block(out)

    assert "goal: prefix-extracted goal" in block
    assert "next action: prefix-extracted next action" in block
    assert "an extra unrelated section" not in block
    assert "more filler" not in block


def test_per_turn_live_state_suppressed_when_working_tier_already_carries_it(tmp_path):
    """A fresh cache with a real next_action carries the goal/next-action
    lines verbatim in the working-tier block -- emit_live_state must not
    repeat them in a second block."""
    _write_snapshot(
        tmp_path, "sess-dedup",
        goal="ship the live-state producer seam",
        next_action="wire memory_capture to update_task",
    )

    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-dedup"}')

    assert "<iai-mcp-working-tier>" in out
    assert "goal: ship the live-state producer seam" in out
    assert "<iai-mcp-live-state>" not in out, (
        "duplicate goal/next-action block must be suppressed when the "
        "working-tier block already fired for the same fresh cache"
    )


def test_per_turn_live_state_skips_empty_block_when_next_action_unset(tmp_path):
    _write_snapshot(
        tmp_path, "sess-none-yet",
        goal="a task with no next_action folded yet", next_action="",
    )

    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-none-yet"}')

    assert "<iai-mcp-live-state>" not in out, (
        "the '(none)' next_action placeholder must never produce a "
        "zero-information live-state block"
    )
    assert "<iai-mcp-working-tier>" in out, (
        "the goal must still surface via the ordinary working-tier block "
        "(the hook must not have no-opped entirely)"
    )


def test_per_turn_live_state_absent_file_is_clean_noop(tmp_path):
    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-none"}')
    assert "<iai-mcp-live-state>" not in out


def test_per_turn_live_state_silent_without_session_id(tmp_path):
    _write_snapshot(
        tmp_path, "sess-any", goal="unreachable without a session", next_action="n/a",
    )
    out = _run_hook(tmp_path, '{"prompt": "hi"}')
    assert "<iai-mcp-live-state>" not in out

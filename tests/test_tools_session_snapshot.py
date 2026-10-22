"""Pure snapshot-builder behavior, plus proof that the follow-up query a
snapshot emits is a real, runnable command against the shipped binary.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_FOLLOW_UP_RE = re.compile(
    r"iai_tools_session search --session (\S+) --category (\S+)"
)


def _session_bin() -> str:
    for profile in ("release", "debug"):
        candidate = REPO / "target" / profile / "iai_tools_session"
        if candidate.exists():
            return str(candidate)
    pytest.skip("iai_tools_session not built; run: cargo build -p iai_mcp_tools_core --bins")


def _event(seq: int, session_id: str, ts: str, category: str, data: str) -> dict:
    return {
        "seq": seq,
        "session_id": session_id,
        "ts": ts,
        "category": category,
        "type": "Edit",
        "data": data,
        "priority": 2,
        "source_hook": "PostToolUse",
        "data_hash": "00000000",
    }


def test_golden_one_section_per_non_empty_category_with_exact_follow_up_shape():
    from iai_mcp.tools import session

    events = [
        _event(0, "sess-golden", "2026-01-01T00:00:00Z", "file", "src/one.py"),
        _event(1, "sess-golden", "2026-01-02T00:00:00Z", "error", "boom"),
        _event(2, "sess-golden", "2026-01-03T00:00:00Z", "file", "src/two.py"),
    ]

    snap = session.build_snapshot_from_events(events)

    assert "file: 2 event(s)" in snap
    assert "error: 1 event(s)" in snap
    # No section for a category with zero events.
    assert "decision:" not in snap
    assert "task:" not in snap

    assert "iai_tools_session search --session sess-golden --category file" in snap
    assert "iai_tools_session search --session sess-golden --category error" in snap

    # Underlying data is never dropped -- both file items are named.
    assert "src/one.py" in snap
    assert "src/two.py" in snap


def test_build_snapshot_from_events_is_pure_with_no_store(tmp_path, monkeypatch):
    """No `IAI_TOOLS_ROOT` is set and no file exists anywhere -- the pure
    builder must still render correctly from the in-memory fixture alone.
    """
    monkeypatch.delenv("IAI_TOOLS_ROOT", raising=False)

    from iai_mcp.tools import session

    events = [_event(0, "sess-pure", "2026-01-01T00:00:00Z", "rule", "always verify")]
    snap = session.build_snapshot_from_events(events)
    assert "rule: 1 event(s)" in snap
    assert "iai_tools_session search --session sess-pure --category rule" in snap


def test_emitted_follow_up_query_actually_retrieves_the_summarized_events(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(tmp_path))

    from iai_mcp.tools import session

    session_id = "sess-execute-proof"
    session.capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/first.py",
            "session_id": session_id,
            "source_hook": "PostToolUse",
        }
    )
    session.capture_event(
        {
            "tool_name": "Edit",
            "file_path": "src/second.py",
            "session_id": session_id,
            "source_hook": "PostToolUse",
        }
    )
    session.capture_event(
        {
            "tool_response": "ValueError: nope",
            "is_error": True,
            "session_id": session_id,
            "source_hook": "PostToolUse",
        }
    )

    snap = session.build_snapshot(session_id)

    match = _FOLLOW_UP_RE.search(snap)
    assert match, f"no follow-up query line found in snapshot: {snap!r}"
    parsed_session, parsed_category = match.group(1), match.group(2)
    assert parsed_session == session_id

    expected = session.query_events(session_id, category=parsed_category)
    assert len(expected) >= 1

    env = {"IAI_TOOLS_ROOT": str(tmp_path)}
    proc = subprocess.run(
        [_session_bin(), "search", "--session", parsed_session, "--category", parsed_category],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr

    returned = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    assert returned == expected
    assert len(returned) == len(expected) > 0
    assert all(e["category"] == parsed_category for e in returned)
    assert all(e["session_id"] == session_id for e in returned)

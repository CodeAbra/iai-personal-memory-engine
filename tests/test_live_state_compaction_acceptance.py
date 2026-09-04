"""Literal post-compaction acceptance: a real composed session-start payload,
carrying a live-state block, is served by the packaged session-recall hook
when Claude Code fires it with source=compact -- and the real $HOME cache is
never read or written by the test.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.community import CommunityAssignment
from iai_mcp.session import _compose_session_start_payload, format_payload_as_markdown
from iai_mcp.store import MemoryStore

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-session-recall.sh"
)

_REAL_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"

FOCAL_GOAL = "compaction acceptance focal goal"
NEXT_ACTION = "prove post-compaction delivery under redirected HOME"


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


def _real_cache_snapshot() -> tuple[bool, float | None]:
    if _REAL_CACHE_PATH.exists():
        return True, _REAL_CACHE_PATH.stat().st_mtime
    return False, None


def _run_session_recall_hook(tmp_home: Path, *, session_id: str, source: str) -> str:
    env = dict(os.environ)
    env["HOME"] = str(tmp_home)
    payload = json.dumps({"session_id": session_id, "source": source})
    proc = subprocess.run(
        ["/bin/sh", str(_HOOK)],
        input=payload,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    assert proc.returncode == 0, f"hook must always exit 0: {proc.stderr}"
    return proc.stdout


def test_compaction_delivers_live_state_from_real_cache_under_redirected_home(tmp_path):
    before_exists, before_mtime = _real_cache_snapshot()

    wt.open_task(FOCAL_GOAL, session_id="s1")
    wt.update_task(next_action=NEXT_ACTION, session_id="s1")

    store = MemoryStore(path=tmp_path / "lancedb")
    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        session_id="s1", profile_state={"wake_depth": "standard"},
    )
    rendered = format_payload_as_markdown(payload)
    assert "## Session continuity (always active)" in rendered
    assert NEXT_ACTION in rendered

    tmp_home = tmp_path / "home"
    cache_dir = tmp_home / ".iai-mcp"
    cache_dir.mkdir(parents=True)
    cache_path = cache_dir / ".session-start-payload.cached.md"
    cache_path.write_text(rendered, encoding="utf-8")

    hook_out = _run_session_recall_hook(tmp_home, session_id="s1", source="compact")

    assert "## Session continuity (always active)" in hook_out
    assert NEXT_ACTION in hook_out
    assert FOCAL_GOAL in hook_out

    after_exists, after_mtime = _real_cache_snapshot()
    assert after_exists == before_exists, (
        "the real $HOME session-start cache must never be created or removed by this test"
    )
    assert after_mtime == before_mtime, (
        "the real $HOME session-start cache must never be modified by this test"
    )

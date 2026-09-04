"""Literal compaction acceptance: a real composed session-start payload,
carrying a running agent alongside the focal phase/step, is served by the
packaged session-recall hook when Claude Code fires it with source=compact.
The composed cache is seeded non-empty so the hook's cache-hit branch (not
the CLI-compose fallback) is what actually delivers the agent block, and the
real $HOME cache is never read or written by the test.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.community import CommunityAssignment
from iai_mcp.daemon_state import register_running_agent
from iai_mcp.session import (
    _compose_session_start_payload,
    format_payload_as_markdown,
    write_continuity_cache,
)
from iai_mcp.store import MemoryStore

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-session-recall.sh"
)

_REAL_CACHE_PATH = Path.home() / ".iai-mcp" / ".session-start-payload.cached.md"

FOCAL_GOAL = "agent registry compaction focal goal"
NEXT_ACTION = "prove the pending agent survives compaction alongside phase/step"
AGENT_ROLE = "research"
AGENT_ARTIFACT = "RESEARCH-280.md"


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


def test_compaction_delivers_agent_registry_alongside_focal_phase_step(tmp_path, monkeypatch):
    before_exists, before_mtime = _real_cache_snapshot()

    tmp_home = tmp_path / "home"
    store_root = tmp_home / ".iai-mcp"
    store_root.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    wt.open_task(FOCAL_GOAL, session_id="s1")
    wt.update_task(next_action=NEXT_ACTION, session_id="s1")
    register_running_agent(
        agent_id="a-compact-1",
        role=AGENT_ROLE,
        expected_artifact=AGENT_ARTIFACT,
        agent_model="claude-sonnet-5",
    )
    # Mirrors the production _persist_entry seam: the eager, session-agnostic
    # continuity file is refreshed with both the focal live-state and the
    # just-registered pending agent.
    write_continuity_cache(SimpleNamespace(root=store_root))

    store = MemoryStore(path=tmp_path / "lancedb")
    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        session_id="s1", profile_state={"wake_depth": "standard"},
    )
    rendered = format_payload_as_markdown(payload)
    assert rendered.strip(), (
        "composed payload must be non-empty to exercise the cache-hit branch, "
        "never weakened to fall through to the CLI-compose fallback"
    )
    assert NEXT_ACTION in rendered

    cache_path = store_root / ".session-start-payload.cached.md"
    cache_path.write_text(rendered, encoding="utf-8")
    assert cache_path.stat().st_size > 0

    hook_out = _run_session_recall_hook(tmp_home, session_id="s1", source="compact")

    assert NEXT_ACTION in hook_out, "the cache-hit branch must still name the focal phase/step"
    assert FOCAL_GOAL in hook_out
    assert "<iai-mcp-agent-registry>" in hook_out, (
        "the eager agent-registry block must be appended at the cache-hit exit"
    )
    assert AGENT_ROLE in hook_out
    assert AGENT_ARTIFACT in hook_out

    after_exists, after_mtime = _real_cache_snapshot()
    assert after_exists == before_exists, (
        "the real $HOME session-start cache must never be created or removed by this test"
    )
    assert after_mtime == before_mtime, (
        "the real $HOME session-start cache must never be modified by this test"
    )

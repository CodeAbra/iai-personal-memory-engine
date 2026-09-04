"""The discriminating /clear acceptance: a brand-new post-/clear session id
has no session-scoped working-tier snapshot, so the per-turn hook must
reconstruct BOTH the current phase/step and the pending agent from the
session-agnostic eager continuity file -- never from a stale composed
session-start cache. A companion test proves the normal same-session case
emits phase/step exactly once (no double injection when the session-scoped
snapshot IS present), and a third proves the eager file's whole-file mtime
bound.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.daemon_state import register_running_agent
from iai_mcp.types import EMBED_DIM, MemoryRecord

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)

AGENT_ROLE = "review"
AGENT_ARTIFACT = "REVIEW-280.md"


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


def _run_hook(root: Path, stdin_json: str) -> str:
    env = dict(os.environ)
    env.pop("IAI_MCP_WORKING_TIER_CACHE", None)
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)
    env["IAI_MCP_STORE"] = str(root)
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


def test_per_turn_hook_names_current_state_not_stale_cache_after_clear(tmp_path, monkeypatch):
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    # Pre-clear session s1 at an OLD step -- this is what a session-start
    # compose would have captured at the time, now stale.
    wt.open_task("hard-clear reconstruction task", session_id="s1")
    wt.update_task(
        focus="OLD phase", next_action="OLD step", session_id="s1",
        store=SimpleNamespace(root=root),
    )
    (root / ".session-start-payload.cached.md").write_text(
        "## Session continuity (always active)\nfocus: OLD phase\nnext action: OLD step\n",
        encoding="utf-8",
    )

    # Work continues in the SAME session: a new step, plus a pending agent.
    # This is the write path a real /clear leaves behind -- the eager file
    # refreshes to CURRENT state.
    register_running_agent(
        agent_id="a-clear-1", role=AGENT_ROLE, expected_artifact=AGENT_ARTIFACT,
        agent_model="claude-sonnet-5",
    )
    wt.update_task(
        focus="NEW phase", next_action="NEW step", session_id="s1",
        store=SimpleNamespace(root=root),
    )

    # /clear mints a brand-new session id s2 -- no snapshot exists for it yet.
    assert not (root / ".working-tier.s2.cached.md").exists()

    out = _run_hook(root, '{"prompt": "hi", "session_id": "s2"}')

    assert "NEW phase" in out
    assert "NEW step" in out
    assert AGENT_ROLE in out
    assert AGENT_ARTIFACT in out
    assert "OLD phase" not in out
    assert "OLD step" not in out


def _make_record(text: str, session_id: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": session_id}],
        created_at=now,
        updated_at=now,
        tags=["role:user"],
        language="en",
    )


def test_per_turn_hook_still_reconstructs_on_turn_two_after_real_record_captured(
    tmp_path, monkeypatch,
):
    """The discriminating regression: the existing acceptance only exercises
    the FIRST per-turn hook call after /clear, before any record for the new
    session has been captured. In real use, the new session's very first
    captured message parks the old focal task and opens a thin fresh entry
    (next_action=""), which must NOT erase the eager file's reconstruction
    source -- turn 2 must still name the pre-clear phase/step."""
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    store = SimpleNamespace(root=root)

    wt.open_task("hard-clear turn-two reconstruction task", session_id="s1")
    register_running_agent(
        agent_id="a-clear-2", role=AGENT_ROLE, expected_artifact=AGENT_ARTIFACT,
        agent_model="claude-sonnet-5",
    )
    wt.update_task(
        focus="NEW phase", next_action="NEW step", session_id="s1", store=store,
    )

    # /clear mints a brand-new session id s2 -- turn 1 has no snapshot for it
    # yet, so the eager file alone must reconstruct.
    out1 = _run_hook(root, '{"prompt": "hi", "session_id": "s2"}')
    assert "NEW phase" in out1
    assert "NEW step" in out1

    # Turn 1's own user message IS captured for s2 -- the real flow: this
    # parks s1 (durables write-through), opens a thin fresh entry for s2
    # (next_action="" until its own state accrues), and persists both,
    # refreshing the session-agnostic eager file on each persist.
    record = _make_record("unrelated turn-two content for session two", session_id="s2")
    wt.update_from_record(record, store=store)

    assert (root / ".working-tier.s2.cached.md").exists()

    # Turn 2: s2 now has its own (still-thin) scoped snapshot.
    out2 = _run_hook(root, '{"prompt": "hi again", "session_id": "s2"}')

    assert out2.count("<iai-mcp-live-state>") == 1, (
        "turn 2 must still emit exactly one live-state block -- no double "
        "injection, and reconstruction must not be a one-shot effect"
    )
    assert out2.count("</iai-mcp-live-state>") == 1
    assert "NEW phase" in out2
    assert "NEW step" in out2


def test_per_turn_hook_emits_phase_step_exactly_once_when_session_snapshot_present(
    tmp_path, monkeypatch,
):
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    wt.open_task("no-double-injection task", session_id="s1")
    wt.update_task(
        focus="SESSION phase", next_action="SESSION step", session_id="s1",
        store=SimpleNamespace(root=root),
    )
    register_running_agent(
        agent_id="a-nodup-1", role=AGENT_ROLE, expected_artifact=AGENT_ARTIFACT,
        agent_model="claude-sonnet-5",
    )
    wt.update_task(session_id="s1", store=SimpleNamespace(root=root))

    assert (root / ".working-tier.s1.cached.md").exists()

    out = _run_hook(root, '{"prompt": "hi", "session_id": "s1"}')

    assert out.count("<iai-mcp-live-state>") == 0, (
        "the session-scoped snapshot is present and fresh -- emit_working_tier "
        "already carries phase/step verbatim, so both emit_live_state and the "
        "eager fallback must stay silent (no double injection)"
    )
    assert "<iai-mcp-working-tier>" in out
    assert "SESSION phase" in out
    assert "SESSION step" in out
    # The agent block is unconditional (session-agnostic) and must still
    # surface even though the phase/step fallback stayed silent.
    assert AGENT_ROLE in out
    assert AGENT_ARTIFACT in out


def test_per_turn_hook_eager_file_past_ttl_surfaces_nothing(tmp_path, monkeypatch):
    root = tmp_path / "store"
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))

    wt.open_task("aged-out task", session_id="s1")
    register_running_agent(
        agent_id="a-aged-1", role=AGENT_ROLE, expected_artifact=AGENT_ARTIFACT,
        agent_model="claude-sonnet-5",
    )
    wt.update_task(
        next_action="aged step", session_id="s1", store=SimpleNamespace(root=root),
    )

    cache = root / ".session-continuity.cached.md"
    assert cache.exists()
    long_ago = time.time() - 30000  # past the 21600s (6h) TTL
    os.utime(cache, (long_ago, long_ago))

    # A fresh post-/clear session with no session-scoped snapshot -- the
    # only source left is the (now aged-out) eager file.
    out = _run_hook(root, '{"prompt": "hi", "session_id": "s2"}')

    assert "<iai-mcp-agent-registry>" not in out
    assert "<iai-mcp-live-state>" not in out
    assert AGENT_ROLE not in out
    assert "aged step" not in out

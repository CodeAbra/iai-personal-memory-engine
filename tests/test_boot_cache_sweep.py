"""Boot-time invalidation of the two per-session staleness surfaces:
a fresh process has no in-RAM focal task, so a pre-restart
.working-tier.<sid>.cached.md and a substantive .session-continuity.cached.md
live-state block must both be rebuilt-from-canonical, not just one of them --
sweeping only the working-tier glob would re-route the same stale focal
framing into the continuity file's 6h fallback window.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from iai_mcp import session
from iai_mcp import working_tier as wt
from iai_mcp.daemon import _rebuild_session_caches_on_boot
from iai_mcp.daemon_state import register_running_agent


def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))


@pytest.fixture(autouse=True)
def _reset_working_tier_singleton():
    wt._reset()
    yield
    wt._reset()


def test_boot_sweep_deletes_working_tier_and_empties_continuity_live_state(
    tmp_path, monkeypatch,
):
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    store = SimpleNamespace(root=store_root)

    # Pre-restart state: a real focal task with a real next_action, seeding
    # both the per-session snapshot AND the eager continuity live-state block.
    wt.open_task("investigate the recall regression", session_id="s1")
    wt.update_task(
        focus="pipeline.py hot path",
        next_action="reproduce locally",
        session_id="s1",
        store=store,
    )

    working_tier_path = store_root / ".working-tier.s1.cached.md"
    continuity_path = store_root / ".session-continuity.cached.md"
    assert working_tier_path.exists()
    assert "reproduce locally" in continuity_path.read_text(encoding="utf-8")

    # Simulate a process restart: the in-RAM focal task is gone (a fresh
    # daemon starts with _FOCAL=None), but the on-disk caches from the prior
    # process are still there.
    wt._reset()

    _rebuild_session_caches_on_boot(store)

    assert not working_tier_path.exists(), (
        "a stale per-session working-tier snapshot must not survive a boot"
    )
    after = continuity_path.read_text(encoding="utf-8")
    assert "focus: pipeline.py hot path" not in after
    assert "next action: reproduce locally" not in after
    live_block = after.split("<iai-mcp-live-state>\n", 1)[1].split(
        "</iai-mcp-live-state>", 1
    )[0]
    assert live_block.strip() == "", (
        "the continuity live-state block must be forced empty on boot, not "
        "re-surfaced via the 6h emit_live_state_fallback window"
    )


def test_boot_sweep_agent_registry_block_still_renders(tmp_path, monkeypatch):
    """Targeted, not blunt: the agent-registry block (sourced independently
    from daemon_state.json) must survive the same boot sweep that empties
    the live-state block."""
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    store = SimpleNamespace(root=store_root)

    register_running_agent(
        agent_id="a1", role="research", expected_artifact="RESEARCH.md",
        agent_model="claude-sonnet-5",
    )

    wt.open_task("a task that will not survive the restart", session_id="s1")
    wt.update_task(next_action="this must be swept", session_id="s1", store=store)
    wt._reset()

    _rebuild_session_caches_on_boot(store)

    continuity_path = store_root / ".session-continuity.cached.md"
    text = continuity_path.read_text(encoding="utf-8")
    agent_block = text.split("<iai-mcp-agent-registry>\n", 1)[1].split(
        "</iai-mcp-agent-registry>", 1
    )[0]
    assert "agent a1" in agent_block
    assert "research" in agent_block
    assert "RESEARCH.md" in agent_block


def test_boot_sweep_deletes_env_override_working_tier_cache(tmp_path, monkeypatch):
    """IAI_MCP_WORKING_TIER_CACHE makes _cache_path return one path verbatim,
    escaping the base-relative glob the sweep otherwise uses -- also a live
    production configuration (the per-turn hook consumes it, not just
    tests). The boot sweep must delete that path directly, or a deployment
    using the override keeps serving a stale pre-restart focal task after
    every restart.
    """
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    store = SimpleNamespace(root=store_root)

    override_path = tmp_path / "custom" / "working-tier-override.md"
    override_path.parent.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv(wt.WORKING_TIER_CACHE_ENV, str(override_path))

    wt.open_task("investigate the override sweep gap", session_id="s1")
    wt.update_task(
        focus="boot sweep coverage", next_action="verify override deletion",
        session_id="s1", store=store,
    )
    assert override_path.exists()

    wt._reset()
    _rebuild_session_caches_on_boot(store)

    assert not override_path.exists(), (
        "a stale env-override working-tier cache must not survive a boot"
    )


def test_clear_preserve_guard_unchanged_outside_boot(tmp_path, monkeypatch):
    """The default (non-boot) write_continuity_cache(store) call, with
    allow_downgrade left at its False default, must still preserve a
    substantive stale block for the /clear reconstruction case -- only the
    boot helper's explicit allow_downgrade=True bypasses that guard."""
    _monkeypatch_env(monkeypatch, tmp_path)
    store_root = tmp_path / "store"
    store_root.mkdir(parents=True, exist_ok=True)
    store = SimpleNamespace(root=store_root)

    wt.open_task("session A's real goal", session_id="s1")
    wt.update_task(
        focus="A's real focus", next_action="A's real next action",
        session_id="s1", store=store,
    )
    continuity_path = store_root / ".session-continuity.cached.md"
    seeded = continuity_path.read_text(encoding="utf-8")
    assert "focus: A's real focus" in seeded

    # A thin park-then-reopen with no focus/next_action yet -- an incidental
    # task switch, not an explicit clear -- must not clobber the prior block.
    wt._reset()
    wt.open_task("a fresh, still-empty task", session_id="s2")

    session.write_continuity_cache(store)

    preserved = continuity_path.read_text(encoding="utf-8")
    assert "focus: A's real focus" in preserved, (
        "the default allow_downgrade=False preserve-guard must stay intact "
        "outside the boot helper"
    )
    assert "next action: A's real next action" in preserved

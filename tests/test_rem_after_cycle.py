"""REM must actually run after a clean sleep cycle.

The nightly insight pathway died when its only call site was dropped in a
refactor: run_rem_cycle stayed imported but never called, so rem_cycle events
and the overnight digest froze. These tests pin the re-wired pass: it fires
when due, spaces routine cycles, honors a fresh force-rem, respects the
kill-switch, and folds its result into the pending digest.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from iai_mcp.store import MemoryStore


def _canned_result() -> dict:
    return {
        "cycle": 1,
        "summaries_created": 4,
        "schemas_induced": 1,
        "schema_candidates": 2,
        "claude_call_used": True,
        "main_insight_text": "insight",
        "timed_out": False,
    }


def _patch_rem(monkeypatch, calls: list) -> None:
    from iai_mcp import daemon as daemon_mod

    async def _fake_rem(store, n, of, session_id, *, is_last, claude_enabled):
        calls.append(
            {"n": n, "of": of, "is_last": is_last, "claude": claude_enabled}
        )
        return _canned_result()

    monkeypatch.setattr(daemon_mod, "run_rem_cycle", _fake_rem)


def test_rem_runs_when_never_ran(tmp_path, monkeypatch):
    from iai_mcp.daemon import _maybe_run_rem

    store = MemoryStore(path=tmp_path)
    calls: list = []
    _patch_rem(monkeypatch, calls)

    result = asyncio.run(_maybe_run_rem(store, {}))
    assert result is not None
    assert calls and calls[0]["claude"] is True


def test_rem_spaced_by_interval(tmp_path, monkeypatch):
    from iai_mcp.daemon import _maybe_run_rem
    from iai_mcp.events import write_event

    store = MemoryStore(path=tmp_path)
    write_event(store, "rem_cycle_completed", {"cycle": 1})
    calls: list = []
    _patch_rem(monkeypatch, calls)

    result = asyncio.run(_maybe_run_rem(store, {}))
    assert result is None, "a REM pass minutes after the last one must skip"
    assert not calls


def test_fresh_force_rem_bypasses_interval(tmp_path, monkeypatch):
    from iai_mcp.daemon import _maybe_run_rem
    from iai_mcp.events import write_event

    store = MemoryStore(path=tmp_path)
    write_event(store, "rem_cycle_completed", {"cycle": 1})
    calls: list = []
    _patch_rem(monkeypatch, calls)

    ds = {
        "force_rem_request": {
            "pending": False,
            "honored_at": datetime.now(timezone.utc).isoformat(),
        }
    }
    result = asyncio.run(_maybe_run_rem(store, ds))
    assert result is not None
    assert calls


def test_stale_force_rem_does_not_bypass(tmp_path, monkeypatch):
    from iai_mcp.daemon import _maybe_run_rem
    from iai_mcp.events import write_event

    store = MemoryStore(path=tmp_path)
    write_event(store, "rem_cycle_completed", {"cycle": 1})
    calls: list = []
    _patch_rem(monkeypatch, calls)

    stale = datetime.now(timezone.utc) - timedelta(hours=6)
    ds = {
        "force_rem_request": {
            "pending": False,
            "honored_at": stale.isoformat(),
        }
    }
    assert asyncio.run(_maybe_run_rem(store, ds)) is None
    assert not calls


def test_kill_switch_disables(tmp_path, monkeypatch):
    from iai_mcp.daemon import _maybe_run_rem

    store = MemoryStore(path=tmp_path)
    calls: list = []
    _patch_rem(monkeypatch, calls)
    monkeypatch.setenv("IAI_MCP_REM_DISABLED", "1")

    assert asyncio.run(_maybe_run_rem(store, {})) is None
    assert not calls


def test_digest_folded_into_daemon_state(tmp_path, monkeypatch):
    from iai_mcp.daemon import _maybe_run_rem
    from iai_mcp.daemon_state import load_state

    store = MemoryStore(path=tmp_path)
    calls: list = []
    _patch_rem(monkeypatch, calls)

    result = asyncio.run(_maybe_run_rem(store, {}))
    assert result is not None
    digest = (load_state() or {}).get("pending_digest") or {}
    assert digest.get("rem_cycles_completed", 0) >= 1
    assert digest.get("claude_call_used") is True

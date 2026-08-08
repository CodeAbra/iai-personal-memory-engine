"""Ambient injection stays attributed to its origin.

Parallel conversations share one store; the anticipation pack and the
recent-work feed are the two surfaces where one session's thread can leak
into another as if it were its own. The pack is per-session (the global
file remains a fallback for hosts without a session id), and every
recent-work line names its origin so the reader attributes instead of
guessing.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
from pathlib import Path

import pytest

from iai_mcp import foresight
from iai_mcp.capture import _drain_write_pending, capture_turn
from iai_mcp.store import MemoryStore, flush_record_buffer

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)


@pytest.fixture(autouse=True)
def _foresight_floor(monkeypatch):
    monkeypatch.setenv(foresight.FORESIGHT_MIN_COS_ENV, "0.45")
    monkeypatch.setenv(foresight.FORESIGHT_GOAL_WEIGHT_ENV, "0")


@pytest.fixture(autouse=True)
def _fresh_working_tier():
    from iai_mcp import working_tier

    working_tier._reset()
    yield
    working_tier._reset()


def _turn(store, text, session):
    result = capture_turn(
        store, cue="", text=text, tier="episodic",
        session_id=session, role="user", live_turn=True,
    )
    flush_record_buffer(store)
    return result


def test_parallel_sessions_each_keep_their_own_pack(tmp_path):
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "old-a")
    _turn(store, "Coral reefs host a quarter of all known marine species.", "old-b")

    _turn(store, "How does sleep consolidation work in the hippocampus?", "sess-A")
    pack_a = foresight.pack_path(store, "sess-A")
    assert pack_a.is_file(), "session A must get its own pack file"
    body_a = pack_a.read_text(encoding="utf-8")
    assert "hippocampus consolidates" in body_a

    _turn(store, "Tell me about coral reef marine biodiversity.", "sess-B")
    pack_b = foresight.pack_path(store, "sess-B")
    assert pack_b.is_file(), "session B must get its own pack file"
    body_b = pack_b.read_text(encoding="utf-8")
    assert "Coral reefs host" in body_b

    # B's refresh must not have starved A: A's file still serves A's topic.
    assert "hippocampus consolidates" in pack_a.read_text(encoding="utf-8")
    # Sid-carrying sessions own ONLY their per-session file — the global
    # pack/state pair belongs to id-less writers alone, so two parallel
    # refreshes can never tear it.
    assert not foresight.pack_path(store).exists()


def test_sessionless_writer_owns_the_global_pack(tmp_path):
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "old-a")

    _turn(store, "How does hippocampal sleep consolidation work?", "-")
    global_pack = foresight.pack_path(store)
    assert global_pack.is_file(), "an id-less refresh must serve the global pack"
    assert "hippocampus consolidates" in global_pack.read_text(encoding="utf-8")


def test_session_id_is_sanitized_into_the_pack_filename(tmp_path):
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "old-a")

    hostile = "../evil/../sess?*id"
    _turn(store, "Tell me about hippocampal memory consolidation.", hostile)
    from iai_mcp.working_tier import _sanitize_session_id

    safe = foresight.pack_path(store, hostile)
    assert _sanitize_session_id(hostile) in safe.name
    assert "/" not in safe.name and ".." not in safe.name.replace(
        ".next-turn-pack.", ""
    ).replace(".cached.md", "")
    assert safe.parent == pathlib.Path(store.root)
    assert safe.is_file(), "the sanitized pack file must be the one written"


def test_per_session_state_isolates_served_ledgers(tmp_path):
    store = MemoryStore(path=tmp_path)
    _turn(store, "The hippocampus consolidates memories during sleep.", "old-a")

    _turn(store, "Explain hippocampal consolidation in sleep.", "sess-A")
    assert "hippocampus consolidates" in foresight.pack_path(
        store, "sess-A"
    ).read_text(encoding="utf-8")

    # A different session asking the same thing is NOT blocked by A's ledger.
    _turn(store, "How does hippocampal memory consolidation happen?", "sess-B")
    pack_b = foresight.pack_path(store, "sess-B")
    assert pack_b.is_file()
    assert "hippocampus consolidates" in pack_b.read_text(encoding="utf-8"), (
        "session B never saw this memory; A's served ledger must not block it"
    )


def test_stale_session_packs_are_garbage_collected(tmp_path):
    store = MemoryStore(path=tmp_path)
    stale_pack = Path(store.root) / ".next-turn-pack.dead-sess.cached.md"
    stale_state = Path(store.root) / ".next-turn-pack.dead-sess.state.json"
    stale_pack.write_text("- old\n", encoding="utf-8")
    stale_state.write_text("{}", encoding="utf-8")
    old = time.time() - 4 * 24 * 3600
    os.utime(stale_pack, (old, old))
    os.utime(stale_state, (old, old))
    fresh_global = Path(store.root) / ".next-turn-pack.cached.md"
    fresh_global.write_text("- global\n", encoding="utf-8")
    os.utime(fresh_global, (old, old))

    foresight._gc_stale_session_packs(store)

    assert not stale_pack.exists()
    assert not stale_state.exists()
    # The unsuffixed global pack is out of the sweep's reach by construction.
    assert fresh_global.exists()


def test_hook_prefers_this_sessions_pack(tmp_path):
    root = tmp_path
    (root / ".next-turn-pack.sidA.cached.md").write_text(
        "- own-session anticipation line\n", encoding="utf-8"
    )
    (root / ".next-turn-pack.sidA.state.json").write_text(
        json.dumps({"session_id": "sidA", "served": {}}), encoding="utf-8"
    )
    (root / ".next-turn-pack.cached.md").write_text(
        "- other-session global line\n", encoding="utf-8"
    )
    (root / ".next-turn-pack.state.json").write_text(
        json.dumps({"session_id": "sidB", "served": {}}), encoding="utf-8"
    )
    env = dict(os.environ)
    env["IAI_MCP_STORE"] = str(root)
    env.pop("IAI_MCP_ROOT", None)
    env.pop("IAI_MCP_FORESIGHT_PACK", None)
    env["IAI_MCP_WORKING_TIER_CACHE"] = str(root / "absent.md")
    env.pop("IAI_MCP_PER_TURN_SOCKET_ACCEL", None)

    proc = subprocess.run(
        [str(_HOOK)], input='{"session_id": "sidA", "prompt": "x"}',
        capture_output=True, text=True, env=env, timeout=10,
    )
    assert proc.returncode == 0
    assert "own-session anticipation line" in proc.stdout
    assert "other-session global line" not in proc.stdout

    # A session with no pack of its own and a foreign global stays silent.
    proc = subprocess.run(
        [str(_HOOK)], input='{"session_id": "sidC", "prompt": "x"}',
        capture_output=True, text=True, env=env, timeout=10,
    )
    assert proc.returncode == 0
    assert "anticipation line" not in proc.stdout
    assert "global line" not in proc.stdout


def test_recent_work_lines_carry_origin_labels(tmp_path):
    from iai_mcp.session import _recent_thread_segment

    store = MemoryStore(path=tmp_path)
    capture_turn(
        store, cue="", text="Refactoring the payment gateway retry loop today.",
        tier="episodic", session_id="abcdef123456", role="user",
        provenance_extra={"cwd": "/x/projA"},
    )
    capture_turn(
        store, cue="", text="Debugging the flaky websocket reconnect test.",
        tier="episodic", session_id="fedcba654321", role="user",
    )
    flush_record_buffer(store)

    segment = _recent_thread_segment(store)
    assert "[projA] " in segment, segment
    assert "[s:fedcba] " in segment, segment


def test_origin_label_never_breaks_on_odd_records():
    from iai_mcp.session import _origin_label

    class _Bare:
        pass

    assert _origin_label(_Bare()) == ""

    class _PendingLike:
        provenance = None
        session_id = "abcdef99"

    assert _origin_label(_PendingLike()) == "[s:abcdef] "

    class _NoSid:
        provenance = [{"session_id": "-"}]

    assert _origin_label(_NoSid()) == ""


def test_drain_write_pending_threads_cwd_provenance(tmp_path):
    from uuid import UUID

    store = MemoryStore(path=tmp_path)
    result = _drain_write_pending(
        store,
        cue="",
        text="A deferred turn drained with its project directory attached.",
        session_id="sess-drain",
        role="user",
        provenance_extra={"cwd": "/x/projB"},
    )
    assert result["status"] == "inserted"
    rec = store.get(UUID(result["record_id"]))
    assert any(
        isinstance(p, dict) and p.get("cwd") == "/x/projB"
        for p in (rec.provenance or [])
    ), rec.provenance

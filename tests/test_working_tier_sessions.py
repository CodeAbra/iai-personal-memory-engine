"""Working tier under concurrent sessions: focal task + bounded park.

A turn from a non-focal session is an explicit task switch — the focal task
is parked (durables written through), the arriving session's task is
restored or opened fresh. A parked task is never fed, never injected into
another session, and its snapshot file is its own. The per-turn hook reads
only its own session's snapshot.
"""
from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import working_tier as wt
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

_HOOK = (
    Path(__file__).resolve().parents[1]
    / "src/iai_mcp/_deploy/hooks/iai-mcp-per-turn-recall.sh"
)


def _make(text: str, session_id: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": session_id}],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


@pytest.fixture(autouse=True)
def _fresh_tier(tmp_path, monkeypatch):
    monkeypatch.delenv(wt.WORKING_TIER_CACHE_ENV, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    wt._reset()
    yield
    wt._reset()


class _FakeStoreRoot:
    def __init__(self, root: Path):
        self.root = root


def test_cross_session_turn_parks_focal_and_restores(tmp_path):
    wt.update_from_record(_make("session A opening goal", "sess-a"))
    wt.update_from_record(_make("session A second turn", "sess-a"))
    wt.update_from_record(_make("session B opening goal", "sess-b"))

    focal = wt.read_task()
    assert focal is not None
    assert focal.session_id == "sess-b"
    assert focal.goal == "session B opening goal"
    assert "session A opening goal" not in " ".join(focal.raw_sensory)

    wt.update_from_record(_make("session A third turn", "sess-a"))
    focal = wt.read_task()
    assert focal is not None
    assert focal.session_id == "sess-a"
    assert focal.goal == "session A opening goal", (
        "switching back must restore the parked task, not open a fresh one"
    )
    joined = " ".join(focal.raw_sensory)
    assert "session A third turn" in joined
    assert "session B opening goal" not in joined


def test_read_task_by_session_never_switches_focus():
    wt.update_from_record(_make("task of session A", "sess-a"))
    wt.update_from_record(_make("task of session B", "sess-b"))

    parked_view = wt.read_task(session_id="sess-a")
    assert parked_view is not None
    assert parked_view.goal == "task of session A"

    focal = wt.read_task()
    assert focal is not None
    assert focal.session_id == "sess-b", "a read must never switch focus"

    assert wt.read_task(session_id="sess-unknown") is None


def test_park_write_through_consolidates_durables(tmp_path):
    store = MemoryStore(path=tmp_path / "store")
    try:
        store.insert(_make("session A durable work", "sess-a"))
        wt.update_task(result="durable finding Z: the rebuild lock was held across the swap")

        store.insert(_make("session B interrupts", "sess-b"))

        found = [
            r for r in store.iter_records(where="tier = 'episodic'")
            if "durable finding Z" in (r.literal_surface or "")
        ]
        assert len(found) == 1, (
            "parking must write the focal task's durable results through "
            f"to the episodic store; found {len(found)}"
        )

        parked = wt.read_task(session_id="sess-a")
        assert parked is not None
        assert parked.results == [], "delivered durables must not be re-consolidated later"

        outcome = wt.close_task(store)
        assert outcome.get("undelivered") in (None, []), (
            f"nothing should remain undelivered after write-through: {outcome}"
        )
        found_after = [
            r for r in store.iter_records(where="tier = 'episodic'")
            if "durable finding Z" in (r.literal_surface or "")
        ]
        assert len(found_after) == 1, "close must not duplicate the written-through result"
    finally:
        store.close()


def test_park_is_lru_bounded_and_eviction_closes(tmp_path):
    for i in range(wt.WORKING_TIER_PARK_SLOTS + 3):
        wt.update_from_record(_make(f"goal of session {i}", f"sess-{i}"))

    assert wt.read_task(session_id="sess-0") is None, (
        "the oldest parked task must be evicted once the park bound is hit"
    )
    live = [
        sid for sid in (f"sess-{i}" for i in range(wt.WORKING_TIER_PARK_SLOTS + 3))
        if wt.read_task(session_id=sid) is not None
    ]
    assert len(live) == wt.WORKING_TIER_PARK_SLOTS + 1, (
        f"focal + park bound must cap live tasks, got {live}"
    )


def test_per_session_snapshot_files(tmp_path):
    store = _FakeStoreRoot(tmp_path)
    wt.update_from_record(_make("goal for session A", "sess-a"), store=store)
    wt.update_from_record(_make("goal for session B", "sess-b"), store=store)

    file_a = tmp_path / ".working-tier.sess-a.cached.md"
    file_b = tmp_path / ".working-tier.sess-b.cached.md"
    assert file_a.is_file() and file_b.is_file()
    assert "goal for session A" in file_a.read_text(encoding="utf-8")
    assert "goal for session B" in file_b.read_text(encoding="utf-8")
    assert "goal for session B" not in file_a.read_text(encoding="utf-8")

    wt.close_task(store=store)
    assert not file_a.exists() and not file_b.exists(), (
        "closing the tier must remove every session's snapshot"
    )


def test_legacy_global_snapshot_swept(tmp_path):
    store = _FakeStoreRoot(tmp_path)
    legacy = tmp_path / ".working-tier.cached.md"
    legacy.write_text("stale global task\n", encoding="utf-8")
    wt._legacy_swept = False

    wt.update_from_record(_make("fresh per-session task", "sess-a"), store=store)
    assert not legacy.exists(), "the unsuffixed global snapshot must be removed"


def test_session_id_sanitized_for_snapshot_filename(tmp_path):
    store = _FakeStoreRoot(tmp_path)
    wt.update_from_record(_make("hostile session id", "../../etc/passwd"), store=store)
    escaped = [p for p in tmp_path.rglob("*") if "passwd" in p.name and "working-tier" not in p.name]
    assert escaped == [], f"session id must never escape the snapshot dir: {escaped}"
    assert any(
        p.name.startswith(".working-tier.") for p in tmp_path.iterdir()
    ), "a sanitized snapshot must still be written"


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


def test_hook_injects_only_own_session_snapshot(tmp_path):
    (tmp_path / ".working-tier.sess-mine.cached.md").write_text(
        "goal: my own task\n", encoding="utf-8"
    )
    (tmp_path / ".working-tier.sess-other.cached.md").write_text(
        "goal: someone else's task\n", encoding="utf-8"
    )

    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-mine"}')
    assert "my own task" in out
    assert "someone else's task" not in out

    out_foreign = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-third"}')
    assert "<iai-mcp-working-tier>" not in out_foreign, (
        "a session with no snapshot of its own must get nothing, never a foreign task"
    )


def test_hook_silent_without_session_id(tmp_path):
    (tmp_path / ".working-tier.sess-any.cached.md").write_text(
        "goal: unreachable without a session\n", encoding="utf-8"
    )
    out = _run_hook(tmp_path, '{"prompt": "hi"}')
    assert "<iai-mcp-working-tier>" not in out


def test_hook_ignores_stale_per_session_snapshot(tmp_path):
    cache = tmp_path / ".working-tier.sess-old.cached.md"
    cache.write_text("goal: yesterday's task\n", encoding="utf-8")
    old = time.time() - 3 * 3600
    os.utime(cache, (old, old))
    out = _run_hook(tmp_path, '{"prompt": "hi", "session_id": "sess-old"}')
    assert "yesterday" not in out

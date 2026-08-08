"""Ambient freshness plumbing: every record write stamps the store-advance
sidecar the per-turn hooks watch, the drain feeds the working tier, and a
drain pass refreshes the next-turn pack from the newest user turn. Hooks read
sidecars only — they must never parse the store file, whose engine format is
not a hook contract."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    from iai_mcp import working_tier

    working_tier._reset()
    yield tmp_path
    working_tier._reset()


def test_emit_read_round_trip_and_monotone(tmp_path):
    from iai_mcp.store_watermark import emit, read

    root = tmp_path / "sidecar"
    assert read(root) is None
    emit(root, "2026-07-01T00:00:00+00:00")
    assert read(root) == "2026-07-01T00:00:00+00:00"

    # A later stamp advances; an earlier one never downgrades.
    emit(root, "2026-07-02T00:00:00+00:00")
    emit(root, "2026-06-01T00:00:00+00:00")
    assert read(root) == "2026-07-02T00:00:00+00:00"

    emit(root, "")
    assert read(root) == "2026-07-02T00:00:00+00:00"


def test_store_insert_stamps_the_sidecar(tmp_path):
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import MemoryStore
    from iai_mcp.store_watermark import read

    store = MemoryStore(path=tmp_path / "store")
    try:
        result = capture_turn(
            store, cue="", text="watermark stamp probe line one",
            tier="episodic", session_id="wm-sess", role="user",
        )
        assert result["status"] == "inserted"
        hippo_dir = getattr(store.db, "_hippo_dir")
        stamped = read(hippo_dir)
        assert stamped is not None, "insert must stamp the store-advance sidecar"
        datetime.fromisoformat(stamped)
    finally:
        store.close()


def _write_drainable(home: Path, session_id: str, text: str, role: str = "user") -> Path:
    deferred_dir = home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out = deferred_dir / f"{session_id}-{int(time.time())}.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "cwd": "/tmp",
    }
    event = {
        "text": text,
        "cue": f"session {session_id} cue",
        "tier": "episodic",
        "role": role,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    out.write_text(
        json.dumps(header, ensure_ascii=False) + "\n"
        + json.dumps(event, ensure_ascii=False) + "\n"
    )
    return out


def test_drain_stamps_sidecar_and_feeds_working_tier(tmp_path):
    from iai_mcp import working_tier
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.store import MemoryStore
    from iai_mcp.store_watermark import read

    _write_drainable(tmp_path, "drain-sess", "the drain must feed the working tier with this turn")

    store = MemoryStore(path=tmp_path / "store")
    try:
        counts = drain_deferred_captures(store)
        assert counts["events_inserted"] == 1, counts

        stamped = read(getattr(store.db, "_hippo_dir"))
        assert stamped is not None, "drained pending row must stamp the sidecar"

        entry = working_tier.read_task()
        assert entry is not None, "ambient drained turn must open/update the active task"
        assert "feed the working tier" in entry.goal

        snapshot = working_tier._cache_path(store, entry.session_id)
        assert snapshot.is_file(), "working-tier snapshot must persist for the per-turn hook"
    finally:
        store.close()


def test_drain_stashes_anchor_and_wake_refreshes_the_pack(tmp_path):
    from iai_mcp import foresight
    from iai_mcp.capture import capture_turn, drain_deferred_captures
    from iai_mcp.embed import embedder_for_store
    from iai_mcp.store import MemoryStore, flush_record_buffer

    store = MemoryStore(path=tmp_path / "store")
    try:
        # Prior-session memory so the pack has something related to serve.
        capture_turn(
            store, cue="", text="The hippocampus consolidates memories during sleep.",
            tier="episodic", session_id="past", role="user",
        )
        flush_record_buffer(store)

        _write_drainable(
            tmp_path, "live-sess",
            "How does sleep consolidation strengthen hippocampal memory?",
        )
        counts = drain_deferred_captures(store)
        assert counts["events_inserted"] == 1, counts

        # The drain itself never embeds: it stashes the newest user turn and
        # the wake-sequence caller pays the one embed.
        anchor = getattr(store, "_foresight_anchor", None)
        assert anchor is not None, "drain must stash the newest user turn"
        assert "sleep consolidation" in anchor[1]

        assert foresight.refresh_from_anchor(store, embedder_for_store(store))
        assert getattr(store, "_foresight_anchor", None) is None, "anchor is consumed"

        pack = foresight.pack_path(store, "live-sess")
        assert pack.is_file(), "the wake pass must refresh the next-turn pack"
        assert "hippocampus consolidates" in pack.read_text(encoding="utf-8")
    finally:
        store.close()


def test_replayed_history_never_touches_live_attention(tmp_path):
    """A drained turn with an old timestamp is history, not the active task:
    it must neither open/overwrite the working tier nor anchor the pack —
    otherwise a backlog replay injects a past conversation as the current one."""
    from iai_mcp import working_tier
    from iai_mcp.capture import drain_deferred_captures
    from iai_mcp.store import MemoryStore

    deferred_dir = tmp_path / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    out = deferred_dir / f"old-sess-{int(time.time())}.jsonl"
    header = {
        "version": 1,
        "deferred_at": datetime.now(timezone.utc).isoformat(),
        "session_id": "old-sess",
        "cwd": "/tmp",
    }
    event = {
        "text": "an ancient replayed turn that must stay out of live attention",
        "cue": "session old-sess cue",
        "tier": "episodic",
        "role": "user",
        "ts": "2026-01-01T00:00:00+00:00",
    }
    out.write_text(
        json.dumps(header, ensure_ascii=False) + "\n"
        + json.dumps(event, ensure_ascii=False) + "\n"
    )

    store = MemoryStore(path=tmp_path / "store")
    try:
        counts = drain_deferred_captures(store)
        assert counts["events_inserted"] == 1, counts

        assert working_tier.read_task() is None, (
            "replayed history must not open the active task"
        )
        assert not working_tier._cache_path(store).exists(), (
            "no live snapshot may be persisted from a replayed turn"
        )
        assert getattr(store, "_foresight_anchor", None) is None, (
            "a replayed turn must not anchor the next-turn pack"
        )
    finally:
        store.close()


def test_turn_hook_gate_reads_sidecar_not_the_store_file():
    repo = Path(__file__).resolve().parent.parent
    hook = repo / "src" / "iai_mcp" / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"
    src = hook.read_text(encoding="utf-8")
    assert ".max-created-at" in src
    assert "sqlite3" not in src, (
        "the hook must never open the store file — the engine format is not a hook contract"
    )
